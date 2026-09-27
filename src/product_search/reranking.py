"""Candidate-only cross-encoding, stable ordering and explicit fallback contract."""
from dataclasses import dataclass
import math
from time import perf_counter
import numpy as np
from .retrieval.common import check_request, product_text

@dataclass(frozen=True)
class RankedHit:
    product_id: int
    reranker_score: float | None
    final_rank: int
    retrieval_score: float
    retrieval_rank: int
    retrieval_source: str

@dataclass(frozen=True)
class RerankResult:
    hits: tuple[RankedHit, ...]
    fallback_used: bool
    error_type: str | None
    timings: dict

class CrossEncoderReranker:
    def __init__(self, scorer, products, representation="A", batch_size=32, max_length=256, max_candidates=100):
        if representation not in ("A","B") or any(type(v) is not int or v <= 0 for v in (batch_size,max_length,max_candidates)):
            raise ValueError("Invalid reranker configuration")
        self.scorer=scorer
        self.products={p.product_id:p for p in products}
        if len(self.products)!=len(products): raise ValueError("Duplicate product IDs")
        self.representation=representation; self.batch_size=batch_size
        self.max_length=max_length; self.max_candidates=max_candidates

    def rerank(self, query, candidates, top_k=10, *, fallback=True):
        check_request(query,top_k)
        candidates=list(candidates)
        if len(candidates)>self.max_candidates or top_k>self.max_candidates:
            raise ValueError("Candidate depth exceeds configured maximum")
        ids=[h.product_id for h in candidates]
        if len(ids)!=len(set(ids)) or any(pid not in self.products for pid in ids):
            raise ValueError("Invalid candidate membership")
        if any(h.rank!=rank or not math.isfinite(h.score) for rank,h in enumerate(candidates,1)):
            raise ValueError("Candidates must preserve contiguous retrieval ranks and finite scores")
        if not candidates: return RerankResult((),False,None,{'total_ms':0.0,'postprocess_ms':0.0})
        if not query.strip(): raise ValueError("Nonempty query required for reranking")
        start=perf_counter()
        try:
            pairs=[(query,product_text(self.products[h.product_id],self.representation)) for h in candidates]
            scores,timings=self.scorer.score_pairs(pairs,self.batch_size,self.max_length)
            scores=np.asarray(scores)
            if scores.shape!=(len(candidates),) or not np.isfinite(scores).all():
                raise ValueError("Malformed model output")
            post=perf_counter()
            order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),candidates[i].rank,candidates[i].product_id))
            hits=tuple(RankedHit(candidates[i].product_id,float(scores[i]),rank,
                candidates[i].score,candidates[i].rank,candidates[i].source) for rank,i in enumerate(order[:top_k],1))
            end=perf_counter()
            return RerankResult(hits,False,None,{**timings,'postprocess_ms':(end-post)*1000,'total_ms':(end-start)*1000})
        except Exception as exc:
            if not fallback: raise
            # Never present retrieval scores as CE logits. Do not swallow input validation errors.
            hits=tuple(RankedHit(h.product_id,None,rank,h.score,h.rank,h.source) for rank,h in enumerate(candidates[:top_k],1))
            return RerankResult(hits,True,type(exc).__name__,{'total_ms':(perf_counter()-start)*1000})

class TransformerPairScorer:
    """Raw query-document relevance logits using the pretrained sequence classifier."""
    def __init__(self, model_name, revision, device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch=torch
        self.device=device
        self.model_name=model_name; self.revision=revision
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name,revision=revision,trust_remote_code=False)
        self.model.to(device=device,dtype=torch.float32).eval()
        if self.model.config.num_labels!=1: raise ValueError("Single relevance logit model required")
        self.tokenizer.truncation_side='right'

    def synchronize(self):
        if str(self.device).startswith('cuda'): self.torch.cuda.synchronize()

    def score_pairs(self,pairs,batch_size,max_length):
        if type(batch_size) is not int or batch_size<=0 or type(max_length) is not int or max_length<=0:
            raise ValueError("Invalid batching/truncation configuration")
        timings={'tokenization_ms':0.0,'transfer_ms':0.0,'gpu_forward_ms':0.0,'batch_sizes':[]}
        scores=[]
        for offset in range(0,len(pairs),batch_size):
            batch=pairs[offset:offset+batch_size]; timings['batch_sizes'].append(len(batch))
            t=perf_counter()
            encoded=self.tokenizer([q for q,d in batch],[d for q,d in batch],padding=True,
                                   truncation='longest_first',max_length=max_length,return_tensors='pt')
            timings['tokenization_ms']+=(perf_counter()-t)*1000
            t=perf_counter(); encoded={k:v.to(self.device) for k,v in encoded.items()}; self.synchronize()
            timings['transfer_ms']+=(perf_counter()-t)*1000
            t=perf_counter()
            with self.torch.inference_mode(): logits=self.model(**encoded).logits
            self.synchronize(); timings['gpu_forward_ms']+=(perf_counter()-t)*1000
            t=perf_counter(); values=logits.detach().cpu().numpy()
            timings['transfer_ms']+=(perf_counter()-t)*1000
            if values.shape!=(len(batch),1): raise ValueError("Malformed classifier logits")
            scores.extend(values[:,0].tolist())
        return np.asarray(scores,dtype=np.float32),timings

    def truncation_stats(self,pairs,max_length):
        lengths=[]
        for offset in range(0,len(pairs),128):
            batch=pairs[offset:offset+128]
            encoded=self.tokenizer([q for q,d in batch],[d for q,d in batch],padding=False,truncation=False)
            lengths.extend(len(x) for x in encoded['input_ids'])
        return {'pairs':len(lengths),'pairs_over_max_length':sum(n>max_length for n in lengths),
                'max_untruncated_length':max(lengths,default=0),'max_length':max_length}

def validate_reranker_config(actual,expected):
    if actual!=expected: raise ValueError("Reranker manifest/config mismatch")
