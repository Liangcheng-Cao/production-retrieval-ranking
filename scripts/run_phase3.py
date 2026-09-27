"""Bounded offline reranking experiment; development selection precedes validation."""
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'artifacts/model_cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from product_search.development_data import BoundaryGuard,load_development
GUARD=BoundaryGuard(ROOT).install()
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits
from product_search.data.io import file_hash
from product_search.retrieval.common import Hit,product_text
from product_search.retrieval.lexical import BM25Retriever
from product_search.retrieval.dense import DenseRetriever
from product_search.retrieval.hybrid import HybridRetriever,rrf
from product_search.reranking import CrossEncoderReranker,TransformerPairScorer,validate_reranker_config
from product_search.evaluation import evaluate,qrels,query_metrics
from product_search.paired import paired_comparison

PLAN=json.loads((ROOT/'configs/phase3_plan.json').read_text())
SELECT2=json.loads((ROOT/'configs/phase2_selected.json').read_text())
REPORT=ROOT/'reports/phase3'
ART=ROOT/'artifacts/phase3'

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def initialize():
    torch.manual_seed(42); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    start=perf_counter()
    cfg=SELECT2['dense']['config']
    encoder=SentenceTransformer(cfg['model'],revision=cfg['revision'],device='cuda',trust_remote_code=False)
    encoder.max_seq_length=cfg['max_seq_length']; encoder.eval()
    dense=DenseRetriever.load(ROOT/'artifacts/phase2'/SELECT2['dense']['name'],cfg,encoder)
    bm25=BM25Retriever.load(ROOT/'artifacts/phase2'/SELECT2['bm25']['name'],SELECT2['bm25']['config'])
    hybrid=HybridRetriever(bm25,dense,SELECT2['hybrid']['config']['constant'],100)
    retrieval_ms=(perf_counter()-start)*1000
    start=perf_counter(); scorer=TransformerPairScorer(PLAN['model'],PLAN['revision'],PLAN['device']); scorer.synchronize()
    startup={'retrieval_startup_ms':retrieval_ms,'cross_encoder_startup_ms':(perf_counter()-start)*1000,
             'parameters':sum(p.numel() for p in scorer.model.parameters()),'model_class':type(scorer.model).__name__,
             'dtype':str(next(scorer.model.parameters()).dtype),'device':str(next(scorer.model.parameters()).device)}
    return hybrid,scorer,startup

def candidates(hybrid,data,partition):
    path=ART/f'candidates_{partition}.json'
    provenance={'phase1_sha256':file_hash(ROOT/'data/processed/data_manifest.json'),
                'phase2_sha256':file_hash(ROOT/'artifacts/phase2/manifest.json'),
                'selection_sha256':file_hash(ROOT/'configs/phase2_selected.json')}
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['provenance']!=provenance: raise ValueError('Candidate cache provenance differs')
        result={int(qid):[Hit(**h) for h in hits] for qid,hits in saved['hits'].items()}
        if set(result)!={q.query_id for q in data.queries}: raise ValueError('Candidate query coverage mismatch')
        return result
    # Same batched query encoding as Phase 2 quality evaluation.
    vectors=hybrid.dense.encoder.encode([q.query for q in data.queries],batch_size=128,
        normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    result={q.query_id:rrf([hybrid.lexical.search(q.query,100),hybrid.dense.search_vector(vectors[i],100)],100,hybrid.constant)
            for i,q in enumerate(data.queries)}
    dump(path,{'provenance':provenance,'hits':{qid:[asdict(h) for h in hits] for qid,hits in result.items()}})
    return result

def ids(hits): return {qid:[h.product_id for h in values] for qid,values in hits.items()}

def scores_to_ranking(candidates,scores,depth):
    chosen=candidates[:depth]
    return [h.product_id for h in sorted(chosen,key=lambda h:(-scores[h.product_id],h.rank,h.product_id))]

def quality(rankings,labels,depth):
    result=evaluate(rankings,labels)
    for cutoff in (50,100):
        if cutoff>depth:
            result['metrics'][f'Recall@{cutoff}']=None
            result['excluded_queries'][f'Recall@{cutoff}']=None
    result['returned_candidate_depth']=depth
    return result

def benchmark(hybrid,reranker,data,depth):
    queries=data.queries[:PLAN['latency']['query_count']]
    for query in queries[:PLAN['latency']['warmup']]:
        reranker.rerank(query.query,hybrid.search(query.query,100)[:depth],depth,fallback=False)
    observations=[]
    for _ in range(PLAN['latency']['repeats']):
        for query in queries:
            t=perf_counter(); hits=hybrid.search(query.query,100)[:depth]; retrieved=perf_counter()
            result=reranker.rerank(query.query,hits,depth,fallback=False)
            elapsed=(perf_counter()-t)*1000
            observations.append({**result.timings,'retrieval_ms':(retrieved-t)*1000,'pipeline_ms':elapsed})
    names=[name for name in observations[0] if name.endswith('_ms')]
    return {'samples':len(observations),'candidate_depth':depth,
            'effective_batch_sizes':observations[0]['batch_sizes'],
            'stages':{name:{'p50_ms':float(np.percentile([o[name] for o in observations],50)),
                            'p95_ms':float(np.percentile([o[name] for o in observations],95))} for name in names}}

def depth_recall_invariance(candidates,rankings,labels,depth):
    before=[]; after=[]
    for qid,ranking in rankings.items():
        original={h.product_id for h in candidates[qid][:depth]}
        if len(ranking)!=len(original) or set(ranking)!=original: raise ValueError('Reranking changed candidate membership')
        positives={pid for pid,g in labels[qid].items() if g>=1}
        if positives:
            before.append(len(original&positives)/len(positives))
            after.append(len(set(ranking)&positives)/len(positives))
    assert before==after
    return {'before':float(np.mean(before)),'after':float(np.mean(after)),'all_queries_invariant':True}

def development():
    if (ROOT/'configs/phase3_selected.json').exists(): raise ValueError('Phase 3 selection already frozen')
    data=load_development(ROOT/'data/processed','train'); labels=qrels(data.judgments)
    hybrid,scorer,startup=initialize(); pool=candidates(hybrid,data,'train')
    trials=[]
    for representation in PLAN['representations']:
        reranker=CrossEncoderReranker(scorer,data.products,representation,PLAN['batch_size'],PLAN['max_length'])
        pairs=[(q.query,product_text(reranker.products[h.product_id],representation)) for q in data.queries for h in pool[q.query_id]]
        truncation=scorer.truncation_stats(pairs,PLAN['max_length'])
        score_map={}
        for index,query in enumerate(data.queries):
            result=reranker.rerank(query.query,pool[query.query_id],100,fallback=False)
            score_map[query.query_id]={h.product_id:h.reranker_score for h in result.hits}
            if (index+1)%48==0: print(f'{representation}: scored {index+1}/{len(data.queries)} queries',flush=True)
        dump(ART/f'scores_train_{representation}.json',score_map)
        for depth in PLAN['candidate_depths']:
            rankings={qid:scores_to_ranking(hits,score_map[qid],depth) for qid,hits in pool.items()}
            trial={'representation':representation,'candidate_depth':depth,'batch_size':PLAN['batch_size'],
                   'quality':quality(rankings,labels,depth),'candidate_recall':depth_recall_invariance(pool,rankings,labels,depth),
                   'truncation_top100':truncation,'latency':benchmark(hybrid,reranker,data,depth)}
            trials.append(trial)
            dump(REPORT/'development.json',{'plan':PLAN,'startup':startup,'trials':trials,
                 'hybrid_baseline':evaluate(ids(pool),labels),
                 'depth_score_policy':'score each representation top-100 once; derive depth subsets from those logits; each latency trial performs real depth-specific inference'})
            print('TRIAL',representation,depth,json.dumps(trial['quality']['metrics']),trial['latency']['stages']['total_ms'],flush=True)
    eligible=[t for t in trials if t['latency']['stages']['total_ms']['p95_ms']<=PLAN['latency_budget_rerank_p95_ms']]
    if not eligible: raise ValueError('No reranking configuration meets predefined latency budget')
    selected=sorted(eligible,key=lambda t:(-t['quality']['metrics']['NDCG@10'],-t['quality']['metrics']['NDCG@20'],t['candidate_depth'],t['representation']))[0]
    dump(ROOT/'configs/phase3_selected.json',{'selected':selected,'plan_sha256':file_hash(ROOT/'configs/phase3_plan.json'),
         'phase1_manifest_sha256':file_hash(ROOT/'data/processed/data_manifest.json'),
         'phase2_manifest_sha256':file_hash(ROOT/'artifacts/phase2/manifest.json'),
         'selected_at_utc':datetime.now(timezone.utc).isoformat(),'rule':PLAN['selection_rule']})
    print('PHASE 3 DEVELOPMENT SELECTION FROZEN',flush=True)

def validation():
    if (REPORT/'validation.json').exists(): raise ValueError('Phase 3 validation already recorded')
    selection=json.loads((ROOT/'configs/phase3_selected.json').read_text())
    validate_reranker_config({'plan':file_hash(ROOT/'configs/phase3_plan.json'),'p1':file_hash(ROOT/'data/processed/data_manifest.json'),'p2':file_hash(ROOT/'artifacts/phase2/manifest.json')},
        {'plan':selection['plan_sha256'],'p1':selection['phase1_manifest_sha256'],'p2':selection['phase2_manifest_sha256']})
    chosen=selection['selected']; depth=chosen['candidate_depth']
    data=load_development(ROOT/'data/processed','validation'); labels=qrels(data.judgments)
    hybrid,scorer,startup=initialize(); pool=candidates(hybrid,data,'validation')
    baseline=evaluate(ids(pool),labels)
    old=json.loads((ROOT/'reports/phase2/validation.json').read_text())['pipelines']
    assert all(abs(value-old['Hybrid']['metrics'][name])<1e-10 for name,value in baseline['metrics'].items())
    reranker=CrossEncoderReranker(scorer,data.products,chosen['representation'],PLAN['batch_size'],PLAN['max_length'])
    rankings={}; results={}
    for query in data.queries:
        result=reranker.rerank(query.query,pool[query.query_id][:depth],depth,fallback=False)
        rankings[query.query_id]=[h.product_id for h in result.hits]
        results[query.query_id]=[asdict(h) for h in result.hits]
    comparison={}
    baseline_per={qid:query_metrics([h.product_id for h in hits],labels[qid]) for qid,hits in pool.items()}
    reranked_per={qid:query_metrics(ranking,labels[qid]) for qid,ranking in rankings.items()}
    for metric in ('NDCG@10','NDCG@20'):
        comparison[metric]=paired_comparison({qid:row[metric] for qid,row in baseline_per.items()},
            {qid:row[metric] for qid,row in reranked_per.items()},seed=PLAN['paired_bootstrap']['seed'],samples=PLAN['paired_bootstrap']['samples'])
    delta=sorted(labels,key=lambda qid:(reranked_per[qid]['NDCG@10']-baseline_per[qid]['NDCG@10'],qid))
    examples=[]; queries={q.query_id:q.query for q in data.queries}; products={p.product_id:p for p in data.products}
    for qid in delta[:3]+delta[-3:][::-1]:
        def row(pid,rank): return {'product_id':pid,'rank':rank,'label_grade':labels[qid].get(pid),'product_name':products[pid].product_name[:140]}
        examples.append({'query_id':qid,'query':queries[qid],'ndcg10_before':baseline_per[qid]['NDCG@10'],
            'ndcg10_after':reranked_per[qid]['NDCG@10'],'delta':reranked_per[qid]['NDCG@10']-baseline_per[qid]['NDCG@10'],
            'before':[row(h.product_id,i) for i,h in enumerate(pool[qid][:5],1)],
            'after':[row(pid,i) for i,pid in enumerate(rankings[qid][:5],1)]})
    pairs=[(q.query,product_text(products[h.product_id],chosen['representation'])) for q in data.queries for h in pool[q.query_id][:depth]]
    dump(ART/'rankings_validation.json',results)
    dump(REPORT/'paired.json',comparison)
    dump(REPORT/'error_examples.json',examples)
    dump(REPORT/'validation.json',{'partition':'validation','startup':startup,'candidate_depth':depth,
        'representation':chosen['representation'],'baselines_from_phase2':old,'hybrid_recomputed':baseline,
        'hybrid_cross_encoder':quality(rankings,labels,depth),'candidate_recall':depth_recall_invariance(pool,rankings,labels,depth),
        'truncation':scorer.truncation_stats(pairs,PLAN['max_length']),
        'selection_sha256':file_hash(ROOT/'configs/phase3_selected.json'),'fallback_count':0,
        'recorded_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'quality':quality(rankings,labels,depth),'paired':comparison},indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['development','validation']); args=parser.parse_args()
    try:
        with threadpool_limits(limits=1): (development if args.stage=='development' else validation)()
    finally:
        dump(REPORT/f'boundary_{args.stage}.json',{'opened_canonical_files':sorted(GUARD.opened),
             'blocked_attempts':GUARD.blocked,'test_labels_accessed':False,
             'phase1_manifest_sha256':file_hash(ROOT/'data/processed/data_manifest.json'),
             'phase2_manifest_sha256':file_hash(ROOT/'artifacts/phase2/manifest.json'),
             'shared_queries_policy':'hash/byte-ID scan only; no test query text decoding or inference'})

if __name__=='__main__': main()
