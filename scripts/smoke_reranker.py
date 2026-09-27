"""Opt-in real-model integration smoke test; synthetic products only, no WANDS labels."""
import json
import os
from pathlib import Path
from dataclasses import asdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'artifacts/model_cache'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from product_search.development_data import BoundaryGuard
GUARD=BoundaryGuard(ROOT).install()
from product_search.data.schema import Product
from product_search.retrieval.common import Hit
from product_search.reranking import CrossEncoderReranker,TransformerPairScorer

def main():
    plan=json.loads((ROOT/'configs/phase3_plan.json').read_text())
    scorer=TransformerPairScorer(plan['model'],plan['revision'],'cuda')
    texts=['red leather sofa','blue ceramic vase','gaming keyboard','red fabric sofa','black wooden dining table','outdoor metal chair']
    products=[Product(i,t,'','','','',None,None,None) for i,t in enumerate(texts)]
    hits=[Hit(i,1/(i+1),i+1,'synthetic') for i in range(len(products))]
    a=CrossEncoderReranker(scorer,products,batch_size=1).rerank('red leather sofa',hits,6,fallback=False)
    b=CrossEncoderReranker(scorer,products,batch_size=32).rerank('red leather sofa',hits,6,fallback=False)
    assert [h.product_id for h in a.hits]==[h.product_id for h in b.hits]
    assert np.allclose([h.reranker_score for h in a.hits],[h.reranker_score for h in b.hits],atol=1e-4,rtol=1e-4)
    q=['red sofa']; d=['very long product description '*400]
    x=scorer.tokenizer(q,d,truncation='longest_first',padding=True,max_length=256,return_tensors='pt')
    y=scorer.tokenizer(q,d,truncation='longest_first',padding=True,max_length=256,return_tensors='pt')
    assert x['input_ids'].shape[1]==256 and x['input_ids'].equal(y['input_ids'])
    report={'passed':True,'model':plan['model'],'revision':plan['revision'],'device':str(next(scorer.model.parameters()).device),
            'batched_ranking_equal':True,'scores_close_atol_rtol':1e-4,'batch1':a.timings['batch_sizes'],'batch32':b.timings['batch_sizes'],
            'real_tokenizer_truncation_deterministic':True,'truncated_length':256,
            'synthetic_only':True,'opened_canonical_files':sorted(GUARD.opened),'test_labels_accessed':False}
    path=ROOT/'reports/phase3/integration_smoke.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
