"""Independent DCG/IDCG arithmetic audit; no model selection or test access."""
import json
import math
from pathlib import Path
from product_search.development_data import BoundaryGuard, load_development
from product_search.retrieval.lexical import BM25Retriever
from product_search.evaluation import qrels, query_metrics

ROOT=Path(__file__).resolve().parents[1]

def main():
    guard=BoundaryGuard(ROOT).install()
    selected=json.loads((ROOT/'configs/phase2_selected.json').read_text())['bm25']
    data=load_development(ROOT/'data/processed','validation')
    labels=qrels(data.judgments)
    model=BM25Retriever.load(ROOT/'artifacts/phase2'/selected['name'],selected['config'])
    calculations=[]; values={10:[],20:[]}; with_hits=0
    for query in data.queries:
        hits=model.search(query.query,20)
        grades=[labels[query.query_id].get(h.product_id,0) for h in hits]
        ideal=sorted(labels[query.query_id].values(),reverse=True)
        with_hits+=any(g>0 for g in grades[10:20])
        example={'query_id':query.query_id,'grades_at_ranks_1_20':grades,'positive_hits_at_11_20':sum(g>0 for g in grades[10:20]),'cutoffs':{}}
        for cutoff in (10,20):
            # Independent scalar arithmetic, no evaluator helper for DCG/IDCG.
            terms=[(pow(2,g)-1)/math.log(rank+1,2) for rank,g in enumerate(grades[:cutoff],1)]
            ideal_terms=[(pow(2,g)-1)/math.log(rank+1,2) for rank,g in enumerate(ideal[:cutoff],1)]
            dcg,idcg=math.fsum(terms),math.fsum(ideal_terms)
            result=dcg/idcg if idcg else None
            reference=query_metrics([h.product_id for h in hits],labels[query.query_id])[f'NDCG@{cutoff}']
            assert result is not None and abs(result-reference)<1e-12
            values[cutoff].append(result)
            example['cutoffs'][str(cutoff)]={'dcg_terms':terms,'idcg_terms':ideal_terms,'dcg':dcg,'idcg':idcg,'ndcg':result}
        if len(calculations)<5: calculations.append(example)
    measured={f'NDCG@{k}':math.fsum(v)/len(v) for k,v in values.items()}
    old=json.loads((ROOT/'reports/phase2/validation.json').read_text())['pipelines']['BM25']['metrics']
    assert all(abs(measured[k]-old[k])<1e-12 for k in measured)
    result={'correct':True,'historical_phase2_numbers_changed':False,'query_count':len(data.queries),
            'measured':measured,'difference_ndcg20_minus_ndcg10':measured['NDCG@20']-measured['NDCG@10'],
            'queries_with_relevant_at_11_20':with_hits,'manual_arithmetic_examples':calculations,
            'finding':'Four-decimal rounding coincidence; DCG and IDCG both use actual cutoff. Independent sums match all 96 query metrics.',
            'opened_files':sorted(guard.opened),'blocked_attempts':guard.blocked,'test_labels_accessed':False}
    target=ROOT/'reports/phase3/metric_sanity.json'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='manual_arithmetic_examples'},indent=2))
    print(json.dumps([{'qid':e['query_id'],**{k:{n:v for n,v in c.items() if n in ('dcg','idcg','ndcg')} for k,c in e['cutoffs'].items()}} for e in calculations],indent=2))

if __name__=='__main__': main()
