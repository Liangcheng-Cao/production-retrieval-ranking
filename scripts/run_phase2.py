"""Bounded development/validation retrieval experiment. Never loads test labels."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/model_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from product_search.development_data import BoundaryGuard, load_development
GUARD = BoundaryGuard(ROOT).install()
import numpy as np
import torch
import bm25s
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits
from product_search.data.io import file_hash
from product_search.retrieval.common import product_text, TEXT_VERSION, LEXICAL_VERSION
from product_search.retrieval.lexical import BM25Retriever
from product_search.retrieval.dense import DenseRetriever, normalize
from product_search.retrieval.hybrid import HybridRetriever, rrf
from product_search.evaluation import evaluate, qrels, complementarity

ART = ROOT / "artifacts/phase2"
REPORTS = ROOT / "reports/phase2"
PLAN = json.loads((ROOT / "configs/phase2_plan.json").read_text(encoding="utf-8"))
DATA_HASH = file_hash(ROOT / "data/processed/data_manifest.json")

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8", newline="\n")

def encoder():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SentenceTransformer(PLAN["model"], revision=PLAN["revision"], device=PLAN["device"], trust_remote_code=False)
    model.max_seq_length = PLAN["max_seq_length"]
    model.eval()
    if model.get_sentence_embedding_dimension() != PLAN["dimension"]:
        raise ValueError("Model dimension differs from frozen experiment plan")
    return model

def config(kind, representation, **kwargs):
    return {"kind":kind,"data_manifest_sha256":DATA_HASH,"text_version":TEXT_VERSION,
            "representation":representation,**kwargs}

def lexical(data, representation, k1, b):
    name=f"bm25_{representation}_{k1}_{b}"
    cfg=config("bm25",representation,k1=k1,b=b,implementation="bm25s",version=bm25s.__version__,
               method="lucene",lexical_version=LEXICAL_VERSION)
    path=ART/name
    start=perf_counter()
    if (path/'manifest.json').exists():
        result=BM25Retriever.load(path,cfg)
        build_ms=None
    else:
        result=BM25Retriever.build(data.products,representation,k1,b)
        build_ms=(perf_counter()-start)*1000
        result.save(path,cfg)
    return name,result,cfg,build_ms

def dense(data,representation,model):
    name=f"dense_{representation}"
    cfg=config("dense",representation,model=PLAN['model'],revision=PLAN['revision'],
               dimension=PLAN['dimension'],normalization="L2",pooling="attention-mask mean",
               max_seq_length=PLAN['max_seq_length'],dtype="float32",model_license=PLAN['license'],
               sentence_transformers=sentence_transformers.__version__)
    path=ART/name
    if (path/'manifest.json').exists():
        return name,DenseRetriever.load(path,cfg,model),cfg,None
    texts=[product_text(p,representation) for p in data.products]
    start=perf_counter()
    vectors=model.encode(texts,batch_size=PLAN['batch_size'],normalize_embeddings=True,
                         convert_to_numpy=True,show_progress_bar=True)
    elapsed=(perf_counter()-start)*1000
    result=DenseRetriever([p.product_id for p in data.products],normalize(vectors),model)
    result.save(path,cfg)
    return name,result,cfg,elapsed

def rank_queries(retriever, queries, vectors=None):
    return {q.query_id: (retriever.search(q.query,100) if vectors is None else retriever.search_vector(vectors[i],100))
            for i,q in enumerate(queries)}

def id_rankings(hits):
    return {qid:[h.product_id for h in rows] for qid,rows in hits.items()}

def choose(trials):
    return sorted(trials,key=lambda t:(-t['quality']['metrics']['Recall@100'],
                                      -t['quality']['metrics']['NDCG@20'],t['name']))[0]

def development():
    if (ROOT/'configs/phase2_selected.json').exists():
        raise ValueError("Development selection already frozen; do not silently retune")
    data=load_development(ROOT/'data/processed','train')
    labels=qrels(data.judgments)
    trials=[]; hit_cache={}; instances={}
    start=perf_counter(); model=encoder(); model_startup=(perf_counter()-start)*1000
    query_vectors=model.encode([q.query for q in data.queries],batch_size=128,
                               normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    model_info={'model':PLAN['model'],'revision':PLAN['revision'],'max_seq_length':model.max_seq_length,
                'dimension':model.get_sentence_embedding_dimension(),'modules':str(model),
                'first_model_load_including_download_ms':model_startup}
    dump(REPORTS/'model.json',model_info)
    def record(name,retriever,cfg,build_ms,vectors=None):
        hits=rank_queries(retriever,data.queries,vectors)
        trial={'name':name,'config':cfg,'build_ms':build_ms,'quality':evaluate(id_rankings(hits),labels)}
        hit_cache[name]=hits; instances[name]=retriever; trials.append(trial)
        dump(REPORTS/'development.json',{'plan':PLAN,'trials':trials})
        print(name,json.dumps(trial['quality']['metrics']),flush=True)
        return trial
    for representation in PLAN['representations']:
        args=lexical(data,representation,**PLAN['bm25_base'])
        record(*args)
    best_base=choose(trials)
    record(*lexical(data,best_base['config']['representation'],**PLAN['bm25_additional_on_best_representation']))
    best_lex=choose(trials)
    dense_trials=[]
    for representation in PLAN['representations']:
        dense_trials.append(record(*dense(data,representation,model),vectors=query_vectors))
    best_dense=choose(dense_trials)
    hybrid_trials=[]
    for constant in PLAN['rrf_constants']:
        name=f"hybrid_rrf_{constant}"
        hits={qid:rrf([hit_cache[best_lex['name']][qid],hit_cache[best_dense['name']][qid]],100,constant) for qid in labels}
        cfg={'method':'rrf','constant':constant,'candidate_depth':100,'weights':[1,1],
             'bm25':best_lex['name'],'dense':best_dense['name']}
        trial={'name':name,'config':cfg,'build_ms':None,'quality':evaluate(id_rankings(hits),labels)}
        trials.append(trial); hybrid_trials.append(trial)
        print(name,json.dumps(trial['quality']['metrics']),flush=True)
    selected={'selection_rule':PLAN['selection'],'selected_at_utc':datetime.now(timezone.utc).isoformat(),
              'plan_sha256':file_hash(ROOT/'configs/phase2_plan.json'),'data_manifest_sha256':DATA_HASH,
              'bm25':best_lex,'dense':best_dense,'hybrid':choose(hybrid_trials)}
    dump(REPORTS/'development.json',{'plan':PLAN,'trials':trials})
    dump(ROOT/'configs/phase2_selected.json',selected)
    print('DEVELOPMENT SELECTION FROZEN',flush=True)

def latency(selected,model):
    data=load_development(ROOT/'data/processed','train')
    start=perf_counter(); bm=BM25Retriever.load(ART/selected['bm25']['name'],selected['bm25']['config']); bm_start=(perf_counter()-start)*1000
    start=perf_counter(); de=DenseRetriever.load(ART/selected['dense']['name'],selected['dense']['config'],model); de_start=(perf_counter()-start)*1000
    hy=HybridRetriever(bm,de,selected['hybrid']['config']['constant'],100)
    queries=[q.query for q in data.queries[:PLAN['latency']['query_count']]]
    output={}
    for name,retriever in [('BM25',bm),('Dense',de),('Hybrid',hy)]:
        for query in queries[:PLAN['latency']['warmup']]: retriever.search(query,100)
        measurements=[]
        for _ in range(PLAN['latency']['repeats']):
            for query in queries:
                if name=='BM25':
                    begin=perf_counter(); retriever.search(query,100); elapsed=(perf_counter()-begin)*1000
                    stages={'search_ms':elapsed,'total_ms':elapsed}
                else:
                    _,stages=retriever.search_timed(query,100)
                measurements.append(stages)
        output[name]={key:{'p50_ms':float(np.percentile([m[key] for m in measurements],50)),
                           'p95_ms':float(np.percentile([m[key] for m in measurements],95))} for key in measurements[0]}
        output[name]['sample_count']=len(measurements)
    return {'protocol':{**PLAN['latency'],'top_k':100,'query_ids':[q.query_id for q in data.queries[:len(queries)]],
                        'scope':'single-process sequential warm retrieval; no QPS or production claim'},
            'startup_ms':{'bm25_artifact_load_and_hash':bm_start,'dense_artifact_load_hash_norm_check':de_start},'pipelines':output}

def validation():
    if (REPORTS/'validation.json').exists():
        raise ValueError("Validation already recorded; no silent repeated selection")
    selected=json.loads((ROOT/'configs/phase2_selected.json').read_text(encoding='utf-8'))
    if selected['data_manifest_sha256']!=DATA_HASH or selected['plan_sha256']!=file_hash(ROOT/'configs/phase2_plan.json'):
        raise ValueError("Selection/data/plan mismatch")
    start=perf_counter(); model=encoder(); cached_model_ms=(perf_counter()-start)*1000
    data=load_development(ROOT/'data/processed','validation'); labels=qrels(data.judgments)
    bm=BM25Retriever.load(ART/selected['bm25']['name'],selected['bm25']['config'])
    de=DenseRetriever.load(ART/selected['dense']['name'],selected['dense']['config'],model)
    query_vectors=model.encode([q.query for q in data.queries],batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    a=rank_queries(bm,data.queries); b=rank_queries(de,data.queries,query_vectors)
    c={qid:rrf([a[qid],b[qid]],100,selected['hybrid']['config']['constant']) for qid in labels}
    rankings={name:id_rankings(hits) for name,hits in [('BM25',a),('Dense',b),('Hybrid',c)]}
    quality={name:evaluate(ranks,labels) for name,ranks in rankings.items()}
    dump(REPORTS/'validation.json',{'partition':'validation','selection_sha256':file_hash(ROOT/'configs/phase2_selected.json'),
         'pipelines':quality,'measured_at_utc':datetime.now(timezone.utc).isoformat()})
    dump(REPORTS/'complementarity.json',{'partition':'validation','depths':complementarity(rankings['BM25'],rankings['Dense'],rankings['Hybrid'],labels)})
    timing=latency(selected,model)
    timing['startup_ms']['model_from_local_cache_to_gpu']=cached_model_ms
    timing['startup_ms']['dense_total']=cached_model_ms+timing['startup_ms']['dense_artifact_load_hash_norm_check']
    timing['startup_ms']['hybrid_total']=timing['startup_ms']['dense_total']+timing['startup_ms']['bm25_artifact_load_and_hash']
    dump(REPORTS/'latency.json',timing)
    print(json.dumps(quality,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['development','validation']); args=parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            torch.set_num_threads(1)
            (development if args.stage=='development' else validation)()
    finally:
        dump(REPORTS/f'boundary_{args.stage}.json',{'opened_canonical_files':sorted(GUARD.opened),
             'blocked_attempts':GUARD.blocked,'test_labels_accessed':False,
             'query_text_policy':'shared file hash/byte ID scan; only requested train/validation rows JSON-decoded, encoded or evaluated',
             'frozen_manifest_sha256_after':file_hash(ROOT/'data/processed/data_manifest.json')})

if __name__=='__main__': main()
