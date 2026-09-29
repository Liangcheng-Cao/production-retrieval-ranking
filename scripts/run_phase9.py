"""One frozen final evaluation, at most one identical rerun, no model selection.

Stages: preflight (train only), freeze (no test labels), primary, rerun, seal.
Existing metric/retrieval/reranking implementations are imported unchanged.
All final-label opens are guarded and recorded before the OS open takes place.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/phase9'
TMP = ROOT/'reports/tmp/phase9'
PROTOCOL = ROOT/'configs/phase9_protocol.json'
METRICS = ('Recall@10','Recall@20','Recall@50','Recall@100','NDCG@10','NDCG@20')
PIPELINES = ('BM25','Dense','Hybrid','Hybrid + CE-20')
DETERMINISTIC = ('final_quality.json','per_query_final.json','paired_final.json',
                 'paired_deltas_final.json','complementarity_final.json','ranking_diagnostics_final.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    return sha(path)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()


def historical_integrity():
    baseline=read(OUT/'starting_baseline.json')
    assert git('rev-parse','HEAD')==baseline['git_head']
    changed=[p for p,h in baseline['tracked_file_sha256'].items() if sha(ROOT/p)!=h]
    assert not changed, f'Historical frozen file changed: {changed}'
    return baseline


def reference(path):
    return {'path':Path(path).relative_to(ROOT).as_posix(),'sha256':sha(path)}


class LabelGuard:
    def __init__(self, root=ROOT):
        self.root=Path(root).resolve()
        self.allowed=False
        self.callback=None
        self.opens=0
        self.blocked=[]

    def check(self,event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(self.root):
            return
        relative=path.relative_to(self.root).as_posix()
        if (relative.startswith('data/raw/') and relative!='data/raw/.gitkeep') or path.name=='judgment_conflicts.jsonl':
            self.blocked.append(relative)
            raise PermissionError('Raw data and global conflicts are outside Phase 9 scope')
        if path==self.root/'data/processed/judgments.test.jsonl':
            if not self.allowed or self.opens:
                self.blocked.append(relative)
                raise PermissionError('Final labels require a sealed freeze and one authorized open per run')
            self.opens+=1
            self.callback()


def selected_queries(partition, limit=None):
    manifest=read(ROOT/'data/processed/data_manifest.json')
    ids=set(manifest['query_ids'][partition])
    if limit is not None:
        ids=set(sorted(ids)[:limit])
    path=ROOT/'data/processed/queries.jsonl'
    assert sha(path)==manifest['artifacts']['queries.jsonl']['sha256']
    rows={}
    with path.open('rb') as handle:
        for line in handle:
            match=re.search(rb'"query_id":([0-9]+)(?:,|})',line)
            assert match
            qid=int(match[1])
            if qid in ids:
                rows[qid]=json.loads(line)['query']
    assert set(rows)==ids
    return dict(sorted(rows.items()))


def labels_from_bytes(payload, ids):
    from product_search.data.schema import Judgment
    from product_search.evaluation import qrels
    rows=[Judgment(**json.loads(line)) for line in payload.splitlines() if line.strip()]
    assert all(j.query_id in ids for j in rows)
    labels=qrels(rows)
    return {qid:labels.get(qid,{}) for qid in sorted(ids)},len(rows)


def predict(queries, check_core=False):
    from product_search.deployment import configure_process
    from product_search.search_engine import SearchEngine
    from product_search.retrieval.hybrid import rrf
    from threadpoolctl import threadpool_limits
    configure_process()
    rankings={p:{} for p in PIPELINES}
    scores={p:{} for p in PIPELINES}
    with threadpool_limits(limits=1):
        engine=SearchEngine.from_config(ROOT/'configs/runtime.json')
        try:
            r=engine._resources
            assert r.candidate_depth==100 and r.reranker_depth==20
            artifacts=dict(engine.version['validated_artifact_sha256'])
            for qid,query in queries.items():
                bm=r.bm25.search(query,100)
                dense=r.hybrid.dense.search(query,100)
                hybrid=rrf([bm,dense],100,r.hybrid.constant)
                ce=r.reranker.rerank(query,hybrid[:20],20,fallback=False)
                assert not ce.fallback_used and len(ce.hits)<=20
                assert {h.product_id for h in ce.hits}=={h.product_id for h in hybrid[:20]}
                for mode,hits in zip(PIPELINES,(bm,dense,hybrid,ce.hits)):
                    rankings[mode][qid]=[h.product_id for h in hits]
                    scores[mode][qid]=[float(h.reranker_score if mode==PIPELINES[-1] else h.score) for h in hits]
                if check_core:
                    for mode,pipeline,k in [('BM25','bm25',100),('Hybrid','hybrid',100),('Hybrid + CE-20','hybrid_rerank',20)]:
                        result=engine.search(query,k,pipeline)
                        assert not result.fallback_used
                        assert [h.product_id for h in result.results]==rankings[mode][qid]
            environment={'device':engine.config.device,'software':dict(engine.version['software']),
                         'numerical_runtime':dict(engine.version['numerical_runtime']),
                         'runtime_manifest_sha256':engine.config.manifest_sha256}
        finally:
            engine.close()
    return rankings,scores,artifacts,environment


def summarize(rankings,labels,protocol):
    import numpy as np
    from product_search.evaluation import evaluate,query_metrics,complementarity
    from product_search.paired import paired_comparison
    from product_search.monitoring import ranking_difference
    per={p:{qid:{m:v for m,v in query_metrics(ids,labels[qid]).items() if m in METRICS}
            for qid,ids in rows.items()} for p,rows in rankings.items()}
    quality={}
    for mode in PIPELINES:
        aggregated=evaluate(rankings[mode],labels)
        quality[mode]={'query_count':aggregated['query_count'],
            'metrics':{m:aggregated['metrics'][m] for m in METRICS},
            'excluded_queries':{m:aggregated['excluded_queries'][m] for m in METRICS}}
        if mode==PIPELINES[-1]:
            for metric in ('Recall@50','Recall@100'):
                quality[mode]['metrics'][metric]=None
                quality[mode]['excluded_queries'][metric]=None
                for row in per[mode].values():row[metric]=None
        for metric in METRICS:
            values=[row[metric] for row in per[mode].values() if row[metric] is not None]
            actual=quality[mode]['metrics'][metric]
            assert (actual is None and not values) or abs(actual-math.fsum(values)/len(values))<1e-12
    assert all(per['Hybrid'][qid]['Recall@20']==per[PIPELINES[-1]][qid]['Recall@20'] for qid in labels)
    paired={};deltas={}
    for metric in ('NDCG@10','NDCG@20'):
        a={qid:row[metric] for qid,row in per['Hybrid'].items()}
        b={qid:row[metric] for qid,row in per[PIPELINES[-1]].items()}
        paired[metric]=paired_comparison(a,b,seed=protocol['bootstrap']['seed'],samples=protocol['bootstrap']['samples'])
        deltas[metric]={qid:b[qid]-a[qid] for qid in a if a[qid] is not None and b[qid] is not None}
        assert sum(paired[metric][k] for k in ('improved','unchanged','worsened'))==len(deltas[metric])
        assert abs(paired[metric]['mean_delta']-math.fsum(deltas[metric].values())/len(deltas[metric]))<1e-12
    differences=[ranking_difference(rankings['Hybrid'][qid],rankings[PIPELINES[-1]][qid],10) for qid in labels]
    displacements=[r['shared_mean_rank_displacement'] for r in differences if r['shared_mean_rank_displacement'] is not None]
    diagnostics={'query_count':len(labels),'ordering_changed_fraction':float(np.mean([r['ordering_changed'] for r in differences])),
        'mean_overlap_at_10':float(np.mean([r['overlap_at_k'] for r in differences])),
        'mean_shared_rank_displacement':float(np.mean(displacements)) if displacements else None,
        'shared_displacement_excluded_queries':len(labels)-len(displacements)}
    return {'final_quality.json':{'partition':'test','test_query_count':len(labels),'pipelines':quality},
        'per_query_final.json':per,'paired_final.json':paired,'paired_deltas_final.json':deltas,
        'complementarity_final.json':complementarity(rankings['BM25'],rankings['Dense'],rankings['Hybrid'],labels),
        'ranking_diagnostics_final.json':diagnostics}


def preflight():
    from product_search.development_data import load_development
    from product_search.evaluation import qrels
    historical_integrity()
    protocol=read(PROTOCOL)
    queries=selected_queries('train',6)
    rankings,scores,artifacts,environment=predict(queries,check_core=True)
    data=load_development(ROOT/'data/processed','train')
    all_labels=qrels(data.judgments)
    results=summarize(rankings,{qid:all_labels.get(qid,{}) for qid in queries},protocol)
    # Exercise deterministic serialization and independent aggregation with train only.
    assert digest(results)==digest(summarize(rankings,{qid:all_labels.get(qid,{}) for qid in queries},protocol))
    tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True,check=True)
    pip=subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,capture_output=True,text=True,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    write_once(OUT/'preflight.json',{'passed':True,'created_utc':now(),'partition':'train','query_count':len(queries),
        'production_core_ordering_parity_comparisons':3*len(queries),'pytest':tests.stdout,'pip_check':pip.stdout,
        'validated_artifact_sha256':artifacts,'environment':environment,'final_test_labels_accessed':False,
        'source_sha256':{p:sha(ROOT/p) for p in ('scripts/run_phase9.py','tests/test_phase9.py','configs/phase9_protocol.json')},
        'git_status':git('status','--short')})


def freeze():
    baseline=historical_integrity()
    pre=read(OUT/'preflight.json');assert pre['passed']
    for p,h in pre['source_sha256'].items():assert sha(ROOT/p)==h
    p1=read(ROOT/'data/processed/data_manifest.json')
    s2=read(ROOT/'configs/phase2_selected.json');p3=read(ROOT/'configs/phase3_plan.json');s3=read(ROOT/'configs/phase3_selected.json')['selected']
    protocol=read(PROTOCOL)
    assert protocol['bootstrap']['seed']==p3['paired_bootstrap']['seed']==42
    assert protocol['bootstrap']['samples']==p3['paired_bootstrap']['samples']==10000
    assert protocol['relevance_mapping']==p1['relevance_mapping'] and protocol['ndcg_gain']==p1['ndcg_gain']
    assert s3['candidate_depth']==protocol['ce_depth']==20
    files=git('ls-files','--cached','--others','--exclude-standard').splitlines()
    code={p:sha(ROOT/p) for p in files if p.startswith(('src/','scripts/','tests/','configs/','deployment/')) or p in ('pyproject.toml','Dockerfile','compose.yaml','.dockerignore')}
    freeze_record={'created_utc':now(),'created_monotonic_ns':time.monotonic_ns(),'git_head':git('rev-parse','HEAD'),
        'git_branch':git('branch','--show-current'),'working_tree_clean':not bool(git('status','--porcelain')),
        'clean_phase8_starting_baseline':reference(OUT/'starting_baseline.json'),
        'uncommitted_phase9_additions_authorized':True,'git_status':git('status','--short'),
        'preaccess_file_sha256':{p:sha(ROOT/p) for p in files},'source_config_sha256':code,
        'phase1_data_manifest_sha':sha(ROOT/'data/processed/data_manifest.json'),
        'phase2_retrieval_manifest_sha':sha(ROOT/'artifacts/phase2/manifest.json'),
        'phase3_reranking_manifest_sha':sha(ROOT/'artifacts/phase3/manifest.json'),
        'phase4_runtime_manifest_sha':sha(ROOT/'artifacts/phase4/manifest.json'),
        'phase8_deployment_evidence_identity':reference(ROOT/'reports/phase8/docker_runtime.json'),
        'docker_image_id':read(ROOT/'reports/phase8/docker_runtime.json')['validation']['image_id'],
        'selected_bm25_config':s2['bm25']['config'],'selected_dense_model':s2['dense']['config']['model'],
        'selected_dense_revision':s2['dense']['config']['revision'],'selected_dense_representation':s2['dense']['config']['representation'],
        'selected_hybrid_config':s2['hybrid']['config'],'selected_ce_model':p3['model'],'selected_ce_revision':p3['revision'],
        'selected_ce_depth':s3['candidate_depth'],'selected_ce_representation':s3['representation'],
        'relevance_mapping':p1['relevance_mapping'],'recall_positive_definition':p1['positive_labels'],
        'ndcg_gain_definition':p1['ndcg_gain'],'bootstrap_seed':42,'bootstrap_iterations':10000,
        'metric_protocol_version':protocol['metric_protocol_version'],'protocol':reference(PROTOCOL),
        'test_query_count':len(p1['query_ids']['test']),'test_query_ids':p1['query_ids']['test'],
        'test_partition_checksum':p1['partition_checksums']['test'],'test_artifact_expected':p1['artifacts']['judgments.test.jsonl'],
        'pipelines_to_evaluate':list(PIPELINES),'authorization_timestamp':baseline['authorization_received_at_utc'],
        'authorization_timestamp_semantics':baseline['authorization_timestamp_semantics'],
        'artifact_sha256':pre['validated_artifact_sha256'],'preflight':reference(OUT/'preflight.json')}
    write_once(OUT/'freeze_record.json',freeze_record)
    write_once(OUT/'freeze_record_seal.json',{'freeze_record':reference(OUT/'freeze_record.json'),
        'write_completed_utc':now(),'write_completed_monotonic_ns':time.monotonic_ns(),
        'freeze_file_mtime_ns':(OUT/'freeze_record.json').stat().st_mtime_ns,'final_test_labels_opened':False})
    verify_freeze()


def verify_freeze():
    historical_integrity()
    seal=read(OUT/'freeze_record_seal.json')
    assert sha(OUT/'freeze_record.json')==seal['freeze_record']['sha256']
    record=read(OUT/'freeze_record.json')
    assert record['git_head']==git('rev-parse','HEAD')
    for p,h in record['preaccess_file_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in record['artifact_sha256'].items():assert sha(ROOT/p)==h,p
    # Forbid adding new executable/configuration files after the freeze as well.
    files=git('ls-files','--cached','--others','--exclude-standard').splitlines()
    current={p:sha(ROOT/p) for p in files if p.startswith(('src/','scripts/','tests/','configs/','deployment/')) or p in ('pyproject.toml','Dockerfile','compose.yaml','.dockerignore')}
    assert current==record['source_config_sha256']
    return record


def final_run(kind,guard):
    record=verify_freeze()
    target=OUT if kind=='primary' else TMP/'rerun'
    if kind=='rerun':
        assert read(OUT/'primary_seal.json')['run']=='primary'
        verify_seal(OUT/'primary_seal.json')
    write_once(OUT/(kind+'_started.json'),{'run':kind,'started_utc':now(),'git_head':record['git_head'],'freeze_record_sha256':sha(OUT/'freeze_record.json')})
    queries=selected_queries('test')
    assert list(queries)==sorted(record['test_query_ids'])
    rankings,scores,artifacts,environment=predict(queries)
    assert artifacts==record['artifact_sha256']
    raw=TMP/kind
    write_once(raw/'rankings.json',rankings)
    write_once(raw/'scores.json',scores)
    # Freeze and all executable bytes are rechecked immediately before first labels.
    verify_freeze()
    expected=record['test_artifact_expected']
    def audit_before_open():
        seal=read(OUT/'freeze_record_seal.json')
        event={'event':'python_audit_open_before_os_open','timestamp':now(),'monotonic_ns':time.monotonic_ns(),
            'script':'scripts/run_phase9.py','script_sha256':sha(Path(__file__)),'git_head':git('rev-parse','HEAD'),
            'freeze_record_sha':sha(OUT/'freeze_record.json'),'freeze_write_completed_utc':seal['write_completed_utc'],
            'freeze_write_monotonic_ns':seal['write_completed_monotonic_ns'],
            'test_artifact_path':'data/processed/judgments.test.jsonl','test_artifact_expected_sha256':expected['sha256'],
            'query_count':len(queries),'expected_label_row_count':expected['rows'],'run':kind,'test_set_spent':True}
        assert event['monotonic_ns']>seal['write_completed_monotonic_ns']
        assert event['timestamp']>seal['write_completed_utc']
        write_once(target/'label_open_event.json',event)
    guard.callback=audit_before_open;guard.allowed=True
    try:payload=(ROOT/'data/processed/judgments.test.jsonl').read_bytes()
    finally:guard.allowed=False
    observed=hashlib.sha256(payload).hexdigest();assert observed==expected['sha256']
    labels,count=labels_from_bytes(payload,set(queries));assert count==expected['rows']
    del payload
    write_once(target/'test_access_audit.json',{'first_access_event':read(target/'label_open_event.json'),
        'read_complete_utc':now(),'test_artifact_sha256':observed,'label_row_count':count,'query_count':len(labels),
        'label_file_open_count_this_process':guard.opens,'blocked_attempts':guard.blocked,'rows_decoded_only_by_evaluator':True})
    results=summarize(rankings,labels,read(PROTOCOL))
    for name,value in results.items():write_once(target/name,value)
    write_once(target/'run_environment.json',environment)
    write_once(target/('primary_seal.json' if kind=='primary' else 'rerun_seal.json'),{
        'run':kind,'sealed_utc':now(),'git_head':git('rev-parse','HEAD'),'freeze_record_sha256':sha(OUT/'freeze_record.json'),
        'files':{name:sha(target/name) for name in (*DETERMINISTIC,'test_access_audit.json','label_open_event.json','run_environment.json')},
        'raw_files':{p.relative_to(ROOT).as_posix():sha(p) for p in (raw/'rankings.json',raw/'scores.json')}})
    if kind=='primary':compare_validation()
    else:compare_rerun()
    print(json.dumps({'run':kind,'completed':True,'query_count':len(labels),'label_rows':count}))


def verify_seal(path):
    record=read(path)
    for p,h in record['files'].items():assert sha(Path(path).parent/p)==h
    for p,h in record.get('raw_files',{}).items():assert sha(ROOT/p)==h
    return record


def compare_validation():
    primary=verify_seal(OUT/'primary_seal.json')
    began=now();assert began>primary['sealed_utc']
    test=read(OUT/'final_quality.json')['pipelines']
    old=read(ROOT/'reports/phase2/validation.json')['pipelines']
    old[PIPELINES[-1]]=read(ROOT/'reports/phase3/validation.json')['hybrid_cross_encoder']
    comparison={}
    for p in PIPELINES:
        comparison[p]={}
        for metric in ('Recall@20','Recall@100','NDCG@10','NDCG@20'):
            a=old[p]['metrics'][metric];b=test[p]['metrics'][metric]
            comparison[p][metric]={'validation':a,'final_test':b,'delta_test_minus_validation':None if a is None or b is None else b-a}
    write_once(OUT/'validation_vs_test.json',{'comparison_began_utc':began,'primary_sealed_utc':primary['sealed_utc'],
        'primary_seal':reference(OUT/'primary_seal.json'),'interpretation':'descriptive only; no tuning or automatic overfitting diagnosis','pipelines':comparison})
    benchmark=read(ROOT/'reports/phase6/benchmark_full.json')
    combined={}
    for p,mode in zip(PIPELINES,('bm25',None,'hybrid','hybrid_rerank')):
        rows=[c for run in benchmark['runs'] for c in run['cases'] if c['pipeline']==mode]
        combined[p]={'final_NDCG@10':test[p]['metrics']['NDCG@10'],'final_Recall@100':test[p]['metrics']['Recall@100']}
        for c in (1,8):
            vals=[r['client_success']['p95_ms'] for r in rows if r['concurrency']==c]
            combined[p][f'local_C{c}_P95_ms_range']=[min(vals),max(vals)] if vals else None
        qps=[r['successful_requests_per_second'] for r in rows]
        combined[p]['QPS_range']=[min(qps),max(qps)] if qps else None
    write_once(OUT/'quality_serving.json',{'quality_source':'Phase 9 frozen offline final test','performance_source':'Phase 6 local controlled HTTP benchmark; separate train traffic; both runs, concurrency 1/2/4/8',
        'benchmark':reference(ROOT/'reports/phase6/benchmark_full.json'),'pipelines':combined,'dense_performance':'N/A: Dense is not an existing HTTP serving pipeline'})


def max_difference(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        return max((max_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max((max_difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,(int,float)) and not isinstance(a,bool):return abs(a-b)
    assert a==b
    return 0.


def compare_rerun():
    primary=verify_seal(OUT/'primary_seal.json');repeated=verify_seal(TMP/'rerun/rerun_seal.json')
    assert primary['git_head']==repeated['git_head'] and primary['freeze_record_sha256']==repeated['freeze_record_sha256']
    rows={name:{'primary_sha256':primary['files'][name],'rerun_sha256':repeated['files'][name],
                'hash_equal':primary['files'][name]==repeated['files'][name],
                'max_absolute_difference':max_difference(read(OUT/name),read(TMP/'rerun'/name))} for name in DETERMINISTIC}
    raw={}
    for name in ('rankings.json','scores.json'):
        a=TMP/'primary'/name;b=TMP/'rerun'/name
        raw[name]={'hash_equal':sha(a)==sha(b),'primary_sha256':sha(a),'rerun_sha256':sha(b),'max_absolute_difference':max_difference(read(a),read(b))}
    write_once(OUT/'reproducibility.json',{'performed':True,'primary_runs':1,'exact_reruns':1,'same_head_protocol_code_models':True,
        'all_deterministic_hashes_equal':all(r['hash_equal'] for r in rows.values()),'files':rows,'raw_outputs':raw,
        'primary_seal':reference(OUT/'primary_seal.json'),'rerun_seal':reference(TMP/'rerun/rerun_seal.json')})


def seal():
    record=verify_freeze()
    primary=verify_seal(OUT/'primary_seal.json')
    repeated=verify_seal(TMP/'rerun/rerun_seal.json')
    repro=read(OUT/'reproducibility.json')
    assert repro['exact_reruns']==1
    from product_search.paired import paired_comparison
    per=read(OUT/'per_query_final.json');paired=read(OUT/'paired_final.json')
    for metric in ('NDCG@10','NDCG@20'):
        a={int(q):r[metric] for q,r in per['Hybrid'].items()};b={int(q):r[metric] for q,r in per[PIPELINES[-1]].items()}
        assert paired_comparison(a,b,42,10000)==paired[metric]
    quality=read(OUT/'final_quality.json')['pipelines']
    for mode in PIPELINES:
        for metric in METRICS:
            values=[r[metric] for r in per[mode].values() if r[metric] is not None]
            value=quality[mode]['metrics'][metric]
            assert (value is None and not values) or abs(value-math.fsum(values)/len(values))<1e-12
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    write_once(OUT/'integrity_audit.json',{'passed':True,'checked_utc':now(),'source_files_changed_after_first_label_access':False,
        'model_configs_changed':False,'retrieval_parameters_changed':False,'ce_depth_changed':False,'metric_semantics_changed':False,
        'tuning_using_test_results':False,'frozen_historical_files_unchanged':True,'metric_aggregation_validated':True,
        'paired_bootstrap_validated':True,'primary_and_rerun_checksums_verified':True,'git_status':git('status','--short'),
        'primary_first_access_utc':read(OUT/'test_access_audit.json')['first_access_event']['timestamp'],
        'label_opens':{'primary':1,'exact_rerun':1},'git_head':git('rev-parse','HEAD')})
    linked=[ROOT/'data/processed/data_manifest.json',ROOT/'artifacts/phase2/manifest.json',ROOT/'artifacts/phase3/manifest.json',
        ROOT/'artifacts/phase4/manifest.json',ROOT/'reports/phase6/benchmark_full.json',ROOT/'reports/phase7/checks.json',
        ROOT/'reports/phase7/monitoring.json',ROOT/'reports/phase7/observability_verified.json',ROOT/'reports/phase8/docker_runtime.json',
        ROOT/'reports/phase8/precommit_checks.json',ROOT/'reports/phase8/checks.json',ROOT/'reports/phase8/bundle.json',
        ROOT/'deployment/base_image.json',ROOT/'deployment/linux-cp314-cu130.lock',ROOT/'Dockerfile',ROOT/'compose.yaml',PROTOCOL]
    evidence=[p for p in sorted(OUT.glob('*.json'))]
    manifest={'version':'release-candidate-evidence-v1','git_head':record['git_head'],
        'uncommitted_evaluation_code':{'path':'scripts/run_phase9.py','sha256':record['source_config_sha256']['scripts/run_phase9.py']},
        'code_traceability':'Phase 8 committed implementation plus pre-access hash-frozen Phase 9 evaluation harness; no Phase 9 commit authorized',
        'freeze_record':reference(OUT/'freeze_record.json'),'model_names_revisions':{
            'dense':{'name':record['selected_dense_model'],'revision':record['selected_dense_revision']},
            'ce':{'name':record['selected_ce_model'],'revision':record['selected_ce_revision']}},
        'artifact_sha256':record['artifact_sha256'],'deployment_image_id':record['docker_image_id'],
        'linked_evidence':[reference(p) for p in linked+evidence],
        'first_test_access_timestamp':read(OUT/'test_access_audit.json')['first_access_event']['timestamp'],
        'test_query_count':record['test_query_count'],'evaluation_runs':{'primary':1,'exact_rerun':1},
        'no_tuning_after_access':True,'phase10_started':False,'git_push':False,'image_push':False}
    write_once(OUT/'release_candidate_manifest.json',manifest)
    for item in manifest['linked_evidence']:assert sha(ROOT/item['path'])==item['sha256']
    files={p.name:sha(p) for p in sorted(OUT.glob('*.json'))}
    write_once(OUT/'evidence_checksums.json',{'algorithm':'SHA-256','sealed_utc':now(),'files':files,
        'release_manifest_validation':'all linked evidence exists and SHA256 matches','verdict':'PASS'})
    print(json.dumps({'verdict':'PASS','manifest_sha256':sha(OUT/'release_candidate_manifest.json'),'freeze_sha256':sha(OUT/'freeze_record.json')}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('preflight','freeze','primary','rerun','seal'))
    args=parser.parse_args()
    guard=LabelGuard();sys.addaudithook(guard.check)
    if args.stage in ('primary','rerun'):
        final_run(args.stage,guard)
    else:
        {'preflight':preflight,'freeze':freeze,'seal':seal}[args.stage]()


if __name__=='__main__':main()
