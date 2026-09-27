"""Assemble provenance from completed experiment reports without loading any labels."""
import json
import importlib.metadata
from pathlib import Path
from product_search.data.io import file_hash

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/phase2'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')

def main():
    selected=read(ROOT/'configs/phase2_selected.json')
    development=read(REPORTS/'development.json')
    validation=read(REPORTS/'validation.json')
    latency=read(REPORTS/'latency.json')
    complement=read(REPORTS/'complementarity.json')
    artifacts={}
    for p in sorted((ROOT/'artifacts/phase2').glob('*/manifest.json')):
        manifest=read(p)
        artifacts[p.parent.name]={'manifest_sha256':file_hash(p),'manifest':manifest,
            'file_bytes':{name:(p.parent/name).stat().st_size for name in manifest['files']}}
    boundaries={stage:read(REPORTS/f'boundary_{stage}.json') for stage in ('development','validation')}
    frozen=file_hash(ROOT/'data/processed/data_manifest.json')
    assert frozen==selected['data_manifest_sha256']
    assert all(b['frozen_manifest_sha256_after']==frozen and not b['blocked_attempts'] for b in boundaries.values())
    assert validation['selection_sha256']==file_hash(ROOT/'configs/phase2_selected.json')
    model_files={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted((ROOT/'artifacts/model_cache').rglob('*'))
                 if p.is_file() and '/snapshots/' in p.as_posix()}
    source_paths=list((ROOT/'src/product_search/retrieval').glob('*.py'))+[
        ROOT/'src/product_search/development_data.py',ROOT/'src/product_search/evaluation.py',ROOT/'scripts/run_phase2.py']
    manifest={'phase':'Phase 2 candidate retrieval only','phase1_data_manifest_sha256':frozen,
        'environment':read(ROOT/'reports/phase2_environment.json'),
        'software_versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','bm25s','torch','transformers','sentence-transformers','huggingface-hub','tokenizers','safetensors','threadpoolctl')},
        'development_plan':read(ROOT/'configs/phase2_plan.json'),
        'development_configurations_tried':[{'name':t['name'],'config':t['config'],'build_ms':t['build_ms']} for t in development['trials']],
        'selected_validation_configurations':selected,'artifacts':artifacts,'downloaded_model_file_sha256':model_files,
        'source_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for p in source_paths},
        'report_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(REPORTS.glob('*.json'))},
        'test_boundary':boundaries,
        'serialization_notes':'BM25S serialized vocabulary ordering need not be byte-identical across rebuilds; stored exact bytes verified on load. IDs/ties deterministic. GPU re-encoding bit identity across runtime/hardware not claimed.'}
    save(ROOT/'artifacts/phase2/manifest.json',manifest)
    lines=['# Phase 2: candidate retrieval evidence','',
        'Validation-only results; no final-test evaluation or production benchmark exists.',
        'The experiment plan and development selection were saved before validation. No validation retuning occurred.','',
        '## Development experiments (288 queries; zero excluded)','',
        '| Configuration | Recall@100 | NDCG@20 | Build seconds |','|---|---:|---:|---:|']
    for t in development['trials']:
        m=t['quality']['metrics']; seconds='—' if t['build_ms'] is None else f"{t['build_ms']/1000:.3f}"
        lines.append(f"| {t['name']} | {m['Recall@100']:.6f} | {m['NDCG@20']:.6f} | {seconds} |")
    lines+=['','A=name; B=name plus nonempty product_class/category_hierarchy (exact duplicate fields omitted);',
        'C=B plus description/features. Numeric ratings/reviews are never included.',
        'BM25 B k1=1.2 b=0.5 and dense A were selected by development Recall@100, then NDCG@20.',
        'RRF 20 and 60 tied on Recall@100; 60 won the prespecified NDCG@20 tie-break.',
        'Only one small pretrained model was tested; representations, not additional models, were explored.',
        'Dense B/C reduced recall and took longer to encode. No additional trials were pursued.','',
        '## Validation (96 queries; zero excluded for every metric)','',
        '| Pipeline | Recall@10 | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,result in validation['pipelines'].items():
        lines.append('| '+name+' | '+' | '.join(f"{result['metrics'][key]:.6f}" for key in ('Recall@10','Recall@20','Recall@50','Recall@100','NDCG@10','NDCG@20'))+' |')
    lines+=['','Recall uses all judged Partial/Exact items as the denominator, not K.',
        'NDCG gain is 2**grade-1. Unjudged returned items contribute zero measured gain but are not known negatives.',
        'Undefined metrics are excluded with explicit counts. Judged fraction at 100 is '+', '.join(f"{n} {v['metrics']['judged_fraction@100']:.4f}" for n,v in validation['pipelines'].items())+'.',
        'Incomplete qrels can bias comparisons; no significance, online traffic or A/B claim is made.',
        'Hybrid improves Recall@100 and NDCG here, but Recall@20 is slightly below BM25.','',
        '## Candidate complementarity','',
        'Means per validation query; relevant means judged Partial/Exact. Recovery/loss is relative to BM25 at the same K.',
        '| K | Overlap | Jaccard | BM25-only relevant | Dense-only relevant | Shared relevant | Hybrid relevant | Hybrid recovered | Hybrid lost |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('20','50','100'):
        c=complement['depths'][k]
        keys=('overlap_count','jaccard','bm25_only_relevant','dense_only_relevant','shared_relevant','hybrid_relevant','hybrid_recovered_beyond_bm25','hybrid_lost_vs_bm25')
        lines.append('| '+k+' | '+' | '.join(f"{c[key]['mean_per_query']:.4f}" for key in keys)+' |')
    lines+=['','RRF takes each component top-100 union, then orders/truncates to 100. It does not preserve the complete union.',
        'At K=20/50 the hybrid can use deeper component candidates; recovery is not necessarily confined to dense top-K.',
        'Mean relevant-hit gains and macro Recall gains differ because queries have different positive denominators.','',
        '## Single-query warm latency','',
        '24 fixed train queries, five warmups per pipeline, three repeats (72 observations), top_k=100.',
        'Sequential process; one CPU BLAS thread; GPU query encoding, CPU exact float32 matrix search.',
        'No query-vector cache in this timing; quality evaluation alone uses batched query vectors.',
        '| Pipeline | P50 ms | P95 ms |','|---|---:|---:|']
    for name,v in latency['pipelines'].items(): lines.append(f"| {name} | {v['total_ms']['p50_ms']:.4f} | {v['total_ms']['p95_ms']:.4f} |")
    lines+=['','Stage percentiles are in latency.json; stage percentiles must not be added to derive total percentiles.',
        'Startup excludes Python process/import time and catalog loading. It includes artifact hash validation.',
        'Model startup loads cached weights and may include Hub metadata checks; it is not a cold download measurement.',
        'Dense/hybrid startup totals are sums of separately measured initialization stages, not process-start wall times.',
        'Single startup measurements (ms): '+json.dumps(latency['startup_ms'],sort_keys=True)+'.',
        'Initial model acquisition/load took '+str(round(read(REPORTS/'model.json')['first_model_load_including_download_ms']/1000,3))+' seconds, reported separately.',
        'The ~43k x 384 exact index is sufficient for this observed single-query workload; no ANN was introduced.',
        'This small benchmark is diagnostic, not an SLA, throughput test or production P95.','',
        '## Test boundary','',
        'Final-test relevance labels accessed: NO. The runner blocks raw data, final-test JSONL and the global conflict table.',
        'The shared canonical queries file is hashed/byte-scanned to identify rows; test query strings are never JSON-decoded, encoded or evaluated.',
        'This is not a claim that no shared-file bytes were read. Only train/validation judgment files were opened.',
        'Phase 1 unit tests use synthetic temporary test partitions, not the frozen WANDS test labels.',
        'The frozen data layer, manifest and split remain unchanged. Access logs are boundary_development.json and boundary_validation.json.','',
        '## Artifacts and reproduction','',
        'artifacts/phase2/manifest.json records software, model revision, source hashes, all tried configurations, selection and artifact hashes.',
        'Only small manifest files are Git-eligible; models, arrays, indexes and caches are ignored.',
        'The selected dense matrix is 42,994 x 384 float32 (66,038,912 bytes including NPY header).',
        'Run with Python 3.14.3 and the exact CUDA wheel described in the README. No cross-device bitwise embedding guarantee is claimed.',
        'The driver refuses silent repeated development selection or validation. For an explicitly authorized reproduction, use a separate checkout/output workspace and preserve the existing results.',
        'No CrossEncoder, API, learned ranking, concurrency benchmark or Phase 3 code has been implemented.']
    (REPORTS/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Wrote retrieval manifest and report without reading any labels')

if __name__=='__main__': main()
