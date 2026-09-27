"""Build the Phase 3 evidence report/provenance from recorded experiments only."""
import json
import importlib.metadata
import platform
from pathlib import Path
from product_search.data.io import file_hash

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/phase3'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def fmt(value): return 'N/A' if value is None else f'{value:.6f}'

def main():
    plan=read(ROOT/'configs/phase3_plan.json'); selection=read(ROOT/'configs/phase3_selected.json')
    dev=read(REPORT/'development.json'); val=read(REPORT/'validation.json'); paired=read(REPORT/'paired.json')
    p2=read(ROOT/'reports/phase2/development.json'); p2lat=read(ROOT/'reports/phase2/latency.json')
    selected2=read(ROOT/'configs/phase2_selected.json')
    models=ROOT/'artifacts/model_cache/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2'
    artifact_files={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':file_hash(p)}
                    for p in sorted((ROOT/'artifacts/phase3').glob('*')) if p.is_file() and p.name!='manifest.json'}
    model_files={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(models.rglob('*')) if p.is_file() and '/snapshots/' in p.as_posix()}
    sources=[ROOT/'src/product_search/reranking.py',ROOT/'src/product_search/paired.py',ROOT/'scripts/run_phase3.py',ROOT/'scripts/check_phase2_metrics.py',ROOT/'scripts/smoke_reranker.py',ROOT/'scripts/summarize_phase3.py']
    reports={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(REPORT.iterdir()) if p.suffix in ('.json','.md') and p.name!='REPORT.md'}
    manifest={'phase':'Phase 3 offline cross-encoding only','phase1_manifest_sha256':file_hash(ROOT/'data/processed/data_manifest.json'),
        'phase2_retrieval_manifest_sha256':file_hash(ROOT/'artifacts/phase2/manifest.json'),
        'plan':plan,'selected':selection,'parameters':val['startup']['parameters'],
        'tokenization':{'pair_format':'original query + chosen product text','truncation':'longest_first','truncation_side':'right','max_length':plan['max_length'],'padding':'longest within each batch','tokenizer_revision':plan['revision']},
        'validation_configuration':{'representation':val['representation'],'candidate_depth':val['candidate_depth'],'full_returned_ranking_depth':val['candidate_depth'],'cutoffs_beyond_depth':'N/A; no tail appended'},
        'software':{'python':platform.python_version(),**{name:importlib.metadata.version(name) for name in ('torch','transformers','sentence-transformers','tokenizers','numpy','huggingface-hub','safetensors')}},
        'dtype':'float32','device':'cuda:0','gpu':'NVIDIA GeForce RTX 4070','model_files_sha256':model_files,
        'experimental_artifacts':artifact_files,'source_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},
        'report_sha256':reports,'plan_sha256':file_hash(ROOT/'configs/phase3_plan.json'),'selection_sha256':file_hash(ROOT/'configs/phase3_selected.json'),
        'no_test_label_access':True,'no_training':True,'fallback_contract':'On model failure return original retrieval prefix, reranker_score=null, fallback_used=true, error_type set; malformed requests fail validation',
        'determinism_limit':'stable tie handling and seeded paired bootstrap; real batch logits checked within tolerance, not bitwise cross-platform guarantees'}
    assert manifest['phase1_manifest_sha256']==selection['phase1_manifest_sha256']
    assert manifest['phase2_retrieval_manifest_sha256']==selection['phase2_manifest_sha256']
    dump(ROOT/'artifacts/phase3/manifest.json',manifest)
    lines=['# Phase 3 — CrossEncoder reranking evidence','',
        'Validation-only offline experiment. No final-test labels, model training, API, concurrency test or production SLA.',
        'Phase 2 was committed and the repository clean before implementation. Its metrics/artifacts remain unchanged.','',
        '## Metric sanity','',
        'BM25 validation NDCG@10=0.6671693286442933 and NDCG@20=0.6671984569895968.',
        'The apparent equality is four-decimal rounding. Independent scalar sums match all 96 queries,',
        'including five stored worked examples. 86 queries have judged positives at ranks 11–20.',
        'Both DCG and IDCG use the actual cutoff. Historical Phase 2 values were not changed.','',
        '## Fixed experiment design','',
        f"Model: {plan['model']}, revision {plan['revision']}, Apache-2.0, {val['startup']['parameters']:,} parameters.",
        'Raw relevance logits from BertForSequenceClassification; no sigmoid, training or fine-tuning.',
        'RTX 4070, float32, batch 32, max length 256, right-side longest-first paired truncation.',
        'A=name; B=name+class/category. Only these two short representations were tested; no description/features.',
        'Depths 20/50/100. Primary selection: development NDCG@10 subject to reranker P95 <=150 ms,',
        'then NDCG@20, smaller depth, representation name. All trials met the budget; it was not binding.',
        'Batch 32 was fixed for this compact model, not claimed optimal. No second model was tested.',
        'Development scores were computed for top-100 once per representation and reused for nested depth subsets;',
        'latency uses real depth-specific inference. Actual batch invariance is covered by a separate synthetic integration smoke.',
        'Selection was written before Phase 3 validation; the prior validation-only metric sanity check did not guide selection.','',
        '## Development quality and latency (288 queries)','',
        '| Text | Depth | NDCG@10 | NDCG@20 | Recall@20 | Recall@50 | Recall@100 | Rerank P95 ms | Pipeline P95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in dev['trials']:
        m=t['quality']['metrics']; timing=t['latency']['stages']
        lines.append(f"| {t['representation']} | {t['candidate_depth']} | "+' | '.join(fmt(m[k]) for k in ('NDCG@10','NDCG@20','Recall@20','Recall@50','Recall@100'))+f" | {timing['total_ms']['p95_ms']:.3f} | {timing['pipeline_ms']['p95_ms']:.3f} |")
    lines+=['','Selected: A, candidate depth 20. Deeper reranking did not improve the primary development objective.',
        'All candidate sets and Recall@candidate_depth remained identical before/after reordering.',
        'A had 0/28,800 truncated development pairs (max 73 tokens); B had 0/28,800 (max 89).',
        f"Selected validation had {val['truncation']['pairs_over_max_length']}/{val['truncation']['pairs']} truncated pairs.",
        'Metrics beyond returned depth are N/A, not zero and not inherited from the original top-100 pool.','',
        '## Validation quality (96 queries, zero undefined-metric exclusions)','',
        '| Pipeline | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in {**val['baselines_from_phase2'],'Hybrid + CE-20':val['hybrid_cross_encoder']}.items():
        lines.append('| '+name+' | '+' | '.join(fmt(row['metrics'][k]) for k in ('Recall@20','Recall@50','Recall@100','NDCG@10','NDCG@20'))+' |')
    lines+=['','Actual path: Hybrid retrieve top-100 → take first 20 → CrossEncoder → ranked 20, with top-10 as a prefix.',
        'Full membership is retained at depth 20, so Recall@20 is invariant. No unscored tail is appended.',
        'Do not label the original candidate-pool Recall@100 as the returned CE-20 Recall@100.','',
        '## Paired query comparison','',
        '| Metric | Mean delta | Median | Improved | Unchanged | Worsened | Paired bootstrap 95% CI |',
        '|---|---:|---:|---:|---:|---:|---|']
    for metric,c in paired.items():
        ci=c['paired_bootstrap_95pct_ci']
        lines.append(f"| {metric} | {c['mean_delta']:.6f} | {c['median_delta']:.6f} | {c['improved']} | {c['unchanged']} | {c['worsened']} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines+=['','10,000 paired query resamples, seed 42, percentile interval, 96 eligible pairs.',
        'Intervals describe query sampling uncertainty on this validation set, not production effects or model-selection uncertainty.',
        'Incomplete judgments remain a limitation; unjudged products contribute zero observed gain but are not known negatives.',
        'See error_analysis.md: category/modifier improvements and both annotation-coverage and graded-ordering regressions.','',
        '## Sequential latency','',
        '24 fixed train queries, five warmups per configuration, three repeats: 72 observations each.',
        'One BLAS thread. Timings include real query encoding and retrieval, not cached candidates.',
        'GPU forward is explicitly synchronized. H2D/D2H transfers are separately reported.',
        'Below, A representation; each cell is P50 / P95 milliseconds.','',
        '| Depth | Tokenization | GPU forward | Transfers | Postprocess | Rerank total | Pipeline total | Effective batches |',
        '|---|---|---|---|---|---|---|---|']
    for t in dev['trials']:
        if t['representation']!='A': continue
        s=t['latency']['stages']
        cells=[f"{s[k]['p50_ms']:.3f} / {s[k]['p95_ms']:.3f}" for k in ('tokenization_ms','gpu_forward_ms','transfer_ms','postprocess_ms','total_ms','pipeline_ms')]
        lines.append(f"| {t['candidate_depth']} | "+' | '.join(cells)+f" | {t['latency']['effective_batch_sizes']} |")
    lines+=['',f"First CE startup including acquisition: {dev['startup']['cross_encoder_startup_ms']/1000:.3f} s; cached startup in validation: {val['startup']['cross_encoder_startup_ms']/1000:.3f} s.",
        'Startup excludes process/import time and is separate from warm latency. Cached model loading can include Hub metadata checks.',
        'Percentiles of individual stages must not be summed. No concurrency or throughput experiment was run.','',
        '## Quality/latency frontier — development quality only','',
        'All quality values below use the same 288 train queries. Baseline timings are the recorded Phase 2 diagnostic run;',
        'CE timings are Phase 3 with the same query list/protocol, a different session. This is a descriptive comparison, not a controlled SLA.',
        '| Pipeline | NDCG@10 | Recall@100 | Pipeline P50 ms | Pipeline P95 ms |',
        '|---|---:|---:|---:|---:|']
    for key,name in [('bm25','BM25'),('hybrid','Hybrid')]:
        trial=next(t for t in p2['trials'] if t['name']==selected2[key]['name']); timing=p2lat['pipelines'][name]['total_ms']
        lines.append(f"| {name} | {trial['quality']['metrics']['NDCG@10']:.6f} | {trial['quality']['metrics']['Recall@100']:.6f} | {timing['p50_ms']:.3f} | {timing['p95_ms']:.3f} |")
    for t in dev['trials']:
        if t['representation']!='A': continue
        timing=t['latency']['stages']['pipeline_ms']
        lines.append(f"| Hybrid + CE-{t['candidate_depth']} | {t['quality']['metrics']['NDCG@10']:.6f} | {fmt(t['quality']['metrics']['Recall@100'])} | {timing['p50_ms']:.3f} | {timing['p95_ms']:.3f} |")
    lines+=['','CE-20 is preferable for this tested top-10 quality/latency objective. CE-50 has higher development NDCG@20,',
        'so CE-20 does not dominate every objective. Hybrid remains the inexpensive full top-100 candidate path;',
        'BM25 remains substantially faster. Candidate-depth coverage differs, so a blanket Pareto dominance claim would be misleading.',
        'Validation supports retaining CE-20 as an optional quality-oriented ranking mode, not making every request use it.','',
        '## Reliability contract and scope','',
        'The reranker never retrieves. It accepts typed candidates and preserves retrieval score/rank/source.',
        'Score ties retain retrieval rank, then product ID. Full reranking preserves IDs; top_k truncation is explicit.',
        'Inference errors/malformed outputs return the original retrieval prefix with null CE scores and an explicit fallback flag/error type.',
        'Input contract errors raise. Experiments use fallback=False to avoid hiding failed measurements.',
        'Fallback tests inject synthetic failures; no production reliability claim is made.','',
        '## Freeze and provenance','',
        'Final-test relevance labels accessed: NO. Only train/validation label files were opened.',
        'The shared query JSONL is byte-scanned for IDs and hashed; test query strings are not decoded/encoded/evaluated.',
        'Phase 1 split/code/manifests and Phase 2 evidence remain unchanged. Access logs and metric sanity provenance are retained.',
        'artifacts/phase3/manifest.json records both prior manifest hashes, the model revision, configuration, source/report hashes and ignored artifact checksums.',
        'Weights, caches, candidate pools and per-pair scores remain ignored. Only compact reports/configs/manifests are Git-eligible.',
        'Normal tests do not download a model. scripts/smoke_reranker.py uses real model weights with synthetic text only.',
        'Phase 4 production search core is not implemented.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Phase 3 provenance and report written from existing evidence')

if __name__=='__main__': main()
