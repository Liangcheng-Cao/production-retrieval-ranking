"""Two independent actual-model offline monitoring builds, train boundary enforced."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from product_search.development_data import BoundaryGuard, load_development
GUARD = BoundaryGuard(ROOT).install()
from product_search.data.io import file_hash
from product_search.monitoring import (VERSION, digest, distribution, embedding_summary, centroid_distance,
    scalar_distance, total_variation, query_features, scenarios, ranking_difference, envelope)
from product_search.runtime_contracts import plain
from product_search.search_engine import SearchEngine
from product_search.evaluation import qrels, evaluate


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def run(plan, data, provenance):
    import numpy as np
    engine = SearchEngine.from_config(ROOT/'configs/runtime.json')
    try:
        traffic, dominant = scenarios(data.queries, plan['long_suffix'])
        output, times = {}, {}
        baseline_vectors = None
        for name, rows in traffic.items():
            vectors, top1, margins, scores = [], [], [], []
            rankings = {p: {} for p in ('hybrid', 'hybrid_rerank')}
            timings = {p: [] for p in rankings}
            diagnostics = {p: {'result_counts': [], 'fallback_count': 0, 'effective_pipeline_counts': Counter()} for p in rankings}
            differences = []
            for index, (query, _) in enumerate(rows):
                vectors.append(engine._resources.hybrid.dense.encoder.encode([query], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)[0])
                for mode in rankings:
                    result = engine.search(query, plan['replay_top_k'], mode)
                    rankings[mode][index] = [h.product_id for h in result.results]
                    timings[mode].append(result.timing['total_ms'])
                    diag = diagnostics[mode]
                    diag['result_counts'].append(len(result.results))
                    diag['fallback_count'] += result.fallback_used
                    diag['effective_pipeline_counts'][result.effective_pipeline] += 1
                    if mode == 'hybrid_rerank' and not result.fallback_used:
                        logits = [h.reranker_score for h in result.results]
                        scores.extend(logits)
                        if logits:
                            top1.append(logits[0])
                        if len(logits) > 1:
                            margins.append(logits[0]-logits[1])
                differences.append(ranking_difference(rankings['hybrid'][index], rankings['hybrid_rerank'][index]))
            emb = embedding_summary(np.asarray(vectors))
            if name == 'baseline':
                baseline_vectors = emb
            for diag in diagnostics.values():
                counts = diag.pop('result_counts')
                diag['result_count'] = distribution(counts)
                diag['empty_result_rate'] = counts.count(0)/len(counts) if counts else None
                diag['fallback_rate'] = diag['fallback_count']/len(rows) if rows else None
            comparison = {'query_count': len(rows), 'request_count': 2*len(rows),
                'ordering_changed_fraction': float(np.mean([d['ordering_changed'] for d in differences])),
                'mean_overlap_at_10': float(np.mean([d['overlap_at_k'] for d in differences])),
                'mean_shared_rank_displacement': distribution([d['shared_mean_rank_displacement'] for d in differences if d['shared_mean_rank_displacement'] is not None])['mean'],
                'rankings_sha256': digest(rankings), 'pipelines': diagnostics}
            if name == 'baseline':
                labels = qrels(data.judgments)
                ids = [q.query_id for q in sorted(data.queries, key=lambda q: q.query_id)]
                comparison['train_quality'] = {}
                for mode, ranking in rankings.items():
                    evaluation = evaluate({qid: ranking[i] for i, qid in enumerate(ids)}, {qid: labels.get(qid, {}) for qid in ids})
                    # Both APIs returned K20: never misrepresent Recall50/100 as full-depth recall.
                    wanted = ('Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20')
                    comparison['train_quality'][mode] = {k: {m: evaluation[k][m] for m in wanted} for k in ('metrics', 'excluded_queries')}
            original = traffic['baseline']
            output[name] = {'version': VERSION, 'provenance': provenance, 'features': query_features(rows),
                'embedding': emb, 'ce_scores': {'top1': distribution(top1), 'top1_top2_margin': distribution(margins), 'all_ce20': distribution(scores)},
                'drift': {'character_wasserstein': scalar_distance([len(t) for t, _ in original], [len(t) for t, _ in rows]),
                    'token_wasserstein': scalar_distance([len(t.split()) for t, _ in original], [len(t.split()) for t, _ in rows]),
                    'class_total_variation': total_variation(Counter(c for _, c in original), Counter(c for _, c in rows)),
                    'centroid_cosine_distance': centroid_distance(baseline_vectors['centroid'], emb['centroid'])},
                'offline_comparison': comparison}
            times[name] = {mode: distribution(values) for mode, values in timings.items()}
            print(json.dumps({'scenario': name, 'queries': len(rows), 'complete': True}), flush=True)
        return {'scenarios': output, 'dominant_source_class': dominant, 'models': plain(engine.version)['dense'] | {'reranker': plain(engine.version)['reranker']}}, times
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'reports/phase7')
    args = parser.parse_args()
    if (args.output/'monitoring.json').exists():
        parser.error('Refusing to overwrite monitoring evidence')
    plan_path = ROOT/'configs/phase7_monitoring.json'
    plan = json.loads(plan_path.read_text())
    if plan['split'] != 'train' or plan['rebuilds'] != 2 or plan['encoder_batch_size'] != 1 or plan['comparison_top_k'] != 10 or plan['replay_top_k'] != 20:
        raise ValueError('Frozen monitoring protocol mismatch')
    data = load_development(ROOT/'data/processed', plan['split'])
    assert len(data.queries) == plan['query_count']
    provenance = {'split': 'train', 'query_count': len(data.queries), 'query_ids_sha256': digest(sorted(q.query_id for q in data.queries)),
        'feature_definitions': plan['feature_definitions'], 'source_sha256': {p: file_hash(ROOT/p) for p in
        ('data/processed/data_manifest.json', 'data/processed/query_splits.json', 'data/processed/queries.jsonl',
         'data/processed/judgments.train.jsonl', 'artifacts/phase4/manifest.json', 'configs/phase7_monitoring.json')}}
    import torch
    from threadpoolctl import threadpool_limits
    torch.manual_seed(plan['seed'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=1):
        first, times1 = run(plan, data, provenance)
        second, times2 = run(plan, data, provenance)
    equal = digest(first) == digest(second)
    # Keep BOTH payloads if unequal: never hide GPU numerical differences.
    if not equal:
        dump(args.output/'monitoring_rebuild_difference.json', {'first': first, 'second': second})
    baseline = envelope({**first['scenarios']['baseline'], 'models': first['models']})
    dump(args.output/'baseline.json', baseline)
    dump(args.output/'monitoring.json', {'payload': first, 'sha256': digest(first), 'protocol': plan,
        'execution': {'created_utc': datetime.now(timezone.utc).isoformat(), 'independent_engine_loads': 2,
            'latency_ms_run1': times1, 'latency_ms_run2': times2, 'deterministic_payload_equal': equal,
            'first_sha256': digest(first), 'second_sha256': digest(second),
            'comparison': 'exact serialized float equality; timings excluded; GPU variation not presumed impossible',
            'boundary': {'opened': sorted(GUARD.opened), 'blocked': GUARD.blocked, 'final_test_labels_accessed': False}}})
    print(json.dumps({'rebuild_equal': equal, 'baseline_sha256': baseline['sha256']}))
    return 0 if equal else 1


if __name__ == '__main__':
    raise SystemExit(main())
