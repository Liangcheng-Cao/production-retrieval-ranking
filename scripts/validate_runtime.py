"""Real sequential startup/smoke/parity check; reads train query text, no labels."""
import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from product_search.development_data import BoundaryGuard
GUARD = BoundaryGuard(ROOT).install()
from product_search.data.io import file_hash
from product_search.runtime_config import RuntimeConfig
from product_search.runtime_contracts import plain
from product_search.search_engine import SearchEngine


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def fixture_queries(count=6):
    manifest = json.loads((ROOT/'data/processed/data_manifest.json').read_text())
    ids = sorted(manifest['query_ids']['train'])[:count]
    path = ROOT/'data/processed/queries.jsonl'
    if file_hash(path) != manifest['artifacts']['queries.jsonl']['sha256']:
        raise ValueError('Query fixture checksum mismatch')
    queries = {}
    with path.open('rb') as handle:
        for line in handle:
            match = re.search(rb'"query_id":([0-9]+)(?:,|})', line)
            if match and int(match[1]) in ids:
                queries[int(match[1])] = json.loads(line)['query']
    if set(queries) != set(ids):
        raise ValueError('Missing development fixture query')
    return dict(sorted(queries.items()))


def resource_snapshot(engine):
    rss = None
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        rss = counters.WorkingSetSize
    gpu = {'allocated_bytes': None, 'reserved_bytes': None}
    if 'torch' in sys.modules:
        import torch
        if engine.config.device.startswith('cuda') and torch.cuda.is_available():
            gpu = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()}
    return {'process_rss_bytes': rss, 'rss_source': 'Windows working set' if os.name == 'nt' else 'unavailable',
            'gpu': gpu, 'loaded_components': engine.readiness().loaded_components,
            'scope': 'one sequential process; diagnostic, not capacity or peak memory'}


def parity(engine, queries):
    import numpy as np
    from product_search.retrieval.lexical import BM25Retriever
    from product_search.retrieval.dense import DenseRetriever
    from product_search.retrieval.hybrid import rrf
    from product_search.retrieval.hybrid import HybridRetriever
    from product_search.reranking import CrossEncoderReranker
    # Reconstruct the offline objects from frozen artifacts. Share already-loaded
    # model weights; do not use SearchEngine dispatch to compute the references.
    r = engine._resources
    s2 = json.loads((ROOT/'configs/phase2_selected.json').read_text())
    s3 = json.loads((ROOT/'configs/phase3_selected.json').read_text())['selected']
    plan = json.loads((ROOT/'configs/phase3_plan.json').read_text())
    lexical = BM25Retriever.load(ROOT/'artifacts/phase2'/s2['bm25']['name'], s2['bm25']['config'])
    dense = DenseRetriever.load(ROOT/'artifacts/phase2'/s2['dense']['name'], s2['dense']['config'], r.hybrid.dense.encoder)
    offline_ce = CrossEncoderReranker(r.reranker.scorer, list(r.products.values()), s3['representation'],
                                     s3['batch_size'], plan['max_length'])
    # Reproduce the historical batch context separately. Production parity uses
    # the existing single-query interface, not a different padding/batch context.
    all_train = fixture_queries(288)
    vectors = dense.encoder.encode(list(all_train.values()), batch_size=128, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
    existing_hybrid = HybridRetriever(lexical, dense, s2['hybrid']['config']['constant'], 100)
    p3 = json.loads((ROOT/'artifacts/phase3/manifest.json').read_text())
    historical = {}
    for name in ('candidates_train.json', 'scores_train_A.json'):
        relative = 'artifacts/phase3/'+name
        if file_hash(ROOT/relative) != p3['experimental_artifacts'][relative]['sha256']:
            raise ValueError('Frozen historical evidence checksum mismatch')
        historical[name] = json.loads((ROOT/relative).read_text())
    comparisons = []
    fingerprints = {}
    batch_diagnosis = []
    for i, (qid, query) in enumerate(queries.items()):
        lex = lexical.search(query, 100)
        historical_hybrid = rrf([lex, dense.search_vector(vectors[i], 100)], 100, s2['hybrid']['config']['constant'])
        historical_ce = offline_ce.rerank(query, historical_hybrid[:20], 20, fallback=False)
        hybrid = existing_hybrid.search(query, 100)
        ce = offline_ce.rerank(query, hybrid[:20], 20, fallback=False)
        cached = historical['candidates_train.json']['hits'][str(qid)]
        if [h.product_id for h in historical_hybrid] != [h['product_id'] for h in cached]:
            raise AssertionError(f'Historical Hybrid membership/order changed: query {qid}')
        old_scores = historical['scores_train_A.json'][str(qid)]
        expected_ce = sorted(cached[:20], key=lambda h: (-old_scores[str(h['product_id'])], h['rank'], h['product_id']))
        if [h.product_id for h in historical_ce.hits] != [h['product_id'] for h in expected_ce]:
            raise AssertionError(f'Historical CE ordering changed: query {qid}')
        if not np.allclose([h.reranker_score for h in historical_ce.hits],
                           [old_scores[str(h.product_id)] for h in historical_ce.hits], atol=1e-4, rtol=1e-4):
            raise AssertionError('Historical CE scores exceed tolerance')
        vector_one = dense.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        normalized_one = vector_one / np.linalg.norm(vector_one)
        normalized_batch = vectors[i] / np.linalg.norm(vectors[i])
        similarities_one = dense.embeddings @ normalized_one
        similarities_batch = dense.embeddings @ normalized_batch
        differences = [{'rank': j+1, 'batch_product_id': h.product_id, 'direct_product_id': hybrid[j].product_id}
                       for j, h in enumerate(historical_hybrid) if h.product_id != hybrid[j].product_id]
        affected = {v[key] for v in differences for key in ('batch_product_id', 'direct_product_id')}
        score_details = [{'product_id': int(pid), 'batch_cosine': float(similarities_batch[j]),
                          'direct_cosine': float(similarities_one[j])}
                         for j, pid in enumerate(dense.product_ids) if int(pid) in affected]
        batch_diagnosis.append({'query_id': qid, 'historical_reproduced_with_original_batch_context': True,
            'hybrid_top100_membership_equal_across_batch_contexts': {h.product_id for h in historical_hybrid} == {h.product_id for h in hybrid},
            'hybrid_order_equal_across_batch_contexts': not differences,
            'ce_order_equal_across_batch_contexts': [h.product_id for h in historical_ce.hits] == [h.product_id for h in ce.hits],
            'max_query_vector_abs_delta': float(np.max(np.abs(vectors[i]-vector_one))),
            'max_dense_cosine_abs_delta': float(np.max(np.abs(similarities_batch-similarities_one))),
            'hybrid_rank_changes': differences, 'affected_dense_scores': score_details})
        for mode, reference, score_field, ks in [('bm25', lex, 'score', (10, 20, 100)),
                ('hybrid', hybrid, 'score', (10, 20, 100)), ('hybrid_rerank', ce.hits, 'reranker_score', (10, 20))]:
            for k in ks:
                result = engine.search(query, k, mode)
                actual_ids = [h.product_id for h in result.results]
                expected_ids = [h.product_id for h in reference[:k]]
                actual_scores = [h.final_score for h in result.results]
                expected_scores = [getattr(h, score_field) for h in reference[:k]]
                tolerance = 1e-4 if mode == 'hybrid_rerank' else 1e-6
                passed = actual_ids == expected_ids and bool(np.allclose(actual_scores, expected_scores, atol=tolerance, rtol=tolerance)) and not result.fallback_used
                comparisons.append({'query_id': qid, 'pipeline': mode, 'top_k': k, 'passed': passed,
                                    'membership_equal': set(actual_ids) == set(expected_ids), 'ordering_equal': actual_ids == expected_ids,
                                    'max_score_abs_error': max((abs(a-b) for a, b in zip(actual_scores, expected_scores)), default=0),
                                    'atol': tolerance, 'rtol': tolerance})
                fingerprints[f'{qid}:{mode}:{k}'] = actual_ids
                if not passed:
                    raise AssertionError(f'Parity mismatch: {comparisons[-1]}')
    return {'query_ids': list(queries), 'query_count': len(queries), 'comparisons': comparisons,
            'all_passed': True, 'historical_phase3_reproduced_with_original_batch_context': True,
            'batch_context_diagnosis': batch_diagnosis,
            'method': 'independent existing offline single-query interfaces; historical batch-128 reproduction checked separately',
            'fingerprints': fingerprints}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/runtime.json')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/phase4/runtime_validation.json')
    parser.add_argument('--parity', action='store_true')
    parser.add_argument('--compare-restart', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Refusing to overwrite existing runtime evidence; choose a fresh report path')
    report = {'passed': False}
    engine = None
    try:
        config = RuntimeConfig.load(args.config)
        if any(p != 'bm25' for p in config.enabled_pipelines):
            import torch
            torch.manual_seed(42)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            engine = SearchEngine.from_config(config)
            report['readiness'] = asdict(engine.readiness())
            report['startup_ms'] = plain(engine.startup_timing)
            report['version'] = plain(engine.version)
            report['resources_after_startup'] = resource_snapshot(engine)
            queries = fixture_queries()
            smoke = {}
            for mode in config.enabled_pipelines:
                result = engine.search('wooden office desk', 10, mode)
                smoke[mode] = {'ids': [h.product_id for h in result.results], 'timing': plain(result.timing),
                               'fallback_used': result.fallback_used, 'effective_pipeline': result.effective_pipeline}
                if result.fallback_used or not result.results:
                    raise AssertionError('Smoke query failed or degraded')
            report['synthetic_smoke'] = smoke
            if args.parity:
                report['parity'] = parity(engine, queries)
            if args.compare_restart:
                previous = json.loads(args.compare_restart.read_text())
                if report['version'] != previous['version']:
                    raise AssertionError('Runtime version changed between independent processes')
                if report.get('parity', {}).get('fingerprints') != previous.get('parity', {}).get('fingerprints'):
                    raise AssertionError('Ranked IDs changed after independent restart')
                report['restart'] = {'passed': True, 'previous_report': str(args.compare_restart),
                                     'separate_processes': True, 'ranked_ids_identical': True}
            report['resources_after_smoke'] = resource_snapshot(engine)
            report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['boundary'] = {'opened_canonical_files': sorted(GUARD.opened), 'blocked_attempts': GUARD.blocked,
                              'final_test_labels_accessed': False,
                              'policy': 'product metadata and train query strings only; no relevance label files'}
        if engine:
            engine.close()
        dump(args.report, report)
    print(json.dumps({'passed': report['passed'], 'report': str(args.report), 'error': report.get('error')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
