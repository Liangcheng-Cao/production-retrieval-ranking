"""Small real-model TCP integration, exact queue visibility and privacy checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from threading import Event
from time import monotonic, sleep

from validate_api import ROOT, GUARD, start, stop, dump, ObservedEngine, fixture_queries
from product_search.api import create_app
from product_search.runtime_config import RuntimeConfig
from product_search.monitoring import distribution
from prometheus_client.parser import text_string_to_metric_families


def samples(text, name):
    return [s for family in text_string_to_metric_families(text) for s in family.samples if s.name == name]


def total(text, name):
    return sum(s.value for s in samples(text, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=ROOT/'reports/phase7/observability_integration.json')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Refusing to overwrite evidence')
    import httpx
    import torch
    # Preload native libraries in main thread, as documented in Phase 6.
    from product_search.retrieval.lexical import BM25Retriever
    from sentence_transformers import SentenceTransformer
    from threadpoolctl import threadpool_limits
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    log_path = ROOT/'reports/tmp'/('phase7-'+args.report.stem+'.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('product_search.service')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Uvicorn dictConfig closes existing handlers. Append mode reopens safely;
    # write mode intentionally refuses to reopen after logging.shutdown/close.
    if log_path.exists():
        parser.error('Refusing to reuse integration log evidence')
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    config = RuntimeConfig.load(ROOT/'configs/runtime.json')
    engines = []
    def factory():
        e = ObservedEngine(config)
        engines.append(e)
        return e
    queries = list(fixture_queries().values())
    # Deliberately unique privacy probe; no relevance judgments are used.
    private_query = 'phase7_private_query_probe_7319 wooden desk'
    app = create_app(engine_factory=factory)
    running = None
    report = {'passed': False, 'scope': 'real loopback service, real frozen models; small controlled integration, no throughput claims'}
    try:
        with threadpool_limits(limits=1):
            running = start(app)
            assert running[0].started
            with httpx.Client(base_url=running[3], timeout=60, trust_env=False) as c:
                def scrape():
                    r = c.get('/metrics')
                    assert r.status_code == 200
                    assert r.headers['content-type'].startswith('text/plain; version=0.0.4')
                    return r.text
                def search(query, mode='hybrid', request_id='obs-request'):
                    r = c.post('/search', json={'query': query, 'pipeline': mode, 'top_k': 10}, headers={'X-Request-ID': request_id})
                    assert r.status_code == 200
                    assert r.headers['x-request-id'] == request_id
                    return r
                before = total(scrape(), 'request_count_total')
                for path in ('/health', '/ready', '/version', '/metrics'):
                    assert c.get(path).status_code == 200
                assert total(scrape(), 'request_count_total') == before == 0
                report['probes_excluded'] = True
                report['pipelines'] = {}
                for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
                    search(private_query, mode, 'obs-'+mode)
                    rows = [s for s in samples(scrape(), 'request_count_total') if s.labels['pipeline'] == mode]
                    assert len(rows) == 1 and rows[0].value == 1 and rows[0].labels['effective_pipeline'] == mode
                    report['pipelines'][mode] = {'request_count_after_one': 1, 'status': 200}
                engine = engines[0]
                original_scorer = engine._resources.reranker.scorer
                class Broken:
                    def score_pairs(self, *args):
                        raise RuntimeError('Injected CE failure')
                engine._resources.reranker.scorer = Broken()
                try:
                    r = search(queries[0], 'hybrid_rerank', 'obs-fallback')
                    assert r.json()['fallback_used'] and r.json()['effective_pipeline'] == 'hybrid'
                    text = scrape()
                    assert total(text, 'reranker_fallback_total') == total(text, 'fallback_count_total') == 1
                    assert total(text, 'degraded_request_total') == 1
                    assert total(text, 'request_error_count_total') == 0
                    assert c.get('/ready').json()['state'] == 'DEGRADED'
                    report['fallback'] = {'status': 200, 'effective_pipeline': 'hybrid', 'reranker_fallback_total': 1, 'degraded_request_total': 1, 'error_count': 0}
                finally:
                    engine._resources.reranker.scorer = original_scorer
                search(queries[0], 'hybrid_rerank', 'obs-recovery')
                assert c.get('/ready').json()['ready']
                original_search = engine.search
                def failed(*args):
                    raise RuntimeError(private_query)
                engine.search = failed
                try:
                    failure = c.post('/search', json={'query': private_query, 'pipeline': 'bm25'}, headers={'X-Request-ID': 'obs-core-failure'})
                    assert failure.status_code == 500 and private_query not in failure.text
                    assert total(scrape(), 'request_error_count_total') == 1
                    report['injected_core_error'] = {'status': 500, 'error_counter_increment': 1, 'exception_query_not_echoed': True}
                finally:
                    engine.search = original_search
                # Small natural C1 baseline, separate from artificially held visibility checks.
                c1 = {}
                for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
                    responses = [search(queries[i % len(queries)], mode, f'obs-c1-{mode}-{i}') for i in range(12)]
                    c1[mode] = {name: distribution([float(r.headers[header]) for r in responses]) for name, header in
                        [('queue_ms', 'x-queue-wait-ms'), ('http_to_headers_ms', 'x-http-app-ms'), ('engine_ms', 'x-engine-ms')]}
                report['natural_c1_12_per_pipeline'] = c1
                report['controlled_visibility'] = []
                for concurrency in (1, 4):
                    entered, release = Event(), Event()
                    original_search = engine.search
                    def held(*a, **kw):
                        entered.set()
                        if not release.wait(10):
                            raise RuntimeError('Controlled hold timed out')
                        return original_search(*a, **kw)
                    engine.search = held
                    try:
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            futures = [pool.submit(search, queries[i], 'hybrid_rerank', f'obs-held-{concurrency}-{i}') for i in range(concurrency)]
                            try:
                                assert entered.wait(10)
                                deadline = monotonic()+5
                                while True:
                                    text = scrape()
                                    observed = {n: total(text, n) for n in ('in_flight_requests', 'requests_waiting_for_engine', 'engine_active_calls')}
                                    if observed == {'in_flight_requests': concurrency, 'requests_waiting_for_engine': concurrency-1, 'engine_active_calls': 1}:
                                        break
                                    assert monotonic() < deadline, observed
                                    sleep(.01)
                            finally:
                                release.set()
                            responses = [f.result() for f in futures]
                    finally:
                        release.set()
                        engine.search = original_search
                    final = scrape()
                    assert all(total(final, n) == 0 for n in ('in_flight_requests', 'requests_waiting_for_engine', 'engine_active_calls'))
                    report['controlled_visibility'].append({'concurrency': concurrency, 'while_first_call_held': observed,
                        'queue_ms': distribution([float(r.headers['x-queue-wait-ms']) for r in responses]),
                        'http_to_headers_ms': distribution([float(r.headers['x-http-app-ms']) for r in responses]),
                        'gauges_after_completion': 0, 'injection': 'test-only Python wrapper holds actual callable until gauges sampled; not latency benchmark'})
                text = scrape()
                assert private_query not in text and 'C:' not in text and str(ROOT) not in text
                assert all(set(s.labels) <= {'pipeline', 'effective_pipeline', 'status', 'le'} for f in text_string_to_metric_families(text) for s in f.samples)
                for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
                    assert any(s.labels['pipeline'] == mode and s.value > 0 for s in samples(text, 'engine_latency_seconds_count'))
                report['metrics_privacy_passed'] = True
                report['final_counts'] = {n: total(text, n) for n in ('request_count_total', 'request_error_count_total', 'fallback_count_total', 'degraded_request_total', 'engine_latency_seconds_count', 'queue_wait_seconds_count')}
            stop(*running[:3])
            running = None
            # Same app, new lifespan, registry reset. One independent engine initialization.
            running = start(app)
            assert running[0].started
            with httpx.Client(base_url=running[3], timeout=30, trust_env=False) as c:
                text = c.get('/metrics').text
                assert total(text, 'request_count_total') == total(text, 'in_flight_requests') == total(text, 'requests_waiting_for_engine') == 0
                report['restart_registry_reset'] = True
            stop(*running[:3])
            running = None
            assert all(e.initialize_count == e.close_count == 1 for e in engines)
            handler.flush()
            log = log_path.read_text(encoding='utf-8')
            assert private_query not in log and all(q not in log for q in queries if len(q) > 5)
            records = [json.loads(line) for line in log.splitlines()]
            assert all('request_id' in r for r in records if r['event'].startswith('search_'))
            required = {'service_starting', 'engine_loading', 'service_ready', 'search_completed', 'search_fallback', 'search_failed', 'service_shutdown'}
            assert required <= {r['event'] for r in records}
            report['logging'] = {'structured_json': True, 'query_text_absent': True, 'request_ids_present': True, 'events': sorted({r['event'] for r in records})}
            report['passed'] = True
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        raise
    finally:
        if running:
            stop(*running[:3])
        report['boundary'] = {'opened': sorted(GUARD.opened), 'blocked': GUARD.blocked, 'final_test_labels_accessed': False}
        dump(args.report, report)
        handler.close()
        logger.removeHandler(handler)
    print(json.dumps({'passed': report['passed'], 'counts': report['final_counts']}))


if __name__ == '__main__':
    main()
