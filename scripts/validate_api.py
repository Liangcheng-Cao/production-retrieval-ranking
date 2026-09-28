"""Sequential real HTTP/core parity and fault checks; no load benchmark or labels."""
import argparse
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
import threading
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from product_search.development_data import BoundaryGuard
GUARD = BoundaryGuard(ROOT).install()
from product_search.runtime_config import RuntimeConfig
from product_search.search_engine import SearchEngine
from product_search.api import create_app
from product_search.api_schemas import public_version
from validate_runtime import fixture_queries


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


class ObservedEngine(SearchEngine):
    initialize_count = 0
    close_count = 0
    def initialize(self):
        self.initialize_count += 1
        return super().initialize()
    def close(self):
        self.close_count += 1
        return super().close()


def start(app):
    import uvicorn
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, access_log=False, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = monotonic()+60
    while not server.started and thread.is_alive() and monotonic() < deadline:
        sleep(.05)
    return server, thread, sock, f'http://127.0.0.1:{port}'


def stop(server, thread, sock):
    server.should_exit = True
    thread.join(30)
    sock.close()
    if thread.is_alive():
        raise RuntimeError('Local service failed graceful shutdown')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=ROOT/'reports/phase5/http_integration.json')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Refusing to overwrite HTTP integration evidence')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # Raw execution logs are temporary, ignored; reports retain compact evidence.
    log_path = ROOT/'reports/tmp/phase5-service.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    service_logger = logging.getLogger('product_search.service')
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    service_logger.addHandler(handler)
    import torch
    import numpy as np
    import httpx
    from threadpoolctl import threadpool_limits
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'passed': False, 'execution': 'sequential real loopback TCP requests, one service worker'}
    direct = None
    running = None
    try:
        config = RuntimeConfig.load(ROOT/'configs/runtime.json')
        with threadpool_limits(limits=1):
            direct = SearchEngine.from_config(config)
            owned = []
            def factory():
                engine = ObservedEngine(config)
                owned.append(engine)
                return engine
            app = create_app(engine_factory=factory)
            running = start(app)
            server, thread, sock, url = running
            if not server.started:
                raise AssertionError('Real API did not start')
            with httpx.Client(base_url=url, timeout=60, trust_env=False) as client:
                endpoint_checks = []
                for endpoint in ('/health', '/ready', '/version', '/openapi.json'):
                    response = client.get(endpoint)
                    assert response.status_code == 200, (endpoint, response.status_code)
                    endpoint_checks.append({'endpoint': endpoint, 'status': response.status_code,
                        'state': response.json().get('state'), 'request_id_present': bool(response.headers.get('x-request-id'))})
                version_text = client.get('/version').text
                assert 'C:' not in version_text and 'validated_artifact' not in version_text
                assert client.get('/version').json() == public_version(direct.version).model_dump()
                comparisons = []
                for qid, query in fixture_queries().items():
                    for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
                        expected = direct.search(query, 10, mode)
                        response = client.post('/search', json={'query': query, 'top_k': 10, 'pipeline': mode},
                                               headers={'X-Request-ID': f'parity-{qid}-{mode}'})
                        assert response.status_code == 200
                        actual = response.json()
                        assert actual['request_id'] == response.headers['x-request-id'] == f'parity-{qid}-{mode}'
                        ids = [h['product_id'] for h in actual['results']]
                        assert ids == [h.product_id for h in expected.results], (qid, mode, 'order')
                        assert actual['effective_pipeline'] == expected.effective_pipeline
                        assert actual['fallback_used'] == expected.fallback_used
                        tolerance = 1e-4 if mode == 'hybrid_rerank' else 1e-6
                        errors = {}
                        for field in ('final_score', 'retrieval_score', 'reranker_score'):
                            got = [h[field] for h in actual['results']]
                            want = [getattr(h, field) for h in expected.results]
                            assert [v is None for v in got] == [v is None for v in want]
                            pairs = [(a, b) for a, b in zip(got, want) if a is not None]
                            assert np.allclose([a for a, b in pairs], [b for a, b in pairs], atol=tolerance, rtol=tolerance), (qid, mode, field)
                            errors[field] = max((abs(a-b) for a, b in pairs), default=0)
                        comparisons.append({'query_id': qid, 'pipeline': mode, 'status': response.status_code,
                            'ordered_ids': ids, 'max_score_abs_errors': errors, 'passed': True,
                            'fallback_used': actual['fallback_used'], 'effective_pipeline': actual['effective_pipeline'],
                            'http_app_ms': float(response.headers['x-http-app-ms']),
                            'engine_total_ms': actual['timing_ms']['total_ms'],
                            'serialization_ms': float(response.headers['x-response-serialization-ms'])})
                invalid = []
                for name, override in [('pipeline', {'pipeline': 'dense'}), ('top_k', {'top_k': 101}),
                    ('empty', {'query': ' '}), ('ce_depth', {'pipeline': 'hybrid_rerank', 'top_k': 21})]:
                    response = client.post('/search', json={'query': 'synthetic desk', 'pipeline': 'hybrid', 'top_k': 10, **override})
                    assert response.status_code == 422
                    invalid.append({'case': name, 'status': response.status_code, 'error': response.json()['error']})
                bad_json = client.post('/search', content='{bad', headers={'Content-Type': 'application/json'})
                assert bad_json.status_code == 422
                invalid.append({'case': 'malformed_json', 'status': bad_json.status_code})
                # Test-only injection after ordinary requests; no diagnostic HTTP endpoint.
                engine = owned[0]
                original = engine._resources.reranker.scorer
                class Broken:
                    def score_pairs(self, *args):
                        raise RuntimeError('synthetic CE failure')
                engine._resources.reranker.scorer = Broken()
                try:
                    payload = {'query': 'wooden office desk', 'top_k': 10, 'pipeline': 'hybrid'}
                    hybrid = client.post('/search', json=payload).json()
                    fallback_response = client.post('/search', json={**payload, 'pipeline': 'hybrid_rerank'})
                    fallback = fallback_response.json()
                    assert fallback_response.status_code == 200 and fallback['fallback_used']
                    assert fallback['requested_pipeline'] == 'hybrid_rerank' and fallback['effective_pipeline'] == 'hybrid'
                    assert [h['product_id'] for h in fallback['results']] == [h['product_id'] for h in hybrid['results']]
                    assert all(h['reranker_score'] is None and h['final_score'] == h['retrieval_score'] for h in fallback['results'])
                    degraded = client.get('/ready')
                    assert degraded.status_code == 200 and degraded.json()['state'] == 'DEGRADED'
                    result['fallback'] = {'status': 200, 'requested_pipeline': fallback['requested_pipeline'],
                        'effective_pipeline': fallback['effective_pipeline'], 'fallback_used': True,
                        'hybrid_order_equal': True, 'ready_status': degraded.status_code, 'readiness': degraded.json()}
                finally:
                    engine._resources.reranker.scorer = original
                recovery = client.post('/search', json={**payload, 'pipeline': 'hybrid_rerank'})
                assert recovery.status_code == 200 and not recovery.json()['fallback_used']
                assert client.get('/ready').json()['ready']
                result.update(endpoints=endpoint_checks, invalid_requests=invalid,
                    parity={'query_count': 6, 'comparison_count': len(comparisons), 'all_passed': True, 'comparisons': comparisons},
                    public_version=client.get('/version').json())
            stop(*running[:3])
            running = None
            assert owned[0].initialize_count == owned[0].close_count == 1
            result['lifecycle'] = {'initialize_count': 1, 'close_count': 1, 'shutdown_state': owned[0].readiness().state}
            # Missing required manifest under a temporary root, no frozen file altered.
            with tempfile.TemporaryDirectory(prefix='phase5-startup-') as directory:
                bad_config = replace(config, root=Path(directory), manifest='missing-runtime.json')
                broken = []
                def bad_factory():
                    e = ObservedEngine(bad_config)
                    broken.append(e)
                    return e
                bad_app = create_app(engine_factory=bad_factory)
                running = start(bad_app)
                failed_server, failed_thread, failed_socket, failed_url = running
                assert not failed_server.started and not failed_thread.is_alive()
                assert not (Path(directory)/'missing-runtime.json').exists()
                assert broken[0].close_count == 1
                result['startup_failure'] = {'missing_required_artifact': True, 'server_started': False,
                    'startup_thread_exited': True, 'silent_rebuild': False, 'engine_closed': True}
                stop(*running[:3])
                running = None
            result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        logging.exception('HTTP integration failed')
    finally:
        if running:
            stop(*running[:3])
        if direct:
            direct.close()
        result['boundary'] = {'opened_canonical_files': sorted(GUARD.opened), 'blocked_attempts': GUARD.blocked,
                              'final_test_labels_accessed': False, 'relevance_labels_opened': False}
        dump(args.report, result)
        handler.close()
        service_logger.removeHandler(handler)
    print(json.dumps({'passed': result['passed'], 'report': str(args.report), 'error': result.get('error')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
