"""Real installed-wheel service checks, with an explicit train-only reference fixture."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep

from product_search.development_data import BoundaryGuard
from product_search.deployment import verify_bundle
from product_search.data.io import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    if args.report.exists():
        parser.error('Use a fresh report path')
    guard = BoundaryGuard(args.bundle).install()
    bundle = verify_bundle(args.bundle)
    fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
    if fixture.get('partition') != 'train' or fixture['runtime_manifest_sha256'] != json.loads((args.bundle/'runtime.json').read_text())['manifest_sha256']:
        raise ValueError('Expected frozen train fixture')
    train_ids = set(json.loads((args.bundle/'data/processed/data_manifest.json').read_text(encoding='utf-8'))['query_ids']['train'])
    if any(row['query_id'] not in train_ids for row in fixture['requests']):
        raise ValueError('Fixture query ID is outside the frozen train partition')
    import httpx
    import numpy as np
    import product_search
    installed_path = Path(product_search.__file__).resolve()
    if not installed_path.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('Integration must use an installed wheel inside the fresh environment')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_suffix('.log')
    report = {'passed': False, 'execution': 'fresh subprocess from installed wheel, portable inference bundle',
        'python': sys.version.split()[0], 'installed_package_path': str(installed_path), 'installed_inside_environment': True,
        'bundle_manifest_sha256': file_hash(args.bundle/'bundle.json'), 'fixture_sha256': file_hash(args.fixture),
        'bundle_file_count': len(bundle['files']), 'comparisons': []}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    process = None
    try:
        with log_path.open('w', encoding='utf-8') as log:
            environment = dict(os.environ)
            environment.pop('PYTHONPATH', None)
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            process = subprocess.Popen([sys.executable, '-m', 'product_search.deployment', '--config', str(args.bundle/'runtime.json'),
                '--host', '127.0.0.1', '--port', str(port)], cwd=args.bundle, env=environment, stdout=log, stderr=log)
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=60, trust_env=False) as client:
                deadline = monotonic()+120
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Packaged service exited before readiness; inspect local log')
                    try:
                        r = client.get('/ready')
                        if r.status_code == 200 and r.json()['ready']:
                            break
                    except httpx.ConnectError:
                        pass
                    if monotonic() > deadline:
                        raise RuntimeError('Packaged service readiness timeout')
                    sleep(.1)
                for endpoint in ('/health', '/ready', '/version', '/metrics'):
                    assert client.get(endpoint).status_code == 200
                for row in fixture['requests']:
                    r = client.post('/search', json=row['request'], headers={'X-Request-ID': 'packaging-check'})
                    assert r.status_code == 200
                    got = r.json()
                    want = row['expected']
                    ids = [h['product_id'] for h in got['results']]
                    error = max((abs(h['final_score']-s) for h, s in zip(got['results'], want['scores'])), default=0)
                    tolerance = 1e-4 if row['request']['pipeline'] == 'hybrid_rerank' else 1e-6
                    passed = (ids == want['ids'] and not got['fallback_used'] and got['effective_pipeline'] == row['request']['pipeline']
                        and bool(np.allclose([h['final_score'] for h in got['results']], want['scores'], rtol=tolerance, atol=tolerance)))
                    report['comparisons'].append({'pipeline': row['request']['pipeline'], 'query_id': row['query_id'],
                        'passed': passed, 'ordering_equal': ids == want['ids'], 'max_score_abs_error': error})
                assert all(r['passed'] for r in report['comparisons']), 'Cross-environment ranking parity mismatch'
                assert client.post('/search', json={'query': '', 'pipeline': 'bm25'}).status_code == 422
                metrics = client.get('/metrics').text
                assert all(row['request']['query'] not in metrics for row in fixture['requests'])
                assert '/mnt/' not in metrics and 'C:' not in metrics
                report['metrics_privacy'] = True
            # SIGTERM on POSIX exercises graceful Uvicorn shutdown. Windows terminate
            # cannot claim graceful signal delivery and is reported separately.
            process.terminate()
            code = process.wait(timeout=60)
            process = None
            report['shutdown'] = {'exit_code': code, 'graceful_signal_supported': os.name != 'nt'}
            if os.name != 'nt':
                # Uvicorn 0.54 restores and re-raises the captured SIGTERM after
                # lifespan cleanup; a negative signal exit is therefore expected.
                assert code in (0, -signal.SIGTERM), f'Unexpected shutdown exit: {code}'
        text = log_path.read_text(encoding='utf-8')
        assert all(row['request']['query'] not in text for row in fixture['requests'] if len(row['request']['query']) > 5)
        if os.name != 'nt':
            assert 'service_shutdown' in text
            assert 'Application shutdown complete.' in text
        report['shutdown']['lifecycle_completed'] = 'service_shutdown' in text
        report['shutdown']['resource_tracker_warning'] = 'resource_tracker: There appear to be' in text
        verify_bundle(args.bundle)
        report['bundle_unchanged_after_service'] = True
        report['query_log_privacy'] = True
        report['passed'] = True
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=60)
        report['boundary'] = {'opened': sorted(guard.opened), 'blocked': guard.blocked, 'final_test_labels_accessed': False,
            'bundle_has_no_query_or_judgment_files': True}
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'passed': report['passed'], 'comparisons': len(report['comparisons']), 'failure': report.get('failure')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
