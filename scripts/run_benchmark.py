"""Bounded closed-loop load harness; client and service run in separate processes."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

import httpx
import numpy as np
from product_search.benchmarking import query_schedule, summarize_requests
from product_search.data.io import file_hash

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


async def case(client, engine, mode, concurrency, plan, seed):
    queries = engine['queries']
    ids = sorted(map(int, queries))
    schedule = query_schedule(ids, plan['requests_per_case'], seed)
    async def request(qid, index):
        start = perf_counter()
        row = {'query_id': qid, 'index': index, 'pipeline': mode, 'status': None}
        try:
            response = await client.post('/search', json={'query': queries[str(qid)], 'pipeline': mode, 'top_k': plan['top_k']})
            row['client_ms'] = (perf_counter()-start)*1000
            row['status'] = response.status_code
            if response.status_code == 200:
                data = response.json()
                expected = engine['expected'][f'{qid}:{mode}']
                tolerance = 1e-4 if mode == 'hybrid_rerank' else 1e-6
                row.update(fallback_used=data['fallback_used'],
                    parity_passed=([h['product_id'] for h in data['results']] == expected['ids']
                        and data['effective_pipeline'] == mode
                        and bool(np.allclose([h['final_score'] for h in data['results']], expected['scores'], atol=tolerance, rtol=tolerance))),
                    engine_ms=data['timing_ms']['total_ms'], http_app_ms=float(response.headers['x-http-app-ms']),
                    serialization_ms=float(response.headers['x-response-serialization-ms']))
                row['server_non_engine_ms'] = max(0, row['http_app_ms']-row['engine_ms'])
            else:
                row['error'] = 'http_error'
        except Exception as exc:
            row['client_ms'] = (perf_counter()-start)*1000
            row['error'] = type(exc).__name__
        return row
    for i, qid in enumerate(schedule[:plan['warmup_requests']]):
        warm = await request(qid, -i-1)
        if warm['status'] != 200 or not warm.get('parity_passed') or warm.get('fallback_used'):
            raise RuntimeError(f'Warmup failed: {warm}')
    # Exactly C long-lived closed-loop clients; no unbounded gather of all requests.
    cursor = iter(enumerate(schedule))
    rows = []
    async def worker():
        for index, qid in cursor:
            rows.append(await request(qid, index))
    start = perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = perf_counter()-start
    rows.sort(key=lambda r: r['index'])
    return rows, summarize_requests(rows, elapsed)


async def run(plan, run_id, directory):
    directory.mkdir(parents=True, exist_ok=False)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    log = (directory/'service.log').open('w', encoding='utf-8')
    child = subprocess.Popen([sys.executable, str(ROOT/'scripts/benchmark_service.py'), '--directory', str(directory), '--run', str(run_id)],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, creationflags=flags)
    try:
        deadline = perf_counter()+180
        url = None
        async with httpx.AsyncClient(timeout=1, trust_env=False) as probe:
            while perf_counter() < deadline:
                if child.poll() is not None:
                    raise RuntimeError('Service child exited before readiness; see ignored service log')
                if (directory/'address.json').exists():
                    try:
                        address = json.loads((directory/'address.json').read_text())
                        url = f"http://127.0.0.1:{address['port']}"
                        response = await probe.get(url+'/ready')
                        if response.status_code == 200 and response.json()['ready']:
                            break
                    except (ValueError, httpx.HTTPError):
                        pass
                await asyncio.sleep(.1)
            else:
                raise TimeoutError('Startup readiness deadline exceeded')
        engine = json.loads((directory/'engine.json').read_text())
        combinations = [(m, c) for m in plan['pipelines'] for c in plan['concurrency']]
        random.Random(plan['seed']+run_id).shuffle(combinations)
        result = {'run': run_id, 'startup_wall_ms': engine['startup_wall_ms'], 'startup_stages': engine['startup_stages'],
            'core': engine['core'], 'runtime_version': engine['version'],
            'resources_after_startup': engine['resources_after_startup'], 'resources_after_core': engine['resources_after_core'],
            'query_ids': sorted(map(int, engine['queries'])), 'cases': []}
        async with httpx.AsyncClient(base_url=url, timeout=plan['request_timeout_seconds'], trust_env=False,
                    limits=httpx.Limits(max_connections=max(plan['concurrency']), max_keepalive_connections=max(plan['concurrency']))) as client:
            for mode, concurrency in combinations:
                rows, summary = await case(client, engine, mode, concurrency, plan, plan['seed'])
                raw = directory/f'{mode}-c{concurrency}.json'
                dump(raw, rows)
                result['cases'].append({'pipeline': mode, 'concurrency': concurrency, **summary,
                    'raw_sha256': file_hash(raw), 'raw_path': raw.relative_to(ROOT).as_posix()})
                print(f"run={run_id} {mode} c={concurrency} n={summary['attempts']} p95={summary['client_success']['p95_ms']:.2f}ms qps={summary['successful_requests_per_second']:.1f} errors={summary['errors']}", flush=True)
        return result
    finally:
        if child.poll() is None:
            child.stdin.write('stop\n')
            child.stdin.flush()
            try:
                await asyncio.to_thread(child.wait, timeout=45)
            except subprocess.TimeoutExpired:
                child.terminate()
                await asyncio.to_thread(child.wait, timeout=10)
                raise RuntimeError('Service required forced termination')
        log.close()
        child.stdin.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=ROOT/'reports/phase6/benchmark.json')
    parser.add_argument('--raw-root', type=Path, default=ROOT/'reports/tmp/phase6')
    args = parser.parse_args()
    if args.report.exists() or args.raw_root.exists():
        parser.error('Refusing to overwrite benchmark evidence; choose new paths')
    args.raw_root = args.raw_root.resolve()
    if not args.raw_root.is_relative_to(ROOT/'reports/tmp'):
        parser.error('Raw observations must stay under ignored reports/tmp')
    plan_path = ROOT/'configs/phase6_plan.json'
    plan = json.loads(plan_path.read_text())
    report = {'plan': plan, 'plan_sha256': file_hash(plan_path), 'runs': [], 'passed': False,
        'environment': {'platform': platform.platform(), 'python': platform.python_version(), 'cpu': platform.processor(),
            'logical_cpus': os.cpu_count(), 'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'torch', 'fastapi', 'uvicorn', 'httpx')},
            'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()},
        'source_sha256': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
            [ROOT/'scripts/run_benchmark.py', ROOT/'scripts/benchmark_service.py', ROOT/'src/product_search/benchmarking.py']}}
    try:
        for number in range(1, plan['independent_runs']+1):
            directory = args.raw_root/f'run-{number}'
            result = await run(plan, number, directory)
            result['shutdown'] = json.loads((directory/'shutdown.json').read_text())
            assert result['shutdown']['state'] == 'CLOSED' and not result['shutdown']['blocked_attempts']
            report['runs'].append(result)
            dump(args.report, report)
        assert report['runs'][0]['runtime_version'] == report['runs'][1]['runtime_version']
        report['passed'] = all(c['attempts'] == plan['requests_per_case'] and not
            (c['errors'] or c['fallbacks'] or c['ranking_mismatches']) for r in report['runs'] for c in r['cases'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        dump(args.report, report)
    print(json.dumps({'passed': report['passed'], 'report': str(args.report)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
