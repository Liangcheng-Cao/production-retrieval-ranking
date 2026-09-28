"""Summarize both formal runs without selecting the faster one or averaging tails."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def span(values, digits=2):
    return f'{min(values):.{digits}f}–{max(values):.{digits}f}'


def main():
    source = ROOT/'reports/phase6/benchmark_full.json'
    data = json.loads(source.read_text())
    if not data['passed'] or len(data['runs']) != 2:
        raise ValueError('Both formal benchmark runs must pass before reporting')
    runs = data['runs']
    plan = data['plan']
    rows = []
    for mode in plan['pipelines']:
        for c in plan['concurrency']:
            cases = [next(x for x in r['cases'] if x['pipeline'] == mode and x['concurrency'] == c) for r in runs]
            rows.append({'pipeline': mode, 'concurrency': c, 'requests': sum(x['attempts'] for x in cases),
                'qps_range': [min(x['successful_requests_per_second'] for x in cases), max(x['successful_requests_per_second'] for x in cases)],
                'p50_range_ms': [min(x['client_success']['p50_ms'] for x in cases), max(x['client_success']['p50_ms'] for x in cases)],
                'p95_range_ms': [min(x['client_success']['p95_ms'] for x in cases), max(x['client_success']['p95_ms'] for x in cases)],
                'p99_range_ms': [min(x['client_success']['p99_ms'] for x in cases), max(x['client_success']['p99_ms'] for x in cases)],
                'server_non_engine_p95_range_ms': [min(x['server_non_engine_ms']['p95_ms'] for x in cases), max(x['server_non_engine_ms']['p95_ms'] for x in cases)],
                'errors': sum(x['errors'] for x in cases), 'fallbacks': sum(x['fallbacks'] for x in cases),
                'ranking_mismatches': sum(x['ranking_mismatches'] for x in cases)})
    comparison = {'aggregation': 'min/max of two independent run statistics, not pooled or averaged percentiles', 'rows': rows}
    (ROOT/'reports/phase6/comparison.json').write_text(json.dumps(comparison, indent=2)+'\n', encoding='utf-8')
    total = sum(x['requests'] for x in rows)
    lines = ['# Phase 6 — Reproducible latency and closed-loop load tests', '',
        '**PASS for the bounded measurement protocol.** No SLA, production capacity, open-loop overload or safe inference deadline is established.', '',
        '## Phase 5 release and commits', '',
        'Pre-commit: 122 tests passed (one retained Starlette TestClient deprecation warning); real API release, 18 API/core parity comparisons,',
        'CE fallback and missing-artifact startup integration passed. Phase 1–4 frozen inputs/evidence and artifact exclusions were verified.',
        'Five commits completed; the repository was clean before Phase 6. Full hashes/messages/files are in phase5_commits.json. No push.', '',
        '## Fixed protocol', '',
        f"Two independent service processes; 24 train queries; modes {plan['pipelines']}; K={plan['top_k']}; concurrency={plan['concurrency']}.",
        f"Eight warmups per case; {plan['requests_per_case']} observations per case/run. {total:,} measured HTTP requests and {2*3*plan['requests_per_case']:,} direct-core observations.",
        'Same balanced seeded query schedule per case. HTTP case order is shuffled independently per run. All measured runs are retained.',
        'One Uvicorn process and one engine worker; separate load-generator process on the same host. INFO application logs enabled.',
        'Torch/BLAS one thread, deterministic algorithms on, TF32 off; unchanged frozen runtime and no model tuning.', '',
        '## Environment', '',
        f"- OS: {data['environment']['platform']}",
        f"- CPU: {data['environment']['cpu']}; logical CPUs: {data['environment']['logical_cpus']}",
        f"- GPU/driver/memory: {data['environment']['gpu']}",
        f"- Python: {data['environment']['python']}",
        f"- Runtime manifest SHA256: {runs[0]['runtime_version']['runtime_manifest_sha256']}",
        'Full package versions, source/config hashes, runtime identity, stage metrics and raw-observation checksums are in benchmark_full.json.', '',
        '## Startup and resources', '',
        '| Run | Engine startup ms | Process RSS MiB | GPU allocated MiB | GPU reserved MiB |',
        '|---|---:|---:|---:|---:|']
    for r in runs:
        resources = r['resources_after_startup']
        lines.append(f"| {r['run']} | {r['startup_wall_ms']:.2f} | {resources['process_rss_bytes']/2**20:.2f} | {resources['gpu']['allocated_bytes']/2**20:.2f} | {resources['gpu']['reserved_bytes']/2**20:.2f} |")
    lines += ['', 'Startup includes artifact verification and model/index loading, excludes process/dependency imports and all warmups/measurements.',
              'Fresh processes do not imply cold OS file caches. Resource snapshots are diagnostic, not peak memory or capacity estimates.', '',
              '## Warm direct-core latency', '', '| Mode | P50 ms | P95 ms | P99 ms |', '|---|---:|---:|---:|']
    for mode in plan['pipelines']:
        lines.append('| '+mode+' | '+' | '.join(span([r['core'][mode]['wall_ms'][p] for r in runs]) for p in ('p50_ms', 'p95_ms', 'p99_ms'))+' |')
    lines += ['', 'Each cell is the range across the two runs; percentile values are never averaged into a claimed pooled percentile.', '',
              '## HTTP latency and throughput', '', '| Mode | C | Samples across runs | P50 ms | P95 ms | P99 ms | Successful requests/s |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"| {x['pipeline']} | {x['concurrency']} | {x['requests']} | "+' | '.join(span(x[k]) for k in ('p50_range_ms','p95_range_ms','p99_range_ms','qps_range'))+' |')
    lines += ['', '## Interpretation and stage separation', '',
        'Hybrid/CE throughput plateaus as outstanding requests increase because model work remains serialized. Tail latency grows largely outside engine execution.',
        'BM25 is inexpensive in-core, so client scheduling, HTTP/JSON processing and logging occupy a larger fraction of the observed request cost.',
        'These are plausible interpretations of the measured stage separation, not proof of CPU/GPU saturation or a universal concurrency optimum.',
        'Server non-engine time is HTTP app time minus engine time; it includes waiting AND other adapter work, so it must not be labeled pure queue time.', '',
        '| Mode | C | Server non-engine P95 ms |', '|---|---:|---:|']
    for x in rows:
        if x['concurrency'] in (1,8):
            lines.append(f"| {x['pipeline']} | {x['concurrency']} | {span(x['server_non_engine_p95_range_ms'])} |")
    lines += ['', 'Full direct stage distributions (preprocessing, BM25, encoding, dense search, fusion, CE, hydration) and HTTP serialization are retained in the machine report.',
        'The benchmark does not retune ranking or replace Phase 2/3 quality evidence. It provides no new relevance-quality measurement.', '',
        '## Correctness, errors and boundaries', '',
        f"All {total:,} measured HTTP requests completed successfully; zero unexpected HTTP/transport errors, fallbacks or ranking mismatches.",
        'Every result is compared with the direct-core IDs/order/scores for the same fixture query. Expected mode and fallback state are checked.',
        'Both service processes shut down gracefully to CLOSED. Their boundary logs report no blocked attempts.',
        '**Final-test relevance labels accessed: NO.** No relevance-label files were opened; only products and train query strings were used.', '',
        '## Preserved failures and pilot', '',
        'benchmark.json and benchmark_diagnostic.json preserve two failed pre-measurement starts. Thread traces placed the stall in the SciPy native BLAS import',
        'on the owner worker after threadpool limits. Main-thread dependency pre-import resolved the observed harness issue; the native root cause is not proven.',
        'benchmark_verified.json is the successful 96-request pilot. Its short BM25 runs were variable; the formal plan increased every case uniformly to 960 requests.',
        'Pilot and formal results are separate, with original embedded plans/hashes retained. No ranking/service changes or faster-run selection occurred.', '',
        '## Reproduction and limitations', '',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/run_benchmark.py --report reports/tmp/phase6-repeat-summary.json --raw-root reports/tmp/phase6-repeat',
        '```', '',
        'Paths must be new. Raw per-request observations/logs remain ignored under reports/tmp, referenced by checksum in compact reports.',
        'Closed-loop clients cannot characterize an independently imposed open-loop overload rate (coordinated-omission limitation). The shared desktop and generator can influence results.',
        'Only C≤8, 24 queries and two repetitions were tested. No confidence interval, production SLA, QPS capacity or cross-hardware guarantee is claimed.',
        'No admission cap or hard inference cancellation was implemented. A client timeout does not cancel native/GPU compute.',
        'See docs/benchmarking.md for exact timing, warmup, scheduling, logging and numerical semantics.', '',
        '## Delivery', '',
        'Full unit suite and Git/frozen-artifact audit are recorded in checks.json. Phase 6 changes remain uncommitted; no push.',
        'Recommended groups: protocol/statistics/tests; process-isolated benchmark scripts; compact evidence and commit audit; README and benchmark documentation.',
        'Stop after Phase 6. Follow-on admission/deadline/resource policies need separate implementation and verification.']
    (ROOT/'reports/phase6/REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    print('Formal report written; all repetitions included')


if __name__ == '__main__':
    main()
