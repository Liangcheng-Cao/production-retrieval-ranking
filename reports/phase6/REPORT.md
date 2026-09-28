# Phase 6 — Reproducible latency and closed-loop load tests

**PASS for the bounded measurement protocol.** No SLA, production capacity, open-loop overload or safe inference deadline is established.

## Phase 5 release and commits

Pre-commit: 122 tests passed (one retained Starlette TestClient deprecation warning); real API release, 18 API/core parity comparisons,
CE fallback and missing-artifact startup integration passed. Phase 1–4 frozen inputs/evidence and artifact exclusions were verified.
Five commits completed; the repository was clean before Phase 6. Full hashes/messages/files are in phase5_commits.json. No push.

## Fixed protocol

Two independent service processes; 24 train queries; modes ['bm25', 'hybrid', 'hybrid_rerank']; K=10; concurrency=[1, 2, 4, 8].
Eight warmups per case; 960 observations per case/run. 23,040 measured HTTP requests and 5,760 direct-core observations.
Same balanced seeded query schedule per case. HTTP case order is shuffled independently per run. All measured runs are retained.
One Uvicorn process and one engine worker; separate load-generator process on the same host. INFO application logs enabled.
Torch/BLAS one thread, deterministic algorithms on, TF32 off; unchanged frozen runtime and no model tuning.

## Environment

- OS: Windows-11-10.0.26200-SP0
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel; logical CPUs: 28
- GPU/driver/memory: NVIDIA GeForce RTX 4070, 581.08, 12282 MiB
- Python: 3.14.3
- Runtime manifest SHA256: 7adf8fcaace380db531b60765678da1dbe4461498e71e313c318cf82d52009b0
Full package versions, source/config hashes, runtime identity, stage metrics and raw-observation checksums are in benchmark_full.json.

## Startup and resources

| Run | Engine startup ms | Process RSS MiB | GPU allocated MiB | GPU reserved MiB |
|---|---:|---:|---:|---:|
| 1 | 1082.50 | 1130.43 | 173.31 | 188.00 |
| 2 | 1079.40 | 1129.73 | 173.31 | 188.00 |

Startup includes artifact verification and model/index loading, excludes process/dependency imports and all warmups/measurements.
Fresh processes do not imply cold OS file caches. Resource snapshots are diagnostic, not peak memory or capacity estimates.

## Warm direct-core latency

| Mode | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| bm25 | 0.22–0.22 | 0.54–0.54 | 0.60–0.62 |
| hybrid | 9.78–9.79 | 10.66–10.88 | 12.31–13.16 |
| hybrid_rerank | 15.21–16.01 | 17.34–19.27 | 18.77–21.19 |

Each cell is the range across the two runs; percentile values are never averaged into a claimed pooled percentile.

## HTTP latency and throughput

| Mode | C | Samples across runs | P50 ms | P95 ms | P99 ms | Successful requests/s |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 1 | 1920 | 2.52–2.60 | 2.99–3.21 | 3.18–3.76 | 363.35–379.60 |
| bm25 | 2 | 1920 | 3.67–3.87 | 4.11–4.31 | 4.48–4.71 | 499.85–526.02 |
| bm25 | 4 | 1920 | 7.45–7.80 | 8.99–10.32 | 12.60–13.29 | 486.67–518.44 |
| bm25 | 8 | 1920 | 15.80–15.82 | 27.38–31.51 | 51.92–60.50 | 394.62–405.76 |
| hybrid | 1 | 1920 | 12.64–12.70 | 13.70–14.66 | 16.28–16.55 | 77.26–77.42 |
| hybrid | 2 | 1920 | 21.27–22.72 | 22.83–26.30 | 26.79–30.59 | 86.48–92.65 |
| hybrid | 4 | 1920 | 42.37–43.83 | 44.29–46.70 | 49.09–60.15 | 90.06–93.68 |
| hybrid | 8 | 1920 | 86.62–87.44 | 91.15–95.15 | 108.69–109.37 | 90.50–91.38 |
| hybrid_rerank | 1 | 1920 | 17.84–18.05 | 19.22–19.31 | 20.09–22.86 | 54.95–55.29 |
| hybrid_rerank | 2 | 1920 | 31.72–33.73 | 33.98–38.90 | 35.78–42.64 | 58.09–62.51 |
| hybrid_rerank | 4 | 1920 | 63.43–63.84 | 67.93–69.90 | 71.48–83.85 | 61.69–62.53 |
| hybrid_rerank | 8 | 1920 | 125.78–128.15 | 130.19–135.49 | 145.77–152.80 | 62.01–63.21 |

## Interpretation and stage separation

Hybrid/CE throughput plateaus as outstanding requests increase because model work remains serialized. Tail latency grows largely outside engine execution.
BM25 is inexpensive in-core, so client scheduling, HTTP/JSON processing and logging occupy a larger fraction of the observed request cost.
These are plausible interpretations of the measured stage separation, not proof of CPU/GPU saturation or a universal concurrency optimum.
Server non-engine time is HTTP app time minus engine time; it includes waiting AND other adapter work, so it must not be labeled pure queue time.

| Mode | C | Server non-engine P95 ms |
|---|---:|---:|
| bm25 | 1 | 0.44–0.44 |
| bm25 | 8 | 3.78–4.52 |
| hybrid | 1 | 0.63–0.66 |
| hybrid | 8 | 78.50–81.43 |
| hybrid_rerank | 1 | 0.61–0.62 |
| hybrid_rerank | 8 | 112.09–116.49 |

Full direct stage distributions (preprocessing, BM25, encoding, dense search, fusion, CE, hydration) and HTTP serialization are retained in the machine report.
The benchmark does not retune ranking or replace Phase 2/3 quality evidence. It provides no new relevance-quality measurement.

## Correctness, errors and boundaries

All 23,040 measured HTTP requests completed successfully; zero unexpected HTTP/transport errors, fallbacks or ranking mismatches.
Every result is compared with the direct-core IDs/order/scores for the same fixture query. Expected mode and fallback state are checked.
Both service processes shut down gracefully to CLOSED. Their boundary logs report no blocked attempts.
**Final-test relevance labels accessed: NO.** No relevance-label files were opened; only products and train query strings were used.

## Preserved failures and pilot

benchmark.json and benchmark_diagnostic.json preserve two failed pre-measurement starts. Thread traces placed the stall in the SciPy native BLAS import
on the owner worker after threadpool limits. Main-thread dependency pre-import resolved the observed harness issue; the native root cause is not proven.
benchmark_verified.json is the successful 96-request pilot. Its short BM25 runs were variable; the formal plan increased every case uniformly to 960 requests.
Pilot and formal results are separate, with original embedded plans/hashes retained. No ranking/service changes or faster-run selection occurred.

## Reproduction and limitations

```powershell
.\.venv\Scripts\python.exe scripts/run_benchmark.py --report reports/tmp/phase6-repeat-summary.json --raw-root reports/tmp/phase6-repeat
```

Paths must be new. Raw per-request observations/logs remain ignored under reports/tmp, referenced by checksum in compact reports.
Closed-loop clients cannot characterize an independently imposed open-loop overload rate (coordinated-omission limitation). The shared desktop and generator can influence results.
Only C≤8, 24 queries and two repetitions were tested. No confidence interval, production SLA, QPS capacity or cross-hardware guarantee is claimed.
No admission cap or hard inference cancellation was implemented. A client timeout does not cancel native/GPU compute.
See docs/benchmarking.md for exact timing, warmup, scheduling, logging and numerical semantics.

## Delivery

Full unit suite and Git/frozen-artifact audit are recorded in checks.json. Phase 6 changes remain uncommitted; no push.
Recommended groups: protocol/statistics/tests; process-isolated benchmark scripts; compact evidence and commit audit; README and benchmark documentation.
Stop after Phase 6. Follow-on admission/deadline/resource policies need separate implementation and verification.
