# Phase 7 — Observability, ML Monitoring & Production Hardening

## A. Phase 6 Commits

| Hash | Message |
|---|---|
| dfc7f6c68e1b943605ed09a283704449deda0284 | feat: define reproducible benchmark protocol and aggregation |
| 9de2f64203a2a770505a937d2c45813a7475ab74 | feat: add isolated benchmark client and execution tooling |
| e6d921c1a4243e0029f97ad73c7a7940ec6124a7 | report: record benchmark evidence and repeatability validation |
| 77ec3acd58f7ec80741e3472b6763233b2e82dd3 | docs: document benchmark methodology and Phase 6 results |

Repository was clean before Phase 7. HEAD remains the last Phase 6 commit.

## B. Observability Architecture

HTTP → RequestContextMiddleware (request ID, in-flight, HTTP lifetime)
→ create_app.search → WaitingTicket (admission/actual-start wait)
→ existing asyncio.Lock + single ThreadPoolExecutor worker
→ unchanged SearchEngine → ServiceMetrics + terminal JSON event.

Each app lifespan owns a new CollectorRegistry. No workers, ranking stages or models
were added. SearchEngine source and frozen selection/configuration remain unchanged.

## C. Metrics

P=pipeline, E=effective_pipeline, S=status. P/E bounded to three modes plus unknown;
S bounded to 200/400/404/405/413/422/499/500/503/other.

| Metric | Type | Labels |
|---|---|---|
| request_count_total | counter | P,E,S |
| request_error_count_total | counter | P,E,S |
| fallback_count_total | counter | P,E,S |
| reranker_fallback_total | counter (fallback alias) | P,E,S |
| degraded_request_total | counter | P,E,S |
| empty_result_total | counter | P,E,S |
| request_latency_seconds | histogram | P,E,S |
| engine_latency_seconds | histogram | P |
| queue_wait_seconds | histogram | P |
| result_count | histogram | P,E,S |
| query_character_length | histogram | P |
| query_token_count | histogram | P |
| bm25_latency_seconds | histogram | P |
| dense_encoding_latency_seconds | histogram | P |
| dense_search_latency_seconds | histogram | P |
| fusion_latency_seconds | histogram | P |
| reranking_latency_seconds | histogram | P |
| hydration_latency_seconds | histogram | P |
| in_flight_requests | gauge | none |
| requests_waiting_for_engine | gauge | none |
| engine_active_calls | gauge | none |

Standard histogram _bucket/_sum/_count/_created and counter _created samples apply;
bucket samples add only bounded `le`. No high-cardinality or sensitive labels.
Logical request_count/request_error_count/fallback_count use standard _total suffixes.
See docs/observability.md for exact population, failure and cancellation semantics.

## D. /metrics Endpoint

HTTP200, Prometheus text0.0.4. Only serializes bounded in-memory metrics; no engine
readiness, inference, artifact hashing or initialization. Real TCP integration
passed response/content-type/privacy checks. Probes do not increment search counts.
Counters reset with each new lifespan; restart validation passed.

## E. Queue / Saturation Visibility

Exact admitted-call wait ends at executor callable entry, including both lock and
dispatch waits. Active ASGI calls and actual engine callables are measured separately;
native work can continue after caller cancellation. No inferred internal queue depth.

| Controlled condition | In-flight | Waiting | Actual engine calls | Queue P95 ms |
|---|---:|---:|---:|---:|
| C1 | 1 | 0 | 1 | .0684 |
| C4 | 4 | 3 | 1 | 78.3985 |

First callable was deliberately held until metrics were sampled. HTTP-to-headers P95
was40.7987/99.7000ms; these are controlled visibility observations, **not** natural
benchmark results. All gauges returned to zero. They expose Phase6's single-owner
waiting mechanism; Phase6 remains the evidence for the throughput plateau.
Separate natural C1 exact queue P95 (12/mode): BM25 .029825ms, hybrid .033740ms,
CE .041960ms. No throughput optimization or broad benchmark rerun.

## F. Logging

service_starting, engine_loading, service_ready, search_completed, search_fallback,
search_failed, service_shutdown verified as JSON. Terminal search records include
request_id, pipeline/effective_pipeline, K, query length, queue/engine/HTTP timings,
fallback, result count, status; unavailable fields are null. Existing request-ID
validation/UUID replacement remains unchanged. Query text/results are not persisted;
request-path exception messages/tracebacks are suppressed to prevent query leakage.
Request IDs never become metric labels. No online A/B test has been performed.

## G. ML Monitoring Baseline

All288 frozen train queries; source query-ID/checksum provenance and codepoint/token/
class distributions, normalized384D embedding centroid/mean norm, CE score summaries.
Models/revisions and complete feature definitions are in baseline.json and the plan.
Five source query_class values are empty strings; these remain an explicit category.
Only train labels support original-query offline quality diagnostics.

Version: monitoring-v1. Baseline canonical payload SHA256:
`1923f498d2a222eca6b2154ff5713808c0eefb14a2f39de15d1008906e321994`.
Replay payload SHA256:
`7deee251d3bc91930effc1f4ebcf6443246352ea71430c97d5dcba27a0f015aa`.
Two independent engine loads in one process produced exactly equal payloads including
float summaries and ranking checksums; timestamps/timings are outside this comparison.

## H. Synthetic Drift Scenarios

Each scenario has288 rows derived only from train:

- long_modifiers: append the protocol's fixed69-character/10-token modifier suffix.
- category_mix: cycle the10 Accent Chairs queries (most common class, alphabetical
  tie rule) to288 rows; intentional repeated traffic, 100% target category.
- short_generic: first whitespace token of every source query.

Modified texts retain source class annotations, not newly inferred semantic labels.
Original judgments are never applied to transformed text for quality claims.

## I. Drift Results

| Scenario | Character Wasserstein | Token Wasserstein | Class TV | Centroid distance |
|---|---:|---:|---:|---:|
| long_modifiers | 69.0000 | 10.0000 | 0 | .395037 |
| category_mix | 2.0486 | .3160 | .965278 | .274991 |
| short_generic | 14.9688 | 2.3681 | 0 | .328702 |

These are synthetic/offline drift signals, not observed production drift.

## J. Embedding / Reranker Monitoring

Mean L2 norms1.00000002–1.00000003 across populations. No embedding retraining.
Mean CE top1 / top1-top2 margin: baseline4.925546/1.350461;
long-.382326/.842044; category4.738119/1.424939; short2.315998/1.029670.
Count/mean/P05/P50/P95 for returned CE20 scores are recorded in monitoring.json.
CrossEncoder logits are not calibrated probabilities. Score shifts do not establish
relevance-rate changes; centroid drift does not prove ranking-quality degradation.
Exact local rebuild equality is not a cross-hardware floating-point guarantee.

## K. Champion / Challenger Offline Replay

Original train:288 queries,576 search calls/build. Champion hybrid versus existing
hybrid_rerank challenger atK20; frozen CE20. 100% of Top10 ordered lists differ,
mean Top10 overlap=.628819; mean absolute shared-item rank displacement=2.597973.
Both returned20 results/query, zero empty results/fallback, requested=effective mode.

| Diagnostic | hybrid | hybrid_rerank |
|---|---:|---:|
| Train NDCG10 | .707188 | .725127 |
| Train NDCG20 | .698653 | .705218 |
| Train Recall10 | .061278 | .061176 |
| Train Recall20 | .115067 | .115067 |
| Core mean ms, run1 | 9.823 | 15.695 |
| Core P95 ms, run1 | 11.045 | 17.732 |
| Core mean ms, run2 | 9.614 | 15.183 |
| Core P95 ms, run2 | 10.771 | 16.981 |

Zero excluded queries for shown quality metrics. Existing evaluation definitions
are unchanged. K20 output cannot support full-depth Recall50/100 claims. This is
in-sample **offline replay**, not model selection, unbiased test evaluation or online
A/B testing. Four populations/two builds total4,608 searches. Latency is sequential
core diagnostic timing, includes possible initial warmup, and is not HTTP latency.

## L. Diagnostic Thresholds

configs/phase7_diagnostics.json defines local diagnostics: any fallback/error>0,
readiness!=READY; queueP95>5ms; requestP95 BM25>10ms, hybrid>50ms, CE>60ms.
Latency requires >=20 observations/mode in an inspected5-minute window. No traffic
means unknown. Request thresholds round ~3x Phase6 C1 RTT maxima upward; applying
them to server ASGI time is explicitly a conservative proxy. Exact queue C1 samples
from this phase support a coarse5ms warning; Phase6 residual is not queue time.
Histogram P95 is interpolated. No production SLA/SLO or deployed alert manager.

## M. Integration Validation

observability_verified.json:47 requests,46 successful core results, one intentionally
injected500, one intentionally injected CE fallback HTTP200, two degraded requests
(fallback and recovery admitted during DEGRADED). Zero unexpected errors/fallbacks.
Every mode, probes, queue/in-flight, labels/privacy, JSON logs and restart passed.

The earlier observability_integration.json is retained with passed=false. Metrics,
queue and restart succeeded but its file log was empty: Uvicorn dictConfig closed
the harness's write-mode FileHandler. Append mode fixes safe reopening; independent
rerun passed. This harness issue did not change engine/service execution semantics.
Separate API integration also passed startup-failure/cleanup and fallback checks.

## N. Ranking Parity

**NO ranking change.** 18HTTP/core and48independent offline/core comparisons passed,
exact ordered IDs and maximum score absolute difference0. Existing CE tolerance1e-4
and other score tolerance1e-6 were retained. Historical Phase3 original batch context
was checked separately; its known near-tie caveat is not erased. Engine source,
frozen BM25/dense/RRF/CE configs/models/depths and Phase1–6 evidence are unchanged.

## O. Tests

`.\.venv\Scripts\python.exe -m pytest -q`:143 passed,0 failed, one existing Starlette
httpx deprecation warning. Unit tests use fake engines/vectors; real models are only
loaded by integration scripts. `.\.venv\Scripts\python.exe -m pip check`:
No broken requirements found. `git diff --check`:passed.
scripts/validate_phase7.py validates frozen files, ignore policy, rebuild hashes,
integration/parity evidence, boundary reports and runs fullpytest/pipcheck.
Prometheus-client0.26.0 is pinned in the API extra; existing SciPy1.18.1 is now an
explicit direct dependency because monitoring uses scipy.stats.wasserstein_distance.

## P. Test Boundary Verification

**Final-test relevance labels accessed: NO.** No final-test query text is decoded
or used. BoundaryGuard records no blocked actual-file attempts. Monitoring opens
only train judgments; serving/parity opens no relevance labels. Frozen canonical
checksums and all tracked prior phase configs/evidence are verified unchanged.
Raw WANDS, embeddings, BM25 indexes, HF weights/cache, venv and temporary experiment
files remain ignored and untracked. Neither raw benchmark observations nor detailed
monitoring/parity artifacts were added to Git.

## Q. Changed Files

Modified: README.md, pyproject.toml, src/product_search/api.py,
src/product_search/api_logging.py.

Added: configs/phase7_monitoring.json, configs/phase7_diagnostics.json;
src/product_search/observability.py, src/product_search/monitoring.py;
tests/test_observability.py, tests/test_monitoring.py;
scripts/run_monitoring.py, scripts/validate_observability.py, scripts/validate_phase7.py;
docs/observability.md, docs/ml_monitoring.md;
reports/phase7/baseline.json, monitoring.json, observability_integration.json,
observability_verified.json, parity.json, checks.json, REPORT.md.

These are intended version-controlled files, currently modified/untracked and
unstaged. No Phase7 commit was made. All new reports are small (<100KB each before
release metadata), with no per-query ranked lists or vectors.

## R. Recommended Commits

Not executed; separate user authorization is required.

1. `feat: instrument search service with bounded metrics and private logs`
   — pyproject.toml; src/product_search/{api.py,api_logging.py,observability.py};
   tests/test_observability.py.
2. `feat: add deterministic train-only drift and ranking replay`
   — configs/phase7_monitoring.json; src/product_search/monitoring.py;
   scripts/run_monitoring.py; tests/test_monitoring.py.
3. `test: validate real observability queue visibility and release boundaries`
   — scripts/validate_observability.py; scripts/validate_phase7.py;
   configs/phase7_diagnostics.json.
4. `report: retain monitoring rebuild parity and integration evidence`
   — reports/phase7/{baseline.json,monitoring.json,observability_integration.json,
   observability_verified.json,parity.json,checks.json}. Keep failed and passing
   integration evidence together.
5. `docs: explain observability privacy and offline monitoring limits`
   — README.md; docs/observability.md; docs/ml_monitoring.md; reports/phase7/REPORT.md.

## S. Phase 7 Verdict

**PASS.** Ready for Phase8 — Docker, clean-environment reproducibility, and deployment
packaging, after Phase7 review/commit authorization. Phase7 is complete and remains
uncommitted. No push was performed. Docker and Phase8 have not begun.
