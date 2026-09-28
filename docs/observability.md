# Local service observability

No real production traffic is available.
No online A/B test has been performed.

Phase 7 instruments the existing FastAPI adapter, not ranking. `RequestContextMiddleware`
owns request identity, ASGI lifetime and in-flight accounting. `WaitingTicket` measures
admission to actual executor callable entry. `ServiceMetrics` owns one private
Prometheus `CollectorRegistry` per app lifespan. `SearchEngine` returns the same
rankings and timings as before. The existing `asyncio.Lock` and one-worker executor
remain in place. There is no deployed monitoring stack, admission controller, timeout,
new worker or throughput optimization.

## Metric contract

`P` = `pipeline`, one of `bm25`, `hybrid`, `hybrid_rerank`, `unknown`.
`E` = `effective_pipeline`, the same bounded set. `S` = `status`, one of
200/400/404/405/413/422/499/500/503/other. Invalid bodies use unknown modes;
their supplied strings never become labels. All request metrics count only POST
`/search`; GET probes, `/metrics`, OpenAPI, and other routes do not count.

| Exposed metric family | Type | Labels | Meaning |
|---|---|---|---|
| `request_count_total` | counter | P,E,S | Finished ASGI search calls, including cancellation |
| `request_error_count_total` | counter | P,E,S | Non-2xx, including invalid requests and cancellation |
| `fallback_count_total` | counter | P,E,S | Responses using CE fallback |
| `reranker_fallback_total` | counter | P,E,S | Alias of fallback count; do not sum the aliases |
| `degraded_request_total` | counter | P,E,S | Admitted/completed DEGRADED, or fallback |
| `empty_result_total` | counter | P,E,S | HTTP200 responses with zero results |
| `request_latency_seconds` | histogram | P,E,S | ASGI entry to application return after response body |
| `engine_latency_seconds` | histogram | P | Returned core `total_ms`, converted to seconds |
| `queue_wait_seconds` | histogram | P | Admission to actual executor callable entry |
| `result_count` | histogram | P,E,S | Hits on successfully serialized HTTP200 responses |
| `query_character_length` | histogram | P | Validated query codepoints |
| `query_token_count` | histogram | P | Validated whitespace-delimited token count |
| `bm25_latency_seconds` | histogram | P | Core BM25 stage |
| `dense_encoding_latency_seconds` | histogram | P | Core dense encoding stage |
| `dense_search_latency_seconds` | histogram | P | Core dense search stage |
| `fusion_latency_seconds` | histogram | P | Core RRF fusion stage |
| `reranking_latency_seconds` | histogram | P | Core reranking stage, including fallback handling |
| `hydration_latency_seconds` | histogram | P | Core metadata hydration stage |
| `in_flight_requests` | gauge | none | Active POST /search ASGI calls |
| `requests_waiting_for_engine` | gauge | none | Live admitted calls not yet entering executor callable |
| `engine_active_calls` | gauge | none | Actually executing callable, even if caller cancelled |

Prometheus standard suffixes: counters have `_total` and `_created`; histograms
have `_bucket` (additional bounded `le` label), `_sum`, `_count`, `_created`.
The requested logical `request_count`, `request_error_count`, `fallback_count`
therefore appear with `_total`. No default process/Python collectors are registered.
There are no query, hash, product ID, request ID, path, or model-revision labels.

Timing buckets (seconds): .0001/.0005/.001/.0025/.005/.01/.025/.05/.1/.25/.5/1/2.5/5/+Inf.
Core stage metrics observe zeros for unused stages; filter P before interpreting.
Core/stage histograms have no observation when the engine raises before returning a
timing object; the HTTP error and request latency still count. Core histograms may
include a completed native call whose caller already cancelled; result/fallback
metrics describe delivered ASGI outcomes. Validated query features count even if the
engine is unavailable; malformed bodies only increment HTTP outcome metrics.

## Queue and cancellation semantics

Waiting includes lock wait **and** executor dispatch wait, excluding request parsing
before admission. It is not an estimate from residual HTTP time, and is not a queue
length inside an unseen GPU scheduler. A thread-safe ticket decrements exactly once,
on callable entry or cancellation before entry. Cancelled-before-entry calls have no
queue histogram observation. ASGI cancellation before response headers uses synthetic
status 499 for accounting; it does not claim a 499 response was delivered. A client
disconnect is not guaranteed to cancel ASGI/native work. Existing cancellation and
execution behavior is unchanged; `engine_active_calls` remains positive while native
work continues after a cancelled caller. Unit tests exercise this case.

The controlled real-model integration holds the first Python engine callable only
until a metric scrape observes the expected state. C1 observes in-flight=1, waiting=0,
active=1; C4 observes 4/3/1. Both finish at zero. This intentionally held scenario
demonstrates measurement, not capacity or natural latency. Phase 6 remains the throughput
evidence. The natural C1 sample (12 searches per mode) is separate.

## Endpoint and timing

`GET /metrics` returns HTTP200 and Prometheus text 0.0.4 using `generate_latest`.
It only serializes the in-memory registry; no inference, artifact reads/hashes,
model initialization or engine readiness calls occur. It works on an uninitialized
app too, but a real server whose lifespan fails cannot accept HTTP connections.
Scraping costs are bounded by the above label sets and bucket counts.
New lifespans reset counters/gauges/histograms; restart discontinuities are expected.

`X-Queue-Wait-Ms` measures direct wait. Existing `X-HTTP-App-Ms` stops at response
headers; the new histogram and terminal structured log stop after the application
returns. Neither is client RTT. Existing `X-Engine-Ms` and serialization headers
retain their meanings. Do not subtract these independently measured quantities and
call the difference exact queue wait.

## Logs and privacy

Lifecycle: service_starting, engine_loading, service_ready, service_shutdown.
Search terminal events: search_completed, search_fallback, search_failed, with
request_id, pipeline, effective_pipeline, top_k, query_length, queue_wait_ms,
engine_total_ms, http_total_ms, fallback_used, result_count, status. Missing fields
are null for failed/unvalidated work. Earlier rejection/failure events supply a
bounded reason, and existing http_response events record time to headers. Thus a
failure can have an early reason event and a terminal timing event; use counters,
not raw event counts, for error rates.

The existing single ASCII X-Request-ID policy (1–64 safe characters) is unchanged;
missing, invalid, duplicate or unsafe IDs are replaced by UUIDs. IDs belong only in
logs/headers, never metric labels. Callers should supply opaque IDs without secrets.
Request-path exception messages/tracebacks are no longer logged by default because
an exception can embed the query. Client failure bodies remain sanitized. Startup
diagnostic tracebacks remain local and may contain installation paths.

Default policy: do not persist full query text, full ranked product lists, or
request-specific product IDs in metrics. Responses still contain the requested
query/results as required by the API. No query fingerprints are added. Service logs
prefer lengths, opaque IDs, modes, latency, counts and fallback. The normal runner
writes logs to its configured output, not a permanent query store. Local validation
logs and detailed parity fixtures live under Git-ignored `reports/tmp/`; keep them
only for local diagnosis and remove manually when no longer needed. No automated
purge or deployment retention service is claimed. Compact aggregate evidence is
retained in Git; no per-query vectors or ranked lists are added to Phase 7 reports.
This is a portfolio/local policy, not a regulatory-compliance claim.

## Local diagnostic thresholds

See `configs/phase7_diagnostics.json`. Inspect 5-minute deltas; require at least 20
samples per mode for a latency warning. Empty traffic has undefined rates/quantiles.
Investigate any fallback/error rate >0 or readiness state !=READY. A fallback can
correctly return HTTP200; a recovery request admitted while DEGRADED also counts as
degraded even when it succeeds and restores READY.

Request P95 warnings: BM25>10ms, hybrid>50ms, CE>60ms. These round roughly three
times Phase 6 maximum C1 client RTT P95 (3.21/14.66/19.31ms) upward. Server ASGI timing
differs from client RTT, so these are conservative local proxies. Queue P95>5ms is a
coarse material wait relative to 10–20ms engine work, well above the small Phase 7
natural C1 exact-wait P95 sample (.030/.034/.042ms). Phase 6 residual time is **not**
used as an exact queue baseline. Histogram quantiles interpolate coarse buckets.
These are local diagnostics, not production SLOs, alert deployment or capacity claims.

## Reproduce

Install the pinned `api` extra (adds prometheus-client 0.26.0), then:

```powershell
.\.venv\Scripts\python.exe scripts/validate_observability.py --report reports/tmp/obs-repeat.json
.\.venv\Scripts\python.exe scripts/validate_api.py --report reports/tmp/api-repeat.json
.\.venv\Scripts\python.exe -m pytest -q
```

Use fresh report paths. The real check verifies endpoints, bounded labels, each mode,
CE fallback, injected core error, queue gauges, privacy and independent engine restart.
The first Phase 7 attempt retained in `observability_integration.json` failed only
its logging harness check: Uvicorn dictConfig closed a write-mode FileHandler.
The fixed append-mode handler reopens safely; `observability_verified.json` is the
passing independent rerun. Failed evidence was preserved, not rewritten.
