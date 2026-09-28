# Local HTTP search service

Phase 5 exposes the unchanged Phase 4 SearchEngine through FastAPI. It is a local
service, not cloud deployed; no Docker image, load study, inference deadline,
throughput/capacity or SLA is claimed.

## Run locally

From the repository, with the existing provisioned/frozen artifacts:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval,api]"
.\.venv\Scripts\python.exe scripts/serve.py --port 8000
```

The runner binds 127.0.0.1, uses one Uvicorn worker, disables access logs that might
include URL query strings, enables basic structured application logs, and forces
Hugging Face offline mode. It does not build or fetch artifacts. `--config PATH`
or `SEARCH_RUNTIME_CONFIG` selects a runtime config; the runner's default resolves
against the repository, without a machine-specific absolute path. Runtime artifact
paths retain Phase 4's config-relative behavior. Stop with Ctrl+C for graceful shutdown.

`product_search.api:create_app` also supports application-factory use; its default
config is `configs/runtime.json` relative to the working directory, or the environment
override. Importing the module creates only routes/schema, never a model or index.

## Lifespan and ownership

FastAPI lifespan creates a dedicated single-worker executor, loads RuntimeConfig,
constructs one SearchEngine and calls initialize. Only successful artifact validation
and component loading permit lifespan to yield and Uvicorn to accept requests.
Startup errors are logged internally and propagated; the engine is closed and no
partially initialized service accepts traffic. Shutdown closes the engine on its
owner worker, then joins the executor. Repeated application lifecycles create fresh
engines; requests within a lifecycle reuse the same one.

Search dispatch awaits a single-worker executor and an application lock. It does not
run synchronous GPU work directly on the event loop, so cheap health endpoints do
not themselves run inference. This preserves the sequential core ownership contract;
it is not a concurrency optimization, load test or bounded admission-control system.
Further worker/concurrency choices belong in Phase 6.

Implementation follows FastAPI's [lifespan](https://fastapi.tiangolo.com/advanced/events/)
pattern. Request validation uses Pydantic [strict mode](https://pydantic.dev/docs/validation/latest/concepts/strict_mode/).

## Endpoints

| Endpoint | Contract |
|---|---|
| POST /search | Validate request, invoke existing engine, serialize its result |
| GET /health | Always 200 while the app can answer; `{ "alive": true }`; no model/hash check |
| GET /ready | 200 for READY/DEGRADED; 503 for NOT_INITIALIZED/LOADING/FAILED/CLOSED |
| GET /version | Compact cached provenance, 200 when searchable; otherwise 503 |

`/ready` includes ready, searchable, state, enabled_pipelines and a generic detail.
For DEGRADED, ready=false but searchable=true; a healthy path or explicit CE fallback
can still work, while a particular failed request may return 500. Core error reasons
and filesystem paths are never echoed by this probe. Uvicorn normally serves no HTTP
at all before successful startup; non-ready state mappings are also tested through ASGI.

FastAPI provides `/openapi.json` and its standard documentation routes. No custom
frontend or service identity endpoint is added.

## Request contract

```json
{"query":"wooden office desk","top_k":10,"pipeline":"hybrid_rerank"}
```

Pipeline is required and is exactly bm25, hybrid or hybrid_rerank. top_k defaults to
10. Query must be a strict string with 1–512 Unicode code points, containing at least
one non-whitespace character. Text is passed unchanged to the engine; whitespace
validation never strips or normalizes the actual query. The bound is an HTTP input
contract, separate from the frozen models' existing token truncation policies.

Top-K is a strict integer: 1–100 for BM25/Hybrid and 1–20 for hybrid_rerank.
Booleans, floats and numeric strings are rejected, as are unknown JSON fields,
invalid pipelines, malformed JSON and valid-but-disabled modes. Discriminated
request schemas encode each pipeline's limit in OpenAPI. No CE tail is appended.

## Response contract

The response keeps core names: query, requested_pipeline, effective_pipeline,
fallback_used, fallback_reason, results, timing_ms and version, plus request_id.
Results contain product_id (integer), title, final_rank, final_score, retrieval_rank,
retrieval_score, reranker_score and source. The adapter copies core hits and timings;
it neither sorts nor recomputes scores. Empty core results serialize as `results: []`.

Public fallback_reason is the stable code `reranker_unavailable`, not a Python
exception message. Score meanings remain BM25/RRF/raw CE logits as documented in
Phase 4. Successful CE fallback has effective_pipeline=hybrid, null reranker_score,
retrieval final scores, and HTTP 200.

The public version is a cached whitelist: API/core/package identity; runtime,
dataset, retrieval and reranking manifest hashes; BM25 scoring config; dense model
and revision; CE model and revision. It omits filesystem paths, environment,
full manifests, source-file inventories and model cache contents.

## Errors

All controlled errors use `{ "error": "code", "message": "safe message", "request_id": "..." }`.

| Condition | Status | Error code |
|---|---:|---|
| Invalid fields/JSON or disabled mode | 422 | invalid_request |
| Engine not searchable | 503 | not_ready |
| Core search or response serialization failure | 500 | search_failed |
| Unexpected adapter failure before headers | 500 | internal_error |
| HTTP route/method error | 404/405 etc. | http_error |
| Successful CE fallback | 200 | none; fallback metadata present |

No traceback, raw validation input or internal path is sent to clients. Internal
tracebacks are logged separately with request IDs. Startup failure aborts serving,
rather than presenting a misleading ready service or rebuilding artifacts.

## Request IDs, logs and timings

An incoming X-Request-ID is accepted only when it occurs once and matches
`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. Otherwise a UUID is generated. It appears in the
response header, search/error body and logs, including validation and controlled
failure paths. It is a correlation identifier, not distributed tracing.

JSON events include service_starting, engine_loading, service_ready,
search_completed, search_fallback, search_failed, search_rejected, http_response
and service_shutdown. Search fields include request_id, requested/effective pipeline,
top_k, query_length, fallback_used, total_ms and result_count. Neither query text
nor full product results are emitted by the application logger. Internal tracebacks
are server-only diagnostics; keep those logs private. Temporary integration logs
are ignored by Git.

Core stage durations remain in timing_ms. Headers expose:

- X-HTTP-App-Ms: ASGI entry to response headers, including body read, validation,
  scheduling/waiting, engine execution and response construction; not network RTT
  or client download completion.
- X-Engine-Ms: the core's existing total_ms.
- X-Response-Serialization-Ms: response projection, schema validation and JSON
  encoding; a subset of HTTP overhead, not all HTTP overhead.

All use perf_counter. Startup stays separate. Individual smoke samples are only
diagnostics and must not be presented as percentile latency, QPS or SLA evidence.

## Timeout semantics

No hard request/inference timeout is implemented. Client disconnect or HTTP-client
timeout cannot safely preempt already running native/GPU computation. Canceling an
await does not cancel that work; the single worker continues and shutdown waits for
owned work. Phase 6 must address admission, waiting, deadlines and load behavior.
The integration client's 60-second timeout is a test harness bound, not a server
inference guarantee. Uvicorn keep-alive behavior is likewise not an inference timeout.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_api.py --report reports/tmp/http-check.json
```

Unit tests use fake engines. The integration command runs a real Uvicorn loopback
listener with actual frozen models and sends sequential HTTP requests. It compares
six train queries across three pipelines against an independently initialized direct
SearchEngine, tests invalid requests, injects a CE scorer failure through the local
test harness (no HTTP fault endpoint), verifies recovery and engine close, then tests
startup failure using a temporary root with a missing runtime manifest. Real frozen
assets are never damaged. The command refuses to overwrite existing reports.

Parity compares IDs/order, effective mode, fallback and final/retrieval/CE scores,
using 1e-6 retrieval and 1e-4 CE tolerances. The harness holds the same numerical
runtime settings as Phase 4; this does not imply cross-hardware or cross-batch
equivalence. The known Phase 4 batch-context limitation still applies.

No relevance labels are read; only product metadata and six train query strings.
Boundary guards block raw labels, final-test judgments and the global conflict audit.
Phase 1–4 frozen data, configs, core source and evidence remain unchanged.
