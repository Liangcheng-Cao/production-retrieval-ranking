# Phase 5 — FastAPI Search Service & HTTP Lifecycle

Verdict: **PASS**. The HTTP adapter exposes the unchanged Phase 4 core. Real tests
use loopback TCP with local frozen artifacts; this is not a cloud deployment,
load benchmark or production SLA. No Dockerization or Phase 6 work was performed.

## A. Phase 4 Commits

Pre-commit full suite: 88 passed in 1.57 s. After these six logical commits,
`git status --porcelain` was empty before any Phase 5 implementation:

| Hash | Message |
|---|---|
| d06c372 | feat: define runtime contracts and sealed artifact configuration |
| 5b40127 | feat: load frozen artifacts into reusable search engine |
| 6b80eb4 | feat: add explicit runtime packaging and startup validation |
| 58fcec2 | report: preserve batch parity investigation evidence |
| b2701dd | report: verify runtime parity restart and artifact boundaries |
| cdb4a74 | docs: describe search lifecycle and numerical parity limits |

Full commit hashes/file lists: phase4_commits.json. No push.

## B. HTTP Architecture

```text
Client → RequestContextMiddleware (ID / HTTP timing)
       → FastAPI discriminated strict request schema
       → one lifespan-owned SearchEngine via single-worker executor
       → copy core hit/timing contracts + public version whitelist
       → validated JSON response
```

Actual modules: api.py, api_schemas.py, api_logging.py. The adapter never retrieves,
reranks, sorts or computes scores. Core source and frozen ML parameters are unchanged.
Pinned API dependencies: FastAPI 0.141.1, Starlette 1.7.0, Pydantic 2.13.5,
Uvicorn 0.54.0, HTTPX 0.28.1.

## C. Lifespan

create_app builds routes only. Lifespan creates the owner executor, loads config,
constructs and initializes one engine, then permits serving. Required artifact
failure propagates out of startup. Shutdown closes that engine and joins its worker.
Real integration observed initialize_count=1 and close_count=1 across many requests.
Unit tests also verify distinct engines across repeated application lifecycles.

## D. Endpoint Contracts

| Endpoint | Behavior |
|---|---|
| POST /search | Validate, call existing engine and return structured results |
| GET /health | 200 alive=true; no model invocation or file hashing |
| GET /ready | 200 when READY/DEGRADED; otherwise 503 |
| GET /version | Cached concise public provenance; 503 when not searchable |

FastAPI-generated OpenAPI and standard docs are available. No custom UI or root
identity endpoint was added.

## E. Request / Response Schema

Required query: strict string, 1–512 Unicode characters, non-whitespace content;
passed unchanged to the core. Required pipeline: bm25, hybrid or hybrid_rerank.
Optional top_k defaults to 10; strict integer 1–100 for retrieval and 1–20 for CE.
Numeric strings, floats, booleans and extra fields fail validation. Discriminated
OpenAPI schemas advertise pipeline-specific limits, not just a generic K maximum.

Response: request_id, query, requested/effective pipeline, fallback flag/reason,
results, timing_ms and compact version. Hits retain product_id/title/final_rank/
final_score/retrieval_rank/retrieval_score/reranker_score/source. No large product
fields are copied. Public fallback reason is the stable code reranker_unavailable.

## F. Error Mapping

| Condition | HTTP status | Response behavior |
|---|---:|---|
| Invalid fields/JSON or disabled mode | 422 | Sanitized invalid_request |
| Engine not searchable | 503 | not_ready |
| Core or response-serialization failure | 500 | search_failed, no traceback/path |
| Unexpected adapter error before headers | 500 | internal_error with request ID |
| Artifact/startup failure | No serving | Abort lifespan; no silent rebuild |
| Successful CE fallback | 200 | Effective Hybrid, explicit fallback metadata |

HTTP route/method errors also use the controlled error envelope. Internal detail
goes only to server-side logs; malformed request input is not echoed by validation errors.

## G. Request IDs / Logging

Accept a single safe X-Request-ID matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`;
otherwise generate a UUID. Return it in headers and search/error bodies. Both
successful and failed requests correlate to logs.

JSON lifecycle/search events include service_starting, engine_loading, service_ready,
search_completed, search_fallback, search_failed, search_rejected, http_response,
service_shutdown. Fields include request ID, modes, K, query_length, fallback,
result_count and timings. No application log contains full query text or result lists.
Internal exception traces are separate private diagnostics; raw test logs remain ignored.

## H. Readiness / Health

All states tested: NOT_INITIALIZED, LOADING, FAILED and CLOSED return ready HTTP 503;
READY returns 200 with ready=true; DEGRADED returns 200 with ready=false and
searchable=true. Generic detail replaces core exception reasons. Health remains
200 independently, with no inference. A degraded path can still fail an individual
search; 200 readiness does not claim every pipeline is healthy.

## I. Version Endpoint

Public whitelist includes API/core/package versions; runtime/dataset/retrieval/
reranking manifest SHAs; BM25 implementation/version/method/k1/b/representation;
dense and CE model IDs/revisions. Paths, secrets and full artifact inventories are
excluded. Unit tests inject private fields and real HTTP checks verify sanitization.
Runtime manifest remains 7adf8fcaace380db531b60765678da1dbe4461498e71e313c318cf82d52009b0.

## J. HTTP Timing

Core stages stay in response timing_ms. X-HTTP-App-Ms measures ASGI entry through
response-header emission; X-Engine-Ms repeats core total; X-Response-Serialization-Ms
measures result projection/schema validation/JSON encoding. HTTP timing includes
body read and scheduling/waiting, but not network RTT or client download completion.

Three diagnostic samples from the final real integration (ms):

| Pipeline | HTTP application | Engine | Serialization |
|---|---:|---:|---:|
| BM25 | 1.4375 | .3369 | .0862 |
| Hybrid | 13.4120 | 12.7935 | .0819 |
| Hybrid+CE | 16.7251 | 16.0641 | .0896 |

No throughput, concurrency study or percentile claims. Startup is separate.
There is no hard inference deadline: client timeout/cancellation cannot interrupt
running native/GPU work. Robust timeout/admission behavior is deferred to Phase 6.

## K. API / Core Parity

Six fixed train queries (IDs 0, 1, 2, 3, 5, 6), all three pipelines, K=10:
**18/18 comparisons passed** against an independently initialized direct SearchEngine.
IDs/order, effective mode and fallback match; maximum observed final/retrieval/CE
score difference is zero. Tolerances: 1e-6 retrieval, 1e-4 CE, both absolute/relative.

Machine-readable evidence: api_core_parity.json and http_release.json. Existing
Phase 4 batch-context ordering caveats remain; this verifies the single-query
HTTP/core contract under a fixed numerical runtime, not cross-batch equivalence.

## L. Real Service Integration

```powershell
.\.venv\Scripts\python.exe scripts/validate_api.py --report reports/phase5/http_release.json
```

Exit 0, passed=true. Started actual Uvicorn on an ephemeral loopback TCP port and
sent sequential real HTTP requests. /health, /ready, /version, /openapi.json all 200;
search for each mode 200. Invalid pipeline/K, blank query, CE K>20 and malformed
JSON all 422. The listener was shut down after testing; no background server remains.

Normal runner: `.\.venv\Scripts\python.exe scripts/serve.py --port 8000`.
It uses the existing config or --config / SEARCH_RUNTIME_CONFIG, with no embedded
machine-specific paths and no runtime artifact acquisition.

## M. Failure Integration

Actual service engine CE scorer was replaced temporarily by a throwing test scorer,
using only the harness (no fault-control HTTP endpoint). Search returned 200,
requested=hybrid_rerank, effective=hybrid, fallback_used=true; IDs match Hybrid
ordering and CE scores are null. Readiness was DEGRADED/200. Restoring the scorer
and making a successful CE request recovered READY.

A second service used a temporary runtime root with a missing manifest. It never
became ready/listening, startup thread exited, engine closed, and no artifact was
rebuilt. Frozen real assets were not altered. These are controlled integration
checks, not evidence of production high availability.

## N. Tests

```text
.\.venv\Scripts\python.exe -m pytest -q
122 passed, 1 warning in 1.23s

.\.venv\Scripts\python.exe -m pip check
No broken requirements found
```

Warning: Starlette deprecates its HTTPX TestClient compatibility path. Tests work;
the warning was retained rather than suppressed. Fake-engine tests cover schema,
lifespan once/close, readiness, public version sanitization, request IDs, controlled
exceptions including serialization, fallback, parity, startup failure and OpenAPI.

## O. Test Boundary Verification

**Final-test relevance labels accessed: NO.** No relevance-label file was opened
in Phase 5. Real integration opened product metadata, dataset manifest and the shared
queries file, decoding only six train strings. Boundary guard blocked-attempt list
is empty. Core/frozen files from Phases 1–4 remain unchanged.

## P. Changed Files

Modified README.md and pyproject.toml (API optional dependencies only).
Added src/product_search/api.py, api_logging.py, api_schemas.py; tests/test_api.py;
scripts/serve.py and validate_api.py; docs/http_service.md; compact reports/phase5
commit, integration, parity, verification and summary evidence.

Raw WANDS, processed JSONL, model cache/weights, indexes, embeddings, .venv and
temporary execution logs remain ignored. Final Git audit is in checks.json.

## Q. Recommended Commits

Phase 5 remains uncommitted. Suggested small logical groups:

1. API schemas/application/middleware plus dependency extra (four files).
2. API tests, local runner and real-service integration script (three files).
3. Integration/parity evidence, checks and Phase 4 commit record.
4. README, HTTP documentation and phase report.

No push. The dirty working tree consists of intentional Phase 5 work.

## R. Phase 5 Verdict

**PASS.** Ready to enter Phase 6 — reproducible latency benchmark and concurrency/load
testing. Phase 6 has not started. Hard cancellation, bounded admission and throughput
claims remain outside this phase.
