# Production search core and artifact lifecycle

This is a sequential local core, not a deployed service. No HTTP endpoints,
concurrency guarantees, load tests, monitoring stack or production SLA exist.

## Contracts

```python
from product_search.search_engine import SearchEngine

engine = SearchEngine.from_config("configs/runtime.json")
result = engine.search("wooden office desk", top_k=10, pipeline="hybrid_rerank")
print(result.effective_pipeline, result.results, engine.readiness())
engine.close()
```

The production path is frozen ML artifacts → RuntimeConfig → ArtifactLoader →
SearchEngine → future API. It delegates ranking to the existing BM25Retriever,
DenseRetriever, HybridRetriever and CrossEncoderReranker implementations.

| Mode | Actual behavior | Allowed top_k |
|---|---|---|
| bm25 | Existing BM25 scoring, omit nonpositive scores, return prefix | 1–100 |
| hybrid | BM25 top-100 + Dense top-100 → frozen equal-weight RRF → prefix | 1–100 |
| hybrid_rerank | Hybrid top-100 → first 20 → frozen CE → prefix | 1–20 |

CE-20 never appends an unscored tail. Unsupported K, booleans/floats, unknown or
disabled modes and empty/whitespace queries fail before invoking a model. The
original query is preserved, including surrounding whitespace; existing retrievers
own their frozen preprocessing. No query normalization was added to the core.

SearchResult contains the query, requested/effective pipeline, tuple of SearchHit,
immutable stage timings, shared immutable version identity, fallback flag/reason.
SearchHit exposes only product ID/title, final rank/score, original retrieval
rank/score, optional CE score and retrieval source. Score scale depends on mode:
BM25 score, RRF score or raw CE logit. Fallback uses RRF scores and null CE scores.
Metadata hydration occurs after ranking; full descriptions are not copied into
responses. Unknown product IDs are explicit errors, never silently dropped.

## Startup and local packaging

1. Load JSON RuntimeConfig. Paths resolve relative to its configured artifact root.
2. Validate the pinned runtime manifest SHA256, frozen Phase 1/2/3 anchors,
   selected configuration compatibility and software versions.
3. Validate products, selected index manifests/files and enabled model snapshots.
4. Load products and BM25 once; verify exact catalog membership.
5. If required, load dense embeddings, then the local encoder and Hybrid.
6. If required, load CE; mark READY only after every enabled component succeeds.

`scripts/seal_runtime.py` is an explicit, one-time packaging action over existing
artifacts. It creates a small manifest and pins its checksum in runtime.json;
it refuses overwrite. It builds no index, downloads no model and opens no labels.
Normal startup never calls packaging or building. Future artifact replacement
requires a separately reviewed package/config version and a new engine.

Frozen Phase 2 and Phase 3 selections supply all ranking settings. Deployment
config specifies paths, enabled/default pipelines, default K, device and whether
CE fallback is allowed. The tested device is CUDA on RTX 4070; CPU configuration
is permitted but cross-device numerical equivalence has not been measured.

Loading is **eager for enabled components**. Set enabled_pipelines to `["bm25"]`
and default_pipeline to `"bm25"` for a lexical-only deployment; it does not validate
or load dense/CE payloads. Enabling hybrid without hybrid_rerank requires dense
but not CE weights. Small frozen manifest metadata remains required for provenance.
This avoids lazy-load races and first-request model startup costs. Phase 3 measured
cached CE startup around 0.68 s; loading it only in instances that offer CE makes
resource ownership explicit. There is no lazy model initialization inside search.

The core reads the sealed local model snapshot paths. Dense loading explicitly
uses local_files_only; the existing CE scorer receives a local snapshot directory.
Validation commands additionally force Hugging Face offline mode. Missing files,
checksum mismatches, changed snapshot inventories, incompatible configs/datasets
or software versions fail startup. No automatic repair or fallback hides corruption.

## Version identity, readiness and failures

Runtime provenance is recursively immutable and shared across results. It includes
the core contract/package version and source hashes; dataset/retrieval/reranking
manifest identities; frozen configs/models/revisions; validated artifact hashes;
software/Python versions; device/dtype; numeric runtime flags and enabled policies.
Hashes are validated at startup, not recomputed per request. The deployment must
keep artifacts stable for an engine's lifetime; hot replacement is unsupported.

States: NOT_INITIALIZED → LOADING → READY or FAILED. Request failures may produce
DEGRADED. `ready` is true only in READY; a degraded engine can still execute healthy
paths and explicit CE fallback. `initialize()` is idempotent after successful load;
failed/closed instances require replacement. `close()` drops owned resource
references and sets CLOSED; it does not globally clear PyTorch's shared allocator.

| Condition | Behavior |
|---|---|
| Missing/corrupt/incompatible required artifact | Startup raises ArtifactError; no rebuild |
| Dense/Hybrid inference failure | SearchError, DEGRADED; no implicit BM25 substitution |
| CE inference failure, fallback enabled | Hybrid prefix, effective_pipeline=hybrid, fallback_used=true, reason recorded |
| CE failure, fallback disabled | SearchError, DEGRADED |
| Invalid request | ValueError before invocation; readiness unchanged |
| Zero results | Valid empty tuple and diagnostics |
| Missing metadata | SearchError, DEGRADED; no invented title or skipped hit |

Degradation from a retrieval path/CE clears after a subsequent successful invocation
of that path/component. Success in an unrelated path does not clear it. Empty
candidates do not count as successful CE inference. Metadata inconsistency is
sticky until the engine is replaced. Fallback does not establish production reliability.

## Timing and resources

Monotonic perf_counter measures request preprocessing, BM25, dense encoding,
dense search, fusion, reranking, hydration and total. Unused stages are zero.
Per-request total excludes startup. Dense encoding synchronizes through NumPy
transfer; CE's existing scorer synchronizes GPU execution. These diagnostics
are not a benchmark, capacity estimate, throughput measurement or SLA.

Startup timings are separate. The validation command records Windows working-set
RSS and PyTorch GPU allocated/reserved memory immediately after engine loading,
plus another snapshot after smoke checks. Allocated memory is not total device
usage; reserved memory is not peak capacity. The core does not change process-wide
Torch/BLAS numerical settings; the integration harness fixes and records them.

## Parity and reproduction limits

The parity check uses six fixed train query IDs: 0, 1, 2, 3, 5, 6. It reconstructs
offline retrievers from the frozen artifacts and invokes the existing single-query
interfaces directly, independently of SearchEngine dispatch. BM25/Hybrid compare
K=10/20/100; CE compares K=10/20: 48 checks per process. IDs and ordering must match;
score tolerances are 1e-6 for retrieval and 1e-4 for CE (absolute and relative).
Two independent processes verify the same runtime identity and ranked IDs.

An initial extra check against historical batch-evaluation caches failed for query 1.
The evidence is preserved in runtime_validation.json and batch_parity_diagnosis.json.
Investigation reproduced the original batch-128 context across train queries and
recovered the historical Hybrid and CE ordering. Single-query encoding changes dense
cosines by at most 2.384185791015625e-7 for query 1. Near-tied dense ranks change RRF
ordering; its top-100 membership remains unchanged, and CE tie ordering can change.
The original offline single-query interface exhibits the same behavior as the core.

Consequently, PASS means parity with the existing **single-query contract**, not
universal equality with batched experiment ordering. The check separately verifies
historical results under their original batch context. Phase 2/3 quality values
remain historical batch-evaluation evidence, not a fresh runtime quality evaluation.
No precision rounding, model retuning, sorting change or historical report edit was
introduced to hide the mismatch. GPU scores are not promised bit-identical across
hardware, dependency versions or different batch shapes.

Run from the repository in PowerShell (choose fresh output paths to preserve evidence):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_runtime.py --parity --report reports/tmp/runtime-check.json
.\.venv\Scripts\python.exe scripts/validate_runtime.py --parity --report reports/tmp/runtime-restart.json --compare-restart reports/tmp/runtime-check.json
```

The startup command exits nonzero on failure and writes structured diagnostics.
Its synthetic smoke query exercises every enabled mode. Full parity requires all
three enabled modes. Reports refuse overwrite. Unit tests use tiny local artifacts
and fake scorers; no real GPU model is required by normal pytest.

Final-test relevance labels are never opened. Runtime needs only product metadata;
the integration harness additionally byte-scans the shared query file and decodes
train strings, never test strings, to reproduce development batch context. No label
files are needed. Access guards and logs remain active throughout real checks.
