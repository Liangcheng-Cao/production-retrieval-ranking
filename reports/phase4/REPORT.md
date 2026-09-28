# Phase 4 — Production Search Core & Artifact Lifecycle

Verdict: **PASS for the existing single-query contract**, with a documented
pre-existing cross-batch numerical ordering limitation. No API, deployment,
concurrency/load test, monitoring integration, model tuning or label evaluation.

## A. Phase 3 Commits

Before implementation, full tests passed (48 in 0.40 s), artifact exclusions were
checked, Phase 3 was committed, and `git status --porcelain` was empty.

| Commit | Message |
|---|---|
| 4664a0f | test: verify Phase 2 metric cutoffs independently |
| 972c878 | feat: add reusable cross-encoder reranking and paired analysis |
| 2800251 | exp: freeze bounded cross-encoder development selection |
| 0c86041 | report: record validation reranking gains and regressions |
| b70e085 | data: seal reranking provenance and access boundaries |
| 274a1f7 | docs: document Phase 3 quality latency and verification |

Full hashes and file lists: phase3_commits.json. Nothing was pushed.

## B. Production Core Architecture

```text
RuntimeConfig.load
  → ArtifactLoader.validate / load
  → SearchEngine
      ├─ BM25Retriever
      ├─ DenseRetriever + one SentenceTransformer
      ├─ HybridRetriever (shares BM25 and Dense)
      └─ CrossEncoderReranker + one TransformerPairScorer
  → immutable SearchResult / SearchHit / version / timing
```

Core modules: runtime_config.py, runtime_artifacts.py, runtime_contracts.py and
search_engine.py. All Phase 2/3 ranking source and frozen settings remain unchanged.

## C. Pipeline Contracts

| Mode | Behavior | K |
|---|---|---|
| bm25 | Existing BM25 scoring and prefix | 1–100 |
| hybrid | BM25-100 + Dense-100 → frozen RRF → prefix | 1–100 |
| hybrid_rerank | Hybrid-100 → first 20 → frozen CE → prefix | 1–20 |

No CE tail is appended. Reject K>20 for CE, invalid numeric types, empty queries
and invalid/disabled modes before model invocation. Results preserve query text,
retrieval provenance, minimal title metadata and final score/rank.

## D. Startup Lifecycle

Validate config → pinned manifests/configs → products/indexes/model checksums →
load products → BM25 → dense index → dense encoder → Hybrid → CE → READY.
Load only the enabled dependencies, eagerly and once. Lexical-only instances
do not need model payloads; Hybrid-only instances do not need CE payloads.
initialize() is idempotent after successful startup; search never initializes models.
Model/index building and explicit package sealing are separate from normal startup.

Final fully enabled startup: 4,184.46 ms, including verification and local loading,
excluding interpreter/import time before engine initialization. This is one observation.

## E. Artifact Validation

Validate runtime-manifest hash; Phase 1/2/3 manifests and selected configs; products
checksum/count/membership; BM25/dense manifests/configs and every referenced file;
enabled model IDs/revisions/snapshot inventories/checksums; software compatibility;
dense embedding dimension and catalog consistency. Missing or invalid artifacts
raise ArtifactError. There is no silent rebuild, download or substitute model.

Explicit packaging command: `python scripts/seal_runtime.py` (already executed;
refuses overwrite). It uses existing artifacts and writes only a small runtime
manifest and deployment configuration.

## F. Runtime Version / Provenance

Recursively immutable version object, retained once per engine, includes:
core contract/package version and source hashes; frozen dataset/retrieval/reranking
manifest paths/hashes; BM25/dense/Hybrid/CE configuration; model revisions; validated
artifact hashes; installed software/Python versions; device/dtype and numeric runtime
flags; enabled modes and fallback/load policy. No huge hashes run per request.

Runtime manifest SHA256:
`7adf8fcaace380db531b60765678da1dbe4461498e71e313c318cf82d52009b0`.

## G. Readiness

NOT_INITIALIZED, LOADING, READY, DEGRADED, FAILED and CLOSED are observable.
`ready` is true only for READY. Degradation records component/path errors;
successful later execution of that path clears its error. Unrelated-path success
and empty CE candidate sets do not claim CE recovery. Metadata inconsistency
requires replacement. All lifecycle behaviors have unit coverage.

## H. Failure / Fallback Behavior

| Condition | Outcome |
|---|---|
| Missing artifact / bad checksum / incompatibility | Startup failure, no repair/rebuild |
| Invalid request | ValueError before models |
| Dense/Hybrid inference failure | SearchError, DEGRADED, no BM25 substitution |
| CE inference failure | Explicit Hybrid prefix fallback if configured, otherwise SearchError |
| Missing product metadata | SearchError; never silently omit results |
| Zero results | Valid empty SearchResult |

CE fallback records requested_pipeline=hybrid_rerank, effective_pipeline=hybrid,
fallback_used=true and exception type as fallback_reason. CE scores become null;
final scores use RRF. Unit tests inject synthetic failures, not production incidents.

## I. Timing Instrumentation

Monotonic stage timing fields: preprocessing_ms, bm25_ms, dense_encoding_ms,
dense_search_ms, fusion_ms, reranking_ms, hydration_ms and total_ms.
Unused stages are zero. Startup has its own separate timing object.

One CE smoke observation (ms): preprocessing .0021, BM25 .3250, dense encoding
4.7759, dense search 4.9490, fusion .0987, reranking 25.6794, hydration .0135,
total 35.8697. This un-warmed smoke sample is neither a benchmark nor an SLA.

## J. Offline / Production Parity

Six fixed train query IDs: 0, 1, 2, 3, 5, 6. Independent offline objects use the
unchanged single-query methods. BM25 and Hybrid test K=10/20/100; CE tests K=10/20.
**48/48 checks pass** per process: candidate membership, ordering and score tolerance.
Observed maximum score difference is 0 for all three modes. Allowed absolute/relative
tolerances are 1e-6 for retrieval and 1e-4 for CE. No fallback occurred.

A fresh process reproduced identical ranked IDs and runtime identity. See parity.json,
runtime_final.json and runtime_final_restart.json.

### Investigated initial mismatch — retained evidence

The initial runtime_validation.json failed an extra historical-batch-cache comparison
for query 1. That check encoded six queries together, unlike the original experiment
which encoded all train queries in batches of 128. Investigation showed the original
offline single-query method also differs from that historical ordering.

For query 1, single-query versus original batch query vectors differ by at most
8.102506399154663e-8; corpus cosine scores differ by at most 2.384185791015625e-7.
Products 608/5173/20513/22679/33290 are an example near-tie cluster: batch cosines
vary around .81937760/.81937766, while single-query scores tie at .81937754.
The resulting dense ranks change RRF ordering, with unchanged Hybrid top-100
membership. The CE final ordering can change through its original tie rule.

Reproducing the original batch-128 context recovers historical Hybrid and CE results
for all six fixture queries. Production and the existing offline **single-query**
interfaces agree exactly, so no model, precision, tie-breaking or ranking source
change was needed. The parity reference now explicitly uses the correct request
contract; a separate check retains original batch-context reproduction.

This does not establish universal cross-batch ordering equivalence. Existing Phase 2/3
quality tables remain historical batch-evaluation evidence, not freshly measured
production-core quality. No labels were opened to estimate a new quality delta.
Initial failure and diagnostic reports are preserved rather than overwritten.

## K. Resource Snapshot

Immediately after fully enabled engine startup:

| Resource | Bytes | MiB |
|---|---:|---:|
| Process RSS / Windows working set | 1,172,353,024 | 1,118.04 |
| PyTorch GPU allocated | 181,724,160 | 173.31 |
| PyTorch GPU reserved | 197,132,288 | 188.00 |

Loaded: product metadata, BM25, dense index/encoder, Hybrid and CE. Diagnostic only;
not peak memory, total GPU usage, service capacity or memory optimization evidence.

## L. Runtime Validation

Real commands executed from the repository:

```powershell
.\.venv\Scripts\python.exe scripts/validate_runtime.py --parity --report reports/phase4/runtime_final.json
.\.venv\Scripts\python.exe scripts/validate_runtime.py --parity --report reports/phase4/runtime_final_restart.json --compare-restart reports/phase4/runtime_final.json
```

Both exit 0 with passed=true. They initialize actual local GPU models, verify READY,
run a synthetic smoke query through all modes, perform parity checks and record
resource snapshots/access logs. Second command verifies independent restart stability.
Hugging Face offline mode is enabled; no runtime artifact acquisition occurs.

## M. Tests

```text
.\.venv\Scripts\python.exe -m pytest -q
88 passed in 0.78s

.\.venv\Scripts\python.exe -m pip check
No broken requirements found
```

Coverage includes real tiny-artifact startup; missing/corrupt/incompatible inputs;
dispatch/limits/empty queries; load-once behavior; immutable provenance; readiness;
fallback metadata/recovery; hydration/missing metadata; empty results; timing fields;
offline parity. Ordinary unit tests do not download or initialize GPU models.

## N. Test Boundary Verification

**Final-test relevance labels accessed: NO.** No relevance-label files at all were
opened in Phase 4. Runtime requires products; integration additionally reads train
query strings to reproduce batch context. Shared query bytes are hashed/scanned,
but test query strings are not decoded or inferred. Guards report no blocked attempts.
Phase 1 frozen split/manifests and Phase 2/3 configs/source/evidence are unchanged.

## O. Changed Files

Modified README.md. Added docs/runtime.md; configs/runtime.json;
artifacts/phase4/manifest.json; four runtime/core source modules; scripts/seal_runtime.py
and scripts/validate_runtime.py; tests/test_runtime_artifacts.py and tests/test_search_engine.py;
compact Phase 4 reports including all failed/intermediate/final evidence and commit records.

checks.json contains the detailed file and exclusion audit. Raw WANDS, processed JSONL,
embeddings, BM25 indexes, model snapshots/caches, virtualenv and large experiment
outputs remain untracked/ignored. Only small manifests/configs/reports are Git-eligible.

## P. Recommended Commits

Phase 4 remains uncommitted. Suggested logical groups:

1. Runtime contracts/configuration and sealed artifact package (four files).
2. Artifact loader, SearchEngine and their two test modules (four files).
3. Package sealing/startup validation scripts and Phase 3 commit record.
4. Initial parity failure, diagnosis and intermediate verification evidence.
5. Final runtime/restart/parity/check reports.
6. README, lifecycle documentation and Phase 4 report.

No push. Final working tree contains these deliberate Phase 4 changes.

## Q. Phase 4 Verdict

**PASS.** Ready for Phase 5 — FastAPI search service and HTTP lifecycle, subject to
retaining the documented single-query/batch distinction and numerical runtime identity.
Phase 5 has not started. No production reliability, latency or deployment claim is made.
