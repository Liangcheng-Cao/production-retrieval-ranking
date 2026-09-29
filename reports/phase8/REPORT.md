# Phase 8 Completion — Docker Runtime Validation

## A. Docker Runtime

PASS. Docker Engine/client 29.8.1, Docker Desktop 4.93.0 (240920), Compose 5.5.1;
Linux amd64 on WSL2 kernel 6.18.33.2-microsoft-standard-WSL2. NVIDIA runtime present.
The minimal CUDA container prerequisite and the actual project GPU container passed.

## B. Image Build

Actual unchanged Compose project build succeeded in 504.071 seconds. Tag:
`production-retrieval-ranking:phase8-local`. Local image identity:
`sha256:d14f681b4c5dad9d62193f55dbbd3d0773508a4c691958918004e9f77d91180a`. Docker-reported size: 9,684,249,609 bytes.

Base: `python:3.14.4-slim-trixie@sha256:2409290aa375de35f6492db84c700067d5c4c2aacfaf770c155d7528fb68bcf1`.
Dependency and application layers executed; no prior project image was reused.
The first resumed build still hit registry authentication reset. Directly pulling
the exact pinned base succeeded, followed by the successful project build.

An optional alternate SOURCE_DATE_EPOCH builder attempt was cancelled when it
invalidated the dependency cache. Two clean application wheel rebuilds subsequently
passed inside the finished image with no network, no pip cache and no dependency
installation. Both hashes: `83bb15043638c830989197d4b49006c61418263b0088bacfd3a9e7cb6d7e1bb7`.
This is application rebuild evidence, not a full Docker `--no-cache` rebuild.
No Dockerfile, dependency lock, ML configuration or artifact-layout change was needed.
OS packages remain resolved at build time; bit-identical Docker images are not claimed.

## C. Container Environment

Measured inside the actual project container: Python 3.14.4, PyTorch 2.14.0+cu130,
CUDA runtime 13.0, RTX 4070, driver 581.08, transformers 5.17.0,
sentence-transformers 6.1.0. CUDA available: true. UID: 10001.
The installed package is under /opt/venv/lib/python3.14/site-packages.

## D. Image Audit

Inspected 45,166 actual filesystem files plus retained build-layer history. No
forbidden raw-data/judgment/model-weight paths, project .venv, Git/SSH credential
paths, Hugging Face cache paths, raw experiment report paths or token environment
keys were found. Application source contains no Windows home paths. This targeted
packaging audit is not a general security vulnerability scan.

Application package and healthcheck exist in the image. As designed, runtime.json,
small provenance/manifests and inference models/indexes are supplied by the read-only
/runtime bundle, not embedded in the image. Their mounted presence was verified.
The inference bundle has 35 payload files, no queries or relevance judgments.

## E. Container Startup

| Start | Container ID | Compose return / READY seconds | Docker health |
|---|---|---|---|
| 1 | 3669a34f45b5 | 1.114 / 10.672 | healthy |
| 2 | cfa3b5729d11 | 1.408 / 8.971 | healthy |

Times begin before Compose startup; image build time is separate. Both use the same
image, UID10001, read-only root and read-only bundle mount. Startup reaches READY.

## F. Endpoint Validation

Both starts: /health, /ready, /version and /metrics return HTTP200. All three search
pipelines return HTTP200 without fallback. Empty query, unknown pipeline and rerank
K21 each return HTTP422. Strict Docker healthcheck and metrics/log query privacy pass.

## G. GPU Validation

Inside-container CUDA tensor sum-of-squares returns 14.0 on GPU. Real dense encoding
and CrossEncoder scoring execute through the HTTP search service using the existing
frozen CUDA configuration. Dense/reranking timings are positive and reranker scores
are present. The no-GPU failure below independently confirms no silent CPU fallback.

## H. Native / Container Parity

The fixed six-query train fixture is unchanged. Native HTTP is compared against
18 requests per container start across bm25, hybrid and hybrid_rerank (K10).
Across two starts: 36/36 ordered-result comparisons match, count mismatches 0,
rank mismatches 0, maximum absolute score difference 0, fallback/effective-pipeline
parity exact. This finite fixture does not establish universal numerical equality.

## I. Restart Reproducibility

Two separately created containers with different IDs reproduce the same public
version, runtime/artifact identity and fixture rankings. Request counters start at
0 in each lifecycle, ending at 57 and 21 respectively (first includes smoke calls).
Both become READY and healthy. The bundle inventory/checksums remain unchanged.

## J. Failure Validation

Missing manifest, corrupt manifest/checksum and no GPU each reject startup with
exit3 and no service_ready event. Log reasons respectively identify the missing
required artifact, checksum mismatch and missing NVIDIA driver. No GPU devices
were requested for the last case. Test layouts remain byte-identical after each
case; there is no artifact rebuild, silent repair or CPU fallback.

## K. Graceful Shutdown

Normal `docker stop --time 60` completes both FastAPI lifespan and SearchEngine
cleanup, confirmed by service_shutdown and application-shutdown-complete logs.
Both exit0 in 2.020 and 1.992 seconds.
No semaphore cleanup warning occurred in these container runs. Earlier standalone
Linux signal-exit/checker failures and its semaphore warning remain historical evidence.
All validation service/fault containers and their Compose network were removed.

## L. Resource Snapshot

After inference: Docker RAM 2.733 GiB of 15.5 GiB; serving PID1 RSS/high-water mark
1,641,064 kB, 6 threads. GPU device memory used 2,870 MiB of 12,282 MiB; this is a
shared device-wide snapshot, not exclusive serving-process memory. Product metadata,
BM25, dense encoder/index and CrossEncoder loading are confirmed by READY and real
pipeline execution. Serving allocator bytes are not exposed. The separate CUDA
probe allocates 512 bytes and reserves 2 MiB; those are not attributed to PID1.
No resource optimization was performed.

## M. Performance Smoke

Twelve sequential warmed requests per pipeline per environment. Client wall-clock
milliseconds, P50 / P95; approximate diagnostic only, no Phase 6 benchmark replacement.
Native Windows service stops before the container smoke begins.

| Pipeline | Native P50 / P95 ms | Container P50 / P95 ms |
|---|---|---|
| bm25 | 1.31 / 1.64 | 1.47 / 1.84 |
| hybrid | 11.90 / 13.42 | 11.27 / 12.49 |
| hybrid_rerank | 16.54 / 17.40 | 16.36 / 17.77 |

Small sample size, shared host/GPU activity and loopback/container networking limit
performance conclusions. No production traffic, capacity claim or online A/B test.

## N. Tests / Checks

- Final Windows full pytest: 156 passed, one existing Starlette/httpx warning.
- Full tests against installed image code: 156 passed, same warning, using an
  ephemeral three-package hash-locked pytest overlay. Production image unchanged.
- Host and project-container pip check: no broken requirements.
- Installed Compose config, image audit, real GPU integration, parity, restart,
  fault cases, bundle verification and git diff --check: passed.
- All prior tracked files except the landing README unchanged relative to Phase 7
  HEAD d5f22df1bb2a51a57ddfc9fcf8f74cb5e3facf7a. Frozen Phase 1–7 evidence/configs intact.
- Raw WANDS, embeddings, BM25 index, model/cache, .venv and large temporary artifacts
  remain ignored/untracked. No forbidden or large prospective Git files found.

Preserved diagnostic failures: prior missing Engine/registry pulls; resumed registry
reset; cancelled optional rebuild; initial wheel checker requiring cross-platform
byte identity (only METADATA CRLF/LF and derived RECORD differ); first image pytest
run lacking empty repository directories (155 passed/1 failed). The isolated test
harness now creates those empty directories and omits host bytecode. No real data
is mounted. Two subsequent clean Linux wheel rebuilds match each other and installed
image code/metadata exactly. Application payloads also match the Windows reference.

## O. Test Boundary

**Final-test relevance labels accessed: NO.**

**Final-test relevance artifacts packaged in runtime image: NO.**

Only fixed frozen train queries are used for real ranking checks. Synthetic unit
fixtures do not contain the final-test data. No Phase 9 execution, Git push or image push.

## P. Changed Files

Additional completion work: scripts/validate_docker_runtime.py,
scripts/validate_image_wheel.py, scripts/validate_image_tests.py; release-gate updates
to scripts/validate_phase8.py; README.md and docs/deployment.md; compact Phase 8
runtime/check reports and this report. docker_runtime_blocked.json and
REPORT_before_runtime.md preserve the previously blocked evidence and narrative.
Raw build/service/failure/test logs and responses remain under ignored reports/tmp.
No deployment implementation or frozen ML file changed during this completion.
The accepted Phase 8 work is retained in logical local commits. Release JSON preserves
its precommit snapshot, including the historical phase8_committed=false field.
precommit_checks.json records the authorized repeat of real native/container parity,
startup failures and boundary checks. Use validate_phase8.py --check-only after
committing to validate the evidence without changing the working tree.
Phase 7 commit hashes/messages/files remain in phase7_commits.json.

## Q. Phase 8 Verdict

**PASS.** A real local project image exists and real GPU-container validation passed.
The repository is ready for Phase 9 — frozen final evaluation and release-candidate
evidence — after the user's review/authorization. Phase 9 has not started.

Machine-readable evidence: [docker_runtime.json](docker_runtime.json),
[checks.json](checks.json). Historical context: [REPORT_before_runtime.md](REPORT_before_runtime.md).
