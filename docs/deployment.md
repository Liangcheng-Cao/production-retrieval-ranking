# Phase 8 packaging and clean-environment reproduction

Status: **PASS** for local Phase 8 validation. Docker Engine 29.8.1 / Compose 5.5.1
built the actual project image and ran two independent GPU service containers.
Both reached READY, matched native HTTP rankings exactly and shut down gracefully.
The image full suite passed 156 tests with an ephemeral pinned pytest overlay.
Earlier missing-socket and registry failures remain retained as historical evidence.
Do not interpret the Linux virtual-environment results as a successful Docker build.

No real production traffic is available. No online A/B test has been performed.
No image was pushed and no cloud service was deployed. The accepted Phase 8 work is
retained in logical local commits; release JSON records preserve their precommit state.
Temporary validation containers were stopped and removed; the project image remains
local. Phase 9 has not started.

## Distribution boundaries

- The application ships as an ordinary, non-editable Python wheel. No repository
  checkout is needed on the service import path.
- Docker's build context is allowlisted to the Dockerfile, Python source/package
  metadata and deployment support files. Raw WANDS, processed data, indexes, model
  weights/cache, .venv, reports, Git history, credentials and fixtures are excluded.
- Runtime artifacts are a separately provisioned, read-only bind mount at `/runtime`.
  Neither data nor weights are included in any intended Docker image layer.
- The portable bundle contains only the exact validated inference closure: 35 files,
  352,611,504 payload bytes, plus runtime.json and bundle.json. It contains products,
  selected BM25/dense indexes, two sealed model snapshots and frozen provenance.
  It contains **no query file or relevance-label file**. Manifest hashes/IDs describe
  provenance and do not constitute access to final-test labels.
- Bundle packaging copies files, dereferencing the existing model cache layout.
  It refuses overwrite and verifies the copied hashes. Moving the whole bundle
  preserves relative paths; startup still uses the unchanged ArtifactLoader.
- A checksum proves identity/integrity relative to the reviewed manifest, not
  authenticity against an attacker who can replace every trusted manifest/config.
  Keep the reviewed bundle-manifest SHA256 separately in release evidence.

## Reproduction target and locks

Target: Linux amd64, CPython 3.14, NVIDIA GPU, frozen torch2.14.0+cu130 and models.
The real independent validation used Ubuntu26.04 under WSL, Python3.14.4, RTX4070,
driver581.08. Original reference environment is Windows Python3.14.3. Cross-environment
rankings matched on 18 train-fixture requests; this finite check is not a proof for
every possible query, GPU, driver, numerical runtime or operating system.

The official python:3.14.4-slim-trixie amd64 image is pinned to digest
`sha256:2409290aa375de35f6492db84c700067d5c4c2aacfaf770c155d7528fb68bcf1`.
The exact registry response identity is recorded in deployment/base_image.json.
deployment/linux-cp314-cu130.lock pins the full 72-wheel Linux runtime closure with
HTTPS URLs and SHA256 hashes, including Linux-specific NVIDIA/Triton packages.
deployment/linux-test.lock is a 3-wheel pytest overlay; it requires the runtime lock.
deployment/constraints.txt records reviewed environment versions used during resolution;
the locks, not unconstrained latest resolution, drive subsequent installation.

URLs point only at Python, PyTorch and NVIDIA package hosts; the lock-generation
script rejects unapproved hosts, non-wheels and missing hashes. This is an explicit
platform lock, not a universal Windows/ARM lock. Package installation needs access
to those package sources (or a separately provisioned wheelhouse); service inference
forces HF offline mode and uses sealed local models only.

Python wheels and the base image are hash-pinned. The Dockerfile still obtains two
OS shared libraries (libgomp1, libstdc++6) from Debian repositories during build.
Thus **bit-identical Docker rebuilds are not claimed**, and base/runtime OS compatibility
has not yet been exercised inside Docker. Freeze an OS package snapshot separately
if bit-identical image rebuilding becomes a requirement. No security scan was run.

The application wheel itself was built twice with SOURCE_DATE_EPOCH=1790599546;
both SHA256 values were
`db751bb36fdc75f534f20c191758780f3a039a8eef932da36b1ecbd67e05ef00`.
The clean service test installed a wheel containing the same source before this
timestamp normalization; its ZIP hash differs because of build timestamps. No
editable install, existing Windows .venv or existing model downloads were used by
the Linux environment. The existing model **payloads** were intentionally provisioned
from the frozen bundle; clean-environment reproduction does not mean retraining them.

## Build the inference bundle

From the existing checked environment:

```powershell
.\.venv\Scripts\python.exe scripts/package_runtime.py `
  --output reports/tmp/runtime-package-new `
  --report reports/tmp/runtime-package-new-report.json
```

Use fresh destinations. Raw data/labels are unnecessary. New artifact/model downloads,
index rebuilds and changes to frozen splits/configs are not performed. The bundle
inherits the existing CUDA/all-three-pipelines deployment configuration; there is
no silent CPU or lexical-only substitution when GPU/model dependencies are missing.
Store large bundles, dependency wheels and temporary fixtures outside Git.

## Docker operation once a working engine is available

Prerequisites: an amd64 Linux Docker Engine (or suitable Docker Desktop WSL backend),
Compose and NVIDIA Container Toolkit/GPU exposure. These were **not installed or
changed system-wide** by this task. The standalone Compose validator is an ignored
local tool only; it cannot provide a Docker Engine or GPU container runtime.

```powershell
$env:SEARCH_BUNDLE = (Resolve-Path reports/tmp/phase8-bundle).Path
docker compose config --quiet
docker compose build
docker compose up -d
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/metrics
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search `
  -ContentType 'application/json' `
  -Body '{"query":"wooden office desk","pipeline":"hybrid_rerank","top_k":10}'
docker compose logs search
docker compose stop
```

The image uses UID/GID10001, a read-only root filesystem, a read-only runtime mount,
bounded /tmp tmpfs, dropped Linux capabilities and no-new-privileges. Published HTTP
binds to localhost. There is no public deployment, authentication layer, reverse proxy,
TLS endpoint, scheduler, automatic rollback, hot artifact swap or distributed store.
Ensure the packaged files are readable by UID10001 on the host. Missing bind sources
are rejected instead of silently creating empty directories.

The service entry point is `python -m product_search.deployment`. It configures the
previously validated deterministic/TF32-off/single Torch and BLAS thread settings,
preloads native libraries in the main thread, and starts the unchanged API with one
engine execution worker. HF offline settings prevent model fetches; this is not a
claim that the container has a network firewall. No throughput optimization occurs.

Healthcheck polls `/ready` and requires ready=true/state=READY; HTTP200 DEGRADED is
searchable but intentionally fails this stricter container check. Docker's health
status alone does not automatically restart an unhealthy container in this Compose
configuration. Startup has a120-second grace period; graceful stop allows60seconds.

## Clean Linux verification without Docker

Create a new Linux venv using Python3.14 (the tested one is on WSL's Linux filesystem):

```bash
python3.14 -m venv /path/to/new-venv
/path/to/new-venv/bin/python -m pip install --require-hashes --no-deps -r deployment/linux-cp314-cu130.lock
/path/to/new-venv/bin/python -m pip wheel --no-build-isolation --no-deps --wheel-dir /path/to/wheels .
/path/to/new-venv/bin/python -m pip install --no-deps /path/to/wheels/production_retrieval_ranking-0.1.0-py3-none-any.whl
/path/to/new-venv/bin/python -m pip check
/path/to/new-venv/bin/python -m pip install --require-hashes --no-deps -r deployment/linux-test.lock
/path/to/new-venv/bin/python -m pytest -q
```

Generate the ignored parity fixture using the original reference environment:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_deployment_fixture.py --output reports/tmp/new-train-fixture.json
```

Then run in the new Linux environment (use Linux paths):

```bash
python scripts/validate_bundle_service.py --bundle /path/to/bundle --fixture /path/to/train-fixture.json --report /path/to/fresh-integration.json
python scripts/validate_bundle_failures.py --bundle /path/to/bundle --output /path/to/fresh-negative-evidence
```

The first script requires an installed package inside sys.prefix, launches a separate
server from the bundle directory, checks readiness/health/version/metrics, all three
pipelines against source references, invalid requests, query privacy, and shutdown.
It also verifies the bundle has not changed. Fixture query IDs must belong to frozen
train. The second script uses separate fault bundles to reject a missing manifest,
corrupt manifest or missing model; no silent repair/model download is allowed.
Hard-linked negative fixtures are read-only inputs; the source bundle is reverified.

Actual Linux results:156 unit tests passed,18/18 ordered rankings and scores identical,
all three startup faults rejected with exit3, pip check passed, bundle unchanged.
The source Windows tests also passed156/156. Both retain the pre-existing Starlette
httpx deprecation warning.

## Historical shutdown evidence and completed container gates

The first Linux integration checker incorrectly required exit0 on SIGTERM. Uvicorn
0.54's `capture_signals` re-raises SIGTERM after lifespan shutdown, producing -15
(shell143); logs confirmed service_shutdown and application shutdown complete.
The checker now accepts this documented behavior **and requires lifecycle completion**.
Failed and successful summaries are retained separately, rather than erasing failure.

Linux shutdown also emitted a resource_tracker warning about one loky semaphore
being cleaned up. It is retained as a limitation; no warning filter or claim of a
warning-free shutdown was added. This does not change the measured ranking results.

The actual project image now passes filesystem/history audit, GPU startup, non-root
and read-only checks, health/ready/version/metrics, all three search modes and invalid
request rejection. Two starts produce 36 exact ordered-result comparisons against
the native HTTP service, maximum score difference 0, unchanged version identity and
fresh metrics. Missing manifest, checksum corruption and no-GPU startup all exit3
without becoming READY or changing mounted files. Both normal Docker stops complete
lifespan/engine cleanup and exit0 without the earlier Linux semaphore warning.

The first resumed build still encountered registry authentication failure. Pulling
the exact pinned base explicitly succeeded, and the subsequent unchanged Compose
build completed in 504.07 seconds. No Dockerfile, dependency lock, global Docker
setting or ML configuration was changed. Image ID:
`sha256:d14f681b4c5dad9d62193f55dbbd3d0773508a4c691958918004e9f77d91180a`.

An optional alternate-epoch builder attempt was stopped because it invalidated the
dependency cache. Two clean application wheel rebuilds instead ran inside the actual
image with network disabled and pip cache disabled; both produced SHA256
`83bb15043638c830989197d4b49006c61418263b0088bacfd3a9e7cb6d7e1bb7`.
All application bytes and metadata match the installed image. The Windows reference
wheel differs only in METADATA CRLF/LF and its derived RECORD; cross-platform archive
byte equality is not claimed. No full Docker `--no-cache` rebuild is claimed.

The final image contains the installed application and healthcheck. Runtime config,
small frozen provenance and model/index payloads remain in the read-only inference
bundle, as designed; the image does not embed those runtime files or relevance labels.

To repeat the real checks after building, choose fresh output paths:

```powershell
.\.venv\Scripts\python.exe scripts/validate_docker_runtime.py --output reports/tmp/phase8-docker-repeat
.\.venv\Scripts\python.exe scripts/validate_image_wheel.py --report reports/tmp/phase8-image-wheel-repeat.json
.\.venv\Scripts\python.exe scripts/validate_image_tests.py --log reports/tmp/phase8-image-tests-repeat.log
.\.venv\Scripts\python.exe scripts/validate_phase8.py --check-only
```

The last command checks retained release evidence and the current image identity
without rewriting committed reports; the protected baseline remains Phase 7 HEAD.
new run reports do not automatically replace the reviewed evidence references.
The image test harness mounts only test/code/config support and creates empty
repository-layout directories in tmpfs. Actual raw data and labels are never mounted.
The small sequential latency smoke uses 12 requests per pipeline per environment;
it is diagnostic and does not replace Phase 6 evidence. Serving GPU allocator values
are not exposed; the separate CUDA probe's allocator values are labeled separately.
See reports/phase8/docker_runtime.json and reports/phase8/REPORT.md for measurements.

Implementation references: [Docker build guidance](https://docs.docker.com/build/building/best-practices/),
[Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/) and
[Compose service settings](https://docs.docker.com/reference/compose-file/services/).
