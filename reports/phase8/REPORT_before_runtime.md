# Phase 8 — Docker, clean-environment reproducibility and deployment packaging

## Verdict

**NEEDS FIX — Phase 8 is not complete.**

Packaging implementation and independent Linux/CUDA validation pass. Docker Engine
29.8.1 / Desktop 4.93.0 and Compose 5.5.1 now respond. An actual minimal CUDA container
detects RTX 4070, driver 581.08, CUDA 13.0. However, two real project Compose builds
failed at the pinned base-image metadata stage: auth.docker.io token acquisition
timed out connecting to 199.59.149.203:443. Durations: 44.503 s and 43.421 s.
Direct HTTPS diagnosis also timed out. No project image exists, so project-container
runtime evidence remains unavailable. No image or Git push was performed.
Phase 8 changes remain uncommitted. See docker_runtime.json for compact evidence
and hashes of both retained build logs. The earlier missing-Engine failure below
is historical evidence, not the current blocker.

## Phase 7 commits and clean starting point

| Commit | Message |
|---|---|
| a1f99e0ee1a36f46bcd7622d5186145fb23600ef | feat: instrument search service with bounded metrics and private logs |
| a509d4d76af02f3be0b7cb5cf1ba538b9d4f106a | feat: add deterministic train-only drift and ranking replay |
| b5b55bc4e17ce06facfb1f196e5aeb2eb7f59761 | test: validate real observability queue visibility and release boundaries |
| 03f6eb97cbf280ced433d9e080611faddc21d791 | report: retain monitoring rebuild parity and integration evidence |
| d5f22df1bb2a51a57ddfc9fcf8f74cb5e3facf7a | docs: explain observability privacy and offline monitoring limits |

The repository was confirmed clean after these commits, before any Phase 8 edits.
phase7_commits.json contains each commit's complete file list. The evidence group
also includes release_checks.json recording the new authorized precommit rerun.

Phase 7 precommit checks: 143 tests passed; observability/queue integration passed;
two complete monitoring/drift rebuilds matched each other and the reviewed payload;
18 HTTP/core and 48 offline/core parity checks passed; boundary, ignore policy,
pip check and diff check passed. Phase 1–6 frozen files were unchanged. No push.

## Implemented distribution

- Ordinary Python wheel, tested as a non-editable installed package.
- Portable inference-only bundle, with relative configuration and checksum inventory.
- Hash-pinned official Python amd64 base image and 72-wheel Linux/CUDA runtime lock.
- Three-wheel pytest overlay and dependency-resolution provenance.
- Multi-stage Dockerfile; raw data, queries, judgments, indexes, models, venv, Git
  history and reports are excluded from build context. Runtime payloads are mounted.
- Compose: one service, GPU reservation, localhost publication, non-root UID10001,
  read-only root/runtime mount, tmpfs, capability drop and no-new-privileges.
- Strict READY healthcheck and graceful SIGTERM lifecycle.
- Installed-wheel service validator and independent missing/corrupt-artifact checks.

No retrieval model/configuration, rank logic, candidate depth, frozen split, worker
architecture or throughput tuning was changed. `deployment.py` configures the
already-tested numeric settings and calls the existing API implementation.

## Artifact bundle

35 files; 352,611,504 payload bytes, plus runtime.json and bundle.json.
Contents: frozen inference provenance, product metadata, selected BM25/dense indexes,
the existing dense model and CE20 snapshots. No query or relevance-label files.
Packaging refuses overwrite, copies rather than linking to model caches, verifies
all hashes and preserves the frozen runtime manifest SHA256.

Bundle manifest SHA256:
`789d7f7644c9032ae85d324ede128e04b4f11eec21aedbdc41d07cb2ef3a9e02`.

The actual large bundle lives in ignored reports/tmp; bundle.json in this report
directory contains only its compact inventory/provenance. A checksum establishes
identity relative to the reviewed root of trust, not a digital signature.

## Dependency and wheel reproducibility

Target: Linux amd64, CPython 3.14, torch2.14.0+cu130. The resolver preserved reviewed
versions for existing packages; Linux NVIDIA/Triton transitive dependencies are
explicitly captured. Locks use official HTTPS wheel URLs and SHA256 hashes with
`--require-hashes --no-deps`; pip check independently verifies dependency consistency.

Official base image: python:3.14.4-slim-trixie,
amd64 digest `sha256:2409290aa375de35f6492db84c700067d5c4c2aacfaf770c155d7528fb68bcf1`.
Runtime image OS libraries still resolve from Debian repositories at build time;
bit-identical Docker rebuilds are not claimed.

Two local application wheel builds used SOURCE_DATE_EPOCH=1790599546. Both SHA256:
`db751bb36fdc75f534f20c191758780f3a039a8eef932da36b1ecbd67e05ef00`.
The previously installed test wheel has identical member names/content hashes;
its archive bytes differ only due to timestamp normalization. See checks.json.

## Independent Linux environment

Created a new venv in WSL's Linux filesystem, installed the 72 locked runtime wheels,
then the ordinary project wheel. This was not a reuse/copy of the Windows .venv.
The installed package was confirmed inside the fresh venv's site-packages. The
server subprocess ran from the portable bundle directory with PYTHONPATH removed.
The preprovisioned frozen model payloads are intentionally reused, not downloaded
again or retrained. Source labels are unnecessary for serving.

Environment: Ubuntu26.04 WSL, Python3.14.4, RTX4070, driver581.08. Reference: original
Windows Python3.14.3 runtime. No Docker namespace was involved in this validation.

| Check | Result |
|---|---|
| Windows full pytest | 156 passed, 0 failed |
| Fresh Linux full pytest | 156 passed, 0 failed |
| pip check, both environments | No broken requirements found |
| Real Linux GPU HTTP parity | 18/18 passed, all three pipelines |
| Ordered IDs | Identical to source reference |
| Maximum final-score absolute difference | 0 |
| Health/ready/version/metrics | Passed |
| Invalid request rejection | HTTP422 |
| Query privacy, metrics and logs | Passed |
| Bundle inventory/hash after serving | Unchanged |
| Missing manifest | Startup rejected, exit3 |
| Corrupt manifest | Startup rejected, exit3 |
| Missing model file | Startup rejected, exit3; no silent repair |

Both test environments retain the existing Starlette/httpx deprecation warning.
The parity fixture contains six frozen train queries, three modes, K10. Its raw
queries/result lists remain ignored; tracked evidence retains aggregate comparisons
and checksums. Equality on this fixture does not prove equality for every query,
operating system, driver or GPU.

## Shutdown finding and retained failed check

The first Linux integration run passed all 18 rankings but the checker required
exit0 after SIGTERM. Uvicorn0.54 restores/re-raises SIGTERM after lifecycle cleanup,
so the actual process exit was -15 (shell143). The corrected checker accepts that
exit and also requires both service_shutdown and application shutdown complete.
The independent rerun passed. The failed report hash and diagnosis are retained.

A resource_tracker warning reports cleanup of one loky semaphore on shutdown.
It was not filtered or hidden; no warning-free-shutdown claim is made. Source
ranking/worker behavior was not altered to remove this diagnostic.

## Historical Docker verification and remaining gates

The official standalone Docker Compose v5.5.1 binary was downloaded to an ignored
local tools directory and checked against the release API's SHA256. No system-wide
daemon or NVIDIA container runtime was installed. `compose config --quiet` passed.

Actual `compose build` failed before building an image:

```text
failed to connect to the docker API at unix:///var/run/docker.sock
connect: no such file or directory
```

The absence of buildx also produced a classic-builder fallback warning; the missing
engine is the blocking error. Remaining gates are actual image build/layer inspection,
GPU container startup, UID/read-only-mount behavior, container health/shutdown and
in-container parity. WSL tests and YAML validation do not satisfy these gates.

The Engine/GPU prerequisite has since passed. Restore registry/authentication
connectivity for the pinned Python base image, then run the existing build/run
commands in docs/deployment.md. The normal build never reached dependency or
application layers; a no-cache rebuild cannot resolve that network failure.
No global DNS/proxy/Docker settings, base digest or ML configuration were changed.
No production deployment is authorized or performed by these local instructions.

The project image audit, container Python/PyTorch/CUDA/model probes, all HTTP
endpoints, native/container parity, independent restarts, injected startup failures,
shutdown, resource snapshots and performance smoke have **not** run. Prior Linux
results below must not be presented as project-container evidence. Installed Compose
configuration validation, full Windows pytest, pip check and frozen-boundary/ignore
checks were rerun successfully; final checks.json records this continuation.

## Boundary and Git audit

**Final-test relevance labels accessed: NO.** Only frozen train query text is used
for parity; no relevance judgments are required. Deployment bundles exclude all
query/judgment files. BoundaryGuard reports no blocked attempts in these validations.

All prior tracked files other than the landing README are unchanged relative to
Phase 7 HEAD, including Phase 1–7 source/config/manifests/tests/evidence. Raw WANDS,
embeddings, BM25 indexes, model payloads/cache, .venv, wheel/binary downloads and large
monitoring/benchmark/packaging artifacts remain untracked and ignored.

`git diff --check` passes. The final working tree contains only intended Phase 8
changes, unstaged and uncommitted; it is no longer clean because of this new work.
The clean state immediately after Phase 7 commits is separately recorded.

## Suggested Phase 8 groups (not committed)

1. Portable bundle/entry point: src/product_search/deployment.py,
   scripts/package_runtime.py, tests/test_deployment.py, scripts/lock_deployment.py.
2. Container/dependency definitions: Dockerfile, .dockerignore, compose.yaml,
   deployment/{base_image.json,constraints.txt,healthcheck.py,linux-cp314-cu130.lock,linux-test.lock}.
3. Installed-package verification: scripts/prepare_deployment_fixture.py,
   scripts/validate_bundle_service.py, scripts/validate_bundle_failures.py,
   scripts/validate_phase8.py.
4. Evidence/documentation: reports/phase8/*.json, reports/phase8/REPORT.md,
   docs/deployment.md, README.md. Keep blocked/failed diagnostic evidence visible.

No real production traffic is available. No online A/B test has been performed.
