# Phase 9 — Frozen Final Evaluation & Release-Candidate Evidence

## A. Phase 8 Commits

| Hash | Message |
|---|---|
| e0ee962 | feat: package portable runtime bundles and generate dependency locks |
| 76cae53 | build: pin Linux CUDA dependencies and base image |
| 486427e | build: define non-root GPU container and read-only runtime mounts |
| 1c7cfda | test: validate installed runtime bundles and train-only service parity |
| 3bf542c | test: validate Docker GPU lifecycle and reproducible application builds |
| 5a03f01 | report: retain Docker parity failure and precommit release evidence |
| 75c72d0 | docs: record verified deployment and Phase 8 release boundaries |

Starting branch main, HEAD `75c72d09e9af756a11888ef216afd4c2ee6a6f0d`, working tree clean, as captured in
starting_baseline.json before Phase 9 file creation. The user explicitly approved
uncommitted Phase 9 additions before access; no Phase 9 commit was made.

## B. Pre-Access Freeze

Freeze record: reports/phase9/freeze_record.json.
SHA256: `00214f9d55af60528a8420664204b73bfdccefc63a8f7223e63c1685118dbcea`.
Protocol: configs/phase9_protocol.json; version phase9-frozen-final-v1;
SHA256: `dc986b4974170f9daee6dbc74d7059dc6bd550affcdddf00937f3ebbab15138d`.

| Manifest | SHA256 |
|---|---|
| phase1_data_manifest_sha | `0b79ce2061ada9574885b952e63f06c9516aa38682b1e91ba1f4334a5d6eb845` |
| phase2_retrieval_manifest_sha | `197cf3325df537f10ce917ac604984ff500d2202719626b1aad8e432cbae34b3` |
| phase3_reranking_manifest_sha | `fef7ff4e731b5bddd6beb83c611216191849745db663fabdce3de1ecad2cc4fb` |
| phase4_runtime_manifest_sha | `7adf8fcaace380db531b60765678da1dbe4461498e71e313c318cf82d52009b0` |

Test partition: 96 queries; partition checksum `8bc47afd2e3c2812204738c918b4c9645fb5f1894b6561bb828a9ac6ba5c2181`.
The freeze records all selected BM25/dense/RRF/CE configurations, revisions,
artifact hashes, source/config/test hashes, metric semantics and bootstrap settings.
BM25 B k1=1.2 b=0.5; Dense A/name representation, MiniLM revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41; equal-weight RRF60, depth100;
CE representation A/name, revision233902d25c440f23af6f7d6e94d2946bac0bee0a,
batch32, max length256, depth20. Nothing was selected from final outcomes.

The freeze was flushed/fsynced and its seal written at **2026-09-29T05:33:45.788739+00:00**,
before first label access. UTC timestamps, monotonic timestamps and file metadata
are retained. Authorization time is the locally recorded receipt time, not an
invented platform message timestamp. Working_tree_clean is truthfully false at
freeze because the user-approved Phase 9 code/protocol/evidence additions are
uncommitted; the starting Phase 8 baseline was clean. All additions present at
freeze are individually hashed. There is no hidden pre-evaluation commit.

## C. Final-Test Access Audit

First process audit-open event: **2026-09-29T05:34:03.130126+00:00** UTC;
read/validation completed at 2026-09-29T05:34:03.300998+00:00. The event is recorded before
the OS file open; successful read and observed row count/checksum are recorded after.
Script: scripts/run_phase9.py, SHA256 `2542010941b0dab65e24f8663ee90b75fa438f2735896cb4373d9d32ad3fa649`.
Artifact: data/processed/judgments.test.jsonl;
SHA256 `000c2edd8f81d99d65c4d5a0b260aa378de2be834625c47a9c3205df9bae9dab`; 48,492 rows, 96 queries.
The test set is now spent. Exactly one label-file read occurred in the primary
process and one in the sole exact rerun. No manual test-query/example browsing.

## D. Final Quality Results

Macro metrics, scores in [0,1]. Irrelevant=0, Partial=1, Exact=2; NDCG gain2**grade-1;
Recall positives Partial or Exact. Unjudged pairs have zero evaluation gain but
are not assigned negative labels. Existing Phase 2 evaluation functions are reused.

| Pipeline | Recall@10 | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.060217 | 0.106019 | 0.224848 | 0.348786 | 0.651768 | 0.633845 |
| Dense | 0.061192 | 0.121203 | 0.246117 | 0.363456 | 0.683312 | 0.679263 |
| Hybrid | 0.071381 | 0.131149 | 0.258428 | 0.390499 | 0.721947 | 0.720425 |
| Hybrid + CE-20 | 0.073484 | 0.131149 | N/A | N/A | 0.743074 | 0.731876 |

CE-20 reranks exactly Hybrid Top-20 from its Top-100 candidate list. No untouched
tail is appended. Recall@50/@100 are undefined for this production contract.
Primary outputs were sealed at 2026-09-29T05:34:03.453113+00:00 before the descriptive
validation comparison began at 2026-09-29T05:34:03.481433+00:00.

## E. Excluded Queries

96 total queries. One zero-positive query excluded from every defined Recall macro;
one zero-IDCG query excluded from each NDCG macro, leaving95 eligible queries.
N/A CE deep recall is an unsupported cutoff, not an excluded-query count.
Ranking/complementarity diagnostics still use all96 queries, as in prior definitions.

## F. Validation vs Final Test

Signed absolute-scale delta = test minus validation; not relative percent change.
Descriptive only. No automatic overfitting diagnosis or configuration change.

| Pipeline | Metric | Validation | Final test | Delta |
|---|---|---:|---:|---:|
| BM25 | Recall@20 | 0.120166 | 0.106019 | -0.014147 |
| BM25 | Recall@100 | 0.358722 | 0.348786 | -0.009935 |
| BM25 | NDCG@10 | 0.667169 | 0.651768 | -0.015401 |
| BM25 | NDCG@20 | 0.667198 | 0.633845 | -0.033354 |
| Dense | Recall@20 | 0.104519 | 0.121203 | +0.016684 |
| Dense | Recall@100 | 0.346532 | 0.363456 | +0.016924 |
| Dense | NDCG@10 | 0.663854 | 0.683312 | +0.019458 |
| Dense | NDCG@20 | 0.645367 | 0.679263 | +0.033896 |
| Hybrid | Recall@20 | 0.118693 | 0.131149 | +0.012456 |
| Hybrid | Recall@100 | 0.380477 | 0.390499 | +0.010022 |
| Hybrid | NDCG@10 | 0.712984 | 0.721947 | +0.008963 |
| Hybrid | NDCG@20 | 0.703333 | 0.720425 | +0.017092 |
| Hybrid + CE-20 | Recall@20 | 0.118693 | 0.131149 | +0.012456 |
| Hybrid + CE-20 | Recall@100 | N/A | N/A | N/A |
| Hybrid + CE-20 | NDCG@10 | 0.752176 | 0.743074 | -0.009102 |
| Hybrid + CE-20 | NDCG@20 | 0.717983 | 0.731876 | +0.013893 |

## G. Paired Hybrid vs CE

Challenger CE-20 minus Hybrid; same95 eligible paired query IDs sorted numerically;
seed42,10,000 paired bootstrap samples,95% percentile CI, tie tolerance1e-12.

| Metric | Mean delta | Median delta | Improved / unchanged / worsened | 95% CI |
|---|---:|---:|---:|---|
| NDCG@10 | +0.021127 | 0.000000 | 40 / 36 / 19 | [-0.003351, +0.046499] |
| NDCG@20 | +0.011450 | 0.000000 | 43 / 29 / 23 | [-0.000782, +0.025358] |

Both intervals include0. Higher point estimates do not establish a conclusive
positive mean effect under this frozen interval procedure. The frozen pipelines
and serving default were retained; no tuning or pipeline promotion followed.

## H. Retrieval Complementarity

Exact existing Phase 2 definitions at K20/50/100. Relevant-hit counts below are
sums across query-product pairs, not unique products across the entire dataset.
Hybrid recovered means relevant items in Hybrid Top-K absent from BM25 Top-K.

| K | Mean candidate overlap count | Mean Jaccard | BM25-only relevant | Dense-only relevant | Shared relevant | Hybrid recovered beyond BM25 | Hybrid relevant |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5.396 | 0.180266 | 843 | 1003 | 493 | 693 | 1570 |
| 50 | 17.115 | 0.239101 | 1621 | 1911 | 1519 | 1161 | 3479 |
| 100 | 37.385 | 0.268498 | 2529 | 2787 | 3200 | 1506 | 6026 |

Machine-readable output also preserves per-query means and Hybrid losses versus
BM25. These findings did not change RRF, candidate depth or model choice.

## I. Ranking-Change Diagnostics

Hybrid versus CE-20, existing Phase 7 Top-10 comparison across96 queries:
ordering changed for100%; mean shared Top-10 fraction **0.623958**;
mean shared-item absolute rank displacement **2.597789**.
No queries excluded from displacement. No new qualitative categories were created.

## J. Quality / Serving Tradeoff

Quality source: **Phase 9 frozen offline final test**.
Performance source: **Phase 6 local controlled HTTP benchmark**, separate train
traffic. P95 ranges retain both independent runs; QPS range covers frozen C1/2/4/8
cases across both runs. No final-test load benchmark was run.

| Pipeline | Final NDCG@10 | Final Recall@100 | Local C1 P95 ms range | Local C8 P95 ms range | QPS range |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.651768 | 0.348786 | 2.99–3.21 | 27.38–31.51 | 363.35–526.02 |
| Dense | 0.683312 | 0.363456 | N/A | N/A | N/A |
| Hybrid | 0.721947 | 0.390499 | 13.70–14.66 | 91.15–95.15 | 77.26–93.68 |
| Hybrid + CE-20 | 0.743074 | N/A | 19.22–19.31 | 130.19–135.49 | 54.95–63.21 |

Dense has no standalone HTTP serving pipeline in Phase 6; performance remains N/A.
Local controlled benchmark values are not production capacity or online A/B evidence.

## K. Deployment Evidence

Phase 8 evidence was referenced and hashed, not rebuilt or rerun. Image identity:
`sha256:d14f681b4c5dad9d62193f55dbbd3d0773508a4c691958918004e9f77d91180a`.
Real GPU containers reached READY twice, native/container ordering matched36/36
with score difference0, metrics reset, fault cases rejected startup, and shutdown
completed. reports/phase8/docker_runtime.json SHA256:
`c64565b6a8fd3b9759a115242c794baee61609fc2abb3e95284030b2769692ec`.
The Phase 8 runtime bundle/image contain no final-test relevance labels. No final
labels were added to deployment artifacts or Docker build context in Phase 9.

## L. Release-Candidate Manifest

Path: reports/phase9/release_candidate_manifest.json.
SHA256: `c4df8c3a572ff546dbda8c2bf8cb5d0f21f11c50819db8a3b4a23e2bba3e7aee`.
Links exact committed Phase 8 HEAD plus hash-frozen uncommitted Phase 9 harness,
model revisions and artifact checksums; data/retrieval/reranking/runtime manifests;
freeze/protocol; primary quality/per-query/paired outputs; access and integrity
records; Phase 6 benchmark, Phase 7 monitoring/observability and Phase 8 deployment.
Every linked evidence hash was checked. The manifest does not imply the new harness
already exists at Git HEAD; its distinct source hash is explicit.

## M. Reproducibility Rerun

Exactly1 primary and1 exact rerun. Same HEAD, source/configs, manifests, split,
models, protocol and seeds. All six deterministic metric/diagnostic files are
byte-identical, as are raw ranking and model-score outputs. Maximum numerical
difference0. Audit timestamps intentionally differ. No retries for better results.
Raw rankings/scores and rerun files remain ignored under reports/tmp/phase9;
checksums and comparisons are retained in reproducibility.json and seals.

## N. Test-Access Integrity

- Source files changed after first label access: **NO**.
- Model configurations changed: **NO**.
- Retrieval parameters changed: **NO**.
- CE depth changed: **NO**.
- Metric semantics changed: **NO**.
- Model/config/code-selection changes after access: **NO**.
- Tuning using test results: **NO**.

All pre-existing tracked files, including Phase 1–8 evidence and README, remain
byte-identical to the clean starting baseline. Only aggregate final outcomes were
inspected for reporting. The final set is spent and cannot be reused for development.

## O. Tests / Checks

Before access:

- python -m pytest -q: **165 passed,1 existing Starlette/httpx warning**.
- python -m pip check: **No broken requirements found**.
- run_phase9.py preflight: real train-only four-pipeline execution and18 core parity
  checks; deterministic summaries, synthetic boundary/exclusion/immutability tests.
- run_phase9.py freeze: frozen hashes and pre-access record verified.
- git diff --check passed; status recorded as authorized Phase 9 additions only.

An initial preflight guard rejected the harmless tracked data/raw/.gitkeep directory
marker while hashing historical files. This was corrected and tested **before**
freeze/access; the failed diagnostic log remains reports/tmp/phase9-preflight.log.
No real raw data or final labels were opened by that failed preflight.

After access: run_phase9.py seal independently checked macro aggregation from saved
per-query rows, recomputed the fixed paired bootstrap, verified primary/rerun output
checksums and historical/source/artifact integrity, and validated every release
manifest link. git diff --check passed. No Phase 6 load test or Docker rerun.

## P. Changed Files

New code/protocol/tests, all frozen before labels:

- configs/phase9_protocol.json
- scripts/run_phase9.py
- tests/test_phase9.py

New evidence:

- reports/phase9/complementarity_final.json
- reports/phase9/evidence_checksums.json
- reports/phase9/final_quality.json
- reports/phase9/freeze_record.json
- reports/phase9/freeze_record_seal.json
- reports/phase9/integrity_audit.json
- reports/phase9/label_open_event.json
- reports/phase9/paired_deltas_final.json
- reports/phase9/paired_final.json
- reports/phase9/per_query_final.json
- reports/phase9/preflight.json
- reports/phase9/primary_seal.json
- reports/phase9/primary_started.json
- reports/phase9/quality_serving.json
- reports/phase9/ranking_diagnostics_final.json
- reports/phase9/release_candidate_manifest.json
- reports/phase9/reproducibility.json
- reports/phase9/rerun_started.json
- reports/phase9/run_environment.json
- reports/phase9/starting_baseline.json
- reports/phase9/test_access_audit.json
- reports/phase9/validation_vs_test.json

Plus reports/phase9/REPORT.md and its SHA256 sidecar. These are uncommitted additions;
no existing tracked file changed. README was left unchanged because the report is
self-contained and Phase 10 owns broader presentation work. Raw labels, query text,
models, rankings and large artifacts are not added to Git.

## Q. Recommended Commits

Not executed; no push.

1. Protocol/evaluator/synthetic tests (3 files).
2. Starting baseline, preflight, freeze record/seal and access audit/events.
3. Final quality/per-query metrics, paired results/deltas and primary seal.
4. Complementarity/ranking diagnostics, descriptive validation/serving comparisons.
5. Reproducibility, integrity audit, release manifest, checksum seal and concise report.

Keep all existing phase histories intact. Final evidence JSON is immutable; do not
regenerate it to update timestamps or substitute more favorable results.

## R. Phase 9 Verdict

**PASS.** Protocol frozen before access; correct one-time primary evaluation; one
exact rerun; no post-access tuning; traceable and reproducible evidence.
Ready for **Phase 10 — final README, GitHub release preparation, and recruiter-facing
portfolio polish**, subject to the next user instruction. Phase 10 has not started.
No Git commit, Git push or Docker image push was performed during Phase 9.
