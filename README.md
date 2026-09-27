# Production Retrieval / Ranking System

A compact WANDS product search project demonstrating retrieval, ranking, evaluation,
serving, benchmarking, observability, reliability and deployment. The eventual
comparison is ranking quality versus latency versus compute and system complexity.

Planned flow: catalog → preprocessing → indexes → lexical + dense retrieval →
hybrid candidates → optional CrossEncoder → Top-K → search engine → FastAPI →
benchmarks/load tests → monitoring → Docker.

Planned serving variants: BM25, hybrid, hybrid with CrossEncoder reranking.

## Current status

Phase 3 complete: frozen canonical data, BM25/dense/RRF retrieval and optional
pretrained CrossEncoder reranking. Validation quality, paired comparisons and
sequential single-query latency are measured. No API exists.
**No final-test or production benchmark results exist.** No online
traffic or A/B experiment exists; future simulated comparisons must be described
as offline replay or synthetic traffic simulation.

## Windows setup

Run in this repository using PowerShell. The verified local environment is Python
3.14.3 on Windows. Phase 2 verified torch 2.14.0+cu130, CUDA 13.0 and a real tensor
operation on RTX 4070 (driver 581.08); transformers 5.17.0 and sentence-transformers
6.1.0 import successfully. No Python migration was required.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval]"
.\.venv\Scripts\python.exe scripts/download_wands.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

Direct project dependencies are pinned in pyproject.toml; this is not a complete
transitive lock or a claim of cross-platform reproducibility.

Read [data/README.md](data/README.md) for the verified schema, license and download
details, [reports/dataset_audit.json](reports/dataset_audit.json) for hashes and
measured counts, and [reports/split_protocol.md](reports/split_protocol.md) for the
implemented methodology and frozen evaluation boundary.

## Canonical data and freeze

Build with `.\.venv\Scripts\python.exe scripts/build_data.py`. The raw inputs must
match the pinned hashes. A repeat run verifies identical outputs; changed frozen
outputs are refused. Rebuilds inspect labels globally and are provenance operations,
not routine model-development commands. See the methodology for clean reconstruction.

The supported interface is `product_search.data.load_dataset(directory, partition="train")`.
It returns immutable Product, Query, Judgment and Conflict records, verifies hashes
and validates contracts. Validation is available explicitly; test loading additionally
requires `allow_test=True`. This is an accidental-access guard, not a security boundary.

Graded relevance is Irrelevant=0, Partial=1, Exact=2; proposed NDCG gain is 2**grade-1.
Recall positives are Partial or Exact. Conflicting pairs are excluded, not relabeled.
Query 366 has no positives; retain it but exclude undefined Recall/NDCG from future
macro means and report exclusions. No metrics are implemented in Phase 1.

Seed-42 query-group splits contain 288 train, 96 validation and 96 test queries.
After Phase 1, final-test relevance labels must not guide selection, hyperparameter
tuning or engineering decisions. Use train for engineering, validation for selection.
Freeze model/metric settings before future final evaluation. Current global label
inspection was structural and is explicitly recorded in the manifest.

Configuration lives in configs/default.toml. Call load_config with an explicit
config path; data paths resolve relative to the repository, independent of cwd.
Future pipeline names are declarations only; split rules are versioned in code and
data/processed/data_manifest.json.

Raw data, processed data, weights, embeddings, indexes, caches, virtual environments
and temporary benchmark outputs stay outside Git. Small reviewed reports and
reproducibility manifests belong in Git. Preserve the upstream WANDS license when
redistributing its material and cite Chen et al., ECIR 2022.

## Retrieval architecture and evidence

Query → BM25 and/or Dense → optional RRF → structured product_id/score/rank/source.
All retrievers expose search(query, top_k=100). Models/indexes initialize once;
product embeddings are built once, and query embeddings are computed per search.
Ties use ascending product_id. Blank queries return []; invalid top_k fails.
BM25 omits zero-score results; hybrid top_k must not exceed its fixed depth of 100.

- BM25Retriever: bm25s 0.3.11, Lucene method, k1=1.2, b=0.5; name + class/category.
  NFKC/casefold/alphanumeric tokenization; no stemming or stopword removal.
- DenseRetriever: sentence-transformers/all-MiniLM-L6-v2, revision
  1110a243fdf4706b3f48f1d95db1a4f5529b4d41, Apache-2.0, 384 dimensions, masked mean
  pooling, L2 normalization, float32, batch 128, max 256 wordpieces; product name only.
  GPU encoding and exact NumPy CPU cosine search; no FAISS/ANN.
- HybridRetriever: equal-weight RRF, 1/(60+rank), each component top-100, union then
  truncate to 100. Component scores are not directly mixed.

Development used 288 queries and exactly nine configurations: BM25 A/B/C plus
one parameter alternative on B, dense A/B/C with one model, and RRF constants 20/60.
A=name; B=add class/category; C=add description/features. Longer dense text reduced
development Recall@100 (A .3615, B .3406, C .3349). Selection was fixed before validation.

Validation (96 queries, zero excluded for these metrics):

| Pipeline | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| BM25 | .1202 | .2408 | .3587 | .6672 | .6672 |
| Dense | .1045 | .2162 | .3465 | .6639 | .6454 |
| Hybrid | .1187 | .2442 | .3805 | .7130 | .7033 |

Hybrid improves coverage at 100 and NDCG here, but not Recall@20. At depth 100,
BM25/dense share 35.98 candidates per query; BM25-only and dense-only relevant hits
average 30.39 and 31.83. Hybrid recovers 16.93 relevant items beyond BM25 while losing
14.09, showing both complementarity and truncation cost. Unjudged items get zero
measured gain, not known-negative labels; incomplete qrels limit interpretation.

Warm single-query P50/P95 (ms): BM25 .250/.543, Dense 9.180/9.943, Hybrid 9.779/10.819.
This used 24 train queries x three repeats, five warmups, top-100, one BLAS thread;
it is not a production SLA or QPS benchmark. Startup is reported separately.
See [full measured report](reports/phase2/REPORT.md) and [retrieval manifest](artifacts/phase2/manifest.json).

The experiment entry point is scripts/run_phase2.py with development and validation
stages. It refuses to overwrite completed selections/results. Existing evidence must
be preserved; future authorized reproduction should use a separate workspace.
For normal use, load selected BM25/dense artifacts with the configs in
configs/phase2_selected.json. Persisted artifacts validate config and file checksums.

**Final test remains untouched by Phase 2 retrieval and evaluation. Final-test
relevance labels were not opened.** The scoped loader byte-scans the shared queries
file for IDs but only decodes/encodes train or validation query text. A process guard
blocks raw labels, final-test judgment files and the mixed conflict audit. Phase 1
code and the frozen split were not modified; test-access unit tests use synthetic data.

## CrossEncoder reranking and offline comparison

Actual selected path: query → Hybrid top-100 → first 20 candidates → CrossEncoder
→ ranked 20 (top-10 is a prefix). `CrossEncoderReranker.rerank` in
`src/product_search/reranking.py` accepts candidates without retrieving them and
preserves original score, rank and source. Model errors return retrieval ordering
with null reranker scores and an explicit fallback flag; malformed requests raise.
Experiments disable fallback so failed inference cannot silently affect evidence.

Selected model: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, Apache-2.0, 22,713,601 parameters.
Use product name, raw relevance logits, RTX 4070, float32, batch 32, max length 256,
paired longest-first/right truncation and dynamic batch padding. No fine-tuning.

The predefined development rule maximized NDCG@10 subject to reranker P95 ≤150 ms.
One model, two representations and depths 20/50/100 produced six trials on 288
train queries. Name-only at depth 20 won; selection was frozen before validation.
Batch size was fixed, not exhaustively optimized. No observed pairs were truncated.

Validation (96 queries, zero metric exclusions):

| Pipeline | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| Hybrid | .1187 | .2442 | .3805 | .7130 | .7033 |
| Hybrid + CE-20 | .1187 | N/A | N/A | .7522 | .7180 |

CE-20 returns only 20 items, with no unscored tail. Recall@20 is invariant;
larger cutoffs are N/A, not inherited candidate-pool recall. Paired mean NDCG@10
delta is +.03919, 95% bootstrap CI [.01522, .06373] (45 improved /35 unchanged /
16 worsened). NDCG@20 delta is +.01465, CI [.00483, .02486] (43/27/26).
Both median deltas are zero. These 10,000 paired query resamples quantify validation
query uncertainty, not production effects or uncertainty from model selection.

Warm sequential pipeline P50/P95 is 15.370/17.113 ms for CE-20,
20.803/23.323 ms for CE-50 and 32.200/36.322 ms for CE-100. Measurements use
24 train queries, five warmups and three repeats; startup is separate.
Hybrid alone measured 9.779/10.819 ms in the prior Phase 2 session.
CE-20 supports an optional quality-oriented mode; deeper reranking did not improve
the primary development objective, although CE-50 improved development NDCG@20.
These are diagnostic timings, not production latency, an SLA or an A/B test.
Incomplete relevance judgments can penalize promoted unjudged items, and the
small validation set does not establish general performance beyond WANDS.

See [Phase 3 evidence](reports/phase3/REPORT.md),
[error analysis](reports/phase3/error_analysis.md),
[artifact provenance](artifacts/phase3/manifest.json), and
[Phase 2 commit history](reports/phase2_commits.json).
`scripts/run_phase3.py` separates development and validation and refuses to
overwrite completed results. `scripts/smoke_reranker.py` checks the real model
using synthetic text; ordinary unit tests do not download or load model weights.

**Final-test relevance labels remained untouched throughout Phase 3.** The guard
blocks raw data, actual test judgments and the global conflict audit. Shared query
bytes are hashed/scanned for IDs; only train/validation text is decoded for inference.
Phase 1 frozen files and Phase 2 reports/configuration remain unchanged.

Phase 4 production search core and artifact lifecycle is not started.
No frontend, LLM features, distributed services or additional infrastructure are planned.
