# Production Retrieval / Ranking System

A compact WANDS product search project demonstrating retrieval, ranking, evaluation,
serving, benchmarking, observability, reliability and deployment. The eventual
comparison is ranking quality versus latency versus compute and system complexity.

Planned flow: catalog → preprocessing → indexes → lexical + dense retrieval →
hybrid candidates → optional CrossEncoder → Top-K → search engine → FastAPI →
benchmarks/load tests → monitoring → Docker.

Planned serving variants: BM25, hybrid, hybrid with CrossEncoder reranking.

## Current status

Phase 1 complete: typed canonical data, audited conflict exclusion, deterministic
query-group splits and a frozen split manifest. No retrieval models, embeddings or
API exist. **No final ranking or performance benchmark results exist yet.** No online
traffic or A/B experiment exists; future simulated comparisons must be described
as offline replay or synthetic traffic simulation.

## Windows setup

Run in this repository using PowerShell. The verified local environment is Python
3.14 on Windows. Metadata/wheel checks support retaining Python 3.14; see
reports/python_compatibility.md. No GPU or transformer stack is installed here.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
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

Phase 2 (not started) is lexical/dense/hybrid retrieval. No frontend,
LLM features, distributed services or additional infrastructure are planned.
