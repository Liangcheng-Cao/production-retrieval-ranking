# Phase 3 — CrossEncoder reranking evidence

Validation-only offline experiment. No final-test labels, model training, API, concurrency test or production SLA.
Phase 2 was committed and the repository clean before implementation. Its metrics/artifacts remain unchanged.

## Metric sanity

BM25 validation NDCG@10=0.6671693286442933 and NDCG@20=0.6671984569895968.
The apparent equality is four-decimal rounding. Independent scalar sums match all 96 queries,
including five stored worked examples. 86 queries have judged positives at ranks 11–20.
Both DCG and IDCG use the actual cutoff. Historical Phase 2 values were not changed.

## Fixed experiment design

Model: cross-encoder/ms-marco-MiniLM-L6-v2, revision 233902d25c440f23af6f7d6e94d2946bac0bee0a, Apache-2.0, 22,713,601 parameters.
Raw relevance logits from BertForSequenceClassification; no sigmoid, training or fine-tuning.
RTX 4070, float32, batch 32, max length 256, right-side longest-first paired truncation.
A=name; B=name+class/category. Only these two short representations were tested; no description/features.
Depths 20/50/100. Primary selection: development NDCG@10 subject to reranker P95 <=150 ms,
then NDCG@20, smaller depth, representation name. All trials met the budget; it was not binding.
Batch 32 was fixed for this compact model, not claimed optimal. No second model was tested.
Development scores were computed for top-100 once per representation and reused for nested depth subsets;
latency uses real depth-specific inference. Actual batch invariance is covered by a separate synthetic integration smoke.
Selection was written before Phase 3 validation; the prior validation-only metric sanity check did not guide selection.

## Development quality and latency (288 queries)

| Text | Depth | NDCG@10 | NDCG@20 | Recall@20 | Recall@50 | Recall@100 | Rerank P95 ms | Pipeline P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 20 | 0.725127 | 0.705217 | 0.115067 | N/A | N/A | 6.469 | 17.113 |
| A | 50 | 0.720820 | 0.719809 | 0.116931 | 0.245510 | N/A | 13.402 | 23.323 |
| A | 100 | 0.720386 | 0.716691 | 0.115630 | 0.253388 | 0.381490 | 26.350 | 36.322 |
| B | 20 | 0.719708 | 0.702877 | 0.115067 | N/A | N/A | 7.650 | 17.663 |
| B | 50 | 0.716090 | 0.710089 | 0.113769 | 0.245510 | N/A | 17.248 | 26.895 |
| B | 100 | 0.713135 | 0.707119 | 0.113513 | 0.251308 | 0.381490 | 31.173 | 41.756 |

Selected: A, candidate depth 20. Deeper reranking did not improve the primary development objective.
All candidate sets and Recall@candidate_depth remained identical before/after reordering.
A had 0/28,800 truncated development pairs (max 73 tokens); B had 0/28,800 (max 89).
Selected validation had 0/1920 truncated pairs.
Metrics beyond returned depth are N/A, not zero and not inherited from the original top-100 pool.

## Validation quality (96 queries, zero undefined-metric exclusions)

| Pipeline | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.120166 | 0.240835 | 0.358722 | 0.667169 | 0.667198 |
| Dense | 0.104519 | 0.216166 | 0.346532 | 0.663854 | 0.645367 |
| Hybrid | 0.118693 | 0.244217 | 0.380477 | 0.712984 | 0.703333 |
| Hybrid + CE-20 | 0.118693 | N/A | N/A | 0.752176 | 0.717983 |

Actual path: Hybrid retrieve top-100 → take first 20 → CrossEncoder → ranked 20, with top-10 as a prefix.
Full membership is retained at depth 20, so Recall@20 is invariant. No unscored tail is appended.
Do not label the original candidate-pool Recall@100 as the returned CE-20 Recall@100.

## Paired query comparison

| Metric | Mean delta | Median | Improved | Unchanged | Worsened | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| NDCG@10 | 0.039192 | 0.000000 | 45 | 35 | 16 | [0.015223, 0.063734] |
| NDCG@20 | 0.014650 | 0.000000 | 43 | 27 | 26 | [0.004830, 0.024856] |

10,000 paired query resamples, seed 42, percentile interval, 96 eligible pairs.
Intervals describe query sampling uncertainty on this validation set, not production effects or model-selection uncertainty.
Incomplete judgments remain a limitation; unjudged products contribute zero observed gain but are not known negatives.
See error_analysis.md: category/modifier improvements and both annotation-coverage and graded-ordering regressions.

## Sequential latency

24 fixed train queries, five warmups per configuration, three repeats: 72 observations each.
One BLAS thread. Timings include real query encoding and retrieval, not cached candidates.
GPU forward is explicitly synchronized. H2D/D2H transfers are separately reported.
Below, A representation; each cell is P50 / P95 milliseconds.

| Depth | Tokenization | GPU forward | Transfers | Postprocess | Rerank total | Pipeline total | Effective batches |
|---|---|---|---|---|---|---|---|
| 20 | 0.925 / 1.219 | 3.972 / 5.140 | 0.288 / 0.583 | 0.022 / 0.025 | 5.321 / 6.469 | 15.370 / 17.113 | [20] |
| 50 | 2.161 / 3.112 | 8.167 / 9.818 | 0.498 / 0.690 | 0.044 / 0.052 | 11.009 / 13.402 | 20.803 / 23.323 | [32, 18] |
| 100 | 4.370 / 5.438 | 16.509 / 19.925 | 0.970 / 1.406 | 0.084 / 0.119 | 22.219 / 26.350 | 32.200 / 36.322 | [32, 32, 32, 4] |

First CE startup including acquisition: 13.552 s; cached startup in validation: 0.680 s.
Startup excludes process/import time and is separate from warm latency. Cached model loading can include Hub metadata checks.
Percentiles of individual stages must not be summed. No concurrency or throughput experiment was run.

## Quality/latency frontier — development quality only

All quality values below use the same 288 train queries. Baseline timings are the recorded Phase 2 diagnostic run;
CE timings are Phase 3 with the same query list/protocol, a different session. This is a descriptive comparison, not a controlled SLA.
| Pipeline | NDCG@10 | Recall@100 | Pipeline P50 ms | Pipeline P95 ms |
|---|---:|---:|---:|---:|
| BM25 | 0.637081 | 0.364528 | 0.250 | 0.543 |
| Hybrid | 0.707188 | 0.381490 | 9.779 | 10.819 |
| Hybrid + CE-20 | 0.725127 | N/A | 15.370 | 17.113 |
| Hybrid + CE-50 | 0.720820 | N/A | 20.803 | 23.323 |
| Hybrid + CE-100 | 0.720386 | 0.381490 | 32.200 | 36.322 |

CE-20 is preferable for this tested top-10 quality/latency objective. CE-50 has higher development NDCG@20,
so CE-20 does not dominate every objective. Hybrid remains the inexpensive full top-100 candidate path;
BM25 remains substantially faster. Candidate-depth coverage differs, so a blanket Pareto dominance claim would be misleading.
Validation supports retaining CE-20 as an optional quality-oriented ranking mode, not making every request use it.

## Reliability contract and scope

The reranker never retrieves. It accepts typed candidates and preserves retrieval score/rank/source.
Score ties retain retrieval rank, then product ID. Full reranking preserves IDs; top_k truncation is explicit.
Inference errors/malformed outputs return the original retrieval prefix with null CE scores and an explicit fallback flag/error type.
Input contract errors raise. Experiments use fallback=False to avoid hiding failed measurements.
Fallback tests inject synthetic failures; no production reliability claim is made.

## Freeze and provenance

Final-test relevance labels accessed: NO. Only train/validation label files were opened.
The shared query JSONL is byte-scanned for IDs and hashed; test query strings are not decoded/encoded/evaluated.
Phase 1 split/code/manifests and Phase 2 evidence remain unchanged. Access logs and metric sanity provenance are retained.
artifacts/phase3/manifest.json records both prior manifest hashes, the model revision, configuration, source/report hashes and ignored artifact checksums.
Weights, caches, candidate pools and per-pair scores remain ignored. Only compact reports/configs/manifests are Git-eligible.
Normal tests do not download a model. scripts/smoke_reranker.py uses real model weights with synthetic text only.
Phase 4 production search core is not implemented.
