# Phase 2: candidate retrieval evidence

Validation-only results; no final-test evaluation or production benchmark exists.
The experiment plan and development selection were saved before validation. No validation retuning occurred.

## Development experiments (288 queries; zero excluded)

| Configuration | Recall@100 | NDCG@20 | Build seconds |
|---|---:|---:|---:|
| bm25_A_1.5_0.75 | 0.362919 | 0.632349 | 0.676 |
| bm25_B_1.5_0.75 | 0.364044 | 0.630334 | 0.803 |
| bm25_C_1.5_0.75 | 0.360836 | 0.641300 | 6.879 |
| bm25_B_1.2_0.5 | 0.364528 | 0.633833 | 1.310 |
| dense_A | 0.361542 | 0.666722 | 15.236 |
| dense_B | 0.340631 | 0.637396 | 24.021 |
| dense_C | 0.334899 | 0.629893 | 189.961 |
| hybrid_rrf_20 | 0.381490 | 0.690711 | — |
| hybrid_rrf_60 | 0.381490 | 0.698651 | — |

A=name; B=name plus nonempty product_class/category_hierarchy (exact duplicate fields omitted);
C=B plus description/features. Numeric ratings/reviews are never included.
BM25 B k1=1.2 b=0.5 and dense A were selected by development Recall@100, then NDCG@20.
RRF 20 and 60 tied on Recall@100; 60 won the prespecified NDCG@20 tie-break.
Only one small pretrained model was tested; representations, not additional models, were explored.
Dense B/C reduced recall and took longer to encode. No additional trials were pursued.

## Validation (96 queries; zero excluded for every metric)

| Pipeline | Recall@10 | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.061540 | 0.120166 | 0.240835 | 0.358722 | 0.667169 | 0.667198 |
| Dense | 0.059675 | 0.104519 | 0.216166 | 0.346532 | 0.663854 | 0.645367 |
| Hybrid | 0.064195 | 0.118693 | 0.244217 | 0.380477 | 0.712984 | 0.703333 |

Recall uses all judged Partial/Exact items as the denominator, not K.
NDCG gain is 2**grade-1. Unjudged returned items contribute zero measured gain but are not known negatives.
Undefined metrics are excluded with explicit counts. Judged fraction at 100 is BM25 0.7478, Dense 0.6700, Hybrid 0.7132.
Incomplete qrels can bias comparisons; no significance, online traffic or A/B claim is made.
Hybrid improves Recall@100 and NDCG here, but Recall@20 is slightly below BM25.

## Candidate complementarity

Means per validation query; relevant means judged Partial/Exact. Recovery/loss is relative to BM25 at the same K.
| K | Overlap | Jaccard | BM25-only relevant | Dense-only relevant | Shared relevant | Hybrid relevant | Hybrid recovered | Hybrid lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5.8333 | 0.1935 | 9.5938 | 9.6354 | 5.4062 | 16.0000 | 6.4479 | 5.4479 |
| 50 | 17.4271 | 0.2449 | 19.3021 | 19.0938 | 15.6458 | 36.2292 | 11.0833 | 9.8021 |
| 100 | 35.9792 | 0.2530 | 30.3854 | 31.8333 | 30.8958 | 64.1146 | 16.9271 | 14.0938 |

RRF takes each component top-100 union, then orders/truncates to 100. It does not preserve the complete union.
At K=20/50 the hybrid can use deeper component candidates; recovery is not necessarily confined to dense top-K.
Mean relevant-hit gains and macro Recall gains differ because queries have different positive denominators.

## Single-query warm latency

24 fixed train queries, five warmups per pipeline, three repeats (72 observations), top_k=100.
Sequential process; one CPU BLAS thread; GPU query encoding, CPU exact float32 matrix search.
No query-vector cache in this timing; quality evaluation alone uses batched query vectors.
| Pipeline | P50 ms | P95 ms |
|---|---:|---:|
| BM25 | 0.2498 | 0.5435 |
| Dense | 9.1804 | 9.9428 |
| Hybrid | 9.7786 | 10.8194 |

Stage percentiles are in latency.json; stage percentiles must not be added to derive total percentiles.
Startup excludes Python process/import time and catalog loading. It includes artifact hash validation.
Model startup loads cached weights and may include Hub metadata checks; it is not a cold download measurement.
Dense/hybrid startup totals are sums of separately measured initialization stages, not process-start wall times.
Single startup measurements (ms): {"bm25_artifact_load_and_hash": 19.004799993126653, "dense_artifact_load_hash_norm_check": 87.63179999368731, "dense_total": 3932.9403000010643, "hybrid_total": 3951.945099994191, "model_from_local_cache_to_gpu": 3845.308500007377}.
Initial model acquisition/load took 22.244 seconds, reported separately.
The ~43k x 384 exact index is sufficient for this observed single-query workload; no ANN was introduced.
This small benchmark is diagnostic, not an SLA, throughput test or production P95.

## Test boundary

Final-test relevance labels accessed: NO. The runner blocks raw data, final-test JSONL and the global conflict table.
The shared canonical queries file is hashed/byte-scanned to identify rows; test query strings are never JSON-decoded, encoded or evaluated.
This is not a claim that no shared-file bytes were read. Only train/validation judgment files were opened.
Phase 1 unit tests use synthetic temporary test partitions, not the frozen WANDS test labels.
The frozen data layer, manifest and split remain unchanged. Access logs are boundary_development.json and boundary_validation.json.

## Artifacts and reproduction

artifacts/phase2/manifest.json records software, model revision, source hashes, all tried configurations, selection and artifact hashes.
Only small manifest files are Git-eligible; models, arrays, indexes and caches are ignored.
The selected dense matrix is 42,994 x 384 float32 (66,038,912 bytes including NPY header).
Run with Python 3.14.3 and the exact CUDA wheel described in the README. No cross-device bitwise embedding guarantee is claimed.
The driver refuses silent repeated development selection or validation. For an explicitly authorized reproduction, use a separate checkout/output workspace and preserve the existing results.
No CrossEncoder, API, learned ranking, concurrency benchmark or Phase 3 code has been implemented.
