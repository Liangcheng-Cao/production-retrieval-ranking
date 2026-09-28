# ML monitoring: aggregate features and offline replay

No real production traffic is available.
No online A/B test has been performed.

## Live signals versus offline features

Live API metrics use only aggregate query codepoint/whitespace token counts,
pipeline/outcome counts, result-count histograms, empty responses and fallback.
They do not require relevance labels, infer a query class, retain vectors, or run
extra inference. Pipeline usage is derived from request counter deltas; empty and
fallback rates divide their counters by the matching successful request population.
Use the error counter over all outcomes for error rates. Empty denominators are
unknown. Operational counters are not a claim that live production users exist.

Offline `monitoring.py` computes scalar 1-Wasserstein (original units), categorical
total variation [0,1], mean embedding norm and centroid cosine distance. Empty
populations return null, never zero drift. Zero/nonfinite embeddings and malformed
aggregate contracts are rejected. A checksum envelope protects the baseline;
split, count, histogram/mean consistency and embedding dimensions are validated.

## Frozen baseline and protocol

`configs/phase7_monitoring.json` was written before replay. Version monitoring-v1
uses all 288 frozen **train** query IDs sorted ascending. Codepoint counts use
`len(text)` without normalization; token counts use `str.split()`, not subwords.
Categories are frozen query_class values, including an explicit empty-string source
class for five queries. Source class strings retain the frozen dataset encoding.

`baseline.json` records source split/count, feature definitions, query-ID population
checksum, checksums of the data manifest, split file, shared query artifact, train
judgments, Phase 4 runtime manifest and monitoring plan, plus exact model identities.
Only rows belonging to train are decoded from the shared query artifact; its full
file checksum is verified without decoding/using final-test query text. Only
judgments.train.jsonl is opened for allowed training-quality diagnostics. The
installed BoundaryGuard blocks raw data, final-test judgments and global conflicts.

Dense encoder: sentence-transformers/all-MiniLM-L6-v2 at
1110a243fdf4706b3f48f1d95db1a4f5529b4d41, float32, 384 dimensions, L2 normalized,
query batch size 1. CE: cross-encoder/ms-marco-MiniLM-L6-v2 at
233902d25c440f23af6f7d6e94d2946bac0bee0a; frozen CE20, batch32, max length256.
No new model, selection, fine-tuning or changed retrieval parameters are involved.

## Synthetic drift scenarios

Each population has 288 rows; transformations operate only on train.

1. `long_modifiers`: append the fixed string
   ` modern durable compact versatile easy assembly for everyday home use`
   (69 codepoints, 10 whitespace tokens).
2. `category_mix`: choose the most frequent train class, breaking ties alphabetically.
   This is Accent Chairs (10 original queries). Cycle its sorted rows to 288, producing
   intentional repeated traffic and a 100% category share versus baseline 10/288.
3. `short_generic`: take the first whitespace token from each original query.

Modified queries retain the **source** class annotation; it is not a prediction of
their altered semantic intent. Thus zero category TV for modifier/short scenarios
is expected by construction, not evidence that their meanings are unchanged. These
are intentionally strong synthetic offline shifts, not representative production
incidents. Original relevance judgments are **not** reused to score modified traffic.

| Population | Character Wasserstein | Token Wasserstein | Class TV | Centroid cosine distance |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | ~0 (1.11e-16) |
| long_modifiers | 69.0000 | 10.0000 | 0 | .395037 |
| category_mix | 2.0486 | .3160 | .965278 | .274991 |
| short_generic | 14.9688 | 2.3681 | 0 | .328702 |

## Embedding and CE interpretation

All four mean embedding norms are 1.00000002–1.00000003, as expected from normalized
vectors. A norm shift diagnoses normalization/runtime issues; it does not measure
semantic drift itself. A centroid can hide offsetting subpopulation changes and is
not proof of quality degradation. The artifact retains only a population centroid,
not individual vectors. No drift detector was trained.

| Population | Mean CE top1 | Mean top1−top2 margin |
|---|---:|---:|
| baseline | 4.925546 | 1.350461 |
| long_modifiers | -.382326 | .842044 |
| category_mix | 4.738119 | 1.424939 |
| short_generic | 2.315998 | 1.029670 |

CrossEncoder logits are not calibrated probabilities. Score/margin distributions
are conditional on the frozen retrieved CE20 candidates and source traffic. They
are not probabilities or relevance-rate estimates. We retain count/mean/P05/P50/P95
for top1, margin and all returned CE20 logits; no individual scores or product IDs.

## Champion/challenger offline replay

Champion=existing hybrid; challenger=existing hybrid_rerank. Each build executes
both at K20 on all four populations: 2,304 searches/build, 4,608 across two builds.
Top10 comparison uses changed ordered-list fraction, overlap=intersection/10, and
mean absolute rank displacement **among shared Top10 items**, macro-averaged over
queries with shared items. Missing members are covered by overlap, not silently
assigned ranks. Full ranking lists exist only in working memory; report stores one
aggregate ranking checksum for reproducibility.

On 288 original train queries (576 searches/build), 100% Top10 orderings differ,
mean overlap=.628819, mean shared rank displacement=2.597973. Both pipelines return
20 hits/query, no empty results, no fallback, and effective mode equals requested.

| Train metric | hybrid | hybrid_rerank |
|---|---:|---:|
| NDCG@10 | .707188 | .725127 |
| NDCG@20 | .698653 | .705218 |
| Recall@10 | .061278 | .061176 |
| Recall@20 | .115067 | .115067 |

Existing graded gains/exclusion rules apply; zero excluded queries for these metrics.
Recall50/100 is not reported because both API outputs stop at20. This is an in-sample
training diagnostic, not an unbiased test estimate or a new champion selection.
Unjudged results have observed zero gain, not known-negative relevance.

Run1 core mean/P95 ms: hybrid9.823/11.045, CE15.695/17.732.
Run2: hybrid9.614/10.771, CE15.183/16.981. These sequential offline diagnostic timings
include any initial warmup and are separate from HTTP latency and Phase 6 benchmarks.
No throughput/speedup claim is drawn from them. Synthetic populations likewise had
zero empty results/fallback and 20 hits; detailed aggregates appear in monitoring.json.

## Reproducibility and retention

Two independently loaded engine instances in the same process rebuilt all scenarios.
Payloads compare byte-for-byte after canonical JSON serialization (including all
floating-point summaries and aggregate ranking checksums). Timestamps and latency
measurements live separately under execution. Exact equality held on this local GPU:
baseline payload SHA256 `1923f498d2a222eca6b2154ff5713808c0eefb14a2f39de15d1008906e321994`;
full replay payload SHA256 `7deee251d3bc91930effc1f4ebcf6443246352ea71430c97d5dcba27a0f015aa`.
This does not guarantee equality across GPU/library versions or encoder batch shapes.
No rounding hid floating-point variation: a mismatch retains both payloads and exits
nonzero. The historical Phase3 batched-encoder near-tie caveat remains documented;
single-query runtime parity and original historical batch context are checked separately.

Tracked reports contain aggregate features, population centroids, model/source hashes,
and summaries, not raw queries, per-query embeddings or full rankings. Temporary model
logs/detailed parity fixtures stay under ignored reports/tmp. Existing raw-data/index/
weights/cache/venv ignore policies are preserved. See observability.md for local log
retention policy; no regulatory compliance is asserted.

```powershell
.\.venv\Scripts\python.exe scripts/run_monitoring.py --output reports/tmp/monitoring-repeat
.\.venv\Scripts\python.exe scripts/validate_runtime.py --parity --report reports/tmp/core-repeat.json
.\.venv\Scripts\python.exe scripts/validate_phase7.py
```

Use a fresh monitoring output directory. The release checker validates the canonical
Phase7 evidence paths and ignored detailed parity files produced for this checkout;
regenerate the latter at the documented paths if repeating the entire release.
No final-test labels, Docker, online A/B test, or Phase8 work is part of these commands.
