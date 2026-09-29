# Interview talking points

1. **Lexical and semantic complementarity:** why equal-weight RRF combines distinct relevant candidates; Hybrid final Recall@100 0.3905 versus BM25 0.3488 and Dense 0.3635; truncation still loses candidates.
2. **CE design and optional status:** rerank only the first 20 Hybrid candidates, append no tail; final NDCG@10 rises 0.7219 → 0.7431, but paired CI [-0.0034, +0.0465] includes zero. Default stays Hybrid.
3. **Final-test discipline:** selection precedes freeze; timestamp/hash-sealed protocol before first access; one primary plus one exact rerun; spent test set, no tuning; evaluated commit differs from later documentation HEAD.
4. **Quality and cost:** separate final-test quality from train-query serving measurements; C=1 Hybrid P95 13.70–14.66 ms versus CE 19.22–19.31 ms; local evidence cannot establish an online benefit or SLA.
5. **Queueing bottleneck:** one engine worker serializes inference; more HTTP concurrency chiefly increases tail latency. Non-engine benchmark time includes adapter work; explicit queue visibility comes from observability integration.
6. **Reliable artifact lifecycle:** eager checksum-validated startup, explicit CE→Hybrid fallback, private logs, bounded metrics, read-only non-root GPU runtime and 36 exact parity comparisons; finite fixtures do not prove universal numerical equivalence.
7. **Next experiments:** test parallel execution/batching against queueing, ANN only at a justified catalog scale, and real feedback/online experiments before claiming production impact; synthetic drift alone does not prove relevance degradation.
