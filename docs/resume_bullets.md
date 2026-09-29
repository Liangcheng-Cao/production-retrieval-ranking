# Resume bullets

- Built hybrid product retrieval on WANDS, improving frozen held-out Recall@100 from 0.3488 (BM25) to 0.3905; optional CE-20 reached NDCG@10 0.7431 versus 0.7219, with a paired 95% interval including zero.
- Served three search modes through FastAPI; measured Hybrid HTTP P95 of 13.70–14.66 ms at C=1 on RTX 4070 and identified serialized execution/queueing as the concurrency constraint across 23,040 checked requests.
- Validated non-root GPU Docker serving with 36/36 exact native/container ranking comparisons and startup fault rejection; preserved a pre-access final-test freeze and one exact evaluation rerun with no post-test tuning.
