"""BM25S Lucene scoring with stable product-ID ties."""
from pathlib import Path
import numpy as np
import bm25s
from .common import check_request, tokens, ids_array, ranked, product_text
from .artifacts import seal, verify

class BM25Retriever:
    def __init__(self, index, product_ids):
        self.index = index
        self.product_ids = ids_array(product_ids)
        if index.scores["num_docs"] != len(self.product_ids):
            raise ValueError("BM25 product mapping mismatch")

    @classmethod
    def build(cls, products, representation="A", k1=1.5, b=0.75):
        if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        products = sorted(products, key=lambda p: p.product_id)
        index = bm25s.BM25(k1=k1, b=b, method="lucene", backend="numpy")
        corpus = [tokens(product_text(p, representation)) for p in products]
        if not any(corpus):
            raise ValueError("Empty lexical vocabulary")
        index.index(corpus, show_progress=False)
        return cls(index, [p.product_id for p in products])

    def search(self, query, top_k=100):
        check_request(query, top_k)
        terms = tokens(query)
        if not terms:
            return []
        return ranked(self.product_ids, self.index.get_scores(terms), top_k, "bm25", positive_only=True)

    def save(self, directory, config):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.index.save(str(directory))
        np.save(directory / "product_ids.npy", self.product_ids, allow_pickle=False)
        return seal(directory, config)

    @classmethod
    def load(cls, directory, config):
        verify(directory, config)
        index = bm25s.BM25.load(str(directory), load_corpus=False, mmap=False)
        return cls(index, np.load(Path(directory) / "product_ids.npy", allow_pickle=False))
