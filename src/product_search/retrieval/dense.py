"""Normalized float32 embeddings and exact CPU matrix search; encoder reused."""
from pathlib import Path
from time import perf_counter
import numpy as np
from .common import check_request, ids_array, ranked
from .artifacts import seal, verify

def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Finite two-dimensional embeddings required")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding")
    return array / norms

class DenseRetriever:
    def __init__(self, product_ids, embeddings, encoder=None):
        self.product_ids = ids_array(product_ids)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.product_ids) or self.embeddings.shape[1] == 0:
            raise ValueError("Dense index dimension mismatch")
        if not np.isfinite(self.embeddings).all() or not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-5):
            raise ValueError("Dense vectors must be unit normalized")
        self.encoder = encoder

    def search_vector(self, vector, top_k=100):
        check_request("", top_k)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.embeddings.shape[1],):
            raise ValueError("Query/index dimension mismatch")
        vector = normalize(vector[None, :])[0]
        return ranked(self.product_ids, self.embeddings @ vector, top_k, "dense")

    def search_timed(self, query, top_k=100):
        check_request(query, top_k)
        if not query.strip():
            return [], {"encoding_ms": 0.0, "search_ms": 0.0, "total_ms": 0.0}
        if self.encoder is None:
            raise ValueError("Query encoder required")
        start = perf_counter()
        # convert_to_numpy synchronizes GPU->CPU transfer before the search timer.
        vector = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        encoded = perf_counter()
        hits = self.search_vector(vector, top_k)
        end = perf_counter()
        return hits, {"encoding_ms": (encoded-start)*1000, "search_ms": (end-encoded)*1000, "total_ms": (end-start)*1000}

    def search(self, query, top_k=100):
        return self.search_timed(query, top_k)[0]

    def save(self, directory, config):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings, allow_pickle=False)
        np.save(directory / "product_ids.npy", self.product_ids, allow_pickle=False)
        return seal(directory, config)

    @classmethod
    def load(cls, directory, config, encoder=None):
        verify(directory, config)
        result = cls(np.load(Path(directory)/"product_ids.npy", allow_pickle=False),
                     np.load(Path(directory)/"embeddings.npy", allow_pickle=False), encoder)
        if config.get("dimension", result.embeddings.shape[1]) != result.embeddings.shape[1]:
            raise ValueError("Manifest dimension mismatch")
        return result
