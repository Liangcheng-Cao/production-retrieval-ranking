"""Equal-weight reciprocal rank fusion; component scores are not mixed."""
from time import perf_counter
from .common import Hit, check_request

def rrf(lists, top_k=100, constant=60):
    check_request("", top_k)
    if type(constant) is not int or constant <= 0:
        raise ValueError("RRF constant must be a positive integer")
    scores = {}
    for hits in lists:
        ids = [h.product_id for h in hits]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidates in component")
        for rank, hit in enumerate(hits, 1):
            scores[hit.product_id] = scores.get(hit.product_id, 0.0) + 1 / (constant + rank)
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))[:top_k]
    return [Hit(pid, scores[pid], rank, "hybrid_rrf") for rank, pid in enumerate(ordered, 1)]

class HybridRetriever:
    def __init__(self, lexical, dense, constant=60, candidate_depth=100):
        if type(candidate_depth) is not int or candidate_depth <= 0 or type(constant) is not int or constant <= 0:
            raise ValueError("Invalid hybrid parameters")
        if set(lexical.product_ids) != set(dense.product_ids):
            raise ValueError("Hybrid catalog mismatch")
        self.lexical, self.dense = lexical, dense
        self.constant, self.candidate_depth = constant, candidate_depth

    def search_timed(self, query, top_k=100):
        check_request(query, top_k)
        if top_k > self.candidate_depth:
            raise ValueError("top_k exceeds fixed hybrid candidate depth")
        start = perf_counter()
        lexical = self.lexical.search(query, self.candidate_depth)
        lex_end = perf_counter()
        dense, stages = self.dense.search_timed(query, self.candidate_depth)
        fused_start = perf_counter()
        hits = rrf([lexical, dense], top_k, self.constant)
        end = perf_counter()
        return hits, {"lexical_ms": (lex_end-start)*1000, "encoding_ms": stages["encoding_ms"],
                      "search_ms": stages["search_ms"], "fusion_ms": (end-fused_start)*1000, "total_ms": (end-start)*1000}

    def search(self, query, top_k=100):
        return self.search_timed(query, top_k)[0]
