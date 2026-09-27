"""Judged Recall and graded NDCG; unjudged results have zero gain, not negative labels."""
from collections import defaultdict
import math
import numpy as np
from .data.schema import RELEVANCE, POSITIVE_LABELS

def qrels(judgments):
    result = defaultdict(dict)
    for j in judgments:
        if j.label not in RELEVANCE or j.relevance != RELEVANCE[j.label] or j.product_id in result[j.query_id]:
            raise ValueError("Invalid or duplicate judgment")
        result[j.query_id][j.product_id] = j.relevance
    return dict(result)

def query_metrics(product_ids, judgments):
    if len(product_ids) != len(set(product_ids)) or any(v not in RELEVANCE.values() for v in judgments.values()):
        raise ValueError("Duplicate ranking or invalid gain")
    positives = {pid for pid, grade in judgments.items() if grade >= RELEVANCE[POSITIVE_LABELS[0]]}
    result = {}
    for k in (10, 20, 50, 100):
        result[f"Recall@{k}"] = len(set(product_ids[:k]) & positives) / len(positives) if positives else None
    ideal = sorted(judgments.values(), reverse=True)
    for k in (10, 20):
        dcg = sum((2**judgments.get(pid, 0)-1)/math.log2(rank+2) for rank, pid in enumerate(product_ids[:k]))
        idcg = sum((2**g-1)/math.log2(rank+2) for rank, g in enumerate(ideal[:k]))
        result[f"NDCG@{k}"] = dcg/idcg if idcg else None
    result["judged_fraction@100"] = sum(pid in judgments for pid in product_ids[:100])/len(product_ids[:100]) if product_ids else 0.0
    return result

def evaluate(rankings, judgments):
    if set(rankings) != set(judgments):
        raise ValueError("Evaluation query coverage mismatch")
    rows = {qid: query_metrics(ids, judgments[qid]) for qid, ids in rankings.items()}
    names = ("Recall@10", "Recall@20", "Recall@50", "Recall@100", "NDCG@10", "NDCG@20", "judged_fraction@100")
    return {"query_count": len(rows), "metrics": {name: float(np.mean([row[name] for row in rows.values() if row[name] is not None]))
             if any(row[name] is not None for row in rows.values()) else None for name in names},
            "excluded_queries": {name: sum(row[name] is None for row in rows.values()) for name in names}}

def complementarity(lexical, dense, hybrid, judgments):
    output = {}
    for k in (20, 50, 100):
        totals = defaultdict(list)
        for qid in judgments:
            a,b,h = (set(r[qid][:k]) for r in (lexical,dense,hybrid))
            positive = {pid for pid,g in judgments[qid].items() if g >= 1}
            totals["overlap_count"].append(len(a&b))
            totals["jaccard"].append(len(a&b)/len(a|b) if a|b else 0)
            totals["bm25_only_relevant"].append(len((a-b)&positive))
            totals["dense_only_relevant"].append(len((b-a)&positive))
            totals["shared_relevant"].append(len(a&b&positive))
            totals["hybrid_relevant"].append(len(h&positive))
            totals["hybrid_recovered_beyond_bm25"].append(len((h-a)&positive))
            totals["hybrid_lost_vs_bm25"].append(len((a-h)&positive))
        output[str(k)] = {name:{"mean_per_query":float(np.mean(values)),"sum":sum(values)} for name,values in totals.items()}
    return output
