"""Query-group assignment independent of source row order and library RNG."""
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from .preprocess import query_key

ALGORITHM = "sha256-json-seed-key-largest-remainder-v1"
PARTITIONS = ("train", "validation", "test")
RATIOS = (Fraction(3, 5), Fraction(1, 5), Fraction(1, 5))

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def checksum(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()

def split_queries(queries, seed=42):
    if type(seed) is not int:
        raise ValueError("Integer split seed required")
    groups = defaultdict(list)
    for q in queries:
        if not q.normalized_query or q.normalized_query != query_key(q.query):
            raise ValueError("Invalid normalized query")
        groups[q.normalized_query].append(q.query_id)
    if len(groups) < 3:
        raise ValueError("At least three query groups required")
    keys = sorted(groups, key=lambda k: (checksum([seed, k]), k))
    targets = [len(keys) * ratio for ratio in RATIOS]
    counts = [int(t) for t in targets]
    order = sorted(range(3), key=lambda i: (-(targets[i] - counts[i]), i))
    for i in order[:len(keys) - sum(counts)]:
        counts[i] += 1
    result, start = {}, 0
    for name, count in zip(PARTITIONS, counts):
        result[name] = sorted(qid for key in keys[start:start + count] for qid in groups[key])
        start += count
    validate_splits(queries, result)
    return result

def validate_splits(queries, splits):
    if set(splits) != set(PARTITIONS) or any(not splits[p] for p in PARTITIONS):
        raise ValueError("Three nonempty partitions required")
    flat = [qid for p in PARTITIONS for qid in splits[p]]
    expected = [q.query_id for q in queries]
    if any(type(qid) is not int for qid in flat) or len(flat) != len(set(flat)) or len(expected) != len(set(expected)) or set(flat) != set(expected):
        raise ValueError("Split overlap or incomplete coverage")
    owners = {qid: p for p in PARTITIONS for qid in splits[p]}
    groups = {}
    for q in queries:
        owner = owners[q.query_id]
        if q.normalized_query in groups and groups[q.normalized_query] != owner:
            raise ValueError("Normalized query group leakage")
        groups[q.normalized_query] = owner
    for partition, ratio in zip(PARTITIONS, RATIOS):
        actual = sum(owner == partition for owner in groups.values())
        if abs(actual - len(groups) * ratio) >= 1:
            raise ValueError("Split group proportions outside rounding tolerance")
