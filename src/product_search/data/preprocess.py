"""Strict parsing and conservative canonicalization of WANDS."""
import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from .schema import Product, Query, Judgment, Conflict, Dataset, RAW_COLUMNS, RELEVANCE

SENTINELS = {"nan", "null", "none", "n/a", "na", "<na>"}

def read_raw(path):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS[path.name]:
            raise ValueError(f"Unexpected schema: {path.name}")
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f"Malformed row: {path.name}")
    return rows

def normalize_text(value):
    # Sentinel-looking text is preserved, not guessed to be missing.
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        raise ValueError("Text must be string or null")
    return value if value.strip() else ""

def query_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError("ID must be a canonical nonnegative integer")
    return int(value)

def number(value, integer=False):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, str):
        raise ValueError("Raw numeric value must be a string or null")
    if not value.strip():
        return None
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("Malformed numeric value") from exc
    if not parsed.is_finite() or parsed < 0 or integer and parsed != parsed.to_integral_value():
        raise ValueError("Invalid nonnegative numeric value")
    result = int(parsed) if integer else float(parsed)
    if not integer and not math.isfinite(result):
        raise ValueError("Numeric overflow")
    return result

def missing_audit(rows):
    return {c: dict(sorted(Counter(
        "empty" if r[c] == "" else "whitespace" if not r[c].strip()
        else "sentinel_like" if r[c].strip().casefold() in SENTINELS else "present"
        for r in rows).items())) for c in rows[0]} if rows else {}

def canonicalize(products, queries, annotations):
    ps = tuple(sorted((Product(identifier(r["product_id"]),
        *(normalize_text(r[c]) for c in RAW_COLUMNS["product.csv"][1:6]),
        number(r["rating_count"], True), number(r["average_rating"]), number(r["review_count"], True))
        for r in products), key=lambda p: p.product_id))
    qs = tuple(sorted((Query(identifier(r["query_id"]), normalize_text(r["query"]),
        normalize_text(r["query_class"]), query_key(normalize_text(r["query"]))) for r in queries), key=lambda q: q.query_id))
    groups, ids = defaultdict(list), set()
    pids, qids = {p.product_id for p in ps}, {q.query_id for q in qs}
    for r in annotations:
        aid, qid, pid = identifier(r["id"]), identifier(r["query_id"]), identifier(r["product_id"])
        if aid in ids or qid not in qids or pid not in pids or r["label"] not in RELEVANCE:
            raise ValueError("Invalid annotation ID, foreign key or label")
        ids.add(aid)
        groups[qid, pid].append((aid, r["label"]))
    judgments, conflicts, types = [], [], Counter()
    same_groups = exact_removed = conflict_rows = 0
    for (qid, pid), rows in sorted(groups.items()):
        labels = tuple(sorted({label for _, label in rows}, key=RELEVANCE.get))
        exact_removed += len(rows) - len(labels)
        if len(labels) > 1:
            conflicts.append(Conflict(qid, pid, labels, tuple(sorted(aid for aid, _ in rows))))
            types[" / ".join(labels)] += 1
            conflict_rows += len(rows)
        else:
            same_groups += len(rows) > 1
            judgments.append(Judgment(qid, pid, labels[0], RELEVANCE[labels[0]]))
    data = Dataset(ps, qs, tuple(judgments), tuple(conflicts))
    validate(data)
    audit = {"total_annotation_rows": len(annotations), "unique_query_product_pairs": len(groups),
             "excess_duplicate_rows": len(annotations) - len(groups),
             "same_label_duplicate_groups": same_groups, "conflicting_label_groups": len(conflicts),
             "rows_removed_by_exact_deduplication": exact_removed,
             "conflict_raw_rows": conflict_rows,
             "conflict_rows_after_exact_deduplication": sum(len(c.labels) for c in conflicts),
             "canonical_judgments": len(judgments), "conflict_types": dict(sorted(types.items())),
             "queries_without_positive": sorted(qids - {j.query_id for j in judgments if j.relevance >= 1}),
             "missing_values": {"products": missing_audit(products), "queries": missing_audit(queries)},
             "numeric_malformed": 0, "numeric_nonmissing_parsed": {
                 c: sum(getattr(p, c) is not None for p in ps) for c in ("rating_count", "average_rating", "review_count")}}
    return data, audit

def validate(data):
    pids, qids = [p.product_id for p in data.products], [q.query_id for q in data.queries]
    for ids in (pids, qids):
        if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Nonempty unique integer primary IDs required")
    for p in data.products:
        if any(not isinstance(getattr(p, c), str) for c in ("product_name", "product_class", "category_hierarchy", "product_description", "product_features")):
            raise ValueError("Invalid product text schema")
        if not p.product_name.strip():
            raise ValueError("Empty product name")
        for c in ("rating_count", "review_count", "average_rating"):
            value = getattr(p, c)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0 or c != "average_rating" and type(value) is not int):
                raise ValueError("Invalid canonical numeric value")
    for q in data.queries:
        if not all(isinstance(v, str) for v in (q.query, q.query_class, q.normalized_query)) or not q.query.strip() or q.normalized_query != query_key(q.query):
            raise ValueError("Invalid query identity")
    pids, qids = set(pids), set(qids)
    conflict_pairs = set()
    for c in data.conflicts:
        pair = c.query_id, c.product_id
        if type(c.query_id) is not int or type(c.product_id) is not int or pair in conflict_pairs or c.query_id not in qids or c.product_id not in pids or len(set(c.labels)) < 2 or not set(c.labels) <= RELEVANCE.keys():
            raise ValueError("Invalid conflict")
        if len(c.annotation_ids) < len(c.labels) or len(set(c.annotation_ids)) != len(c.annotation_ids) or any(type(i) is not int or i < 0 for i in c.annotation_ids):
            raise ValueError("Invalid conflict annotation provenance")
        conflict_pairs.add(pair)
    pairs = set()
    for j in data.judgments:
        pair = j.query_id, j.product_id
        if type(j.query_id) is not int or type(j.product_id) is not int or j.query_id not in qids or j.product_id not in pids:
            raise ValueError("Unresolved foreign key")
        if pair in pairs or pair in conflict_pairs:
            raise ValueError("Duplicate or conflicting judgment")
        if j.label not in RELEVANCE or type(j.relevance) is not int or j.relevance != RELEVANCE[j.label]:
            raise ValueError("Invalid label/relevance mapping")
        pairs.add(pair)
    judged = {j.query_id for j in data.judgments}
    if qids != judged:
        raise ValueError("Every query must retain at least one judgment")
