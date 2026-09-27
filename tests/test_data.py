import csv
import json
from dataclasses import replace
import pytest
from product_search.data.schema import RAW_COLUMNS, RELEVANCE, POSITIVE_LABELS, NDCG_GAIN
from product_search.data.preprocess import (canonicalize, normalize_text, number, query_key, read_raw, validate)
from product_search.data.splits import split_queries, validate_splits
from product_search.data.io import file_hash, load_dataset
from product_search.data.pipeline import build

@pytest.fixture
def raw_rows():
    products = [dict(zip(RAW_COLUMNS["product.csv"], [str(i), "Synthetic product", "", "", "", "", "1.0", "4.5", ""])) for i in range(3)]
    queries = [{"query_id": str(i), "query": f"Query {i}", "query_class": ""} for i in range(10)]
    labels = [{"id": str(i), "query_id": str(i), "product_id": "0", "label": "Exact"} for i in range(10)]
    return products, queries, labels

def write_fixture(root, rows):
    root.mkdir()
    for name, values in zip(RAW_COLUMNS, rows):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS[name], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    return {"revision": "synthetic-v1", "files": {name: {"sha256": file_hash(root / name)} for name in RAW_COLUMNS}}

def test_raw_roundtrip_and_schema(tmp_path, raw_rows):
    raw_rows[0][0]["product_description"] = "Quoted\ttext\nnext line"
    write_fixture(tmp_path / "raw", raw_rows)
    assert read_raw(tmp_path / "raw/product.csv") == raw_rows[0]
    (tmp_path / "raw/query.csv").write_text("wrong\n1\n")
    with pytest.raises(ValueError, match="schema"):
        read_raw(tmp_path / "raw/query.csv")

def test_missing_and_numeric_policy():
    for v in (None, "", " \t", float("nan")):
        assert normalize_text(v) == ""
        assert number(v) is None
    assert normalize_text("NaN") == "NaN"
    assert normalize_text("  Original  ") == "  Original  "
    assert number("3.0", True) == 3
    for v in ("NaN", "null", "oops", "inf", "-1", "1.5"):
        with pytest.raises(ValueError):
            number(v, True)

def test_duplicate_conflict_audit_and_exclusion(raw_rows):
    p, q, a = raw_rows
    a.extend([dict(a[0], id="10"), {"id": "11", "query_id": "0", "product_id": "1", "label": "Partial"},
              {"id": "12", "query_id": "0", "product_id": "1", "label": "Exact"},
              {"id": "13", "query_id": "0", "product_id": "1", "label": "Exact"}])
    d, audit = canonicalize(p, q, a)
    assert len(d.judgments) == 10
    assert audit["same_label_duplicate_groups"] == 1
    assert audit["rows_removed_by_exact_deduplication"] == 2
    assert audit["conflicting_label_groups"] == 1
    assert audit["conflict_types"] == {"Partial / Exact": 1}
    assert d.conflicts[0].annotation_ids == (11, 12, 13)
    leaked = replace(d.judgments[0], product_id=1)
    with pytest.raises(ValueError, match="conflicting"):
        validate(replace(d, judgments=d.judgments + (leaked,)))
    reversed_data, reversed_audit = canonicalize(p[::-1], q[::-1], a[::-1])
    assert (reversed_data, reversed_audit) == (d, audit)

def test_relevance_and_query_preservation(raw_rows):
    assert RELEVANCE == {"Irrelevant": 0, "Partial": 1, "Exact": 2}
    assert POSITIVE_LABELS == ("Partial", "Exact")
    assert NDCG_GAIN == "2**relevance - 1"
    raw_rows[1][0]["query"] = "  ＣＨＡＩＲ\twhite  "
    d, _ = canonicalize(*raw_rows)
    assert d.queries[0].query == "  ＣＨＡＩＲ\twhite  "
    assert d.queries[0].normalized_query == "chair white"
    assert len(d.products) == 3  # identical product text never merges identities

@pytest.mark.parametrize("field,value", [("product_id", "99"), ("query_id", "99"), ("label", "Unknown")])
def test_bad_annotations_fail(raw_rows, field, value):
    raw_rows[2][0][field] = value
    with pytest.raises(ValueError, match="foreign key or label"):
        canonicalize(*raw_rows)

@pytest.mark.parametrize("change", ["product_duplicate", "empty_query", "no_judgment", "duplicate_annotation"])
def test_integrity_failures(raw_rows, change):
    p, q, a = raw_rows
    if change == "product_duplicate": p.append(p[0])
    if change == "empty_query": q[0]["query"] = "  "
    if change == "no_judgment": a.pop(0)
    if change == "duplicate_annotation": a.append(a[0])
    with pytest.raises(ValueError): canonicalize(p, q, a)

def test_zero_positive_query_retained_and_reported(raw_rows):
    raw_rows[2][0]["label"] = "Irrelevant"
    data, audit = canonicalize(*raw_rows)
    assert len(data.queries) == 10
    assert audit["queries_without_positive"] == [0]
    assert 0 in sum(split_queries(data.queries).values(), [])

def test_split_reordering_and_groups(raw_rows):
    d, _ = canonicalize(*raw_rows)
    splits = split_queries(d.queries)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [6, 2, 2]
    assert splits == split_queries(d.queries[::-1])
    alias = replace(d.queries[0], query_id=10, query=" QUERY 0 ")
    queries = d.queries + (alias,)
    grouped = split_queries(queries)
    assert next(p for p, ids in grouped.items() if 0 in ids) == next(p for p, ids in grouped.items() if 10 in ids)
    owner = next(p for p, ids in grouped.items() if 10 in ids)
    other = next(p for p in grouped if p != owner)
    grouped[owner].remove(10)
    grouped[other].append(10)
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(queries, grouped)
    splits["test"].append(splits["train"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(d.queries, splits)

def test_manifest_rebuild_load_and_freeze(tmp_path, raw_rows):
    raw = tmp_path / "raw"
    provenance = write_fixture(raw, raw_rows)
    out = tmp_path / "processed"
    first, _ = build(raw, out, provenance, created_at="2026-09-27T00:00:00+00:00")
    before = {p.name: file_hash(p) for p in out.iterdir()}
    build(raw, out, provenance)
    assert before == {p.name: file_hash(p) for p in out.iterdir()}
    second = tmp_path / "second"
    build(raw, second, provenance, created_at=first["creation"]["created_at_utc"])
    assert before == {p.name: file_hash(p) for p in second.iterdir()}
    assert len(load_dataset(out).queries) == 6
    assert len(load_dataset(out, "validation").queries) == 2
    with pytest.raises(ValueError, match="frozen"):
        load_dataset(out, "test")
    assert len(load_dataset(out, "test", allow_test=True).queries) == 2
    with pytest.raises(ValueError, match="refusing overwrite"):
        build(raw, out, provenance, seed=1)
    assert before == {p.name: file_hash(p) for p in out.iterdir()}
    (out / "judgments.train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(out)
    (raw / "query.csv").write_text("bad\n")
    with pytest.raises(ValueError, match="Raw checksum"):
        build(raw, out, provenance)
