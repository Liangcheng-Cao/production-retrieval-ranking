"""Build once; subsequent runs verify byte identity without replacing frozen files."""
import json
import platform
import tempfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from .schema import (SCHEMA_VERSION, NORMALIZATION_VERSION, CONFLICT_POLICY, RELEVANCE,
                     POSITIVE_LABELS, NDCG_GAIN, Product, Query, Judgment, Conflict)
from .preprocess import read_raw, canonicalize
from .splits import split_queries, checksum, json_bytes, ALGORITHM
from .io import file_hash, write_rows

def build(raw, output, provenance, *, seed=42, created_at=None):
    raw, output = Path(raw), Path(output)
    expected = provenance["files"]
    for name in ("product.csv", "query.csv", "label.csv"):
        if file_hash(raw / name) != expected[name]["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {name}")
    data, audit = canonicalize(*(read_raw(raw / name) for name in ("product.csv", "query.csv", "label.csv")))
    splits = split_queries(data.queries, seed)
    if splits != split_queries(tuple(reversed(data.queries)), seed):
        raise ValueError("Split row-order instability")
    existing = output / "data_manifest.json"
    old = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else None
    created_at = created_at or (old["creation"]["created_at_utc"] if old else datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wands-build-", dir=output.parent) as temp:
        stage = Path(temp)
        tables = {"products.jsonl": data.products, "queries.jsonl": data.queries,
                  "judgment_conflicts.jsonl": data.conflicts}
        for partition, ids in splits.items():
            members = set(ids)
            tables[f"judgments.{partition}.jsonl"] = tuple(j for j in data.judgments if j.query_id in members)
        for name, rows in tables.items():
            write_rows(stage / name, rows)
        (stage / "query_splits.json").write_bytes(json_bytes(splits))
        (stage / "data_audit.json").write_bytes(json_bytes(audit))
        artifacts = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size,
                             "rows": len(tables[p.name]) if p.name in tables else None}
                     for p in sorted(stage.iterdir())}
        source_hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {"schema_version": SCHEMA_VERSION, "upstream_revision": provenance["revision"],
                    "raw_sha256": {n: expected[n]["sha256"] for n in sorted(expected)},
                    "split_algorithm": ALGORITHM, "split_seed": seed,
                    "split_ratios": {"train": "3/5", "validation": "1/5", "test": "1/5"},
                    "normalization_version": NORMALIZATION_VERSION, "conflict_policy_version": CONFLICT_POLICY,
                    "text_missing_policy": "null-or-blank-to-empty; preserve nonblank including sentinel-like text",
                    "numeric_policy": "blank-to-null; reject malformed, nonfinite, negative and fractional counts",
                    "relevance_mapping": RELEVANCE, "positive_labels": POSITIVE_LABELS,
                    "ndcg_gain": NDCG_GAIN, "unjudged_policy": "unknown; not supervised negatives",
                    "zero_positive_policy": "Retain queries in splits; exclude from Recall/NDCG macro mean and report excluded counts (undefined denominator/IDCG)",
                    "queries_without_positive": audit["queries_without_positive"],
                    "query_count": len(data.queries), "query_counts": {p: len(ids) for p, ids in splits.items()},
                    "query_ids": splits, "partition_checksums": {p: checksum(ids) for p, ids in splits.items()},
                    "artifacts": artifacts, "source_sha256": source_hashes,
                    "schemas": {cls.__name__: {f.name: str(f.type) for f in fields(cls)} for cls in (Product, Query, Judgment, Conflict)},
                    "creation": {"created_at_utc": created_at, "builder": "product_search.data.pipeline.build", "python": platform.python_version()},
                    "freeze": {"status": "frozen-query-split-v1", "phase1_global_structural_label_access": True,
                               "restriction": "No final-test labels for model selection, tuning or engineering decisions after Phase 1"}}
        (stage / "data_manifest.json").write_bytes(json_bytes(manifest))
        # Preflight every output before publishing anything. No force-overwrite option.
        for p in stage.iterdir():
            target = output / p.name
            if target.exists() and file_hash(target) != file_hash(p):
                raise ValueError(f"Frozen output differs; refusing overwrite: {p.name}")
        if old and any(not (output / p.name).exists() for p in stage.iterdir()):
            raise ValueError("Frozen output incomplete; refusing implicit repair")
        output.mkdir(parents=True, exist_ok=True)
        for p in sorted(stage.iterdir(), key=lambda p: p.name == "data_manifest.json"):
            target = output / p.name
            if not target.exists():
                p.replace(target)
    return manifest, audit
