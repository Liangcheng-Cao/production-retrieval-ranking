"""Canonical JSONL serialization and verified, partition-scoped loading."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from .schema import Dataset, Product, Query, Judgment, Conflict, SCHEMA_VERSION
from .preprocess import validate
from .splits import json_bytes, checksum, validate_splits

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_rows(path, rows):
    with Path(path).open("wb") as handle:
        for row in rows:
            handle.write(json_bytes(asdict(row)))

def load_dataset(directory, partition="train", *, allow_test=False):
    """Test labels require explicit access. This is a guardrail, not access control."""
    if partition not in ("train", "validation", "test"):
        raise ValueError("Explicit train/validation/test partition required")
    if partition == "test" and not allow_test:
        raise ValueError("Final-test labels are frozen; explicit allow_test required")
    directory = Path(directory)
    manifest = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported canonical schema")

    def verified(name):
        path = directory / name
        if file_hash(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        return path

    splits = json.loads(verified("query_splits.json").read_text(encoding="utf-8"))
    if splits != manifest["query_ids"] or any(checksum(ids) != manifest["partition_checksums"][p] for p, ids in splits.items()):
        raise ValueError("Split checksum mismatch")

    def rows(name, cls):
        result = []
        with verified(name).open(encoding="utf-8") as handle:
            for line in handle:
                values = json.loads(line)
                if cls is Conflict:
                    values["labels"] = tuple(values["labels"])
                    values["annotation_ids"] = tuple(values["annotation_ids"])
                result.append(cls(**values))
        return tuple(result)

    queries = rows("queries.jsonl", Query)
    validate_splits(queries, splits)
    ids = set(splits[partition])
    data = Dataset(rows("products.jsonl", Product), tuple(q for q in queries if q.query_id in ids),
                   rows(f"judgments.{partition}.jsonl", Judgment),
                   tuple(c for c in rows("judgment_conflicts.jsonl", Conflict) if c.query_id in ids))
    validate(data)
    return data
