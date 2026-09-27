"""Phase 2 scoped access, using frozen canonical contracts without scanning test labels.

The shared queries file is byte-scanned for IDs; only allowed rows are decoded.
Global conflicts and final-test judgment files are never opened.
"""
import json
import re
import sys
from pathlib import Path
from product_search.data.schema import Dataset, Product, Query, Judgment
from product_search.data.preprocess import validate
from product_search.data.io import file_hash

class BoundaryGuard:
    """Process-wide defense for experiment scripts; synthetic pytest data is separate."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.opened = set()
        self.blocked = []

    def check(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if not path.is_relative_to(self.root):
            return
        relative = path.relative_to(self.root).as_posix()
        forbidden = relative.startswith("data/raw/") or relative.startswith("reports/tmp/phase1-rebuild/") or path.name in ("judgments.test.jsonl", "judgment_conflicts.jsonl")
        if forbidden:
            self.blocked.append(relative)
            raise PermissionError(f"Phase 2 test boundary: {relative}")
        if relative.startswith("data/processed/"):
            self.opened.add(relative)

    def install(self):
        sys.addaudithook(self.check)
        return self

def load_development(directory, partition):
    if partition not in ("train", "validation"):
        raise ValueError("Only train/validation are permitted in Phase 2")
    directory = Path(directory)
    manifest = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
    ids = set(manifest["query_ids"][partition])

    def path(name):
        target = directory / name
        if file_hash(target) != manifest["artifacts"][name]["sha256"]:
            raise ValueError("Frozen canonical artifact checksum mismatch")
        return target

    queries = []
    with path("queries.jsonl").open("rb") as handle:
        for line in handle:
            match = re.search(rb'"query_id":([0-9]+)(?:,|})', line)
            if not match:
                raise ValueError("Missing canonical query ID")
            if int(match[1]) in ids:
                queries.append(Query(**json.loads(line)))
    with path("products.jsonl").open(encoding="utf-8") as handle:
        products = tuple(Product(**json.loads(line)) for line in handle)
    with path(f"judgments.{partition}.jsonl").open(encoding="utf-8") as handle:
        judgments = tuple(Judgment(**json.loads(line)) for line in handle)
    if {q.query_id for q in queries} != ids:
        raise ValueError("Partition query membership mismatch")
    # Conflict exclusion is certified by the unchanged frozen judgment checksum.
    data = Dataset(products, tuple(queries), judgments, ())
    validate(data)
    return data
