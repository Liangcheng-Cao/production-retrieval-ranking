"""Phase 0 structural audit only; no preprocessing, split creation or evaluation."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"


def read_table(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed tab-separated table: {path}")
    return columns, rows


def audit():
    tables, report = {}, {"source": "https://github.com/wayfair/WANDS", "revision": REVISION,
                          "delimiter": "tab", "encoding": "UTF-8", "files": {}}
    for name, key in (("product.csv", "product_id"), ("query.csv", "query_id"), ("label.csv", "id")):
        path = ROOT / "data/raw" / name
        columns, rows = read_table(path)
        tables[name] = rows
        report["files"][name] = {
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "url": f"https://raw.githubusercontent.com/wayfair/WANDS/{REVISION}/dataset/{name}",
            "rows": len(rows), "columns": columns,
            "blank_counts": {c: sum(not r[c].strip() for r in rows) for c in columns},
            "duplicate_id_rows": len(rows) - len({r[key] for r in rows}),
            "non_integer_id_rows": sum(not r[key].isdigit() for r in rows),
        }
    products, queries, labels = (tables[n] for n in ("product.csv", "query.csv", "label.csv"))
    pids, qids = {r["product_id"] for r in products}, {r["query_id"] for r in queries}
    pairs = [(r["query_id"], r["product_id"]) for r in labels]
    pair_labels = {}
    for row in labels:
        pair_labels.setdefault((row["query_id"], row["product_id"]), set()).add(row["label"])
    normalize = lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split())
    report["integrity"] = {
        "label_counts": dict(Counter(r["label"] for r in labels)),
        "duplicate_query_product_rows": len(pairs) - len(set(pairs)),
        "conflicting_query_product_pairs": sum(len(values) > 1 for values in pair_labels.values()),
        "unknown_query_references": sum(r["query_id"] not in qids for r in labels),
        "unknown_product_references": sum(r["product_id"] not in pids for r in labels),
        "queries_without_judgments": len(qids - {r["query_id"] for r in labels}),
        "products_without_judgments": len(pids - {r["product_id"] for r in labels}),
        "duplicate_normalized_query_rows": len(queries) - len({normalize(r["query"]) for r in queries}),
        "duplicate_product_name_rows": len(products) - len({r["product_name"] for r in products}),
        "duplicate_product_text_rows": len(products) - len({(r["product_name"], r["product_description"]) for r in products}),
    }
    return report


if __name__ == "__main__":
    result = json.dumps(audit(), indent=2) + "\n"
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(result, encoding="utf-8", newline="\n")
    else:
        print(result)
