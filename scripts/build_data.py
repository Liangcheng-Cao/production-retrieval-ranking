"""Phase 1 structural build/verification. Never run as a model-tuning helper."""
import argparse
import json
from pathlib import Path
from product_search.config import load_config
from product_search.data.pipeline import build

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--created-at", help="Reuse the original UTC creation value for clean rebuild checks")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/default.toml")
    provenance = json.loads((ROOT / "reports/dataset_audit.json").read_text(encoding="utf-8"))
    manifest, audit = build(config["paths"]["raw"], args.output or config["paths"]["processed"],
                            provenance, seed=config["random_seed"], created_at=args.created_at)
    print(json.dumps({"queries": manifest["query_counts"], "canonical_judgments": audit["canonical_judgments"],
                      "conflict_types": audit["conflict_types"], "status": "validated; built or byte-identical"}, indent=2))

if __name__ == "__main__":
    main()
