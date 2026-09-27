"""Download the audited revision and verify recorded SHA-256 before accepting files."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "reports/dataset_audit.json").read_text(encoding="utf-8"))
    directory = ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    for name, item in manifest["files"].items():
        target = directory / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Existing {name} has unexpected hash; refusing overwrite")
            print(f"Verified existing {name}")
            continue
        temporary = target.with_suffix(".csv.part")
        try:
            with urlopen(item["url"], timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size != item["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Download integrity check failed: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    main()
