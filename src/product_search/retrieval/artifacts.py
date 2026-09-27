"""Verified local artifacts; no pickle and no model execution during loading."""
import json
from pathlib import Path
from product_search.data.io import file_hash
from product_search.data.splits import json_bytes

def seal(directory, config):
    directory = Path(directory)
    manifest = {"format_version": 1, "config": config, "files": {
        p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "manifest.json"}}
    (directory / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest

def verify(directory, expected):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1 or manifest["config"] != expected:
        raise ValueError("Artifact/config compatibility mismatch")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != digest:
            raise ValueError("Artifact checksum mismatch")
    return manifest
