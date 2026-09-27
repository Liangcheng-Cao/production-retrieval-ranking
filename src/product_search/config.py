"""Load repository-relative TOML settings without relying on the working directory."""
from pathlib import Path
import tomllib


def load_config(config_path: str | Path) -> dict:
    """Resolve paths relative to the parent of the configs directory."""
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if type(config.get("random_seed")) is not int:
        raise ValueError("random_seed must be an integer")
    resolved = {}
    for name in ("raw", "processed", "artifacts", "reports"):
        path = (root / config["paths"][name]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path {name} must stay within project root")
        resolved[name] = path
    config["paths"] = resolved
    return config
