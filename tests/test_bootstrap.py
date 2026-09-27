import importlib.util
from pathlib import Path
import pytest
from product_search.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_wands", ROOT / "scripts/audit_wands.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_config_paths_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs/default.toml")
    assert config["paths"]["raw"] == ROOT / "data/raw"
    assert all(p.is_dir() for p in config["paths"].values())
    assert config["random_seed"] == 42


def test_config_rejects_escape(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    config = directory / "bad.toml"
    config.write_text('random_seed = 42\n[paths]\nraw = "../outside"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="within project root"):
        load_config(config)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_tab_separated_csv_handles_quoted_newlines(tmp_path, newline):
    path = tmp_path / "product.csv"
    text = f'product_id\tproduct_name{newline}0\t"Synthetic{newline}product"{newline}'
    path.write_bytes(text.encode("utf-8"))
    columns, rows = audit_module.read_table(path)
    assert columns == ["product_id", "product_name"]
    assert rows == [{"product_id": "0", "product_name": f"Synthetic{newline}product"}]


@pytest.mark.parametrize("body", ['id\tlabel\n1\n', 'id\tlabel\n1\tExact\textra\n'])
def test_malformed_field_count_fails(tmp_path, body):
    path = tmp_path / "label.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        audit_module.read_table(path)
