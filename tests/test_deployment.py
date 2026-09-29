import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest

from product_search.data.io import file_hash
from product_search.runtime_config import RuntimeConfig
from product_search.deployment import package_runtime, verify_bundle, bundle_inventory
import product_search.deployment as deployment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path/'source'
    manifest = root/'artifacts/phase4/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}', encoding='utf-8')
    product = root/'data/processed/products.jsonl'
    product.parent.mkdir(parents=True)
    product.write_text('synthetic product payload\n', encoding='utf-8')
    checked = {'data/processed/products.jsonl': file_hash(product)}
    class FakeLoader:
        def __init__(self, config):
            self.config = config
        def validate(self):
            return {'anchors': {}}, {}, checked, {}
    monkeypatch.setattr(deployment, 'ArtifactLoader', FakeLoader)
    return RuntimeConfig(root, 'artifacts/phase4/manifest.json', file_hash(manifest)), checked


def test_bundle_relocatable_deterministic_no_source_dependencies(sealed, tmp_path):
    config, _ = sealed
    a, b = tmp_path/'one', tmp_path/'two'
    first, second = package_runtime(config, a), package_runtime(config, b)
    assert first == second
    assert file_hash(a/'bundle.json') == file_hash(b/'bundle.json')
    assert RuntimeConfig.load(a/'runtime.json').root == a
    assert verify_bundle(a) == first
    assert not any(p.is_symlink() for p in a.rglob('*'))
    with pytest.raises(ValueError, match='overwrite'):
        package_runtime(config, a)


@pytest.mark.parametrize('name', ['data/raw/label.csv', 'data/processed/judgments.test.jsonl',
    'data/processed/judgments.train.jsonl', 'data/processed/queries.jsonl', 'reports/private.json', '../escape'])
def test_bundle_rejects_non_inference_and_escape_paths(sealed, name):
    config, checked = sealed
    checked[name] = 'a'*64
    with pytest.raises(ValueError):
        bundle_inventory(config)


def test_missing_corrupt_and_extra_bundle_files(sealed, tmp_path):
    config, _ = sealed
    for case in ('missing', 'corrupt', 'extra', 'config'):
        target = tmp_path/case
        package_runtime(config, target)
        if case == 'missing':
            (target/'data/processed/products.jsonl').unlink()
        elif case == 'corrupt':
            (target/'data/processed/products.jsonl').write_text('corrupt', encoding='utf-8')
        elif case == 'extra':
            (target/'unexpected.txt').write_text('extra', encoding='utf-8')
        else:
            (target/'runtime.json').write_text('{}', encoding='utf-8')
        with pytest.raises((ValueError, KeyError, TypeError)):
            verify_bundle(target)


def test_recomputed_bundle_manifest_does_not_bless_modified_frozen_payload(sealed, tmp_path):
    config, _ = sealed
    target = tmp_path/'bundle'
    package_runtime(config, target)
    data = target/'data/processed/products.jsonl'
    data.write_text('different payload', encoding='utf-8')
    manifest = json.loads((target/'bundle.json').read_text())
    manifest['files']['data/processed/products.jsonl'] = {'sha256': file_hash(data), 'bytes': data.stat().st_size}
    (target/'bundle.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen inference closure'):
        verify_bundle(target)


def test_lock_rejects_untrusted_unhashed_or_duplicate_wheels():
    lock_rows = runpy.run_path(str(ROOT/'scripts/lock_deployment.py'))['lock_rows']
    row = {'metadata': {'name': 'test', 'version': '1'}, 'download_info': {
        'url': 'https://files.pythonhosted.org/test.whl', 'archive_info': {'hashes': {'sha256': 'a'*64}}}}
    assert len(lock_rows({'install': [row]})) == 1
    with pytest.raises(ValueError):
        lock_rows({'install': [row, row]})
    row['download_info']['url'] = 'https://unreviewed.invalid/test.whl'
    with pytest.raises(ValueError):
        lock_rows({'install': [row]})


@pytest.mark.parametrize('state,ready,expected', [('READY', True, 0), ('DEGRADED', False, 1), ('LOADING', False, 1)])
def test_healthcheck_requires_ready_not_only_http200(monkeypatch, state, ready, expected):
    from io import StringIO
    class Response(StringIO):
        status = 200
    def urlopen(*a, **kw):
        return Response(json.dumps({'state': state, 'ready': ready}))
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    check = runpy.run_path(str(ROOT/'deployment/healthcheck.py'))['main']
    assert check() == expected
