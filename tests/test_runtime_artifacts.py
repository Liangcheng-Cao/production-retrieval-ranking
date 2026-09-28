"""Tiny local artifacts exercise the real loader without GPU models/downloads."""
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from product_search.data.io import file_hash
from product_search.data.schema import Product
from product_search.retrieval.lexical import BM25Retriever
from product_search.runtime_artifacts import ArtifactLoader
from product_search.runtime_config import RuntimeConfig
from product_search.runtime_contracts import ArtifactError
from product_search.search_engine import SearchEngine


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def package(tmp_path):
    products = [Product(i, name, '', '', '', '', None, None, None)
                for i, name in enumerate(('wood desk', 'office chair', 'desk lamp'))]
    product_path = tmp_path/'products.jsonl'
    product_path.write_text('\n'.join(json.dumps(asdict(p)) for p in products)+'\n')
    p1 = {'artifacts': {'products.jsonl': {'sha256': file_hash(product_path), 'rows': 3}}}
    write(tmp_path/'dataset.json', p1)
    data_sha = file_hash(tmp_path/'dataset.json')
    cfg = {'data_manifest_sha256': data_sha, 'kind': 'bm25', 'representation': 'A'}
    index = BM25Retriever.build(products)
    index_manifest = index.save(tmp_path/'indexes/bm25_fixture', cfg)
    s2 = {'data_manifest_sha256': data_sha, 'bm25': {'name': 'bm25_fixture', 'config': cfg},
          'dense': {'name': 'dense_fixture', 'config': {}},
          'hybrid': {'config': {'candidate_depth': 100, 'method': 'rrf', 'constant': 60,
                               'weights': [1, 1], 'bm25': 'bm25_fixture', 'dense': 'dense_fixture'}}}
    p2 = {'artifacts': {'bm25_fixture': {'manifest': index_manifest,
        'manifest_sha256': file_hash(tmp_path/'indexes/bm25_fixture/manifest.json')}}}
    write(tmp_path/'retrieval.json', p2)
    plan = {'dtype': 'float32', 'model': 'synthetic', 'revision': 'fixed', 'max_length': 256}
    s3 = {'phase1_manifest_sha256': data_sha, 'phase2_manifest_sha256': file_hash(tmp_path/'retrieval.json'),
          'selected': {'candidate_depth': 20, 'representation': 'A', 'batch_size': 32}}
    write(tmp_path/'selection2.json', s2)
    write(tmp_path/'selection3.json', s3)
    write(tmp_path/'plan3.json', plan)
    p3 = {'phase1_manifest_sha256': data_sha, 'phase2_retrieval_manifest_sha256': file_hash(tmp_path/'retrieval.json'),
          'selected': s3, 'plan': plan, 'selection_sha256': file_hash(tmp_path/'selection3.json'),
          'plan_sha256': file_hash(tmp_path/'plan3.json')}
    write(tmp_path/'reranking.json', p3)
    roles = ('dataset', 'retrieval', 'reranking', 'selection2', 'selection3', 'plan3')
    lock = {'format_version': 1, 'anchors': {role: {'path': role+'.json', 'sha256': file_hash(tmp_path/(role+'.json'))} for role in roles},
            'index_root': 'indexes', 'products_path': 'products.jsonl', 'models': {},
            'software': {name: importlib.metadata.version(name) for name in ('numpy', 'bm25s')}}
    write(tmp_path/'manifest.json', lock)
    config = RuntimeConfig(tmp_path, 'manifest.json', file_hash(tmp_path/'manifest.json'),
                           ('bm25',), 'bm25', device='cpu')
    return config, lock


def test_real_bm25_startup_and_offline_parity(package):
    config, _ = package
    engine = SearchEngine.from_config(config)
    assert engine.readiness().ready
    assert engine.readiness().loaded_components == ('products', 'bm25')
    offline = BM25Retriever.load(config.root/'indexes/bm25_fixture',
                                 json.loads((config.root/'selection2.json').read_text())['bm25']['config'])
    for query in ('desk', 'office chair', 'nonexistenttoken'):
        result = engine.search(query, 3)
        reference = offline.search(query, 3)
        assert [h.product_id for h in result.results] == [h.product_id for h in reference]
        assert [h.final_score for h in result.results] == pytest.approx([h.score for h in reference])
    assert not any('model' in p for p in engine.version['validated_artifact_sha256'])


@pytest.mark.parametrize('target', ['manifest.json', 'dataset.json', 'products.jsonl',
                                  'indexes/bm25_fixture/product_ids.npy'])
def test_required_missing(package, target):
    config, _ = package
    (config.root/target).unlink()  # Synthetic files inside pytest temporary directory only.
    with pytest.raises(ArtifactError, match='Missing required artifact'):
        ArtifactLoader(config).load()


@pytest.mark.parametrize('target', ['manifest.json', 'selection2.json', 'products.jsonl',
                                  'indexes/bm25_fixture/product_ids.npy'])
def test_checksum_rejection(package, target):
    config, _ = package
    with (config.root/target).open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ArtifactError, match='checksum mismatch'):
        ArtifactLoader(config).load()


def test_manifest_format(package):
    config, lock = package
    lock['format_version'] = 99
    write(config.root/'manifest.json', lock)
    config = replace(config, manifest_sha256=file_hash(config.root/'manifest.json'))
    with pytest.raises(ArtifactError, match='Incompatible runtime manifest'):
        ArtifactLoader(config).load()


def test_wrong_dataset_even_with_resealed_checksums(package):
    config, lock = package
    path = config.root/'selection2.json'
    value = json.loads(path.read_text())
    value['data_manifest_sha256'] = 'f'*64
    write(path, value)
    lock['anchors']['selection2']['sha256'] = file_hash(path)
    write(config.root/'manifest.json', lock)
    config = replace(config, manifest_sha256=file_hash(config.root/'manifest.json'))
    with pytest.raises(ArtifactError, match='provenance mismatch'):
        ArtifactLoader(config).load()


def test_wrong_selected_config_even_with_resealed_checksums(package):
    config, lock = package
    path = config.root/'selection2.json'
    value = json.loads(path.read_text())
    value['bm25']['config']['representation'] = 'B'
    write(path, value)
    lock['anchors']['selection2']['sha256'] = file_hash(path)
    write(config.root/'manifest.json', lock)
    config = replace(config, manifest_sha256=file_hash(config.root/'manifest.json'))
    with pytest.raises(ArtifactError, match='Incompatible bm25 configuration'):
        ArtifactLoader(config).load()
