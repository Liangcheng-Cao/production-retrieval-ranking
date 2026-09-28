"""Explicit packaging step: seal existing artifacts, never rebuild or evaluate data."""
import importlib.metadata
import json
from pathlib import Path
from product_search.data.io import file_hash
from product_search.development_data import BoundaryGuard

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')


def main():
    BoundaryGuard(ROOT).install()
    target = ROOT/'artifacts/phase4/manifest.json'
    config = ROOT/'configs/runtime.json'
    if target.exists() or config.exists():
        raise ValueError('Runtime package already sealed; use explicit reviewed versioning for replacement')
    paths = {'dataset': 'data/processed/data_manifest.json', 'retrieval': 'artifacts/phase2/manifest.json',
             'reranking': 'artifacts/phase3/manifest.json', 'selection2': 'configs/phase2_selected.json',
             'selection3': 'configs/phase3_selected.json', 'plan3': 'configs/phase3_plan.json'}
    anchors = {role: {'path': p, 'sha256': file_hash(ROOT/p)} for role, p in paths.items()}
    s2 = json.loads((ROOT/paths['selection2']).read_text())
    plan = json.loads((ROOT/paths['plan3']).read_text())
    models = {}
    for role, selected in [('dense', s2['dense']['config']), ('ce', plan)]:
        snapshot = 'artifacts/model_cache/hub/models--'+selected['model'].replace('/', '--')+'/snapshots/'+selected['revision']
        files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted((ROOT/snapshot).rglob('*')) if p.is_file()}
        if not files:
            raise ValueError('Preprovisioned pinned model snapshot missing')
        models[role] = {'model': selected['model'], 'revision': selected['revision'],
                        'snapshot': snapshot, 'files': files}
    lock = {'format_version': 1, 'anchors': anchors, 'models': models,
            'products_path': 'data/processed/products.jsonl', 'index_root': 'artifacts/phase2',
            'software': {n: importlib.metadata.version(n) for n in
                         ('numpy', 'bm25s', 'torch', 'transformers', 'sentence-transformers', 'tokenizers', 'safetensors')}}
    dump(target, lock)
    dump(config, {'root': '..', 'manifest': target.relative_to(ROOT).as_posix(),
                  'manifest_sha256': file_hash(target), 'enabled_pipelines': ['bm25', 'hybrid', 'hybrid_rerank'],
                  'default_pipeline': 'hybrid', 'default_top_k': 10, 'device': 'cuda', 'reranker_fallback': True})
    print('Sealed local runtime package; no labels opened or artifacts rebuilt')


if __name__ == '__main__':
    main()
