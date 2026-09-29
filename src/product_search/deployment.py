"""Deployment boundary: portable sealed artifacts and unchanged single-worker service."""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil

from .data.io import file_hash
from .runtime_config import RuntimeConfig
from .runtime_artifacts import ArtifactLoader


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def bundle_inventory(config):
    """Explicit inference closure only; query/judgment/raw artifacts are never included."""
    lock, _, checked, _ = ArtifactLoader(config).validate()
    files = dict(checked)
    files[config.manifest] = config.manifest_sha256
    files.update({v['path']: v['sha256'] for v in lock['anchors'].values()})
    for name in files:
        if (name.startswith('data/raw/') or Path(name).name.startswith(('queries.', 'judgments.', 'judgment_conflicts.'))
                or name.startswith(('reports/', '.venv/', '.git/'))):
            raise ValueError('Non-inference artifact in deployment closure')
        config.path(name)
    return dict(sorted(files.items()))


def package_runtime(config, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Refusing to overwrite runtime bundle; use a fresh destination')
    files = bundle_inventory(config)
    destination.mkdir(parents=True)
    # Copy instead of linking: the bundle survives relocation without cache symlinks.
    for name, sha in files.items():
        source = config.path(name)
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_hash(target) != sha:
            raise ValueError('Copied runtime artifact hash mismatch')
    runtime = {'root': '.', 'manifest': config.manifest, 'manifest_sha256': config.manifest_sha256,
        'enabled_pipelines': list(config.enabled_pipelines), 'default_pipeline': config.default_pipeline,
        'default_top_k': config.default_top_k, 'device': config.device, 'reranker_fallback': config.reranker_fallback}
    write_json(destination/'runtime.json', runtime)
    manifest = {'format_version': 1, 'scope': 'inference-only portable bundle; no query or relevance label files',
        'runtime_config_sha256': file_hash(destination/'runtime.json'),
        'files': {name: {'sha256': sha, 'bytes': (destination/name).stat().st_size} for name, sha in files.items()}}
    write_json(destination/'bundle.json', manifest)
    verify_bundle(destination)
    return manifest


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Invalid bundle manifest')
    config = RuntimeConfig.load(directory/'runtime.json')
    if config.root != directory or file_hash(directory/'runtime.json') != manifest['runtime_config_sha256']:
        raise ValueError('Bundle config changed or root escaped')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    expected = set(manifest['files']) | {'runtime.json', 'bundle.json'}
    if actual != expected:
        raise ValueError('Bundle file inventory mismatch')
    for name, record in manifest['files'].items():
        path = config.path(name)
        if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError('Bundle checksum mismatch')
    # Re-evaluate frozen root of trust; editing bundle.json cannot bless altered payloads.
    if {p: v['sha256'] for p, v in manifest['files'].items()} != bundle_inventory(config):
        raise ValueError('Bundle does not match frozen inference closure')
    return manifest


def configure_process():
    # These are the already-validated numeric settings, not throughput tuning.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from .retrieval.lexical import BM25Retriever  # Preload native libraries before owner thread.
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def serve(config_path, host, port):
    from .development_data import BoundaryGuard
    config = RuntimeConfig.load(config_path)
    BoundaryGuard(config.root).install()
    configure_process()
    from threadpoolctl import threadpool_limits
    from .api import create_app
    import uvicorn
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    with threadpool_limits(limits=1):
        uvicorn.run(create_app(config_path), host=host, port=port, workers=1, access_log=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(os.environ.get('SEARCH_RUNTIME_CONFIG', '/runtime/runtime.json')))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    serve(args.config, args.host, args.port)


if __name__ == '__main__':
    main()
