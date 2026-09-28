"""Validate/load local frozen inputs. Never builds indexes or fetches models."""
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .data.io import file_hash
from .data.schema import Product
from .runtime_contracts import ArtifactError, freeze


@dataclass
class RuntimeResources:
    products: dict
    bm25: object
    hybrid: object | None
    reranker: object | None
    version: object
    startup: object
    candidate_depth: int
    reranker_depth: int
    loaded: tuple[str, ...]


class ArtifactLoader:
    def __init__(self, config):
        self.config = config

    def checked(self, name, digest):
        path = self.config.path(name)
        if not path.is_file():
            raise ArtifactError(f'Missing required artifact: {name}')
        if file_hash(path) != digest:
            raise ArtifactError(f'Artifact checksum mismatch: {name}')
        return path

    def validate(self):
        c = self.config
        lock = json.loads(self.checked(c.manifest, c.manifest_sha256).read_text(encoding='utf-8'))
        if lock.get('format_version') != 1:
            raise ArtifactError('Incompatible runtime manifest format')
        docs = {role: json.loads(self.checked(v['path'], v['sha256']).read_text(encoding='utf-8'))
                for role, v in lock['anchors'].items()}
        p1, p2, p3 = (docs[k] for k in ('dataset', 'retrieval', 'reranking'))
        s2, s3, plan = (docs[k] for k in ('selection2', 'selection3', 'plan3'))
        dataset_sha = lock['anchors']['dataset']['sha256']
        retrieval_sha = lock['anchors']['retrieval']['sha256']
        if (s2['data_manifest_sha256'] != dataset_sha
                or s3['phase1_manifest_sha256'] != dataset_sha
                or s3['phase2_manifest_sha256'] != retrieval_sha
                or p3['phase1_manifest_sha256'] != dataset_sha
                or p3['phase2_retrieval_manifest_sha256'] != retrieval_sha
                or p3['selected'] != s3 or p3['plan'] != plan
                or p3['selection_sha256'] != lock['anchors']['selection3']['sha256']
                or p3['plan_sha256'] != lock['anchors']['plan3']['sha256']):
            raise ArtifactError('Frozen dataset/selection provenance mismatch')
        hybrid = s2['hybrid']['config']
        if (hybrid['candidate_depth'] != 100 or hybrid['method'] != 'rrf'
                or hybrid['weights'] != [1, 1] or hybrid['bm25'] != s2['bm25']['name']
                or hybrid['dense'] != s2['dense']['name'] or s3['selected']['candidate_depth'] != 20
                or plan['dtype'] != 'float32'):
            raise ArtifactError('Unsupported frozen pipeline semantics')
        checked = {}
        products_path = lock['products_path']
        checked[products_path] = p1['artifacts']['products.jsonl']['sha256']
        self.checked(products_path, checked[products_path])
        need_dense = any(p != 'bm25' for p in c.enabled_pipelines)
        need_ce = 'hybrid_rerank' in c.enabled_pipelines
        for role in ('bm25', 'dense') if need_dense else ('bm25',):
            choice = s2[role]
            record = p2['artifacts'][choice['name']]
            directory = lock['index_root'] + '/' + choice['name']
            manifest_path = directory + '/manifest.json'
            index_manifest = json.loads(self.checked(manifest_path, record['manifest_sha256']).read_text())
            if (index_manifest != record['manifest'] or index_manifest['format_version'] != 1
                    or index_manifest['config'] != choice['config']
                    or choice['config']['data_manifest_sha256'] != dataset_sha):
                raise ArtifactError(f'Incompatible {role} configuration/dataset')
            checked[manifest_path] = record['manifest_sha256']
            for name, digest in index_manifest['files'].items():
                target = directory + '/' + name
                if not c.path(target).is_relative_to(c.path(directory)):
                    raise ArtifactError('Index file escapes artifact directory')
                self.checked(target, digest)
                checked[target] = digest
        for role in (('dense', 'ce') if need_ce else ('dense',) if need_dense else ()):
            model = lock['models'][role]
            expected = s2['dense']['config'] if role == 'dense' else plan
            if model['model'] != expected['model'] or model['revision'] != expected['revision']:
                raise ArtifactError('Model revision mismatch')
            snapshot = c.path(model['snapshot'])
            if not model['files'] or not any(p.endswith('.safetensors') for p in model['files']):
                raise ArtifactError('Missing sealed model weights')
            actual_files = {p.relative_to(c.root).as_posix() for p in snapshot.rglob('*') if p.is_file()}
            if actual_files != set(model['files']):
                raise ArtifactError('Model snapshot file inventory mismatch')
            for name, digest in model['files'].items():
                if not c.path(name).is_relative_to(snapshot):
                    raise ArtifactError('Model file escapes snapshot')
                self.checked(name, digest)
                checked[name] = digest
        required = ['numpy', 'bm25s']
        if need_dense:
            required += ['torch', 'transformers', 'sentence-transformers', 'tokenizers', 'safetensors']
        software = {name: importlib.metadata.version(name) for name in required}
        if any(software[name] != lock['software'][name] for name in required):
            raise ArtifactError('Runtime software differs from sealed environment')
        return lock, docs, checked, software

    def load(self):
        from .retrieval.lexical import BM25Retriever
        start = perf_counter()
        try:
            lock, docs, checked, software = self.validate()
            validated = perf_counter()
            c = self.config
            s2, s3, plan = docs['selection2'], docs['selection3']['selected'], docs['plan3']
            with c.path(lock['products_path']).open(encoding='utf-8') as handle:
                rows = [Product(**json.loads(line)) for line in handle]
            products = {p.product_id: p for p in rows}
            if len(products) != len(rows) or len(rows) != docs['dataset']['artifacts']['products.jsonl']['rows']:
                raise ArtifactError('Invalid product metadata membership')
            hydrated = perf_counter()
            root = c.path(lock['index_root'])
            bm25 = BM25Retriever.load(root / s2['bm25']['name'], s2['bm25']['config'])
            if set(map(int, bm25.product_ids)) != set(products):
                raise ArtifactError('BM25/product metadata catalog mismatch')
            lexical_end = perf_counter()
            loaded = ['products', 'bm25']
            hybrid = reranker = None
            stage = {'validation_ms': (validated-start)*1000, 'metadata_ms': (hydrated-validated)*1000,
                     'bm25_load_ms': (lexical_end-hydrated)*1000}
            if any(p != 'bm25' for p in c.enabled_pipelines):
                from .retrieval.dense import DenseRetriever
                from .retrieval.hybrid import HybridRetriever
                t = perf_counter()
                dense = DenseRetriever.load(root / s2['dense']['name'], s2['dense']['config'])
                stage['dense_index_load_ms'] = (perf_counter()-t)*1000
                t = perf_counter()
                import torch
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(str(c.path(lock['models']['dense']['snapshot'])),
                                              device=c.device, local_files_only=True, trust_remote_code=False)
                encoder.max_seq_length = s2['dense']['config']['max_seq_length']
                encoder.to(dtype=torch.float32).eval()
                if encoder.get_embedding_dimension() != dense.embeddings.shape[1]:
                    raise ArtifactError('Encoder/index dimension mismatch')
                dense.encoder = encoder
                hybrid = HybridRetriever(bm25, dense, s2['hybrid']['config']['constant'],
                                         s2['hybrid']['config']['candidate_depth'])
                stage['dense_encoder_load_ms'] = (perf_counter()-t)*1000
                loaded += ['dense_index', 'dense_encoder', 'hybrid']
            if 'hybrid_rerank' in c.enabled_pipelines:
                from .reranking import CrossEncoderReranker, TransformerPairScorer
                t = perf_counter()
                scorer = TransformerPairScorer(str(c.path(lock['models']['ce']['snapshot'])),
                                               plan['revision'], c.device)
                scorer.synchronize()
                reranker = CrossEncoderReranker(scorer, rows, s3['representation'],
                                               s3['batch_size'], plan['max_length'],
                                               s2['hybrid']['config']['candidate_depth'])
                stage['reranker_load_ms'] = (perf_counter()-t)*1000
                loaded += ['cross_encoder']
            numerical_runtime = {}
            if hybrid is not None:
                numerical_runtime = {'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                    'torch_num_threads': torch.get_num_threads(), 'cuda_runtime': torch.version.cuda}
            version = freeze({'runtime_manifest_sha256': c.manifest_sha256,
                'core_contract_version': 'production-search-core-v1',
                'project_package_version': importlib.metadata.version('production-retrieval-ranking'),
                'runtime_source_sha256': {p.name: file_hash(p) for p in
                    sorted(Path(__file__).parent.glob('runtime_*.py')) + [Path(__file__).parent/'search_engine.py']},
                'anchors': lock['anchors'], 'bm25': s2['bm25']['config'], 'dense': s2['dense']['config'],
                'hybrid': s2['hybrid']['config'], 'reranker': {'model': plan['model'], 'revision': plan['revision'],
                    'representation': s3['representation'], 'depth': s3['candidate_depth'],
                    'batch_size': s3['batch_size'], 'max_length': plan['max_length']},
                'validated_artifact_sha256': checked, 'software': software, 'python': platform.python_version(),
                'device': c.device, 'dtype': 'float32', 'numerical_runtime': numerical_runtime,
                'load_policy': 'eager-enabled-components',
                'enabled_pipelines': c.enabled_pipelines, 'reranker_fallback': c.reranker_fallback})
            stage['total_ms'] = (perf_counter()-start)*1000
            return RuntimeResources(products, bm25, hybrid, reranker, version, freeze(stage),
                                    s2['hybrid']['config']['candidate_depth'], s3['candidate_depth'], tuple(loaded))
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f'Runtime startup failed: {type(exc).__name__}: {exc}') from exc
