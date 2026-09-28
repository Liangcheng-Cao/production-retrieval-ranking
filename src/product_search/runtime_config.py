"""Small JSON deployment config; ranking parameters live in frozen selections."""
import json
from dataclasses import dataclass
from pathlib import Path

PIPELINES = ('bm25', 'hybrid', 'hybrid_rerank')


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    manifest: str
    manifest_sha256: str
    enabled_pipelines: tuple[str, ...] = PIPELINES
    default_pipeline: str = 'hybrid'
    default_top_k: int = 10
    device: str = 'cuda'
    reranker_fallback: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).resolve())
        object.__setattr__(self, 'enabled_pipelines', tuple(self.enabled_pipelines))
        if (not self.enabled_pipelines or len(set(self.enabled_pipelines)) != len(self.enabled_pipelines)
                or any(p not in PIPELINES for p in self.enabled_pipelines)
                or self.default_pipeline not in self.enabled_pipelines):
            raise ValueError('Invalid enabled/default pipeline')
        if type(self.default_top_k) is not int or not 1 <= self.default_top_k <= (20 if self.default_pipeline == 'hybrid_rerank' else 100):
            raise ValueError('Invalid default_top_k')
        if self.device not in ('cpu', 'cuda', 'cuda:0') or type(self.reranker_fallback) is not bool:
            raise ValueError('Invalid device/fallback policy')
        if len(self.manifest_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.manifest_sha256):
            raise ValueError('Pinned manifest SHA256 required')
        self.path(self.manifest)

    def path(self, relative):
        target = (self.root / relative).resolve()
        if Path(relative).is_absolute() or not target.is_relative_to(self.root):
            raise ValueError('Runtime paths must stay inside artifact root')
        return target

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = json.loads(path.read_text(encoding='utf-8'))
        values['root'] = (path.parent / values['root']).resolve()
        return cls(**values)
