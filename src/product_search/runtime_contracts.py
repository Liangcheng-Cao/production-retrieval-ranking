"""Framework-independent, immutable search contracts."""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class SearchHit:
    product_id: int
    title: str
    final_rank: int
    final_score: float
    retrieval_rank: int
    retrieval_score: float
    reranker_score: float | None
    source: str


@dataclass(frozen=True)
class SearchResult:
    query: str
    requested_pipeline: str
    effective_pipeline: str
    results: tuple[SearchHit, ...]
    timing: Mapping
    version: Mapping
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class Readiness:
    state: str
    ready: bool
    loaded_components: tuple[str, ...]
    enabled_pipelines: tuple[str, ...]
    reason: str | None


class ArtifactError(ValueError):
    """Required frozen runtime input is missing, corrupt or incompatible."""


class SearchError(RuntimeError):
    """Search failed without silently changing the requested pipeline."""
