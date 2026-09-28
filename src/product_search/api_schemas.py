"""HTTP-only validation and compact public projection of core contracts."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Pipeline = Literal['bm25', 'hybrid', 'hybrid_rerank']


class RequestBase(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    query: Annotated[str, Field(min_length=1, max_length=512)]

    @field_validator('query')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Query must contain non-whitespace characters')
        return value  # Do not change the core's query preprocessing.


class BM25Request(RequestBase):
    pipeline: Literal['bm25']
    top_k: Annotated[int, Field(ge=1, le=100)] = 10


class HybridRequest(RequestBase):
    pipeline: Literal['hybrid']
    top_k: Annotated[int, Field(ge=1, le=100)] = 10


class RerankRequest(RequestBase):
    pipeline: Literal['hybrid_rerank']
    top_k: Annotated[int, Field(ge=1, le=20)] = 10


SearchRequest = Annotated[BM25Request | HybridRequest | RerankRequest, Field(discriminator='pipeline')]


class PublicVersion(BaseModel):
    api_contract: str = 'search-http-v1'
    package_version: str
    core_contract: str
    runtime_manifest_sha256: str
    dataset_manifest_sha256: str
    retrieval_manifest_sha256: str
    reranking_manifest_sha256: str
    bm25: dict[str, str | float]
    dense: dict[str, str]
    cross_encoder: dict[str, str]


def public_version(version):
    # Whitelist values, never serialize the full runtime object or filesystem paths.
    return PublicVersion(package_version=version['project_package_version'],
        core_contract=version['core_contract_version'], runtime_manifest_sha256=version['runtime_manifest_sha256'],
        dataset_manifest_sha256=version['anchors']['dataset']['sha256'],
        retrieval_manifest_sha256=version['anchors']['retrieval']['sha256'],
        reranking_manifest_sha256=version['anchors']['reranking']['sha256'],
        bm25={k: version['bm25'][k] for k in ('implementation', 'version', 'method', 'k1', 'b', 'representation')},
        dense={k: version['dense'][k] for k in ('model', 'revision')},
        cross_encoder={k: version['reranker'][k] for k in ('model', 'revision')})


class ResultHit(BaseModel):
    product_id: int
    title: str
    final_rank: int
    final_score: float
    retrieval_rank: int
    retrieval_score: float
    reranker_score: float | None
    source: str


class StageTiming(BaseModel):
    preprocessing_ms: float
    bm25_ms: float
    dense_encoding_ms: float
    dense_search_ms: float
    fusion_ms: float
    reranking_ms: float
    hydration_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    request_id: str
    query: str
    requested_pipeline: Pipeline
    effective_pipeline: Pipeline
    fallback_used: bool
    fallback_reason: str | None
    results: list[ResultHit]
    timing_ms: StageTiming
    version: PublicVersion


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str


class HealthResponse(BaseModel):
    alive: bool


class ReadyResponse(BaseModel):
    ready: bool
    searchable: bool
    state: Literal['NOT_INITIALIZED', 'LOADING', 'READY', 'DEGRADED', 'FAILED', 'CLOSED']
    enabled_pipelines: list[Pipeline]
    detail: str | None
