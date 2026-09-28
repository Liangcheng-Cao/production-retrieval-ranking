"""Sequential production-style core using unchanged offline ranking implementations."""
from time import perf_counter
from .runtime_config import RuntimeConfig
from .runtime_artifacts import ArtifactLoader
from .runtime_contracts import Readiness, SearchError, SearchHit, SearchResult, freeze


class SearchEngine:
    def __init__(self, config, loader=None):
        self.config = config
        self._loader = loader if loader is not None else ArtifactLoader(config)
        self._resources = None
        self._state = 'NOT_INITIALIZED'
        self._reason = None
        self._failures = {}

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, RuntimeConfig):
            config = RuntimeConfig.load(config)
        engine = cls(config)
        engine.initialize()
        return engine

    def initialize(self):
        if self._state in ('READY', 'DEGRADED'):
            return self
        if self._state != 'NOT_INITIALIZED':
            raise SearchError('Create a new engine after failed startup or close')
        self._state = 'LOADING'
        try:
            resources = self._loader.load()
            if resources.hybrid is None and any(p != 'bm25' for p in self.config.enabled_pipelines):
                raise SearchError('Dense/Hybrid component required')
            if resources.reranker is None and 'hybrid_rerank' in self.config.enabled_pipelines:
                raise SearchError('Reranker required at startup')
            self._resources = resources
            self._state = 'READY'
        except Exception as exc:
            self._resources = None
            self._state = 'FAILED'
            self._reason = f'{type(exc).__name__}: {exc}'
            raise
        return self

    @property
    def version(self):
        if self._resources is None:
            raise SearchError('Runtime is not initialized')
        return self._resources.version

    @property
    def startup_timing(self):
        if self._resources is None:
            raise SearchError('Runtime is not initialized')
        return self._resources.startup

    def readiness(self):
        return Readiness(self._state, self._state == 'READY',
                         self._resources.loaded if self._resources else (),
                         self.config.enabled_pipelines, self._reason)

    def close(self):
        self._resources = None
        self._state = 'CLOSED'
        self._reason = None

    def _health(self, component, error=None):
        if error is None:
            self._failures.pop(component, None)
        else:
            self._failures[component] = error
        self._state = 'DEGRADED' if self._failures else 'READY'
        self._reason = '; '.join(f'{k}: {v}' for k, v in sorted(self._failures.items())) or None

    def search(self, query, top_k=None, pipeline=None):
        start = perf_counter()
        pipeline = self.config.default_pipeline if pipeline is None else pipeline
        top_k = self.config.default_top_k if top_k is None else top_k
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be a nonempty string')
        if not isinstance(pipeline, str) or pipeline not in self.config.enabled_pipelines:
            raise ValueError('Pipeline is unknown or disabled')
        if type(top_k) is not int or not 1 <= top_k <= (20 if pipeline == 'hybrid_rerank' else 100):
            raise ValueError('top_k must be 1..20 for CE-20, otherwise 1..100')
        if self._state not in ('READY', 'DEGRADED'):
            raise SearchError(f'Engine unavailable: {self._state}')
        # Preserve the original query: retrievers own their frozen preprocessing.
        timing = {k: 0.0 for k in ('preprocessing_ms', 'bm25_ms', 'dense_encoding_ms',
                  'dense_search_ms', 'fusion_ms', 'reranking_ms', 'hydration_ms', 'total_ms')}
        timing['preprocessing_ms'] = (perf_counter()-start)*1000
        r = self._resources
        retrieval_component = 'bm25' if pipeline == 'bm25' else 'hybrid'
        try:
            if pipeline == 'bm25':
                t = perf_counter()
                candidates = r.bm25.search(query, top_k)
                timing['bm25_ms'] = (perf_counter()-t)*1000
            else:
                candidates, stages = r.hybrid.search_timed(query, r.candidate_depth)
                for old, new in [('lexical_ms', 'bm25_ms'), ('encoding_ms', 'dense_encoding_ms'),
                                 ('search_ms', 'dense_search_ms'), ('fusion_ms', 'fusion_ms')]:
                    timing[new] = stages[old]
            self._health(retrieval_component)
        except Exception as exc:
            self._health(retrieval_component, type(exc).__name__)
            raise SearchError(f'{retrieval_component} retrieval failed; no pipeline substitution') from exc
        fallback = False
        reason = None
        effective = pipeline
        if pipeline == 'hybrid_rerank':
            t = perf_counter()
            try:
                reranked = r.reranker.rerank(query, candidates[:r.reranker_depth], top_k,
                                            fallback=self.config.reranker_fallback)
            except Exception as exc:
                self._health('reranker', type(exc).__name__)
                raise SearchError('Reranker failed and fallback is disabled or input is invalid') from exc
            timing['reranking_ms'] = (perf_counter()-t)*1000
            fallback, reason = reranked.fallback_used, reranked.error_type
            effective = 'hybrid' if fallback else pipeline
            if candidates:
                self._health('reranker', reason if fallback else None)
            raw = [(h.product_id, h.reranker_score if h.reranker_score is not None else h.retrieval_score,
                    h.retrieval_rank, h.retrieval_score, h.reranker_score, h.retrieval_source) for h in reranked.hits]
        else:
            raw = [(h.product_id, h.score, h.rank, h.score, None, h.source) for h in candidates[:top_k]]
        t = perf_counter()
        try:
            hits = tuple(SearchHit(pid, r.products[pid].product_name, rank, float(score),
                                  original_rank, float(original_score), ce_score, source)
                         for rank, (pid, score, original_rank, original_score, ce_score, source) in enumerate(raw, 1))
        except KeyError as exc:
            self._health('metadata', 'Missing product ID')
            raise SearchError(f'Missing product metadata: {exc.args[0]}') from exc
        timing['hydration_ms'] = (perf_counter()-t)*1000
        timing['total_ms'] = (perf_counter()-start)*1000
        return SearchResult(query, pipeline, effective, hits, freeze(timing), self.version, fallback, reason)
