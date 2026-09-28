from dataclasses import replace
from types import SimpleNamespace
import pytest

from product_search.data.schema import Product
from product_search.retrieval.common import Hit
from product_search.reranking import CrossEncoderReranker
from product_search.runtime_config import RuntimeConfig
from product_search.runtime_contracts import SearchError, freeze
from product_search.runtime_artifacts import RuntimeResources
from product_search.search_engine import SearchEngine


class Lexical:
    def __init__(self):
        self.calls = 0
        self.hits = [Hit(i, 4.0-i, i+1, 'bm25') for i in range(3)]

    def search(self, query, top_k):
        self.calls += 1
        return self.hits[:top_k]


class Hybrid:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def search_timed(self, query, top_k):
        self.calls += 1
        if self.fail:
            raise RuntimeError('dense unavailable')
        return [Hit(i, .03-i*.001, i+1, 'hybrid_rrf') for i in range(3)], {
            'lexical_ms': .1, 'encoding_ms': .2, 'search_ms': .3, 'fusion_ms': .4}


class Scorer:
    def __init__(self):
        self.fail = False

    def score_pairs(self, pairs, *args):
        if self.fail:
            raise RuntimeError('synthetic inference failure')
        return list(range(len(pairs))), {}


@pytest.fixture
def setup(tmp_path):
    config = RuntimeConfig(tmp_path, 'manifest.json', '0'*64)
    products = [Product(i, f'title {i}', '', '', '', '', None, None, None) for i in range(3)]
    scorer = Scorer()
    r = RuntimeResources({p.product_id: p for p in products}, Lexical(), Hybrid(),
                         CrossEncoderReranker(scorer, products), freeze({'model': {'revision': 'fixed'}}),
                         freeze({'total_ms': 100}), 100, 20, ('products', 'bm25', 'dense', 'ce'))
    class Loader:
        calls = 0
        def load(self):
            self.calls += 1
            assert engine.readiness().state == 'LOADING'
            return r
    loader = Loader()
    engine = SearchEngine(config, loader)
    return engine, r, scorer, loader


def test_lifecycle_and_load_once(setup):
    e, r, scorer, loader = setup
    assert e.readiness().state == 'NOT_INITIALIZED'
    with pytest.raises(SearchError):
        e.search('desk')
    e.initialize().initialize()
    for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
        e.search('desk', 2, mode)
        e.search('chair', 2, mode)
    assert loader.calls == 1 and e.readiness().ready
    e.close()
    assert e.readiness().state == 'CLOSED'
    with pytest.raises(SearchError):
        e.initialize()


@pytest.mark.parametrize('mode', ['bm25', 'hybrid', 'hybrid_rerank'])
def test_dispatch_parity_hydration_timing(setup, mode):
    e, r, _, _ = setup
    e.initialize()
    result = e.search('unchanged query ', 2, mode)
    assert result.query == 'unchanged query '
    expected = ([0, 1] if mode != 'hybrid_rerank' else [2, 1])
    assert [h.product_id for h in result.results] == expected
    assert [h.final_rank for h in result.results] == [1, 2]
    assert all(h.title == f'title {h.product_id}' for h in result.results)
    assert result.requested_pipeline == result.effective_pipeline == mode
    assert result.version is e.version
    assert set(result.timing) == {'preprocessing_ms', 'bm25_ms', 'dense_encoding_ms',
                                 'dense_search_ms', 'fusion_ms', 'reranking_ms', 'hydration_ms', 'total_ms'}
    assert all(v >= 0 for v in result.timing.values())
    if mode == 'hybrid_rerank':
        offline = r.reranker.rerank('unchanged query ', r.hybrid.search_timed('q', 100)[0], 2)
        assert [h.final_score for h in result.results] == [h.reranker_score for h in offline.hits]
    elif mode == 'bm25':
        assert [h.final_score for h in result.results] == [h.score for h in r.bm25.search('q', 2)]
    else:
        assert [h.final_score for h in result.results] == [h.score for h in r.hybrid.search_timed('q', 100)[0][:2]]


@pytest.mark.parametrize('query,k,mode', [('', 1, 'bm25'), ('  ', 1, 'hybrid'), (None, 1, 'hybrid'),
    ('q', 0, 'bm25'), ('q', True, 'hybrid'), ('q', 101, 'hybrid'), ('q', 21, 'hybrid_rerank'),
    ('q', 1.5, 'bm25'), ('q', 1, 'dense'), ('q', 1, [])])
def test_bad_requests_before_models(setup, query, k, mode):
    e, r, _, _ = setup
    e.initialize()
    with pytest.raises(ValueError):
        e.search(query, k, mode)
    assert r.bm25.calls == r.hybrid.calls == 0
    assert e.readiness().ready


def test_fallback_and_recovery(setup):
    e, r, scorer, _ = setup
    e.initialize()
    scorer.fail = True
    result = e.search('q', 2, 'hybrid_rerank')
    assert result.fallback_used and result.fallback_reason == 'RuntimeError'
    assert result.requested_pipeline == 'hybrid_rerank' and result.effective_pipeline == 'hybrid'
    assert [h.product_id for h in result.results] == [0, 1]
    assert all(h.reranker_score is None and h.final_score == h.retrieval_score for h in result.results)
    assert e.readiness().state == 'DEGRADED' and not e.readiness().ready
    e.search('q', 2, 'bm25')
    assert e.readiness().state == 'DEGRADED'  # BM25 success cannot clear CE failure.
    scorer.fail = False
    e.search('q', 2, 'hybrid_rerank')
    assert e.readiness().ready


def test_dense_failure_does_not_substitute_pipeline(setup):
    e, r, _, _ = setup
    e.initialize()
    r.hybrid.fail = True
    with pytest.raises(SearchError, match='no pipeline substitution'):
        e.search('q', 2, 'hybrid')
    assert e.readiness().state == 'DEGRADED'
    assert e.search('q', 2, 'bm25').results
    r.hybrid.fail = False
    e.search('q', 2, 'hybrid')
    assert e.readiness().ready


def test_strict_reranker_failure(setup):
    e, _, scorer, _ = setup
    e.config = replace(e.config, reranker_fallback=False)
    e.initialize()
    scorer.fail = True
    with pytest.raises(SearchError, match='fallback is disabled'):
        e.search('q', 2, 'hybrid_rerank')


def test_missing_metadata_is_explicit(setup):
    e, r, _, _ = setup
    e.initialize()
    del r.products[0]
    with pytest.raises(SearchError, match='Missing product metadata'):
        e.search('q', 2, 'bm25')
    assert e.readiness().state == 'DEGRADED'


def test_empty_result(setup):
    e, r, _, _ = setup
    e.initialize()
    r.bm25.hits = []
    result = e.search('unknown', 2, 'bm25')
    assert result.results == () and not result.fallback_used


def test_empty_candidates_do_not_claim_reranker_recovery(setup):
    e, r, scorer, _ = setup
    e.initialize()
    scorer.fail = True
    e.search('q', 2, 'hybrid_rerank')
    r.hybrid.search_timed = lambda *args: ([], {'lexical_ms': 0, 'encoding_ms': 0, 'search_ms': 0, 'fusion_ms': 0})
    result = e.search('q', 2, 'hybrid_rerank')
    assert result.results == ()
    assert e.readiness().state == 'DEGRADED'


def test_immutable_metadata(setup):
    e, _, _, _ = setup
    e.initialize()
    with pytest.raises(TypeError):
        e.version['model']['revision'] = 'changed'
    with pytest.raises(TypeError):
        e.search('q').timing['total_ms'] = 0


def test_failed_startup_is_observable(setup):
    e, _, _, _ = setup
    def fail():
        raise FileNotFoundError('index missing')
    e._loader = SimpleNamespace(load=fail)
    with pytest.raises(FileNotFoundError):
        e.initialize()
    assert e.readiness().state == 'FAILED' and not e.readiness().loaded_components
    with pytest.raises(SearchError):
        e.search('q')


def test_disabled_pipeline(setup):
    e, _, _, _ = setup
    e.config = replace(e.config, enabled_pipelines=('bm25',), default_pipeline='bm25')
    e.initialize()
    with pytest.raises(ValueError):
        e.search('q', 1, 'hybrid')


@pytest.mark.parametrize('kwargs', [{'enabled_pipelines': ('dense',)}, {'default_top_k': True},
    {'device': 'invalid'}, {'manifest': '../escape.json'}, {'manifest_sha256': 'x'*64}])
def test_invalid_config(tmp_path, kwargs):
    args = dict(root=tmp_path, manifest='m.json', manifest_sha256='0'*64)
    args.update(kwargs)
    with pytest.raises(ValueError):
        RuntimeConfig(**args)
