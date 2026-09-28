from types import SimpleNamespace
from uuid import UUID
import pytest
from fastapi.testclient import TestClient
from product_search.api import create_app
from product_search.runtime_contracts import Readiness, SearchResult, SearchHit, freeze


class FakeEngine:
    def __init__(self):
        self.config = SimpleNamespace(enabled_pipelines=('bm25', 'hybrid', 'hybrid_rerank'))
        self.initialized = self.closed = self.calls = 0
        self.state = 'NOT_INITIALIZED'
        self.fail = self.fallback = self.bad_startup = False
        self.version = freeze({'project_package_version': '0.1.0', 'core_contract_version': 'core-v1',
            'runtime_manifest_sha256': 'a'*64,
            'anchors': {k: {'sha256': 'b'*64, 'path': 'C:/private/path'} for k in ('dataset', 'retrieval', 'reranking')},
            'bm25': {'implementation': 'bm25s', 'version': '0.3.11', 'method': 'lucene', 'k1': 1.2, 'b': .5, 'representation': 'B'},
            'dense': {'model': 'dense', 'revision': 'fixed'}, 'reranker': {'model': 'ce', 'revision': 'fixed'},
            'secret': 'do-not-expose', 'validated_artifact_sha256': {'C:/private/file': 'hash'}})

    def initialize(self):
        self.initialized += 1
        if self.bad_startup:
            self.state = 'FAILED'
            raise ValueError('C:/private/missing-artifact')
        self.state = 'READY'

    def close(self):
        self.closed += 1
        self.state = 'CLOSED'

    def readiness(self):
        return Readiness(self.state, self.state == 'READY', ('fake',), self.config.enabled_pipelines, 'C:/private/reason')

    def search(self, query, top_k, pipeline):
        self.calls += 1
        if self.fail:
            raise RuntimeError('C:/private/path secret-stack-detail')
        fb = self.fallback and pipeline == 'hybrid_rerank'
        if fb:
            self.state = 'DEGRADED'
        ids = [2, 1] if pipeline == 'hybrid_rerank' and not fb else [1, 2]
        hits = tuple(SearchHit(pid, 'title', i+1, 1.0, pid, .03,
                              1.0 if pipeline == 'hybrid_rerank' and not fb else None, 'source') for i, pid in enumerate(ids[:top_k]))
        timing = {k: .01 for k in ('preprocessing_ms', 'bm25_ms', 'dense_encoding_ms',
            'dense_search_ms', 'fusion_ms', 'reranking_ms', 'hydration_ms', 'total_ms')}
        return SearchResult(query, pipeline, 'hybrid' if fb else pipeline, hits, freeze(timing), self.version, fb, 'RuntimeError' if fb else None)


@pytest.fixture
def client():
    engine = FakeEngine()
    app = create_app(engine_factory=lambda: engine)
    with TestClient(app) as client:
        yield client, engine
    assert engine.closed == 1


def test_lifecycle_once_and_parity(client):
    c, engine = client
    for mode in engine.config.enabled_pipelines:
        direct = engine.search(' desk ', 2, mode)
        for _ in range(2):
            response = c.post('/search', json={'query': ' desk ', 'pipeline': mode, 'top_k': 2})
            assert response.status_code == 200
            data = response.json()
            assert data['query'] == direct.query
            assert [h['product_id'] for h in data['results']] == [h.product_id for h in direct.results]
            assert [h['final_score'] for h in data['results']] == [h.final_score for h in direct.results]
            assert data['effective_pipeline'] == direct.effective_pipeline
            assert data['fallback_used'] == direct.fallback_used
    assert engine.initialized == 1


@pytest.mark.parametrize('override', [{'query': ''}, {'query': '   '}, {'query': 'x'*513}, {'query': 123},
    {'top_k': 0}, {'top_k': -1}, {'top_k': True}, {'top_k': '10'}, {'top_k': 10.0},
    {'top_k': 101}, {'pipeline': 'dense'}, {'pipeline': None}, {'pipeline': 'hybrid_rerank', 'top_k': 21}, {'unknown': 1}])
def test_strict_validation(client, override):
    c, engine = client
    response = c.post('/search', json={'query': 'q', 'top_k': 10, 'pipeline': 'hybrid', **override})
    assert response.status_code == 422 and engine.calls == 0
    assert response.json()['request_id'] == response.headers['x-request-id']


def test_malformed_json_and_missing_pipeline(client):
    c, e = client
    assert c.post('/search', content='{broken', headers={'Content-Type': 'application/json'}).status_code == 422
    assert c.post('/search', json={'query': 'q'}).status_code == 422
    assert not e.calls


@pytest.mark.parametrize('state,code', [('NOT_INITIALIZED', 503), ('LOADING', 503), ('FAILED', 503),
                                     ('CLOSED', 503), ('READY', 200), ('DEGRADED', 200)])
def test_health_readiness(client, state, code):
    c, e = client
    e.state = state
    assert c.get('/health').json() == {'alive': True}
    response = c.get('/ready')
    assert response.status_code == code
    assert response.json()['ready'] == (state == 'READY')
    assert 'private' not in response.text
    if code == 503:
        assert c.post('/search', json={'query': 'q', 'pipeline': 'bm25'}).status_code == 503
        assert c.get('/version').status_code == 503
    assert e.calls == 0


def test_without_lifespan_not_ready():
    app = create_app(engine_factory=FakeEngine)
    c = TestClient(app)
    assert c.get('/health').status_code == 200
    assert c.get('/ready').status_code == 503
    assert app.state.engine is None


def test_version_and_request_ids(client):
    c, e = client
    response = c.get('/version')
    assert response.status_code == 200
    assert not any(x in response.text for x in ('private', 'secret', 'validated_artifact', 'path'))
    UUID(response.headers['x-request-id'])
    request = {'query': 'q', 'pipeline': 'bm25'}
    propagated = c.post('/search', json=request, headers={'X-Request-ID': 'fixture-001'})
    assert propagated.headers['x-request-id'] == propagated.json()['request_id'] == 'fixture-001'
    regenerated = c.post('/search', json=request, headers={'X-Request-ID': 'bad request id'})
    UUID(regenerated.headers['x-request-id'])
    assert float(propagated.headers['x-http-app-ms']) >= 0
    assert float(propagated.headers['x-engine-ms']) >= 0
    assert float(propagated.headers['x-response-serialization-ms']) >= 0


def test_controlled_error_and_logs(client, caplog):
    c, e = client
    e.fail = True
    with caplog.at_level('INFO', logger='product_search.service'):
        response = c.post('/search', json={'query': 'private query text', 'pipeline': 'hybrid'}, headers={'X-Request-ID': 'error-fixture'})
    assert response.status_code == 500
    assert 'private' not in response.text and 'traceback' not in response.text.lower()
    assert response.headers['x-request-id'] == 'error-fixture'
    assert 'error-fixture' in caplog.text and 'private query text' not in caplog.text
    assert 'search_failed' in caplog.text


def test_fallback_http_integration(client):
    c, e = client
    e.fallback = True
    hybrid = c.post('/search', json={'query': 'q', 'pipeline': 'hybrid'}).json()
    response = c.post('/search', json={'query': 'q', 'pipeline': 'hybrid_rerank'})
    result = response.json()
    assert response.status_code == 200
    assert result['requested_pipeline'] == 'hybrid_rerank'
    assert result['effective_pipeline'] == 'hybrid'
    assert result['fallback_used'] and result['fallback_reason'] == 'reranker_unavailable'
    assert result['results'] == hybrid['results']
    assert c.get('/ready').json()['state'] == 'DEGRADED'


def test_startup_failure_closes_engine():
    e = FakeEngine()
    e.bad_startup = True
    app = create_app(engine_factory=lambda: e)
    with pytest.raises(ValueError, match='missing-artifact'):
        with TestClient(app):
            pytest.fail('Should never serve after failed startup')
    assert e.initialized == e.closed == 1


def test_openapi_limits_and_errors(client):
    c, _ = client
    schema = c.get('/openapi.json').json()
    schemas = schema['components']['schemas']
    assert schemas['RerankRequest']['properties']['top_k']['maximum'] == 20
    assert schemas['HybridRequest']['properties']['top_k']['maximum'] == 100
    assert schemas['BM25Request']['properties']['query']['maxLength'] == 512
    operation = schema['paths']['/search']['post']
    assert operation['requestBody']['content']['application/json']['schema']['discriminator']['propertyName'] == 'pipeline'
    assert all(str(code) in operation['responses'] for code in (200, 422, 500, 503))


def test_disabled_mode(client):
    c, e = client
    e.config.enabled_pipelines = ('bm25',)
    assert c.post('/search', json={'query': 'q', 'pipeline': 'hybrid'}).status_code == 422
    assert not e.calls


def test_unexpected_probe_error_is_controlled(client):
    c, e = client
    def fail():
        raise RuntimeError('C:/private/probe-detail')
    e.readiness = fail
    response = c.get('/ready', headers={'X-Request-ID': 'probe-error'})
    assert response.status_code == 500
    assert 'private' not in response.text
    assert response.headers['x-request-id'] == response.json()['request_id'] == 'probe-error'


def test_multiple_lifespans_make_distinct_engines():
    engines = []
    def factory():
        e = FakeEngine()
        engines.append(e)
        return e
    app = create_app(engine_factory=factory)
    for _ in range(2):
        with TestClient(app) as c:
            assert c.get('/ready').status_code == 200
    assert len(engines) == 2
    assert all(e.initialized == e.closed == 1 for e in engines)


@pytest.mark.parametrize('headers', [{'X-Request-ID': 'x'*65}, [('X-Request-ID', 'one'), ('X-Request-ID', 'two')]])
def test_unsafe_or_duplicate_request_id_regenerated(client, headers):
    c, _ = client
    response = c.post('/search', json={'query': 'x'*512, 'pipeline': 'bm25'}, headers=headers)
    assert response.status_code == 200
    UUID(response.headers['x-request-id'])


def test_bad_core_response_serialization_remains_controlled(client):
    from dataclasses import replace
    c, e = client
    original = e.search
    def bad(*args):
        result = original(*args)
        return replace(result, results=(replace(result.results[0], final_score=object()),))
    e.search = bad
    response = c.post('/search', json={'query': 'q', 'pipeline': 'bm25'})
    assert response.status_code == 500
    assert response.headers['x-request-id'] == response.json()['request_id']
    assert response.json()['error'] == 'search_failed'
