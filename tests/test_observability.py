import asyncio
import json
from threading import Event
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from product_search.api import create_app
from product_search.observability import ServiceMetrics, WaitingTicket, STAGES
from test_api import FakeEngine


def samples(text, name):
    return [s for family in text_string_to_metric_families(text) for s in family.samples if s.name == name]


def total(metrics, name):
    return sum(s.value for s in samples(metrics.render().decode(), name))


def test_counter_histogram_labels_fallback_and_privacy():
    engine = FakeEngine()
    app = create_app(engine_factory=lambda: engine)
    with TestClient(app) as c:
        for path in ('/metrics', '/health', '/ready', '/version'):
            assert c.get(path).status_code == 200
        m = app.state.metrics
        assert total(m, 'request_count_total') == 0
        for pipeline in ('bm25', 'hybrid', 'hybrid_rerank'):
            assert c.post('/search', json={'query': 'private-query-phrase two', 'pipeline': pipeline}).status_code == 200
        engine.fallback = True
        c.post('/search', json={'query': 'private-query-phrase', 'pipeline': 'hybrid_rerank'})
        c.post('/search', json={'query': 'private-query-phrase', 'pipeline': 'bm25'})
        c.post('/search', json={'query': 'private-query-phrase', 'pipeline': 'arbitrary-high-cardinality'})
        text = c.get('/metrics').text
        assert c.get('/metrics').headers['content-type'].startswith('text/plain; version=0.0.4')
        assert total(m, 'request_count_total') == 6
        assert total(m, 'request_error_count_total') == 1
        assert total(m, 'fallback_count_total') == total(m, 'reranker_fallback_total') == 1
        assert total(m, 'degraded_request_total') == 2
        assert total(m, 'request_latency_seconds_count') == 6
        assert total(m, 'engine_latency_seconds_count') == 5
        assert total(m, 'queue_wait_seconds_count') == 5
        assert total(m, 'result_count_sum') == 10
        assert total(m, 'query_token_count_sum') == 8
        assert total(m, 'requests_waiting_for_engine') == total(m, 'in_flight_requests') == 0
        for stage in STAGES:
            assert total(m, stage+'_latency_seconds_count') == 5
        assert all(token not in text for token in ('private-query', 'arbitrary-high-cardinality', 'C:', 'request_id', 'product_id', 'query_hash', 'secret'))
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                assert set(sample.labels) <= {'pipeline', 'effective_pipeline', 'status', 'le'}
        fallback = samples(text, 'reranker_fallback_total')[0]
        assert fallback.labels == {'pipeline': 'hybrid_rerank', 'effective_pipeline': 'hybrid', 'status': '200'}


def test_registry_restart_and_empty_results():
    engine = FakeEngine()
    original = engine.search
    from dataclasses import replace
    engine.search = lambda *a: replace(original(*a), results=())
    app = create_app(engine_factory=lambda: engine)
    for _ in range(2):
        with TestClient(app) as c:
            assert total(app.state.metrics, 'request_count_total') == 0
            c.post('/search', json={'query': 'q', 'pipeline': 'bm25'})
            assert total(app.state.metrics, 'empty_result_total') == 1
            assert total(app.state.metrics, 'result_count_sum') == 0


def test_structured_logs_no_exception_query_leak(caplog):
    engine = FakeEngine()
    def fail(*args):
        raise RuntimeError('sensitive-query-in-exception')
    engine.search = fail
    with caplog.at_level('INFO', logger='product_search.service'):
        with TestClient(create_app(engine_factory=lambda: engine)) as c:
            response = c.post('/search', json={'query': 'sensitive-query-in-exception', 'pipeline': 'hybrid'}, headers={'X-Request-ID': 'safe-id'})
    assert response.status_code == 500
    assert 'sensitive-query-in-exception' not in caplog.text
    rows = [json.loads(r.message) for r in caplog.records if r.name == 'product_search.service']
    row = next(r for r in rows if r['event'] == 'search_failed' and 'http_total_ms' in r)
    assert row['request_id'] == 'safe-id' and row['status'] == 500 and row['queue_wait_ms'] >= 0
    assert {'service_starting', 'engine_loading', 'service_ready', 'service_shutdown'} <= {r['event'] for r in rows}


def test_ticket_cancellation_idempotent():
    m = ServiceMetrics()
    ticket = WaitingTicket(m, 'user-supplied-unknown')
    assert total(m, 'requests_waiting_for_engine') == 1
    ticket.cancel()
    ticket.cancel()
    ticket.enter()
    assert total(m, 'requests_waiting_for_engine') == total(m, 'queue_wait_seconds_count') == 0


def test_cancellation_does_not_hide_active_native_call_or_leak_waiters():
    import httpx
    async def exercise():
        engine = FakeEngine()
        entered, release = Event(), Event()
        original = engine.search
        def held(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test timeout')
            return original(*args)
        engine.search = held
        app = create_app(engine_factory=lambda: engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as c:
                async def until(predicate):
                    for _ in range(200):
                        if predicate():
                            return
                        await asyncio.sleep(.005)
                    raise AssertionError('timeout')
                a = asyncio.create_task(c.post('/search', json={'query': 'one', 'pipeline': 'bm25'}))
                try:
                    await until(entered.is_set)
                    b = asyncio.create_task(c.post('/search', json={'query': 'two', 'pipeline': 'bm25'}))
                    await until(lambda: total(app.state.metrics, 'requests_waiting_for_engine') == 1)
                    assert total(app.state.metrics, 'in_flight_requests') == 2
                    b.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await b
                    a.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await a
                    assert total(app.state.metrics, 'in_flight_requests') == 0
                    assert total(app.state.metrics, 'requests_waiting_for_engine') == 0
                    assert total(app.state.metrics, 'engine_active_calls') == 1
                    assert total(app.state.metrics, 'request_error_count_total') == 2
                finally:
                    release.set()
                await until(lambda: total(app.state.metrics, 'engine_active_calls') == 0)
    asyncio.run(exercise())
