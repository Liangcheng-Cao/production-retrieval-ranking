import pytest
from product_search.benchmarking import query_schedule, distribution, summarize_requests


def test_balanced_deterministic_schedule():
    a = query_schedule([1, 2, 3, 4], 96, 42)
    assert a == query_schedule([1, 2, 3, 4], 96, 42)
    assert a != query_schedule([1, 2, 3, 4], 96, 43)
    assert all(a.count(i) == 24 for i in (1, 2, 3, 4))
    with pytest.raises(ValueError):
        query_schedule([], 3, 42)


def test_percentile_accounting_and_empty():
    d = distribution([0, 10, 20])
    assert d['p50_ms'] == 10 and d['p95_ms'] == 19 and d['count'] == 3
    assert distribution([])['p95_ms'] is None
    with pytest.raises(ValueError):
        distribution([float('nan')])


def test_errors_and_fallbacks_are_not_hidden():
    rows = [{'status': 200, 'client_ms': 10, 'engine_ms': 5, 'fallback_used': True, 'parity_passed': True},
            {'status': 500, 'client_ms': 20}, {'status': None, 'client_ms': 30, 'error': 'Timeout'},
            {'status': 200, 'client_ms': 15, 'parity_passed': False}]
    value = summarize_requests(rows, 2)
    assert value['attempts'] == 4 and value['errors'] == 2
    assert value['successful_requests_per_second'] == 1
    assert value['error_rate'] == .5 and value['fallbacks'] == value['ranking_mismatches'] == 1
    assert value['client_all_attempts']['count'] == 4 and value['client_success']['count'] == 2
