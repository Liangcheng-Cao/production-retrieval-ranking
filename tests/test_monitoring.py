from types import SimpleNamespace
import pytest
from product_search.monitoring import (VERSION, digest, distribution, scalar_distance, total_variation,
    embedding_summary, centroid_distance, query_features, scenarios, ranking_difference, envelope, validate_artifact)
from product_search.development_data import BoundaryGuard, load_development


def test_scalar_and_categorical_drift_known_answers():
    assert scalar_distance([0, 1], [2, 3]) == 2
    assert scalar_distance([1, 1, 2], [2, 1, 1]) == 0
    assert total_variation({'a': 1, 'b': 1}, {'a': 2}) == .5
    assert total_variation({'a': 1}, {'b': 1}) == 1
    assert scalar_distance([], [1]) is None
    assert total_variation({}, {'a': 1}) is None
    with pytest.raises(ValueError):
        scalar_distance([float('nan')], [1])
    with pytest.raises(ValueError):
        total_variation({'a': -1}, {'a': 2})


def test_embedding_summaries():
    summary = embedding_summary([[1, 0], [0, 1]])
    assert summary == {'count': 2, 'dimension': 2, 'mean_norm': 1, 'centroid': [.5, .5]}
    assert centroid_distance([1, 0], [0, 1]) == 1
    assert centroid_distance([1, 0], [-1, 0]) == 2
    assert centroid_distance([0, 0], [1, 0]) is None
    assert embedding_summary(__import__('numpy').empty((0, 2)))['centroid'] is None
    with pytest.raises(ValueError):
        embedding_summary([[float('nan')]])
    with pytest.raises(ValueError):
        embedding_summary([[0, 0]])


def artifact():
    return envelope({'version': VERSION, 'provenance': {'split': 'train', 'query_count': 2},
        'features': query_features([('abc one', 'a'), ('xy', 'b')]), 'embedding': embedding_summary([[1, 0], [0, 1]])})


def test_baseline_determinism_and_tampering():
    assert artifact() == artifact()
    a = artifact()
    a['payload']['features']['query_count'] = 3
    with pytest.raises(ValueError):
        validate_artifact(a)


@pytest.mark.parametrize('mutation', [lambda p: p['provenance'].update(split='test'),
    lambda p: p['features'].update(query_count=-1), lambda p: p['features']['class_counts'].update(a=-1),
    lambda p: p['embedding'].update(centroid=[1]), lambda p: p['embedding'].update(count=1),
    lambda p: p['features']['characters'].update(mean=None), lambda p: p.pop('features')])
def test_malformed_even_with_recomputed_checksum(mutation):
    a = artifact()
    mutation(a['payload'])
    a['sha256'] = digest(a['payload'])
    with pytest.raises(ValueError):
        validate_artifact(a)


def test_empty_traffic_and_scenarios():
    assert query_features([])['query_count'] == 0
    assert distribution([])['mean'] is None
    assert all(not rows for rows in scenarios([], ' extra')[0].values())
    qs = [SimpleNamespace(query_id=2, query='two words', query_class='b'), SimpleNamespace(query_id=1, query='one word', query_class='a')]
    first, dominant = scenarios(qs, ' extra')
    assert first == scenarios(list(reversed(qs)), ' extra')[0]
    assert dominant == 'a'
    assert first['category_mix'] == [('one word', 'a')]*2
    assert first['short_generic'] == [('one', 'a'), ('two', 'b')]


def test_ranking_difference_with_missing_hits():
    row = ranking_difference([1, 2, 3], [2, 1, 4], k=3)
    assert row == {'ordering_changed': True, 'overlap_at_k': 2/3, 'shared_count': 2, 'shared_mean_rank_displacement': 1}
    assert ranking_difference([], [])['shared_mean_rank_displacement'] is None
    with pytest.raises(ValueError):
        ranking_difference([1, 1], [1, 2])


def test_final_test_boundary_without_reading_actual_labels(tmp_path):
    # Synthetic paths only; check raises before opening anything.
    guard = BoundaryGuard(tmp_path)
    with pytest.raises(PermissionError):
        guard.check('open', (str(tmp_path/'data/processed/judgments.test.jsonl'),))
    with pytest.raises(ValueError):
        load_development(tmp_path, 'test')
