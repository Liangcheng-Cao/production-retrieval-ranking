"""Small offline aggregate feature contract. No model loading or tuning."""
from collections import Counter
import hashlib
import json
import math
import numpy as np
from scipy.stats import wasserstein_distance

VERSION = 'monitoring-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def distribution(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite scalar observations')
    return {'count': len(x), 'mean': float(x.mean()) if len(x) else None,
            'p05': float(np.quantile(x, .05)) if len(x) else None,
            'p50': float(np.quantile(x, .5)) if len(x) else None,
            'p95': float(np.quantile(x, .95)) if len(x) else None}


def scalar_distance(a, b):
    # Empty populations have undefined drift, not evidence of zero drift.
    distribution(a)
    distribution(b)
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else None


def total_variation(a, b):
    for counts in (a, b):
        if any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError('Expected nonnegative integer counts')
    na, nb = sum(a.values()), sum(b.values())
    return sum(abs(a.get(k, 0)/na-b.get(k, 0)/nb) for k in a.keys() | b.keys())/2 if na and nb else None


def embedding_summary(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError('Expected finite embedding matrix')
    if not len(x):
        return {'count': 0, 'dimension': x.shape[1], 'mean_norm': None, 'centroid': None}
    if (np.linalg.norm(x, axis=1) == 0).any():
        raise ValueError('Zero embeddings are invalid')
    return {'count': len(x), 'dimension': x.shape[1], 'mean_norm': float(np.linalg.norm(x, axis=1).mean()),
            'centroid': x.mean(axis=0).tolist()}


def centroid_distance(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid centroids')
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(1-np.dot(a, b)/norm, 0, 2)) if norm else None


def query_features(rows):
    lengths = [len(text) for text, _ in rows]
    tokens = [len(text.split()) for text, _ in rows]
    return {'query_count': len(rows), 'characters': distribution(lengths), 'tokens': distribution(tokens),
            'character_counts': dict(sorted(Counter(map(str, lengths)).items())),
            'token_counts': dict(sorted(Counter(map(str, tokens)).items())),
            'class_counts': dict(sorted(Counter(label for _, label in rows).items()))}


def scenarios(queries, suffix):
    ordered = sorted(queries, key=lambda q: q.query_id)
    original = [(q.query, q.query_class) for q in ordered]
    if not ordered:
        return {'baseline': [], 'long_modifiers': [], 'category_mix': [], 'short_generic': []}, None
    counts = Counter(q.query_class for q in ordered)
    dominant = sorted(counts, key=lambda k: (-counts[k], k))[0]
    selected = [row for row in original if row[1] == dominant]
    # Same population size, deterministic cycling; repetitions are intentional.
    return {'baseline': original, 'long_modifiers': [(text+suffix, cls) for text, cls in original],
            'category_mix': [selected[i % len(selected)] for i in range(len(original))],
            'short_generic': [(text.split()[0], cls) for text, cls in original]}, dominant


def ranking_difference(champion, challenger, k=10):
    if type(k) is not int or k < 1 or len(champion) != len(set(champion)) or len(challenger) != len(set(challenger)):
        raise ValueError('Invalid ranking comparison')
    a, b = champion[:k], challenger[:k]
    shared = set(a) & set(b)
    return {'ordering_changed': a != b, 'overlap_at_k': len(shared)/k,
            'shared_count': len(shared), 'shared_mean_rank_displacement':
            sum(abs(a.index(pid)-b.index(pid)) for pid in shared)/len(shared) if shared else None}


def envelope(payload):
    result = {'payload': payload, 'sha256': digest(payload)}
    validate_artifact(result)
    return result


def validate_artifact(value):
    """Reject changed hashes, wrong split/version, and malformed aggregate dimensions."""
    try:
        p = value['payload']
        if set(value) != {'payload', 'sha256'} or value['sha256'] != digest(p):
            raise ValueError('Monitoring checksum mismatch')
        if p['version'] != VERSION or p['provenance']['split'] != 'train':
            raise ValueError('Monitoring baseline must use frozen train only')
        n = p['features']['query_count']
        if type(n) is not int or n < 0 or p['provenance']['query_count'] != n:
            raise ValueError('Invalid population count')
        for name in ('character_counts', 'token_counts', 'class_counts'):
            counts = p['features'][name]
            if not isinstance(counts, dict) or any(not isinstance(k, str) for k in counts) or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) != n:
                raise ValueError('Invalid feature counts')
        for name in ('characters', 'tokens'):
            summary = p['features'][name]
            if set(summary) != {'count', 'mean', 'p05', 'p50', 'p95'} or summary['count'] != n or any((v is not None if n == 0 else type(v) not in (int, float) or not math.isfinite(v) or v < 0) for k, v in summary.items() if k != 'count'):
                raise ValueError('Invalid scalar summary')
            counts = p['features']['character_counts' if name == 'characters' else 'token_counts']
            if any(not k.isascii() or not k.isdecimal() or str(int(k)) != k for k in counts):
                raise ValueError('Invalid scalar histogram keys')
            # Weighted moments without expanding potentially large populations.
            expected_mean = sum(int(k)*v for k, v in counts.items())/n if n else None
            if summary['mean'] != expected_mean:
                raise ValueError('Inconsistent scalar mean')
        e = p['embedding']
        if e['count'] != n or type(e['dimension']) is not int or e['dimension'] < 1:
            raise ValueError('Invalid embedding dimensions')
        if n and (len(e['centroid']) != e['dimension'] or not np.isfinite(e['centroid']).all() or not math.isfinite(e['mean_norm']) or e['mean_norm'] <= 0):
            raise ValueError('Invalid embedding summary')
        if not n and (e['centroid'] is not None or e['mean_norm'] is not None):
            raise ValueError('Invalid empty embedding summary')
    except (KeyError, TypeError, OverflowError, AttributeError) as exc:
        raise ValueError('Malformed monitoring artifact') from exc
    return value
