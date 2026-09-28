"""Small benchmark accounting helpers; no model or serving behavior changes."""
import math
import random
import numpy as np


def query_schedule(query_ids, count, seed):
    if not query_ids or type(count) is not int or count <= 0:
        raise ValueError('Nonempty queries and positive sample count required')
    if len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate fixture IDs')
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        block = list(query_ids)
        rng.shuffle(block)
        result.extend(block)
    return result[:count]


def distribution(values):
    values = list(values)
    if not values:
        return {'count': 0, 'p50_ms': None, 'p95_ms': None, 'p99_ms': None, 'max_ms': None, 'mean_ms': None}
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Timings must be finite and nonnegative')
    return {'count': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99)),
            'max_ms': float(max(values)), 'mean_ms': float(np.mean(values))}


def summarize_requests(rows, elapsed_seconds):
    if not rows or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError('Nonempty observations and positive elapsed time required')
    success = [r for r in rows if r['status'] == 200 and not r.get('error')]
    result = {'attempts': len(rows), 'successful_responses': len(success),
              'errors': len(rows)-len(success), 'fallbacks': sum(bool(r.get('fallback_used')) for r in success),
              'ranking_mismatches': sum(not r.get('parity_passed', False) for r in success),
              'elapsed_seconds': elapsed_seconds, 'completed_requests_per_second': len(rows)/elapsed_seconds,
              'successful_requests_per_second': len(success)/elapsed_seconds,
              'error_rate': (len(rows)-len(success))/len(rows),
              'client_all_attempts': distribution(r['client_ms'] for r in rows),
              'client_success': distribution(r['client_ms'] for r in success)}
    for name in ('engine_ms', 'http_app_ms', 'serialization_ms', 'server_non_engine_ms'):
        result[name] = distribution(r[name] for r in success if r.get(name) is not None)
    return result
