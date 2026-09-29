"""Ignored train-only parity fixture; source runtime reference, no relevance labels."""
import argparse
import json
from pathlib import Path
from product_search.data.io import file_hash
from product_search.deployment import configure_process
from product_search.search_engine import SearchEngine
from validate_runtime import fixture_queries, GUARD, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((ROOT/'reports/tmp').resolve()):
        p.error('Fixture must be new and under ignored reports/tmp')
    configure_process()
    from threadpoolctl import threadpool_limits
    rows = []
    with threadpool_limits(limits=1):
        engine = SearchEngine.from_config(ROOT/'configs/runtime.json')
        try:
            for qid, query in fixture_queries().items():
                for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
                    result = engine.search(query, 10, mode)
                    assert not result.fallback_used
                    rows.append({'query_id': qid, 'request': {'query': query, 'pipeline': mode, 'top_k': 10},
                        'expected': {'ids': [h.product_id for h in result.results], 'scores': [h.final_score for h in result.results]}})
        finally:
            engine.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'partition': 'train', 'runtime_manifest_sha256': file_hash(ROOT/'artifacts/phase4/manifest.json'),
        'requests': rows, 'boundary': {'opened': sorted(GUARD.opened), 'blocked': GUARD.blocked, 'final_test_labels_accessed': False}}, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'requests': len(rows), 'sha256': file_hash(args.output), 'final_test_labels_accessed': False}))


if __name__ == '__main__':
    main()
