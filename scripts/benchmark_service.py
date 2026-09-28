"""Benchmark-only child process, unchanged HTTP app/core; graceful stdin shutdown."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import faulthandler
import logging
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from product_search.development_data import BoundaryGuard
GUARD = BoundaryGuard(ROOT).install()
from product_search.benchmarking import distribution, query_schedule
from product_search.search_engine import SearchEngine
from product_search.runtime_config import RuntimeConfig
from product_search.runtime_contracts import plain
from product_search.api import create_app
from validate_runtime import fixture_queries, resource_snapshot


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--run', type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    plan = json.loads((ROOT/'configs/phase6_plan.json').read_text())
    import torch
    import uvicorn
    # Initialize native dependency libraries before threadpool limits/owner thread.
    # No model or index is constructed here; imports are excluded from startup timing.
    from product_search.retrieval.lexical import BM25Retriever
    from sentence_transformers import SentenceTransformer
    from threadpoolctl import threadpool_limits
    torch.manual_seed(plan['seed'])
    torch.set_num_threads(plan['torch_threads'])
    torch.use_deterministic_algorithms(plan['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = plan['allow_tf32']
    torch.backends.cudnn.allow_tf32 = plan['allow_tf32']
    queries = fixture_queries(plan['query_count'])
    held = []
    class MeasuredEngine(SearchEngine):
        def initialize(self):
            start = perf_counter()
            super().initialize()
            startup_wall = (perf_counter()-start)*1000
            baseline = resource_snapshot(self)
            expected = {}
            for qid, text in queries.items():
                for mode in plan['pipelines']:
                    result = self.search(text, plan['top_k'], mode)
                    assert not result.fallback_used
                    expected[f'{qid}:{mode}'] = {'ids': [h.product_id for h in result.results],
                                                'scores': [h.final_score for h in result.results]}
            schedule = query_schedule(list(queries), plan['requests_per_case'], plan['seed'])
            core = {}
            rows = []
            for mode in plan['pipelines']:
                for qid in schedule[:plan['warmup_requests']]:
                    self.search(queries[qid], plan['top_k'], mode)
                observations = []
                for qid in schedule:
                    start = perf_counter()
                    result = self.search(queries[qid], plan['top_k'], mode)
                    observations.append({'query_id': qid, 'wall_ms': (perf_counter()-start)*1000, **plain(result.timing)})
                    assert [h.product_id for h in result.results] == expected[f'{qid}:{mode}']['ids']
                core[mode] = {key: distribution(r[key] for r in observations)
                              for key in observations[0] if key.endswith('_ms')}
                rows.extend({'pipeline': mode, **r} for r in observations)
            dump(args.directory/'core_raw.json', rows)
            dump(args.directory/'engine.json', {'run': args.run, 'startup_wall_ms': startup_wall,
                'startup_stages': plain(self.startup_timing), 'version': plain(self.version),
                'resources_after_startup': baseline, 'resources_after_core': resource_snapshot(self),
                'expected': expected, 'core': core, 'queries': queries})
            return self
    def factory():
        e = MeasuredEngine(RuntimeConfig.load(ROOT/'configs/runtime.json'))
        held.append(e)
        return e
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    dump(args.directory/'address.json', {'port': port, 'pid': os.getpid()})
    server = uvicorn.Server(uvicorn.Config(create_app(engine_factory=factory), log_level='warning', access_log=False))
    def stop_listener():
        sys.stdin.readline()
        server.should_exit = True
    threading.Thread(target=stop_listener, daemon=True).start()
    try:
        with threadpool_limits(limits=plan['blas_threads']):
            server.run(sockets=[sock])
    finally:
        sock.close()
        dump(args.directory/'shutdown.json', {'state': held[0].readiness().state if held else 'FAILED',
            'opened_canonical_files': sorted(GUARD.opened), 'blocked_attempts': GUARD.blocked,
            'final_test_labels_accessed': False})


if __name__ == '__main__':
    main()
