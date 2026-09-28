"""Release checks and compact evidence; never reads final-test relevance labels."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
from product_search.development_data import BoundaryGuard
GUARD = BoundaryGuard(ROOT).install()
from product_search.data.io import file_hash
from product_search.monitoring import digest, validate_artifact


def command(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Check failed: {args}: {result.stdout} {result.stderr}')
    return result.stdout.strip()


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def dump(path, value):
    (ROOT/path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def main():
    phase6 = '77ec3acd58f7ec80741e3472b6763233b2e82dd3'
    assert command('git', 'rev-parse', 'HEAD') == phase6, 'Phase7 must remain uncommitted'
    tracked = command('git', 'ls-files').splitlines()
    protected = [p for p in tracked if p.startswith(('data/', 'artifacts/', 'configs/', 'reports/phase'))]
    protected += [p for p in tracked if p.startswith(('src/product_search/retrieval/', 'src/product_search/data/'))]
    protected += ['src/product_search/search_engine.py', 'src/product_search/reranking.py', 'src/product_search/runtime_config.py', 'src/product_search/runtime_artifacts.py', 'src/product_search/development_data.py']
    command('git', 'diff', '--exit-code', phase6, '--', *protected)
    for path, sha in {
        'data/processed/data_manifest.json': '0b79ce2061ada9574885b952e63f06c9516aa38682b1e91ba1f4334a5d6eb845',
        'artifacts/phase2/manifest.json': '197cf3325df537f10ce917ac604984ff500d2202719626b1aad8e432cbae34b3',
        'artifacts/phase3/manifest.json': 'fef7ff4e731b5bddd6beb83c611216191849745db663fabdce3de1ecad2cc4fb',
        'artifacts/phase4/manifest.json': '7adf8fcaace380db531b60765678da1dbe4461498e71e313c318cf82d52009b0',
    }.items():
        assert file_hash(ROOT/path) == sha
    prospective = command('git', 'ls-files', '--cached', '--others', '--exclude-standard').splitlines()
    bad = [p for p in prospective if (p.startswith(('data/raw/', '.venv/', 'reports/tmp/', '.cache/')) and not p.endswith('.gitkeep'))
        or Path(p).suffix in ('.npy', '.npz', '.pt', '.pth', '.safetensors', '.index', '.faiss', '.pkl', '.joblib', '.jsonl')
        or (ROOT/p).stat().st_size > 1_000_000]
    assert not bad, bad
    ignored_examples = ['data/raw/WANDS/label.csv', 'artifacts/phase2/dense/embeddings.npy', 'artifacts/phase2/bm25/index.bin',
        '.cache/huggingface/hub/model.safetensors', '.venv/Scripts/python.exe', 'reports/tmp/phase7-service.log', 'reports/tmp/phase6-full/run-1/core_raw.json']
    ignored = command('git', 'check-ignore', *ignored_examples).splitlines()
    assert set(ignored) == set(ignored_examples)
    baseline = validate_artifact(read('reports/phase7/baseline.json'))
    monitoring = read('reports/phase7/monitoring.json')
    assert digest(monitoring['payload']) == monitoring['sha256']
    execution = monitoring['execution']
    assert execution['deterministic_payload_equal'] and execution['first_sha256'] == execution['second_sha256'] == monitoring['sha256']
    for path, sha in baseline['payload']['provenance']['source_sha256'].items():
        assert file_hash(ROOT/path) == sha
    obs = read('reports/phase7/observability_verified.json')
    api = read('reports/tmp/phase7-http-parity.json')
    core = read('reports/tmp/phase7-runtime-parity.json')
    for value in (obs, api, core):
        assert value['passed'] and value['boundary']['final_test_labels_accessed'] is False
        assert not value['boundary'].get('blocked', value['boundary'].get('blocked_attempts', []))
    assert not execution['boundary']['final_test_labels_accessed'] and not execution['boundary']['blocked']
    assert api['parity']['all_passed'] and core['parity']['all_passed'] and api['startup_failure']['server_started'] is False
    compact = {'passed': True, 'api_comparisons': api['parity']['comparison_count'],
        'api_all_passed': True, 'core_comparisons': len(core['parity']['comparisons']), 'core_all_passed': True,
        'api_max_score_abs_error': max(v for row in api['parity']['comparisons'] for v in row['max_score_abs_errors'].values()),
        'core_max_score_abs_error': max(row['max_score_abs_error'] for row in core['parity']['comparisons']),
        'core_method': core['parity']['method'], 'historical_batch_context_checked': core['parity']['historical_phase3_reproduced_with_original_batch_context'],
        'startup_failure': api['startup_failure'], 'api_boundary': api['boundary'], 'core_boundary': core['boundary'],
        'raw_evidence_sha256': {p: file_hash(ROOT/p) for p in ('reports/tmp/phase7-http-parity.json', 'reports/tmp/phase7-runtime-parity.json')},
        'retention': 'Detailed small comparison fixtures are ignored; only aggregates and file checksums retained here', 'verdict': 'NO ranking change'}
    dump('reports/phase7/parity.json', compact)
    tests = command(sys.executable, '-m', 'pytest', '-q')
    pip = command(sys.executable, '-m', 'pip', 'check')
    command('git', 'diff', '--check')
    report = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(), 'head': phase6,
        'pytest': {'command': '.\\.venv\\Scripts\\python.exe -m pytest -q', 'output': tests}, 'pip_check': pip,
        'diff_check': 'passed', 'frozen_tracked_files_unchanged': protected, 'ignored_examples': ignored,
        'large_or_raw_prospective_files': bad, 'baseline_sha256': baseline['sha256'], 'replay_sha256': monitoring['sha256'],
        'deterministic_rebuild_equal': True, 'parity': compact['verdict'],
        'observability_verified': True, 'retained_failed_observability_run': {'path': 'reports/phase7/observability_integration.json',
            'reason': 'Logging harness write-mode FileHandler closed by Uvicorn dictConfig; metrics and queue checks passed but event check failed. Fixed append-mode handler, independent rerun passed.'},
        'boundary': {'opened_by_release_check': sorted(GUARD.opened), 'blocked': GUARD.blocked, 'final_test_labels_accessed': False},
        'phase7_committed': False, 'pushed': False}
    dump('reports/phase7/checks.json', report)
    print(json.dumps({'passed': True, 'pytest': tests.splitlines()[-1], 'pip_check': pip, 'parity': compact['verdict']}))


if __name__ == '__main__':
    main()
