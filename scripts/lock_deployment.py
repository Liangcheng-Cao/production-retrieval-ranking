"""Convert a reviewed pip resolution report to target-specific wheel/hash locks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlparse
from product_search.data.io import file_hash

HOSTS = {'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org', 'pypi.nvidia.com'}


def lock_rows(report):
    rows = []
    for record in report['install']:
        name = record['metadata']['name']
        if name == 'production-retrieval-ranking':
            continue
        download = record['download_info']
        url = download['url']
        sha = download['archive_info']['hashes']['sha256']
        if urlparse(url).scheme != 'https' or urlparse(url).hostname not in HOSTS or not urlparse(url).path.endswith('.whl'):
            raise ValueError('Expected an official HTTPS wheel source')
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid wheel SHA256')
        rows.append((name.lower().replace('_', '-'), f'{name} @ {url} --hash=sha256:{sha}', record['metadata']['version']))
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError('Duplicate resolved package')
    return sorted(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resolution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.report.exists():
        p.error('Use fresh lock/evidence paths; do not silently replace a release lock')
    source = json.loads(args.resolution.read_text(encoding='utf-8'))
    rows = lock_rows(source)
    args.output.write_text('# Target-specific complete pip closure; use --require-hashes --no-deps.\n'+
        '# Linux x86_64, CPython 3.14, CUDA 13.0; runtime model artifacts are separate.\n'+
        '\n'.join(r[1] for r in rows)+'\n', encoding='utf-8', newline='\n')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'resolver_environment': source['environment'], 'resolution_sha256': file_hash(args.resolution),
        'lock_sha256': file_hash(args.output), 'packages': {r[0]: r[2] for r in rows}}, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(f'Locked {len(rows)} wheels with hashes')


if __name__ == '__main__':
    main()
