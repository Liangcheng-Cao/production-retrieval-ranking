"""Copy only the sealed inference closure to an ignored portable runtime directory."""
import argparse
import json
from pathlib import Path
from product_search.development_data import BoundaryGuard
from product_search.runtime_config import RuntimeConfig
from product_search.deployment import package_runtime
from product_search.data.io import file_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/runtime.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Refusing to overwrite bundle evidence')
    config = RuntimeConfig.load(args.config)
    guard = BoundaryGuard(config.root).install()
    manifest = package_runtime(config, args.output)
    report = {'passed': True, 'file_count': len(manifest['files']), 'payload_bytes': sum(v['bytes'] for v in manifest['files'].values()),
        'bundle_manifest_sha256': file_hash(args.output/'bundle.json'), 'runtime_manifest_sha256': config.manifest_sha256,
        'inventory': manifest, 'boundary': {'opened': sorted(guard.opened), 'blocked': guard.blocked, 'final_test_labels_accessed': False}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: report[k] for k in ('passed', 'file_count', 'payload_bytes', 'bundle_manifest_sha256')}))


if __name__ == '__main__':
    main()
