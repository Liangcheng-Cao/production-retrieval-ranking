"""Installed-wheel startup faults on separate inference bundles, no source mutation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from product_search.deployment import verify_bundle
from product_search.development_data import BoundaryGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        p.error('Use fresh negative evidence directory')
    guard = BoundaryGuard(bundle).install()
    inventory = verify_bundle(bundle)
    output.mkdir(parents=True)
    config = json.loads((bundle/'runtime.json').read_text())
    model = next(k for k in inventory['files'] if k.endswith('.safetensors'))
    rows = []
    for case in ('missing_manifest', 'corrupt_manifest', 'missing_model'):
        root = output/case
        root.mkdir()
        if case != 'missing_manifest':
            for name in inventory['files']:
                if case == 'missing_model' and name == model:
                    continue
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == config['manifest'] and case == 'corrupt_manifest':
                    target.write_text('{}', encoding='utf-8')
                else:
                    # Read-only use of hard links; no existing target is ever mutated.
                    try:
                        os.link(bundle/name, target)
                    except OSError:
                        shutil.copyfile(bundle/name, target)
        (root/'runtime.json').write_text(json.dumps(config), encoding='utf-8')
        with (output/(case+'.log')).open('w', encoding='utf-8') as log:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-m', 'product_search.deployment', '--config', str(root/'runtime.json'),
                '--host', '127.0.0.1', '--port', '0'], cwd=root, env=env, stdout=log, stderr=log, timeout=90)
        text = (output/(case+'.log')).read_text(encoding='utf-8')
        assert result.returncode != 0 and 'Application startup failed' in text
        assert '"event": "service_ready"' not in text
        assert not (root/model).exists() if case == 'missing_model' else True
        rows.append({'case': case, 'exit_code': result.returncode, 'startup_rejected': True, 'no_ready_event': True, 'no_silent_repair': True})
    verify_bundle(bundle)
    report = {'passed': True, 'cases': rows, 'original_bundle_unchanged': True,
        'boundary': {'opened': sorted(guard.opened), 'blocked': guard.blocked, 'final_test_labels_accessed': False}}
    (output/'checks.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
