"""Network-disabled clean application rebuild in the actual project image.

Only application source and pyproject are mounted, both read-only. Dependencies
are reused from the built image; no Dockerfile or dependency lock is changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
from time import monotonic
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'production-retrieval-ranking:phase8-local'
CODE = r'''
import hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
source=Path('/tmp/source');source.mkdir()
shutil.copytree('/input/src',source/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
shutil.copyfile('/input/pyproject.toml',source/'pyproject.toml')
result=subprocess.run([sys.executable,'-m','pip','wheel','--no-build-isolation','--no-deps','--no-cache-dir','--wheel-dir','/tmp/wheels',str(source)],capture_output=True,text=True)
assert result.returncode==0,result.stdout+result.stderr
wheel=next(Path('/tmp/wheels').glob('*.whl'))
import product_search
installed=Path(product_search.__file__).parent.parent
with zipfile.ZipFile(wheel) as z:
    members={name:hashlib.sha256(z.read(name)).hexdigest() for name in sorted(z.namelist())}
    application=[name for name in z.namelist() if name.startswith('product_search/') and name.endswith('.py')]
    assert application and all((installed/name).read_bytes()==z.read(name) for name in application)
    metadata=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
    assert (installed/metadata).read_bytes()==z.read(metadata)
    metadata_lf_sha256=hashlib.sha256(z.read(metadata).replace(b'\r\n',b'\n')).hexdigest()
print(json.dumps(dict(passed=True,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),members=members,application_files=len(application),application_matches_installed_image=True,metadata_matches_installed_image=True,metadata_lf_sha256=metadata_lf_sha256,build_output=result.stdout+result.stderr)))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():p.error('Use a fresh report path')
    started=monotonic()
    command=['docker','run','--rm','--network','none','--read-only','--tmpfs','/tmp:size=128m,mode=1777',
        '--mount',f'type=bind,source={ROOT / "src"},target=/input/src,readonly',
        '--mount',f'type=bind,source={ROOT / "pyproject.toml"},target=/input/pyproject.toml,readonly',
        '-e','SOURCE_DATE_EPOCH=1790599546','--entrypoint','python',IMAGE,'-c',CODE]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
    report={'passed':False,'duration_seconds':monotonic()-started,'exit_code':result.returncode,
        'kind':'fresh application wheel in project image; no pip cache, network disabled, no dependencies installed',
        'full_docker_no_cache_build':False,'source_date_epoch':1790599546}
    if result.returncode==0:
        report.update(json.loads(result.stdout))
        reference=next((ROOT/'reports/tmp/phase8-repro-wheel1').glob('*.whl'))
        with zipfile.ZipFile(reference) as z:
            expected={name:hashlib.sha256(z.read(name)).hexdigest() for name in sorted(z.namelist())}
            metadata=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
            metadata_lf_sha256=hashlib.sha256(z.read(metadata).replace(b'\r\n',b'\n')).hexdigest()
        report['reference_wheel_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
        report['reference_member_payload_equal']=report['members']==expected
        report['reference_archive_equal']=report['wheel_sha256']==report['reference_wheel_sha256']
        differing=sorted(n for n in set(expected)|set(report['members']) if expected.get(n)!=report['members'].get(n))
        report['reference_differing_members']=differing
        report['reference_metadata_newline_normalized_equal']=metadata_lf_sha256==report['metadata_lf_sha256']
        # Windows setuptools writes CRLF metadata, Linux writes LF. RECORD changes
        # because it records that byte hash. Every application member must match.
        allowed={metadata,metadata.replace('/METADATA','/RECORD')}
        report['passed']=report['passed'] and set(differing)<=allowed and report['reference_metadata_newline_normalized_equal']
    else:report['failure']=result.stdout+result.stderr
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:report[k] for k in ('passed','duration_seconds','exit_code')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
