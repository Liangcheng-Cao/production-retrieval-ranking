"""Run full tests against installed image code using an ephemeral test overlay.

Mount only tests, scripts, configs and deployment support. No dataset/model payload
is mounted. Empty data/artifact/report directories satisfy repository-shape tests.
"""
import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import shutil,subprocess,sys
from pathlib import Path
work=Path('/tmp/check')
shutil.copytree('/input',work,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
for name in ('data/raw','data/processed','artifacts','reports'):(work/name).mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','--require-hashes','--no-deps','--no-cache-dir','--target','/tmp/test-deps','-r',str(work/'deployment/linux-test.lock')],check=True)
subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider'],cwd=work,check=True)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--log',type=Path,required=True)
    args=p.parse_args()
    if args.log.exists():p.error('Use a fresh log path')
    command=['docker','run','--rm','--read-only','--tmpfs','/tmp:size=256m,mode=1777',
             '--env','PYTHONPATH=/tmp/test-deps','--entrypoint','python']
    for name in ('tests','scripts','configs','deployment','pyproject.toml'):
        command+=['--mount',f'type=bind,source={ROOT/name},target=/input/{name},readonly']
    command+=['production-retrieval-ranking:phase8-local','-c',CODE]
    args.log.parent.mkdir(parents=True,exist_ok=True)
    with args.log.open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,timeout=180)
    print(f'Image full tests exit code: {result.returncode}')
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
