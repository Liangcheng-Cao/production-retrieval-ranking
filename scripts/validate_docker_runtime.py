"""Real Docker release checks; frozen train fixture only, compact retained evidence.

Build the image first. This command owns only its uniquely named Compose project
and fault containers. Raw responses/logs remain under ignored reports/tmp.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'production-retrieval-ranking:phase8-local'
from product_search.development_data import BoundaryGuard
from product_search.deployment import configure_process, verify_bundle
from product_search.data.io import file_hash


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def command(*args, check=True, timeout=180):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{args[:3]} exited {result.returncode}: {result.stderr[-2000:]}')
    return result


def inspect(cid):
    return json.loads(command('docker', 'inspect', cid).stdout)[0]


def metric_count(text):
    return sum(float(line.rsplit(' ', 1)[1]) for line in text.splitlines() if line.startswith('request_count_total{'))


def compare(got, want, pipeline):
    import numpy as np
    ids = [h['product_id'] for h in got['results']]
    expected_ids = [h['product_id'] for h in want['results']]
    scores = [h['final_score'] for h in got['results']]
    expected_scores = [h['final_score'] for h in want['results']]
    same_size = len(scores) == len(expected_scores)
    error = max((abs(a-b) for a, b in zip(scores, expected_scores)), default=0)
    tol = 1e-4 if pipeline == 'hybrid_rerank' else 1e-6
    fallback = (got['fallback_used'], got['fallback_reason'], got['effective_pipeline']) == (want['fallback_used'], want['fallback_reason'], want['effective_pipeline'])
    passed = same_size and ids == expected_ids and fallback and np.allclose(scores, expected_scores, atol=tol, rtol=tol)
    return {'passed': bool(passed), 'count_equal': same_size, 'ordering_equal': ids == expected_ids,
            'max_score_abs_error': error, 'fallback_equal': fallback, 'pipeline': pipeline}


def exercise(url, fixture, smoke):
    import httpx
    import numpy as np
    with httpx.Client(base_url=url, timeout=60, trust_env=False) as client:
        probes = {p: client.get(p) for p in ('/health', '/ready', '/version', '/metrics')}
        assert all(r.status_code == 200 for r in probes.values())
        assert probes['/ready'].json()['state'] == 'READY'
        initial = metric_count(probes['/metrics'].text)
        assert initial == 0, 'New service must have clean metrics'
        outputs = []
        for row in fixture['requests']:
            r = client.post('/search', json=row['request'])
            assert r.status_code == 200
            got = r.json()
            assert not got['fallback_used']
            assert got['effective_pipeline'] == row['request']['pipeline']
            outputs.append(got)
        invalid = []
        for payload in ({'query': '', 'pipeline': 'bm25'}, {'query': 'synthetic', 'pipeline': 'unknown'},
                        {'query': 'synthetic', 'pipeline': 'hybrid_rerank', 'top_k': 21}):
            status = client.post('/search', json=payload).status_code
            assert status == 422
            invalid.append(status)
        timing = {}
        if smoke:
            for mode in ('bm25', 'hybrid', 'hybrid_rerank'):
                rows = [r for r in fixture['requests'] if r['request']['pipeline'] == mode]
                # Parity calls already warmed all pipelines. Twelve timed requests
                # per pipeline, sequential and diagnostic, not a load benchmark.
                values = []
                for row in rows*2:
                    start = monotonic()
                    response = client.post('/search', json=row['request'])
                    values.append((monotonic()-start)*1000)
                    assert response.status_code == 200 and not response.json()['fallback_used']
                timing[mode] = {'requests': len(values), 'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95))}
        metrics = client.get('/metrics').text
        assert all(r['request']['query'] not in metrics for r in fixture['requests'])
        final = metric_count(metrics)
        assert final == len(outputs)+len(invalid)+(sum(r['requests'] for r in timing.values()))
        return outputs, {'probes': {p: r.status_code for p, r in probes.items()}, 'ready': probes['/ready'].json(),
            'version': probes['/version'].json(), 'initial_request_count': initial, 'final_request_count': final,
            'validation_errors': invalid, 'performance_smoke': timing, 'metrics_query_privacy': True}


AUDIT = r'''
import os,json,importlib.util
from pathlib import Path
bad=[];count=0
for base,dirs,files in os.walk('/'):
    if base=='/': dirs[:]=[d for d in dirs if d not in ('proc','sys','dev')]
    for name in files:
        p=Path(base)/name;count+=1;s=str(p).lower()
        if any(x in s for x in ('/.venv/','/.ssh/','/.git/','/.cache/huggingface/','/data/raw/','/reports/tmp/')) or name in ('.git-credentials','id_rsa','id_ed25519') or name.startswith('judgments.') or name.endswith('.safetensors'):
            bad.append(str(p))
spec=importlib.util.find_spec('product_search')
app=Path(spec.origin).parent
windows_paths=[]
for p in app.rglob('*.py'):
    if 'C:\\Users\\' in p.read_text() or 'C:/Users/' in p.read_text(): windows_paths.append(str(p))
secret_env=[k for k in os.environ if k.upper() in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','AWS_SECRET_ACCESS_KEY','GITHUB_TOKEN')]
print(json.dumps(dict(file_count=count,forbidden_paths=bad,windows_home_paths=windows_paths,secret_environment_keys=secret_env,
application_package=str(app),healthcheck_present=Path('/app/healthcheck.py').is_file(),runtime_payload_baked=Path('/runtime/runtime.json').exists(),uid=os.getuid())))
'''

ENVIRONMENT = r'''
import json,sys,os,torch,transformers,sentence_transformers
from pathlib import Path
assert torch.cuda.is_available()
a=torch.tensor([1.,2.,3.],device='cuda');v=float((a*a).sum().item());assert v==14
print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0),cuda_tensor_result=v,transformers=transformers.__version__,sentence_transformers=sentence_transformers.__version__,uid=os.getuid(),runtime_config_present=Path('/runtime/runtime.json').is_file(),bundle_manifest_present=Path('/runtime/bundle.json').is_file(),probe_process_allocated_bytes=torch.cuda.memory_allocated(),probe_process_reserved_bytes=torch.cuda.memory_reserved())))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists(): parser.error('Use a fresh output directory to retain failed evidence')
    out.mkdir(parents=True)
    guard = BoundaryGuard(ROOT).install()
    bundle = ROOT/'reports/tmp/phase8-bundle'
    verify_bundle(bundle)
    fixture_path = ROOT/'reports/tmp/phase8-train-fixture.json'
    fixture = json.loads(fixture_path.read_text())
    train = set(json.loads((bundle/'data/processed/data_manifest.json').read_text())['query_ids']['train'])
    assert fixture['partition'] == 'train' and all(r['query_id'] in train for r in fixture['requests'])
    os.environ['SEARCH_BUNDLE'] = str(bundle)
    project = 'phase8-'+out.name.lower().replace('_','-')
    compose = ['docker','compose','-p',project]
    owned = []
    report = {'passed': False, 'image': IMAGE, 'fixture_sha256': file_hash(fixture_path),
              'bundle_sha256': file_hash(bundle/'bundle.json'), 'starts': [], 'failures': []}
    try:
        command(*compose, 'config','--quiet')
        meta = json.loads(command('docker','image','inspect',IMAGE).stdout)[0]
        report['image_id'] = meta['Id']
        report['image_bytes'] = meta['Size']
        report['image_config'] = {k:meta['Config'][k] for k in ('User','Entrypoint','WorkingDir','Healthcheck')}
        history = command('docker','history','--no-trunc',IMAGE).stdout
        (out/'image-history.log').write_text(history,encoding='utf-8')
        audit = json.loads(command('docker','run','--rm','--network','none','--read-only','--entrypoint','python',IMAGE,'-c',AUDIT).stdout)
        report['image_audit'] = audit
        assert not audit['forbidden_paths'] and not audit['windows_home_paths'] and not audit['secret_environment_keys']
        assert audit['healthcheck_present'] and not audit['runtime_payload_baked'] and audit['uid']==10001
        report['image_audit'] = dict(audit, passed=True, scope='actual final image filesystem plus retained layer history; runtime config/provenance supplied by read-only bundle')
        print('Image filesystem audit passed', flush=True)
        configure_process()
        from threadpoolctl import threadpool_limits
        from product_search.api import create_app
        from validate_api import start, stop
        handler = logging.FileHandler(out/'native.log',encoding='utf-8')
        logging.getLogger('product_search.service').addHandler(handler)
        with threadpool_limits(limits=1):
            native = start(create_app(ROOT/'configs/runtime.json'))
            try:
                native_outputs, report['native'] = exercise(native[3], fixture, True)
            finally:
                stop(*native[:3])
        handler.close()
        logging.getLogger('product_search.service').removeHandler(handler)
        dump(out/'native-responses.json', native_outputs)
        print('Native HTTP reference and diagnostic smoke completed', flush=True)
        for row,got in zip(fixture['requests'],native_outputs):
            assert [h['product_id'] for h in got['results']]==row['expected']['ids']
        for run in (1,2):
            started = monotonic()
            command(*compose,'up','-d','--no-build','--force-recreate')
            cid = command(*compose,'ps','-q','search').stdout.strip()
            assert cid
            owned.append(cid)
            state = {'container_id': cid, 'compose_start_seconds': monotonic()-started}
            report['starts'].append(state)
            import httpx
            with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5,trust_env=False) as client:
                deadline=monotonic()+180
                while monotonic()<deadline:
                    assert inspect(cid)['State']['Running'], 'Container exited before READY'
                    try:
                        ready=client.get('/ready')
                        if ready.status_code==200 and ready.json()['state']=='READY':break
                    except httpx.HTTPError:pass
                    sleep(.25)
                else:raise RuntimeError('Container READY timeout')
            state['ready_seconds']=monotonic()-started
            print(f'Container start {run} reached READY in {state["ready_seconds"]:.3f}s', flush=True)
            outputs,checks=exercise('http://127.0.0.1:8000',fixture,run==1)
            state.update(checks)
            dump(out/f'container-{run}-responses.json',outputs)
            state['parity']=[compare(a,b,r['request']['pipeline']) for a,b,r in zip(outputs,native_outputs,fixture['requests'])]
            assert len(outputs)==len(native_outputs)==18 and all(r['passed'] for r in state['parity'])
            assert state['version']==report['native']['version']
            assert all(g['timing_ms']['dense_encoding_ms']>0 for g in outputs if g['requested_pipeline']!='bm25')
            assert all(g['timing_ms']['reranking_ms']>0 and all(h['reranker_score'] is not None for h in g['results']) for g in outputs if g['requested_pipeline']=='hybrid_rerank')
            state['real_dense_and_crossencoder_inference']=True
            command('docker','exec',cid,'python','/app/healthcheck.py')
            # Docker health status is independent of directly executing the check.
            deadline=monotonic()+30
            while inspect(cid)['State']['Health']['Status']!='healthy' and monotonic()<deadline:sleep(.5)
            detail=inspect(cid)
            assert detail['State']['Health']['Status']=='healthy'
            assert detail['Config']['User']=='10001:10001' and detail['HostConfig']['ReadonlyRootfs']
            assert any(m['Destination']=='/runtime' and not m['RW'] for m in detail['Mounts'])
            state['docker_health']='healthy'
            state['nonroot_and_readonly_verified']=True
            if run==1:
                report['environment']=json.loads(command('docker','exec',cid,'python','-c',ENVIRONMENT).stdout)
                report['container_pip_check']=command('docker','exec',cid,'python','-m','pip','check').stdout.strip()
                report['resources']={'docker_stats':json.loads(command('docker','stats','--no-stream','--format','{{json .}}',cid).stdout),
                    'service_proc_status':command('docker','exec',cid,'python','-c',"from pathlib import Path; print('\\n'.join(s for s in Path('/proc/1/status').read_text().splitlines() if s.startswith(('VmRSS:','VmHWM:','Threads:'))))").stdout.strip(),
                    'gpu_snapshot':command('docker','exec',cid,'nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total','--format=csv,noheader,nounits').stdout.strip(),
                    'service_allocator':'not exposed; probe-process allocator is reported separately, not attributed to serving PID1',
                    'loaded_components':'products/BM25/dense/CrossEncoder confirmed by READY and successful real three-pipeline inference'}
            stopped=monotonic()
            command('docker','stop','--time','60',cid,timeout=75)
            log=command('docker','logs',cid).stdout+command('docker','logs',cid).stderr
            (out/f'container-{run}.log').write_text(log,encoding='utf-8')
            detail=inspect(cid)
            state['shutdown']={'seconds':monotonic()-stopped,'exit_code':detail['State']['ExitCode'],
                'lifecycle_completed':'service_shutdown' in log and 'Application shutdown complete.' in log,
                'resource_tracker_warning':'resource_tracker: There appear to be' in log}
            assert state['shutdown']['lifecycle_completed'] and state['shutdown']['exit_code'] in (0,143)
            print(f'Container start {run}: parity and graceful shutdown passed', flush=True)
            assert all(r['request']['query'] not in log for r in fixture['requests'] if len(r['request']['query'])>5)
        assert report['starts'][0]['container_id']!=report['starts'][1]['container_id']
        assert report['starts'][0]['version']==report['starts'][1]['version']
        report['restart_reproducibility']=True
        for case in ('missing_manifest','corrupt_manifest','gpu_unavailable'):
            layout=bundle if case=='gpu_unavailable' else ROOT/'reports/tmp/phase8-linux-negative'/case
            before={str(p.relative_to(layout)):file_hash(p) for p in layout.rglob('*') if p.is_file()}
            name=project+'-'+case.replace('_','-')
            args=['docker','run','-d','--name',name,'--read-only','--tmpfs','/tmp:size=256m,mode=1777','--network','none',
                '--mount',f'type=bind,source={layout},target=/runtime,readonly']
            if case!='gpu_unavailable':args+=['--gpus','all']
            cid=command(*args,IMAGE).stdout.strip();owned.append(cid)
            deadline=monotonic()+120
            while inspect(cid)['State']['Running'] and monotonic()<deadline:sleep(.5)
            detail=inspect(cid)
            result=command('docker','logs',cid)
            log=result.stdout+result.stderr
            (out/(case+'.log')).write_text(log,encoding='utf-8')
            rejected=not detail['State']['Running'] and detail['State']['ExitCode']!=0 and 'Application startup failed' in log and '"event": "service_ready"' not in log
            after={str(p.relative_to(layout)):file_hash(p) for p in layout.rglob('*') if p.is_file()}
            report['failures'].append({'case':case,'container_id':cid,'exit_code':detail['State']['ExitCode'],'startup_rejected':rejected,'layout_unchanged':before==after,'gpu_devices_requested':bool(detail['HostConfig'].get('DeviceRequests'))})
            assert rejected and before==after
            print(f'Failure case {case}: startup rejected and layout unchanged', flush=True)
        verify_bundle(bundle)
        report['bundle_unchanged']=file_hash(bundle/'bundle.json')==report['bundle_sha256']
        report['passed']=True
    except Exception as exc:
        import traceback
        (out/'failure-traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        report['failure']=f'{type(exc).__name__}: {exc}'
    finally:
        for cid in owned:
            found=command('docker','inspect',cid,check=False)
            if found.returncode==0:
                command('docker','stop','--time','60',cid,check=False,timeout=75)
                logs=command('docker','logs',cid,check=False)
                (out/(cid[:12]+'.log')).write_text(logs.stdout+logs.stderr,encoding='utf-8')
                command('docker','rm',cid,check=False)
        command(*compose,'down','--remove-orphans',check=False)
        report['boundary']={'opened':sorted(guard.opened),'blocked':guard.blocked,'final_test_labels_accessed':False}
        report['raw_evidence_sha256']={p.name:file_hash(p) for p in out.iterdir() if p.is_file()}
        dump(out/'checks.json',report)
    print(json.dumps({'passed':report['passed'],'failure':report.get('failure'),'report':str(out/'checks.json')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
