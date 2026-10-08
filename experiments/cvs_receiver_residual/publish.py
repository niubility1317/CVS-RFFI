"""Locally committed/pushed release, one transfer, independent remote readback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_receiver_residual import design as d
from experiments.cvs_phase1_overlay.publish import CONNECTION

PACKAGES=('cvs_receiver_residual','cvs_phase1_overlay','cvs_selected_concat','cvs_energy_identity',
    'cvs_equivariant_identity','cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity',
    'cvs_rff_physics','cvs_reference_identity','cvs_residual_identity','cvs_clean_design','cvs_identity_ce')
PREFIXES=['experiments/'+p+'/' for p in PACKAGES]+['experiments/adv3b02_xuc/code/',
    'baselines/common/','baselines/cvcnn_ce/','code/leo_practical/']
FILES={'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'}


def ssh(script):
    compile(script,'remote','exec')
    # Validation imports Torch; never rely on the server's system python3.
    command=['ssh',*CONNECTION,'-T','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -']
    result=subprocess.run(command,input=script.encode('utf-8'),capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+'\n'+result.stdout.decode('utf-8',errors='replace'))
    return result.stdout

REMOTE=r'''
import hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release']
run=project/'runs'/c['run'];logs=project/'logs'/c['run']
if release.exists() or run.exists() or logs.exists(): raise FileExistsError('Reconcile existing delivery before retry')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440': raise ValueError('Host/user mismatch')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']: raise ValueError('Archive transfer differs')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not (release/member.name).resolve().is_relative_to(release.resolve()) or not member.isfile(): raise ValueError('Unsafe archive member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments'),str(release/'baselines'),str(release/'code')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_receiver_residual.smoke','--device','cpu','--output',str(release/'remote_smoke.json')],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_receiver_residual import design as d
from experiments.cvs_receiver_residual.dispatch import validate_matrix
spec=release/'experiments/cvs_receiver_residual/configs/launch_spec.json'
value=json.loads(spec.read_text());validate_matrix(value)
for row in value['rows']:
    cfg=row['config'];d.validate_config(cfg)
    if json.loads((release/'experiments/cvs_receiver_residual/configs'/(row['row_id']+'.json')).read_text())!=cfg: raise ValueError('Entity config differs')
    for k in ('dataset','source_contract'):
        if not Path(cfg[k]).is_file(): raise FileNotFoundError(cfg[k])
for p in (Path(d.VIEWS_ROOT)/'manifest.json',Path(d.VIEWS_ROOT)/'index.npz',Path(d.CAPSULE)/'clean.npy',Path(d.TRUTH)):
    if not p.is_file(): raise FileNotFoundError(p)
# Existing manifest reused, no data builder/revalidation or query/truth content read.
from experiments.cvs_receiver_residual.evaluate import manifest
manifest()
if shutil.disk_usage(project).free<20*1024**3: raise RuntimeError('Insufficient output space')
command=[python,'-u','-m','experiments.cvs_receiver_residual.dispatch','--spec',str(spec)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

INSPECT=r'''
import json,os,subprocess,time
from pathlib import Path
p=Path(PROJECT);run=p/'runs'/RUN;release=p/'releases'/RELEASE
def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None
def process(pid):
    path=Path('/proc')/str(pid)
    if not path.exists():return None
    try:
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0] in ('Z','X','x'):return None
        return dict(pid=pid,cwd=str((path/'cwd').resolve()),argv=[x for x in (path/'cmdline').read_bytes().decode().split('\0') if x],state=stat[0],start_ticks=int(stat[19]))
    except (OSError,UnicodeError):return dict(pid=pid,state='UNKNOWN')
submit=read(release/'submit.json')
active=read(run/'dispatcher_active.json')
owner=active or submit
ownerproc=process(owner['pid']) if owner else None
if ownerproc and (ownerproc.get('cwd')!=owner['cwd'] or ownerproc.get('argv')!=owner['argv'] or (active and ownerproc.get('start_ticks')!=active['start_ticks'])):ownerproc=None
result=dict(read_at=time.time(),run_id=RUN,release_exists=release.exists(),run_exists=run.exists(),
    submit=submit,dispatcher=active or read(run/'dispatcher.json'),dispatcher_original=read(run/'dispatcher.json'),
    dispatcher_process=ownerproc,
    capacity4_handoff=read(run/'capacity4_handoff_v2.json') or read(run/'capacity4_handoff.json'),capacity4_adopted=read(run/'capacity4_adopted.json'),
    remote_smoke=read(release/'remote_smoke.json'),queue=read(run/'queue_state.json'),
    completion=read(run/'completion.json'),failure=read(run/'failure.json'),
    scoring=read(run/'scoring_complete.json'),source_freeze=read(run/'source_matrix_frozen.json'),rows=[])
for kind in ('source','predict'):
    launch=read(run/('launch_'+kind+'.json'))
    for r in (launch or {}).get('rows',[]):
        source=run/r['row_id']/'source';pred=source.parent/'prediction';log=Path(r['log'])
        epochs=source/'epoch_metrics.jsonl';latest=None
        if epochs.exists() and epochs.stat().st_size:
            lines=epochs.read_text().splitlines()
            try:latest=json.loads(lines[-1])
            except json.JSONDecodeError:pass
        result['rows'].append(dict(**r,process=process(r['pid']),initialization=read(source/'initialization.json'),
            resolved=read(source/'resolved_config.json'),epoch=latest,
            source_completion=read(source/'completion.json'),prediction_complete=read(pred/'complete.json'),
            lease=read(run/'capacity_leases'/(kind+'-'+r['row_id']+'.json')),
            log_bytes=log.stat().st_size if log.exists() else 0,
            log_tail=log.read_text(errors='replace').splitlines()[-2:] if log.exists() else []))
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
ownerlog=Path(active['log']) if active else release/'dispatcher.stdout.log'
result['dispatcher_log_tail']=ownerlog.read_text(errors='replace').splitlines()[-8:] if ownerlog.exists() else []
print(json.dumps(result))
'''


def inspect(output):
    code=INSPECT.replace('PROJECT',repr(d.PROJECT)).replace('RELEASE',repr(d.RELEASE)).replace('RUN',repr(d.RUN))
    data=json.loads(ssh(code));output.mkdir(parents=True,exist_ok=True)
    d.write(output/'readback.json',data)
    print(json.dumps(dict(run_id=d.RUN,dispatcher_alive=bool(data['dispatcher_process']),
        completion=data['completion'],failure=data['failure'],queue=data['queue'],
        remote_smoke=data['remote_smoke']['status'] if data['remote_smoke'] else None,
        rows=[dict(row_id=r['row_id'],kind=r['kind'],gpu=r['gpu'],alive=bool(r['process']),
            epoch=r['epoch']['epoch'] if r['epoch'] else None,source_complete=bool(r['source_completion']),
            prediction_complete=bool(r['prediction_complete']),log_bytes=r['log_bytes'],
            log_tail=r['log_tail'] if not r['process'] and not r['source_completion'] else []) for r in data['rows']]),ensure_ascii=False))
    return data


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=d.ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote',
        'origin','refs/heads/'+branch],cwd=d.ROOT,text=True).split()[0]
    if remote!=commit: raise ValueError('Remote branch differs from HEAD')
    if subprocess.check_output(['git','status','--porcelain','--',*PREFIXES,*FILES],cwd=d.ROOT,text=True).strip():
        raise ValueError('Uncommitted release code/config')
    names=subprocess.check_output(['git','ls-files'],cwd=d.ROOT,text=True).splitlines()
    selected=[n for n in names if (any(n.startswith(p) for p in PREFIXES) or n in FILES) and Path(n).suffix in {'.py','.json'}]
    output.mkdir(parents=True,exist_ok=True);archive=output/(d.RELEASE+'.tar.gz')
    if archive.exists(): raise FileExistsError('Archive exists; reconcile before resubmit')
    with tarfile.open(archive,'w:gz') as tar:
        for n in selected:
            blob=subprocess.check_output(['git','show',commit+':'+n],cwd=d.ROOT)
            item=tarfile.TarInfo(n);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    config=dict(project=d.PROJECT,release=d.RELEASE,run=d.RUN,archive=archive.name,sha256=sha,commit=commit)
    d.write(output/'package.json',dict(config,files=len(selected)))
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(config))))
    d.write(output/'submit.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true')
    a=p.parse_args();inspect(a.output) if a.inspect else publish(a.output)
