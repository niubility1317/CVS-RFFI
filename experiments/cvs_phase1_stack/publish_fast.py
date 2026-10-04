"""Commit-first execution release; adopt workers without restarting training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_phase1_stack.design import PROJECT,RUN,RELEASE as WORKER_RELEASE
from experiments.cvs_phase1_stack.capacity16 import WORKER_COMMIT
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh

RELEASE='cvs_reference_stack_fast_20261005_r01'
FILES=['capacity16.py','fast_dispatch.py','fast_source.py','fast_execution.py','speed_checks.py']

REMOTE=r'''
import importlib.util,json,os,signal,subprocess,time,tarfile,hashlib
from pathlib import Path
c=CONFIG
project=Path(c['project']);worker=project/'releases'/c['worker_release'];control=project/'releases'/c['release'];run=project/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if control.exists() or (run/(c['release']+'_owner.json')).exists():raise FileExistsError('Reconcile existing release; never resubmit blindly')
if (worker/'release_commit.txt').read_text().strip()!=c['worker_commit']:raise ValueError('Wrong immutable worker release')
if (run/'failure.json').exists() or (run/'completion.json').exists():raise RuntimeError('Parent is not active and healthy')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
control.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if m.name not in c['files']+['release_commit.txt'] or not m.isfile():raise ValueError('Unexpected release member')
    t.extractall(control)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
subprocess.run([python,'-m','compileall','-q',str(control)],check=True)
env=dict(os.environ,PYTHONPATH=str(worker)+os.pathsep+str(worker/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (control/'compatibility.stdout.log').open('x') as log:
    subprocess.run([python,str(control/'speed_checks.py'),'--worker-root',str(worker),'--output',str(control/'compatibility.json')],cwd=worker,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([python,str(control/'fast_source.py'),'--help'],stdout=subprocess.DEVNULL,check=True)
subprocess.run([python,str(control/'fast_dispatch.py'),'--help'],stdout=subprocess.DEVNULL,check=True)
spec=importlib.util.spec_from_file_location('capacity',control/'capacity16.py');cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)
def read(p):return json.loads(p.read_text())
active=read(run/'dispatcher_active.json');old=cap.proc(active['pid'])
if not old or old['argv']!=active['argv'] or old['cwd']!=str(worker):raise ValueError('Active controller identity mismatch')
if active['control_release']!='cvs_reference_stack_dispatch16_20261004_r01' or active['worker_commit']!=c['worker_commit']:raise ValueError('Unexpected owner')
queue=read(run/'queue_state.json')
if queue['phase']!='r2' or queue['kind']!='source' or queue['failures']:raise ValueError('R2 handoff window changed; reconcile')
def snapshot_workers():
    workers={};completed=[]
    for path in sorted(run.glob('launch_r*.json')):
        for row in read(path)['rows']:
            p=cap.proc(row['pid'])
            if p:
                if p['argv']!=row['argv'] or p['cwd']!=row['cwd']:raise ValueError('Training identity mismatch')
                e=dict(v.split('=',1) for v in Path('/proc',str(p['pid']),'environ').read_bytes().decode().split('\0') if '=' in v)
                if e.get('CUDA_VISIBLE_DEVICES')!=str(row['gpu']):raise ValueError('GPU differs')
                workers[row['row_id']]=p
            else:
                done=run/row['row_id']/'source/completion.json'
                if not done.exists():raise RuntimeError('Dead worker without completion')
                value=read(done)
                if value['status']!='SOURCE_TRAINED' or value['logged_steps']!=44400 or value['optimizer_steps']!=44400:raise ValueError('Invalid completed row')
                completed.append(row['row_id'])
    return workers,completed
before,completed=snapshot_workers()
intent=dict(run_id=c['run'],authorization='User: optimize subsequent training speed; retain live workers',max_active=16,per_gpu_limit=2,worker_commit=c['worker_commit'],control_commit=c['commit'],old_dispatcher=old,workers=before,completed_before=completed,queue_before=queue,time=time.time())
with (control/'handoff_intent.json').open('x') as f:json.dump(intent,f,indent=2)
if cap.proc(old['pid'])!=old:raise RuntimeError('Controller changed before handoff')
# Exactly one controller is signalled; no worker or process-group signal.
os.kill(old['pid'],signal.SIGTERM)
for _ in range(100):
    if cap.proc(old['pid']) is None:break
    time.sleep(.1)
else:raise RuntimeError('Old controller termination UNKNOWN')
after,completed=snapshot_workers()
for rid,p in before.items():
    if rid in after and after[rid]!=p:raise RuntimeError('Worker identity changed during handoff')
    if rid not in after and rid not in completed:raise RuntimeError('Worker disappeared')
handoff=dict(intent,workers=after,completed_at_handoff=completed)
path=control/'handoff.json'
with path.open('x') as f:json.dump(handoff,f,indent=2)
cmd=[python,'-u',str(control/'fast_dispatch.py'),'--worker-root',str(worker),'--handoff',str(path)]
with (control/'dispatcher.stdout.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=worker,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(worker),argv=cmd,control_release=str(control),commit=c['commit'],old_dispatcher_pid=old['pid'],preserved_workers=after,naturally_completed=completed,log=str(control/'dispatcher.stdout.log'))
with (control/'submit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''

def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote branch differs from HEAD')
    paths=['experiments/cvs_phase1_stack/'+f for f in FILES+['publish_fast.py']]
    if subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing delivery; inspect post-state before retry')
    with tarfile.open(archive,'w:gz') as t:
        payloads=[(f,subprocess.check_output(['git','show',commit+':experiments/cvs_phase1_stack/'+f],cwd=ROOT)) for f in FILES]+[('release_commit.txt',(commit+'\n').encode())]
        for name,data in payloads:
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;t.addfile(item,io.BytesIO(data))
    cfg=dict(project=PROJECT,run=RUN,worker_release=WORKER_RELEASE,worker_commit=WORKER_COMMIT,release=RELEASE,commit=commit,archive=archive.name,files=FILES,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.output)
