"""Deliver a committed controller, transfer ownership, leave training PIDs alive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_phase1_stack.design import PROJECT,RUN,RELEASE
from experiments.cvs_phase1_stack.capacity16 import CONTROL_RELEASE,WORKER_COMMIT
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh

REMOTE=r'''
import importlib.util,json,os,signal,subprocess,sys,time,tarfile,hashlib
from pathlib import Path
c=CONFIG
project=Path(c['project']);worker=project/'releases'/c['worker_release'];control=project/'releases'/c['control_release'];run=project/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if control.exists() or (run/'dispatcher_active.json').exists() or (run/'capacity16_owner.json').exists():raise FileExistsError('Reconcile existing capacity change; no repeat mutation')
if (worker/'release_commit.txt').read_text().strip()!=c['worker_commit']:raise ValueError('Wrong worker release')
if (run/'failure.json').exists() or (run/'completion.json').exists():raise RuntimeError('Run no longer active and healthy')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
def read(p):return json.loads(p.read_text())
queue=read(run/'queue_state.json');receipt=read(worker/'submit.json');launched=read(run/'launch_r2.json')['rows']
if queue['phase']!='r2' or queue['kind']!='source' or queue['failures'] or len(launched)!=8 or len(queue['active'])!=8 or len(queue['pending'])!=8:raise ValueError('Current queue differs; reconcile before handoff')
control.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if m.name not in ['capacity16.py','release_commit.txt'] or not m.isfile():raise ValueError('Unexpected controller archive')
    t.extractall(control)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
subprocess.run([python,'-m','py_compile',str(control/'capacity16.py')],check=True)
subprocess.run([python,str(control/'capacity16.py'),'--help'],stdout=subprocess.DEVNULL,check=True)
spec=importlib.util.spec_from_file_location('capacity',control/'capacity16.py');cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)
old=cap.proc(receipt['pid'])
if not old or old['argv']!=receipt['argv'] or old['cwd']!=str(worker) or receipt['commit']!=c['worker_commit']:raise ValueError('Old dispatcher identity mismatch')
workers={}
for row in launched:
    p=cap.proc(row['pid'])
    if not p or p['argv']!=row['argv'] or p['cwd']!=str(worker):raise ValueError('Existing training identity differs')
    env=dict(v.split('=',1) for v in Path('/proc',str(p['pid']),'environ').read_bytes().decode().split('\0') if '=' in v)
    if env.get('CUDA_VISIBLE_DEVICES')!=str(row['gpu']):raise ValueError('Existing GPU reservation differs')
    workers[row['row_id']]=p
gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
handoff=dict(run_id=c['run'],authorization='User 2026-10-04: two experiments per GPU',max_active=16,per_gpu_limit=2,worker_commit=c['worker_commit'],control_commit=c['commit'],old_dispatcher=old,workers=workers,queue_before=queue,gpu_before=gpu_state,time=time.time())
handoff_path=control/'handoff.json'
with handoff_path.open('x') as f:json.dump(handoff,f,indent=2)
if cap.proc(old['pid'])!=old:raise RuntimeError('Dispatcher changed before handoff')
# Only this controller PID is signaled. Workers started in their own sessions.
os.kill(old['pid'],signal.SIGTERM)
for _ in range(100):
    if cap.proc(old['pid']) is None:break
    time.sleep(.1)
else:raise RuntimeError('Dispatcher termination UNKNOWN; do not launch another owner')
for identity in workers.values():
    if cap.proc(identity['pid'])!=identity:raise RuntimeError('Worker identity changed during handoff; reconcile')
env=dict(os.environ,PYTHONPATH=str(worker)+os.pathsep+str(worker/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[python,'-u',str(control/'capacity16.py'),'--worker-root',str(worker),'--handoff',str(handoff_path)]
with (control/'dispatcher.stdout.log').open('x') as log:
    child=subprocess.Popen(command,cwd=worker,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(worker),argv=command,control_release=str(control),commit=c['commit'],worker_commit=c['worker_commit'],old_dispatcher_pid=old['pid'],preserved_worker_pids=[x['pid'] for x in workers.values()],log=str(control/'dispatcher.stdout.log'))
with (control/'submit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote differs from HEAD')
    paths=['experiments/cvs_phase1_stack/capacity16.py','experiments/cvs_phase1_stack/set_capacity16.py']
    if subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted control code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(CONTROL_RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing delivery; reconcile first')
    with tarfile.open(archive,'w:gz') as t:
        for name,data in [('capacity16.py',subprocess.check_output(['git','show',commit+':'+paths[0]],cwd=ROOT)),('release_commit.txt',(commit+'\n').encode())]:
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;t.addfile(item,io.BytesIO(data))
    cfg=dict(project=PROJECT,run=RUN,worker_release=RELEASE,worker_commit=WORKER_COMMIT,control_release=CONTROL_RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.output)
