"""Replace only the verified waiting repair owner; never modify worker release."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh
from experiments.cvs_phase1_repair.design import PROJECT,RUN
RELEASE='cvs_reference_repair_direct_20261008_r01'
WORKER_RELEASE='cvs_reference_repair_20261008_r01'
REMOTE=r'''
from pathlib import Path
import os,json,subprocess,hashlib,tarfile,time,signal
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];worker=p/'releases'/c['worker_release'];run=p/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if release.exists() or (run/'direct_owner.json').exists():raise FileExistsError('Reconcile existing handoff before retry')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for m in tar.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(worker)+os.pathsep+str(worker/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
check="import sys,json; from pathlib import Path; sys.path.insert(0,"+repr(str(release))+"); import direct_controller as c; from experiments.cvs_phase1_repair import design as d; from experiments.cvs_phase1_stack.capacity16 import proc; rows=c.check_pending(d); prior=d.read(d.ROOT/'submit.json'); current=proc(prior['pid']); assert current and current['cwd']==prior['cwd'] and current['argv']==prior['argv']; print(json.dumps(dict(previous_owner=current,run_id=d.RUN,worker_commit=c.WORKER_COMMIT,rows=len(rows))))"
handoff=json.loads(subprocess.check_output([python,'-c',check],cwd=worker,env=env,text=True))
(release/'handoff.json').write_text(json.dumps(handoff,indent=2)+'\n')
old=handoff['previous_owner'];pid=old['pid'];q=Path('/proc')/str(pid)
if str((q/'cwd').resolve())!=old['cwd'] or [s for s in (q/'cmdline').read_bytes().decode().split('\0') if s]!=old['argv']:raise ValueError('Owner identity changed before signal')
if int((q/'stat').read_text().rsplit(')',1)[1].split()[19])!=old['start_ticks']:raise ValueError('Owner PID reused')
if json.loads((run/'queue_state.json').read_text())['phase']!='WAITING_PRIOR_OWNERS' or list(run.glob('launch_*.json')):raise ValueError('Repair workers started; no handoff')
# User explicitly requested direct start of this waiting queue. Only its owner
# receives SIGTERM; the original R5/R6 launchers/workers are never signalled.
os.kill(pid,signal.SIGTERM)
for _ in range(100):
    try:alive=bool((q/'cmdline').read_bytes())
    except FileNotFoundError:alive=False
    if not alive:break
    time.sleep(.1)
else:raise RuntimeError('Waiting owner did not exit; no second owner launched')
cmd=[python,'-u',str(release/'direct_controller.py'),'--worker-root',str(worker)]
logpath=release/'controller.stdout.log'
with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,log=str(logpath),control_commit=c['commit'],worker_commit=handoff['worker_commit'],run_id=c['run'],previous_pid=pid)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote commit differs')
    path='experiments/cvs_phase1_repair/direct_controller.py'
    if subprocess.check_output(['git','status','--porcelain','--',path],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted controller')
    out=ROOT/'local_artifacts'/RELEASE;out.mkdir(exist_ok=False)
    archive=out/(RELEASE+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for name,data in [('direct_controller.py',subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)),
            ('train_direct.py',subprocess.check_output(['git','show',commit+':experiments/cvs_phase1_repair/train_direct.py'],cwd=ROOT)),
            ('release_commit.txt',(commit+'\n').encode())]:
            m=tarfile.TarInfo(name);m.size=len(data);tar.addfile(m,io.BytesIO(data))
    c=dict(project=PROJECT,run=RUN,release=RELEASE,worker_release=WORKER_RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'package.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(c))))
    (out/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':publish()
