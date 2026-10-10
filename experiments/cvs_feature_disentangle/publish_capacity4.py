"""Versioned scheduler-only handoff; never signal a training worker."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
from . import design as d

CONTROL_RELEASE='cvs_feature_capacity4_20261010_r01'
WORKER_COMMIT='d4723b1aa227fce6368fd25680bb74b1190557d7'
REMOTE=r'''
import hashlib,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
c=CONFIG;project=Path(c['project']);control=project/'releases'/c['control_release'];release=project/'releases'/c['worker_release'];base=project/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host/user mismatch')
if control.exists() or (base/'dispatcher_active.json').exists():raise FileExistsError('Existing handoff; reconcile')
if (release/'release_commit.txt').read_text().strip()!=c['worker_commit']:raise ValueError('Scientific release mismatch')
if any((base/name).exists() for name in ('failure.json','completion.json','source_selection.json')):raise ValueError('Unexpected terminal/selected run state')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_receiver_residual_capacity4.control import process,running,read,write,lock
submit=read(release/'submit.json');old=process(submit['pid'])
if not old or old['cwd']!=str(release) or old['argv']!=submit['argv']:raise ValueError('Controller identity mismatch')
if read(base/'dispatcher.json')['pid']!=old['pid'] or read(base/'queue_state.json')['phase']!='train':raise ValueError('Owner or phase changed')
control.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not member.isfile() or not (control/member.name).resolve().is_relative_to(control.resolve()):raise ValueError('Unsafe archive')
 tar.extractall(control)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';entry=control/'experiments/cvs_feature_disentangle/capacity4.py'
subprocess.run([python,'-m','py_compile',str(entry)],check=True)
with lock(base/'capacity4_handoff.lock',nonblocking=True):
 with lock(project/'runs/receiver_residual_capacity4.lock'):
  current=process(old['pid'])
  if not current or any(current[k]!=old[k] for k in ('start_ticks','argv','cwd')):raise ValueError('Controller changed')
  launches=read(base/'launch.json')['rows'];workers=[]
  for item in launches:
   actual=process(item['pid'])
   if actual:
    if actual['cwd']!=item['cwd'] or actual['argv']!=item['argv']:raise ValueError('Worker identity mismatch')
    workers.append(actual)
  write(control/'handoff_before.json',dict(old_controller=old,workers=workers,launches=launches,at=time.time()))
  # Exactly the verified scheduling PID; never a process group or worker PID.
  os.kill(old['pid'],signal.SIGTERM)
  for _ in range(50):
   if not running(old['pid']):break
   time.sleep(.1)
  if running(old['pid']):raise RuntimeError('Old controller still alive; no new owner')
  after=[process(w['pid']) for w in workers]
  for before,now in zip(workers,after):
   if now and any(now[k]!=before[k] for k in ('start_ticks','cwd','argv')):raise ValueError('Worker changed')
  write(control/'handoff.json',dict(status='OLD_CONTROLLER_EXIT_VERIFIED',workers_before=workers,workers_after=after,
    training_workers_signaled=False,old_controller=old,authorization='用户要求每张卡4进程',at=time.time()))
  with lock(base/'capacity4_owner.lock',nonblocking=True):
   cmd=[python,'-u',str(entry),'--release',str(release)]
   env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
   env.pop('CUDA_VISIBLE_DEVICES',None)
   logpath=control/'dispatcher.stdout.log'
   with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state=process(child.pid)
   if not state:raise RuntimeError('New controller absent; reconcile')
   receipt=dict(status='SUBMITTED',pid=child.pid,start_ticks=state['start_ticks'],cwd=str(release),argv=cmd,log=str(logpath),
    control_release=str(control),control_commit=c['commit'],commit=c['worker_commit'],run_id=c['run'],per_gpu_limit=4,
    launch_owner=c['owner'],at=time.time())
   write(base/'dispatcher_active.json',receipt);write(control/'submit.json',receipt)
   spec=read(base/'experiment.json');spec['execution']['gpu_policy']='User override: <=4 total processes/GPU; pre-CUDA reservations; >=12GB free; no global12 cap'
   spec['execution']['controller_pid']=child.pid;spec['execution']['control_commit']=c['commit'];write(base/'experiment.json',spec)
print(json.dumps(receipt))
'''

def publish(output):
 def git(*args):return subprocess.check_output(['git',*args],cwd=d.ROOT,text=True).strip()
 head=git('rev-parse','HEAD');branch=git('branch','--show-current')
 if git('-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch).split()[0]!=head:raise ValueError('Remote OID mismatch')
 path='experiments/cvs_feature_disentangle/capacity4.py'
 if git('status','--porcelain','--',path):raise ValueError('Uncommitted controller')
 output.mkdir(parents=True,exist_ok=True);archive=output/(CONTROL_RELEASE+'.tar.gz')
 if archive.exists():raise FileExistsError('Existing delivery; reconcile before retry')
 with tarfile.open(archive,'w:gz') as tar:
  for name,blob in [(path,subprocess.check_output(['git','show',head+':'+path],cwd=d.ROOT)),('release_commit.txt',(head+'\n').encode())]:
   item=tarfile.TarInfo(name);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
 cfg=dict(project=d.PROJECT,control_release=CONTROL_RELEASE,worker_release=d.RELEASE,worker_commit=WORKER_COMMIT,run=d.RUN,
  owner=d.OWNER,commit=head,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 d.write(output/'package.json',cfg)
 subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True)
 receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))));d.write(output/'submit.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);publish(p.parse_args().output)
