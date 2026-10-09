"""Publish isolated fourteen-model evaluation; no source process mutations."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh
from experiments.cvs_phase1_repair.evaluate_completed14 import PROJECT,RUN
RELEASE='cvs_reference_repair_completed14_eval_20261009_r01'
REMOTE=r'''
from pathlib import Path
import os,json,subprocess,hashlib,tarfile,shutil
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];run=p/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if release.exists() or run.exists():raise FileExistsError('Existing evaluation; reconcile before retry')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
if shutil.disk_usage(p).free<10*1024**3:raise RuntimeError('Insufficient disk')
release.mkdir()
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe member')
 tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','py_compile',str(release/'evaluate_completed14.py')],check=True,env=env)
cmd=[python,'-u',str(release/'evaluate_completed14.py'),'--mode','dispatch'];logpath=release/'controller.log'
with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,log=str(logpath),commit=c['commit'],run_id=c['run'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
def publish():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
 if commit!=remote:raise ValueError('Remote OID differs')
 path='experiments/cvs_phase1_repair/evaluate_completed14.py'
 if subprocess.check_output(['git','status','--porcelain','--',path],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted evaluator')
 out=ROOT/'local_artifacts'/RELEASE;out.mkdir(exist_ok=False);archive=out/(RELEASE+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for name,data in [('evaluate_completed14.py',subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)),('release_commit.txt',(commit+'\n').encode())]:
   m=tarfile.TarInfo(name);m.size=len(data);tar.addfile(m,io.BytesIO(data))
 c=dict(project=PROJECT.as_posix(),run=RUN,release=RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 (out/'package.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
 subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT.as_posix()+'/releases/'+archive.name],check=True)
 receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(c))));(out/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':publish()
