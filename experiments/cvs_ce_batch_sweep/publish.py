"""Publish isolated five-model evaluation; no source process mutations."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh
from experiments.cvs_ce_batch_sweep.design import PROJECT,RUN
RELEASE='cvs_ce_singlepass_batch_20261009_r03'
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
subprocess.run([python,'-m','py_compile',*[str(release/n) for n in ('design.py','source.py','evaluate.py')]],check=True,env=env)
subprocess.run([python,'-c',"import sys;sys.path.insert(0,"+repr(str(release))+");import design as d;assert (d.WORKER/'release_commit.txt').read_text().strip()==d.WORKER_COMMIT;cs=[d.config(b) for b in d.BATCHES];assert len(cs)==4;[d.validate(c) for c in cs];[d.make_args(c) for c in cs]"],check=True,env=env,cwd=release)
check="import sys,json;sys.path.insert(0,"+repr(str(release))+");import design as d;import torch;from experiments.cvs_phase1_overlay.model import native_modules;n=native_modules();from cvsrffi.xuc_fusion.native import role_ids_from_native;from experiments.cvs_phase1_stack.design import SOURCE;a=d.make_args(d.config(256));a.output_dir="+repr(str(release/'preflight'))+";ctx=n._build_ssdg_wisig_data(a,torch.device('cpu'));assert not ctx['named_test_loaders'];assert role_ids_from_native(ctx)==json.load(open(SOURCE))['role_ids'];assert len(ctx['train_loader'].dataset)==6300;from torch.utils.data import DataLoader;assert [len(DataLoader(ctx['train_loader'].dataset,batch_size=b,drop_last=False)) for b in d.BATCHES]==[25,13,7,4];print('ACTUAL_SOURCE_LOADER_PREFLIGHT_PASS')"
subprocess.run([python,'-c',check],check=True,env=env,cwd=release)
cmd=[python,'-u',str(release/'evaluate.py'),'--mode','dispatch'];logpath=release/'controller.log'
with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,log=str(logpath),commit=c['commit'],run_id=c['run'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
def publish():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
 if commit!=remote:raise ValueError('Remote OID differs')
 path='experiments/cvs_ce_batch_sweep/evaluate.py'
 if subprocess.check_output(['git','status','--porcelain','--',path],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted evaluator')
 out=ROOT/'local_artifacts'/RELEASE;out.mkdir(exist_ok=False);archive=out/(RELEASE+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  entries=[(name,subprocess.check_output(['git','show',commit+':experiments/cvs_ce_batch_sweep/'+name],cwd=ROOT)) for name in ('design.py','source.py','evaluate.py')]
  entries.append(('release_commit.txt',(commit+'\n').encode()))
  for name,data in entries:
   m=tarfile.TarInfo(name);m.size=len(data);tar.addfile(m,io.BytesIO(data))
 c=dict(project=PROJECT.as_posix(),run=RUN,release=RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 (out/'package.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
 subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT.as_posix()+'/releases/'+archive.name],check=True)
 receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(c))));(out/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':publish()
