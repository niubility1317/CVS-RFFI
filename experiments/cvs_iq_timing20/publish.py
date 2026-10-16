"""Publish only the committed source-gradient diagnostic, then observe one child."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
from experiments.cvs_rdrop_sinc_units.publish import PYTHON
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
RUN=d.RUN
RELEASE=d.RELEASE
PKG='experiments/cvs_iq_timing20'
def activation_ssh(script):
 compile(script,'remote_activation','exec')
 result=subprocess.run(['ssh',*CONNECTION,'-T','N607',PYTHON+' -'],input=script.encode('utf8'),capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 if result.returncode:raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
 return result.stdout
def main():
 root=Path(__file__).resolve().parents[2];local=root/'local_artifacts/F_timing_preparation_r01';local.mkdir(parents=True,exist_ok=True)
 if subprocess.check_output(['git','status','--porcelain','--',PKG],cwd=root,text=True).strip():raise ValueError('Uncommitted diagnostic code')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();remote=d.PROJECT+'/releases/'+RELEASE
 c=dict(release=remote,archive=remote+'.tar.gz',root=d.PROJECT+'/runs/'+RUN,AF=d.AF_RELEASE,python=PYTHON,commit=commit)
 script=r'''
import json,os,subprocess,shutil
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','root')):raise FileExistsError('Immutable source diagnosis already exists; reconcile, never resubmit')
if not Path(c['AF']).is_dir() or (Path(c['AF'])/'release_commit.txt').read_text().strip()!='09554c3a06c976c53efc4dbc61f5c501a1e7b06e':raise ValueError('Wrong immutable A-F execution dependency')
if shutil.disk_usage(c['AF']).free<1024**3:raise ValueError('Insufficient artifact space')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,immutable_dependency=True,query_access=False,GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True))))
'''
 pre=json.loads(ssh(script.replace('CONFIG',repr(c))));(local/'diagnostic_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
 raw=subprocess.check_output(['git','archive',commit,PKG],cwd=root);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json'],cwd=root);archive=local/(RELEASE+'.tar.gz')
 with tarfile.open(fileobj=io.BytesIO(raw)) as original,tarfile.open(archive,'w:gz') as packed:
  for member in original.getmembers():
   if member.isfile():packed.addfile(member,original.extractfile(member))
  for name,data in [('release_commit.txt',(commit+'\n').encode()),('runtime.json',json.dumps(dict(AF=c['AF'])).encode()),('registration/experiment.json',record)]:
   member=tarfile.TarInfo(name);member.size=len(data);packed.addfile(member,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+c['archive']],check=True,capture_output=True)
 script=r'''
import hashlib,importlib.util,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;r=Path(c['release']);archive=Path(c['archive'])
if r.exists() or Path(c['root']).exists():raise FileExistsError('Reconcile before any possible relaunch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single release archive transfer mismatch')
r.mkdir()
with tarfile.open(archive) as t:
 for m in t.getmembers():
  if not m.isfile() or not (r/m.name).resolve().is_relative_to(r.resolve()):raise ValueError('Invalid archive member')
 t.extractall(r)
subprocess.run([c['python'],'-m','compileall','-q',str(r/'experiments/cvs_iq_timing20')],cwd=r,check=True)
spec=importlib.util.spec_from_file_location('diagnostic_bootstrap',r/'experiments/cvs_iq_timing20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose,process
from experiments.cvs_rdrop_sinc_units.startup import observe_started_child
cmd=[c['python'],'-u',str(r/'experiments/cvs_iq_timing20/runner.py')];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (r/'diagnostic.stdout.log').open('x') as stream:p=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
submit=dict(status='SUBMITTED',pid=p.pid,cwd=str(r),argv=cmd,gpu='',commit=c['commit'],archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',CPU_dispatcher=True,per_GPU_limit=3)
(r/'submit.json').write_text(json.dumps(submit,indent=2)+'\n')
try:
 actual=observe_started_child(p,cwd=str(r),argv=cmd,gpu='',timeout=2.)
 submit.update(status='OBSERVED',actual=actual)
except Exception as e:submit.update(status='OBSERVATION_FAILED',observation_error=str(e))
(r/'submit.json').write_text(json.dumps(submit,indent=2)+'\n');print(json.dumps(submit))
'''
 submit=json.loads(activation_ssh(script.replace('CONFIG',repr(c))));(local/'diagnostic_submit.json').write_text(json.dumps(submit,indent=2)+'\n');print(json.dumps({k:submit[k] for k in ('status','pid','gpu','commit','archive_SHA_comparison','remote_compile')}))
if __name__=='__main__':main()
