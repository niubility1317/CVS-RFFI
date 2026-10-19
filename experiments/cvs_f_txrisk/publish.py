"""Commit-pinned read-only source audit release; one run owner, no retry."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
from experiments.cvs_rdrop_sinc_units.publish import PYTHON

def activate(code):
 compile(code,'source_audit_activation','exec');p=subprocess.run(['ssh',*CONNECTION,'-T','N607',PYTHON+' -'],input=code.encode(),capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
 return json.loads(p.stdout)
def main():
 root=d.ROOT;pkg='experiments/cvs_f_txrisk';local=root/'local_artifacts/F_TXrisk_preparation_r01';local.mkdir(parents=True,exist_ok=True)
 if subprocess.check_output(['git','status','--porcelain','--',pkg],cwd=root,text=True).strip():raise ValueError('Source audit code must be committed')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();c=dict(release=d.PROJECT+'/releases/'+d.RELEASE,archive=d.PROJECT+'/releases/'+d.RELEASE+'.tar.gz',out=str(d.BASE),original=d.ORIGINAL,original_commit=d.ORIGINAL_COMMIT,python=PYTHON,commit=commit,GPUs=list(d.GPUS))
 code=r'''
import os,json,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary verified N607 required')
if any(Path(c[k]).exists() for k in ('release','archive','out')):raise FileExistsError('Immutable run exists; reconcile, do not resubmit')
if (Path(c['original'])/'release_commit.txt').read_text().strip()!=c['original_commit'] or shutil.disk_usage(c['original']).free<1024**3:raise ValueError('Wrong dependency or insufficient artifact space')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,original_execution='VERIFIED',query_access=False,GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True))))
''';pre=activate(code.replace('CONFIG',repr(c)));(local/'release_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
 raw=subprocess.check_output(['git','archive',commit,pkg],cwd=root);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=root);archive=local/(d.RELEASE+'.tar.gz')
 with tarfile.open(fileobj=io.BytesIO(raw)) as old,tarfile.open(archive,'w:gz') as out:
  for member in old.getmembers():
   if member.isfile():out.addfile(member,old.extractfile(member))
  for name,value in [('release_commit.txt',(commit+'\n').encode()),('registration/experiment.json',record)]:
   member=tarfile.TarInfo(name);member.size=len(value);out.addfile(member,io.BytesIO(value))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+c['archive']],check=True,capture_output=True)
 code=r'''
import hashlib,importlib.util,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release'])
if release.exists() or Path(c['out']).exists():raise FileExistsError('Possibly submitted output exists; reconcile')
archive=Path(c['archive'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single release archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as package:
 for m in package.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive path')
 package.extractall(release)
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_f_txrisk')],cwd=release,check=True)
p=release/'experiments/cvs_f_txrisk/runner.py';spec=importlib.util.spec_from_file_location('audit_startup',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose,serial
from experiments.cvs_rdrop_sinc_units.startup import launch_once
with capacity_lock():
 caps={g:v for g,v in capacity().items() if str(g) in c['GPUs']};gpu=choose(caps)
 if gpu is None:raise RuntimeError('GPU2/3 have no legal capacity; preserve extracted release, no worker submitted')
 cmd=[c['python'],'-u',str(p)];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def create():
  stream=(release/'worker.stdout.log').open('x');return subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 def record(v):(release/'submit.json').write_text(json.dumps(dict(v,commit=c['commit'],archive_comparison='VERIFIED',remote_compile='VERIFIED'),indent=2)+'\n')
 child,receipt=launch_once(create,dict(cwd=str(release),argv=cmd,gpu=str(gpu),capacity_before=serial(caps),source_only=True),record,timeout=2.)
print(json.dumps(dict(receipt,commit=c['commit'],archive_comparison='VERIFIED',remote_compile='VERIFIED')))
''';result=activate(code.replace('CONFIG',repr(c)));(local/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','pid','gpu','commit','archive_comparison','remote_compile')}))
if __name__=='__main__':main()
