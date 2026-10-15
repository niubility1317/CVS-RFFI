import hashlib,io,json,subprocess,tarfile
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
from experiments.cvs_sigtime_rdrop_legacy.publish import PYTHON
SHARED=['experiments/cvs_sigtime_rdrop_recovery/'+n for n in ('__init__.py','startup.py')]
def main():
 local=d.ROOT/'local_artifacts'/d.RELEASE;local.mkdir(parents=True,exist_ok=True);git='C:/Program Files/Git/cmd/git.exe';paths=['experiments/cvs_sigtime_rdrop_completion20',*SHARED]
 if subprocess.check_output([git,'-C',str(d.ROOT),'status','--porcelain','--',*paths],text=True).strip():raise ValueError('Commit actual code/config first')
 commit=subprocess.check_output([git,'-C',str(d.ROOT),'rev-parse','HEAD'],text=True).strip();remote=d.PROJECT+'/releases/'+d.RELEASE
 c=dict(release=remote,archive=remote+'.tar.gz',root=d.BASE.as_posix(),model=d.MODEL_RELEASE,model_commit=d.MODEL_COMMIT,dependency=d.DEPENDENCY_RELEASE,dependency_commit=d.DEPENDENCY_COMMIT,fold=d.FOLD_RELEASE,source=d.SOURCE,python=PYTHON,commit=commit)
 preflight=r'''
import json,os,subprocess,shutil
from pathlib import Path
c=CONFIG
assert os.geteuid()!=0 and subprocess.check_output(['whoami'],text=True).strip()=='szu2070436088' and subprocess.check_output(['hostname'],text=True).strip()=='dell-DSS8440'
assert (Path(c['model'])/'release_commit.txt').read_text().strip()==c['model_commit'] and (Path(c['dependency'])/'release_commit.txt').read_text().strip()==c['dependency_commit']
assert Path(c['fold']).is_dir() and Path(c['source']).is_file()
assert not any(Path(c[k]).exists() for k in ('release','archive','root')) and shutil.disk_usage(c['model']).free>2*1024**3
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,ready_rows=4,source_only=True,epochs_per_row=20,optimizer_steps_total=17760,CPU_threads=2,data_loader_workers=0,training_initialization='scratch;code dependencies only',
GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True),GPU_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
'''
 d.write(local/'preflight.json',json.loads(ssh(preflight.replace('CONFIG',repr(c)))))
 raw=subprocess.check_output([git,'-C',str(d.ROOT),'archive',commit,*paths]);record=subprocess.check_output([git,'-C',str(d.ROOT),'show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json']);archive=local/(d.RELEASE+'.tar.gz')
 with tarfile.open(fileobj=io.BytesIO(raw)) as original,tarfile.open(archive,'w:gz') as packed:
  for member in original.getmembers():
   if member.isfile():packed.addfile(member,original.extractfile(member))
  for name,data in [('release_commit.txt',(commit+'\n').encode()),('registration/experiment.json',record)]:
   member=tarfile.TarInfo(name);member.size=len(data);packed.addfile(member,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+c['archive']],capture_output=True,check=True)
 script=r'''
import json,hashlib,tarfile,subprocess,os
from pathlib import Path
c=CONFIG;r=Path(c['release']);archive=Path(c['archive'])
if r.exists() or Path(c['root']).exists():raise FileExistsError('Immutable release/root exists; reconcile before retry')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release archive transfer mismatch')
r.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (r/member.name).resolve().is_relative_to(r.resolve()):raise ValueError('Invalid archive path')
 source.extractall(r)
subprocess.run([c['python'],'-m','compileall','-q',str(r/'experiments')],cwd=r,check=True)
cmd=[c['python'],'-u',str(r/'experiments/cvs_sigtime_rdrop_completion20/runner.py'),str(r/'experiments/cvs_sigtime_rdrop_completion20/configs/matrix.json')];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (r/'dispatcher.stdout.log').open('x') as stream:p=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='SUBMITTED',pid=p.pid,cwd=str(r),argv=cmd,commit=c['commit'],model_code_dependency=c['model_commit'],CUDA_VISIBLE_DEVICES='',archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',source_only=True)
with (r/'submit.json').open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(value))
'''
 value=json.loads(ssh(script.replace('CONFIG',repr(c))));d.write(local/'submit.json',value);print(json.dumps({k:value[k] for k in ('status','pid','commit','archive_SHA_comparison','remote_compile','source_only')}))
if __name__=='__main__':main()
