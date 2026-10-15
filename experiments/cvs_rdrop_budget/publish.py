import hashlib,io,json,subprocess,tarfile
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
SHARED=['experiments/cvs_rdrop_fold20/'+name for name in ('__init__.py','design.py','model.py','resources.py')]
def main():
 local=d.ROOT/'local_artifacts'/d.RELEASE;local.mkdir(parents=True,exist_ok=True);git='C:/Program Files/Git/cmd/git.exe'
 commit=subprocess.check_output([git,'-C',str(d.ROOT),'rev-parse','HEAD'],text=True).strip();paths=['experiments/cvs_rdrop_budget',*SHARED]
 if subprocess.check_output([git,'-C',str(d.ROOT),'status','--porcelain','--',*paths],text=True).strip():raise ValueError('Commit actual budget code/config before release')
 remote=d.PROJECT+'/releases/'+d.RELEASE
 c=dict(release=remote,archive=remote+'.tar.gz',run_root=d.BASE.as_posix(),dependency=d.DEPENDENCY_RELEASE,dependency_commit=d.DEPENDENCY_COMMIT,fold=d.FOLD_RELEASE,source_root=d.PROJECT+'/runs/'+d.SOURCE_RUN,python=PYTHON,commit=commit)
 preflight=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
assert os.geteuid()!=0 and subprocess.check_output(['whoami'],text=True).strip()=='szu2070436088'
assert subprocess.check_output(['hostname'],text=True).strip()=='dell-DSS8440'
assert Path(c['dependency']).is_dir() and Path(c['fold']).is_dir() and (Path(c['dependency'])/'release_commit.txt').read_text().strip()==c['dependency_commit']
source=json.loads((Path(c['source_root'])/'matrix_result.json').read_text())
assert source['status']=='SOURCE_MATRIX_COMPLETE' and source['actual_optimizer_steps']==8880 and source['qualification']['qualified'] and not source['target_access']
assert not any(Path(c[key]).exists() for key in ('release','archive','run_root'))
assert shutil.disk_usage(c['release'].rsplit('/releases/',1)[0]).free>2*1024**3
memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,ready_rows=12,budget_E20_models=6,budget_E50_models=6,total_model_epochs=420,total_steps=93240,
 CPU_threads_per_worker=2,data_loader_workers=0,logical_CPUs=os.cpu_count(),memory_available=memory.get('MemAvailable','').strip(),
 GPU_policy='All8 considered; existing shared lock includes every actual/pre-CUDA owner;cap3/free>=12000MiB;idle first',
 GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),GPU_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
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
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;r=Path(c['release']);archive=Path(c['archive'])
if r.exists() or Path(c['run_root']).exists():raise FileExistsError('Reconcile existing immutable release/run; never resubmit blindly')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release archive transfer mismatch')
r.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (r/member.name).resolve().is_relative_to(r.resolve()):raise ValueError('Invalid archive path')
 source.extractall(r)
subprocess.run([c['python'],'-m','compileall','-q',str(r/'experiments')],cwd=r,check=True)
cmd=[c['python'],'-u',str(r/'experiments/cvs_rdrop_budget/runner.py'),str(r/'experiments/cvs_rdrop_budget/configs/matrix.json')]
log=r/'dispatcher.stdout.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('x') as stream:child=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
p=Path('/proc')/str(child.pid);first=dict(pid=child.pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0')) if p.exists() else None
start=log.stat().st_size;deadline=time.monotonic()+5
while time.monotonic()<deadline and log.stat().st_size<=start and child.poll() is None:time.sleep(.1)
value=dict(status='SUBMITTED',pid=child.pid,cwd=str(r),argv=cmd,commit=c['commit'],dispatcher_CUDA_VISIBLE_DEVICES='',first_process=first,archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',log_first_bytes=start,log_last_bytes=log.stat().st_size,log_grew=log.stat().st_size>start,child_returncode_if_finished=child.poll())
(r/'submit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
 value=json.loads(ssh(script.replace('CONFIG',repr(c))));d.write(local/'submit.json',value);print(json.dumps({k:value[k] for k in ('status','pid','commit','archive_SHA_comparison','remote_compile','log_grew')}))
if __name__=='__main__':main()
