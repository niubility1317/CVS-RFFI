import hashlib,io,json,subprocess,tarfile
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
SHARED=['experiments/cvs_rdrop_fold20/'+n for n in ('__init__.py','design.py','resources.py')]+['experiments/cvs_evidence_compensation20/original_gate.py']
def main():
 local=d.ROOT/'local_artifacts'/d.RELEASE;local.mkdir(parents=True,exist_ok=True);git='C:/Program Files/Git/cmd/git.exe';paths=['experiments/cvs_rx_teacher20',*SHARED]
 if subprocess.check_output([git,'-C',str(d.ROOT),'status','--porcelain','--',*paths],text=True).strip():raise ValueError('Actual teacher code/config must be committed')
 commit=subprocess.check_output([git,'-C',str(d.ROOT),'rev-parse','HEAD'],text=True).strip();remote=d.PROJECT+'/releases/'+d.RELEASE
 c=dict(release=remote,archive=remote+'.tar.gz',root=d.BASE.as_posix(),dependency=d.DEPENDENCY_RELEASE,dependency_commit=d.DEPENDENCY_COMMIT,fold=d.FOLD_RELEASE,python=PYTHON,commit=commit,A0_root=d.PROJECT+'/runs/'+d.parent.RUN+'/BASE_SSL-s'+str(d.SEED)+'/source')
 preflight=r'''
import json,os,subprocess,shutil
from pathlib import Path
c=CONFIG
assert os.geteuid()!=0 and subprocess.check_output(['whoami'],text=True).strip()=='szu2070436088' and subprocess.check_output(['hostname'],text=True).strip()=='dell-DSS8440'
assert Path(c['dependency']).is_dir() and Path(c['fold']).is_dir() and (Path(c['dependency'])/'release_commit.txt').read_text().strip()==c['dependency_commit']
a=Path(c['A0_root']);v=json.loads((a/'completion.json').read_text());assert v['status']=='SOURCE_TRAINED' and v['optimizer_steps']==4440 and v['initialization']=='scratch' and not v['checkpoint_sources'] and not v['ancestors'] and not v['target_access'] and (a/'final_ssdg.pth').is_file()
assert not any(Path(c[k]).exists() for k in ('release','archive','root')) and shutil.disk_usage(c['release'].rsplit('/releases/',1)[0]).free>2*1024**3
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,ready_rows=2,epochs_per_row=20,optimizer_steps_total=8880,CPU_threads_per_worker=2,data_loader_workers=0,source_only=True,
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
import json,hashlib,tarfile,subprocess,os,time
from pathlib import Path
c=CONFIG;r=Path(c['release']);archive=Path(c['archive'])
if r.exists() or Path(c['root']).exists():raise FileExistsError('Existing immutable teacher release/run; reconcile before retry')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release archive transfer mismatch')
r.mkdir()
with tarfile.open(archive) as source:
 for m in source.getmembers():
  if not m.isfile() or not (r/m.name).resolve().is_relative_to(r.resolve()):raise ValueError('Invalid archive path')
 source.extractall(r)
subprocess.run([c['python'],'-m','compileall','-q',str(r/'experiments')],cwd=r,check=True)
cmd=[c['python'],'-u',str(r/'experiments/cvs_rx_teacher20/runner.py'),str(r/'experiments/cvs_rx_teacher20/configs/matrix.json')];log=r/'dispatcher.stdout.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('x') as stream:p=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
proc=Path('/proc')/str(p.pid);first=dict(pid=p.pid,cwd=str((proc/'cwd').resolve()),argv=(proc/'cmdline').read_bytes().decode().split('\0')) if proc.exists() else None
start=log.stat().st_size;deadline=time.monotonic()+5
while time.monotonic()<deadline and p.poll() is None and log.stat().st_size<=start:time.sleep(.1)
v=dict(status='SUBMITTED',pid=p.pid,cwd=str(r),argv=cmd,commit=c['commit'],first_process=first,dispatcher_CUDA_VISIBLE_DEVICES='',archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',log_first_bytes=start,log_last_bytes=log.stat().st_size,log_grew=log.stat().st_size>start,child_returncode_if_finished=p.poll());(r/'submit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''
 v=json.loads(ssh(script.replace('CONFIG',repr(c))));d.write(local/'submit.json',v);print(json.dumps({k:v[k] for k in ('status','pid','commit','archive_SHA_comparison','remote_compile','log_grew')}))
if __name__=='__main__':main()
