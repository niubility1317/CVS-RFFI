"""One ordinary-account CPU release; no GPU allocation or process intervention."""
import hashlib,io,json,subprocess,tarfile
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'

def main():
    local=d.ROOT/'local_artifacts'/d.RELEASE;local.mkdir(parents=True,exist_ok=True)
    git='C:/Program Files/Git/cmd/git.exe';prefix='experiments/cvs_rdrop_source20'
    commit=subprocess.check_output([git,'-C',str(d.ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output([git,'-C',str(d.ROOT),'status','--porcelain','--',prefix],text=True).strip():raise ValueError('Commit actual source diagnostic code/config before release')
    remote=d.PROJECT+'/releases/'+d.RELEASE
    c=dict(release=remote,archive=remote+'.tar.gz',run_root=d.PROJECT+'/runs/'+d.RUN,
        dependency=d.DEPENDENCY_RELEASE,dependency_commit=d.DEPENDENCY_COMMIT,python=PYTHON,commit=commit)
    preflight=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
assert os.geteuid()!=0 and subprocess.check_output(['whoami'],text=True).strip()=='szu2070436088'
assert subprocess.check_output(['hostname'],text=True).strip()=='dell-DSS8440'
assert Path(c['dependency']).is_dir() and (Path(c['dependency'])/'release_commit.txt').read_text().strip()==c['dependency_commit']
assert not any(Path(c[key]).exists() for key in ('release','archive','run_root'))
assert shutil.disk_usage(c['release'].rsplit('/releases/',1)[0]).free>1024**3
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,CPU_threads=2,CUDA_VISIBLE_DEVICES='',GPU_allocation=0,existing_GPU_processes=len(apps.splitlines()),output_root_absent=True)))
'''
    d.write(local/'preflight.json',json.loads(ssh(preflight.replace('CONFIG',repr(c)))))
    raw=subprocess.check_output([git,'-C',str(d.ROOT),'archive',commit,prefix])
    record=subprocess.check_output([git,'-C',str(d.ROOT),'show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'])
    archive=local/(d.RELEASE+'.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(raw)) as original,tarfile.open(archive,'w:gz') as packed:
        for member in original.getmembers():
            if member.isfile():packed.addfile(member,original.extractfile(member))
        for name,data in [('release_commit.txt',(commit+'\n').encode()),('registration/experiment.json',record)]:
            member=tarfile.TarInfo(name);member.size=len(data);packed.addfile(member,io.BytesIO(data))
    c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+c['archive']],capture_output=True,check=True)
    script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;r=Path(c['release']);archive=Path(c['archive'])
if r.exists() or Path(c['run_root']).exists():raise FileExistsError('Reconcile existing release/run before any retry')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release archive transfer mismatch')
r.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (r/member.name).resolve().is_relative_to(r.resolve()):raise ValueError('Invalid archive member path')
 source.extractall(r)
subprocess.run([c['python'],'-m','compileall','-q',str(r/'experiments')],cwd=r,check=True)
command=[c['python'],'-u',str(r/'experiments/cvs_rdrop_source20/runner.py'),str(r/'experiments/cvs_rdrop_source20/configs/matrix.json')]
log=r/'source_rdrop.stdout.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('x') as stream:child=subprocess.Popen(command,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
def process():
 p=Path('/proc')/str(child.pid)
 if not p.exists():return None
 try:return dict(pid=child.pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0'),CUDA_VISIBLE_DEVICES=next((v.split('=',1)[1] for v in (p/'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')),None))
 except OSError:return dict(pid=child.pid,state='UNKNOWN')
first=process();start=log.stat().st_size;deadline=time.monotonic()+8
while time.monotonic()<deadline and log.stat().st_size<=start and child.poll() is None:time.sleep(.1)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
membership=str(child.pid) in {line.strip() for line in apps.splitlines()}
value=dict(status='SUBMITTED',pid=child.pid,cwd=str(r),argv=command,commit=c['commit'],CPU_threads=2,CUDA_VISIBLE_DEVICES='',GPU_allocation=0,
 own_PID_in_GPU_processes=membership,archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',first_process=first,last_process=process(),
 log_first_bytes=start,log_last_bytes=log.stat().st_size,log_grew=log.stat().st_size>start,child_returncode_if_finished=child.poll())
(r/'submit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
    value=json.loads(ssh(script.replace('CONFIG',repr(c))));d.write(local/'submit.json',value)
    print(json.dumps({key:value[key] for key in ('status','pid','commit','CPU_threads','GPU_allocation','archive_SHA_comparison','remote_compile','log_grew','own_PID_in_GPU_processes')}))

if __name__=='__main__':main()
