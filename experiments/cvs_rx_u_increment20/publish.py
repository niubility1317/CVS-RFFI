"""One immutable ordinary-account release for the source-only RX1 fold."""
import hashlib,io,json,subprocess,tarfile
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'

def main():
    local=d.ROOT/'local_artifacts'/d.RELEASE;local.mkdir(parents=True,exist_ok=True)
    git='C:/Program Files/Git/cmd/git.exe';commit=subprocess.check_output([git,'-C',str(d.ROOT),'rev-parse','HEAD'],text=True).strip()
    remote=d.PROJECT+'/releases/'+d.RELEASE;archive_remote=remote+'.tar.gz'
    config=dict(release=remote,archive=archive_remote,run_root=d.PROJECT+'/runs/'+d.RUN,dependency=d.DEPENDENCY_RELEASE,python=PYTHON)
    preflight=r'''
import json,subprocess,shutil
from pathlib import Path
c=CONFIG
assert subprocess.check_output(['whoami'],text=True).strip()=='szu2070436088'
assert subprocess.check_output(['hostname'],text=True).strip()=='dell-DSS8440'
assert Path(c['dependency']).is_dir() and not any(Path(c[k]).exists() for k in ('release','archive','run_root'))
assert shutil.disk_usage(c['release'].rsplit('/releases/',1)[0]).free>2*1024**3
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
choices=[]
for line in gpus.splitlines():
 index,uuid,free=[part.strip() for part in line.split(',')]
 pids={row.split(',')[0].strip() for row in apps.splitlines() if uuid in row}
 if len(pids)<3 and float(free)>3072:choices.append((len(pids),-float(free),int(index),uuid))
if not choices:raise RuntimeError('No source-fold GPU capacity now; no process intervention')
count,negative_free,index,uuid=min(choices)
print(json.dumps(dict(status='VERIFIED',GPU_index=index,GPU_uuid=uuid,total_existing_GPU_processes=count,free_MiB=-negative_free,total_process_cap=3)))
'''
    pre=json.loads(ssh(preflight.replace('CONFIG',repr(config))));d.write(local/'preflight.json',pre)
    raw=subprocess.check_output([git,'-C',str(d.ROOT),'archive',commit,'experiments/cvs_rx_u_increment20','experiments/cvs_feature_sequence50/source.py'])
    archive=local/(d.RELEASE+'.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as packed:
        for member in source.getmembers():
            if member.isfile():packed.addfile(member,source.extractfile(member))
        data=(commit+'\n').encode();member=tarfile.TarInfo('release_commit.txt');member.size=len(data);packed.addfile(member,io.BytesIO(data))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+archive_remote],check=True,capture_output=True)
    script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;r=Path(c['release']);archive=Path(c['archive'])
if r.exists() or Path(c['run_root']).exists():raise FileExistsError('Reconcile existing release/run; preserve all output')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release archive transfer mismatch')
r.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not member.isfile() or not (r/member.name).resolve().is_relative_to(r.resolve()):raise ValueError('Invalid archive path')
 tar.extractall(r)
subprocess.run([c['python'],'-m','compileall','-q',str(r/'experiments')],cwd=r,check=True)
request=r/'experiments/cvs_rx_u_increment20/configs/source_fold.json'
command=[c['python'],'-u',str(r/'experiments/cvs_rx_u_increment20/runner.py'),str(request)]
log=r/'source_fold.stdout.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(c['GPU_index']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('x') as stream:child=subprocess.Popen(command,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
def process():
 p=Path('/proc')/str(child.pid)
 if not p.exists():return None
 try:return dict(pid=child.pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0'))
 except OSError:return dict(pid=child.pid,state='UNKNOWN')
first=process();start=log.stat().st_size;deadline=time.monotonic()+3
while time.monotonic()<deadline and log.stat().st_size<=start and child.poll() is None:time.sleep(.1)
v=dict(status='SUBMITTED',pid=child.pid,cwd=str(r),argv=command,commit=(r/'release_commit.txt').read_text().strip(),GPU_index=c['GPU_index'],GPU_uuid=c['GPU_uuid'],
 archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',first_process=first,last_process=process(),log_first_bytes=start,log_last_bytes=log.stat().st_size,log_grew=log.stat().st_size>start,child_returncode_if_finished=child.poll())
(r/'submit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''
    config.update(sha256=digest,GPU_index=pre['GPU_index'],GPU_uuid=pre['GPU_uuid'])
    result=json.loads(ssh(script.replace('CONFIG',repr(config))));d.write(local/'submit.json',result)
    print(json.dumps(dict(status=result['status'],pid=result['pid'],GPU_index=result['GPU_index'],commit=result['commit'],archive_SHA_comparison=result['archive_SHA_comparison'],remote_compile=result['remote_compile'],log_grew=result['log_grew'])))
if __name__=='__main__':main()
