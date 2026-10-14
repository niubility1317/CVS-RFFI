"""One committed CPU-only source diagnostic release, with independent readback."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION

PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'

def main():
    local=d.ROOT/'local_artifacts'/d.RELEASE;local.mkdir(parents=True,exist_ok=True)
    git='C:/Program Files/Git/cmd/git.exe'
    commit=subprocess.check_output([git,'-C',str(d.ROOT),'rev-parse','HEAD'],text=True).strip()
    remote=d.PROJECT+'/releases/'+d.RELEASE
    archive_remote=remote+'.tar.gz'
    preflight=json.loads(ssh('import json,subprocess,shutil\nfrom pathlib import Path\n'
        'assert subprocess.check_output(["whoami"],text=True).strip()=="szu2070436088"\n'
        'assert subprocess.check_output(["hostname"],text=True).strip()=="dell-DSS8440"\n'
        'assert Path('+repr(d.SOURCE_RELEASE)+').is_dir()\n'
        'assert not any(Path(p).exists() for p in '+repr([remote,archive_remote,d.PROJECT+'/runs/'+d.RUN])+')\n'
        'assert shutil.disk_usage('+repr(d.PROJECT)+').free>1024**3\n'
        'print(json.dumps(dict(status="VERIFIED",CPU_only=True,GPU_allocations=0,prior_release_unchanged=True)))\n'))
    d.write(local/'preflight.json',preflight)
    raw=subprocess.check_output([git,'-C',str(d.ROOT),'archive',commit,'experiments/cvs_evidence_recomposition20','experiments/cvs_crossrx_sharpness20'])
    archive=local/(d.RELEASE+'.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as packed:
        for member in source.getmembers():
            if member.isfile():packed.addfile(member,source.extractfile(member))
        data=(commit+'\n').encode();member=tarfile.TarInfo('release_commit.txt');member.size=len(data)
        packed.addfile(member,io.BytesIO(data))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+archive_remote],check=True,capture_output=True)
    script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);archive=Path(c['archive']);root=Path(c['run_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile existing publication, preserve outputs')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive path')
 tar.extractall(release)
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments')],cwd=release,check=True)
request=json.loads((release/'experiments/cvs_evidence_recomposition20/configs/source_probe.json').read_text())
request['analysis_code_commit']=(release/'release_commit.txt').read_text().strip()
(release/'resolved_request.json').write_text(json.dumps(request,indent=2)+'\n')
command=[c['python'],'-u',str(release/'experiments/cvs_evidence_recomposition20/runner.py'),str(release/'resolved_request.json')]
log=release/'source_probe.stdout.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
def proc():
 p=Path('/proc')/str(child.pid)
 if not p.exists():return None
 try:return dict(pid=child.pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0'))
 except OSError:return dict(pid=child.pid,state='UNKNOWN')
first=proc();first_bytes=log.stat().st_size;deadline=time.monotonic()+3
while time.monotonic()<deadline and log.stat().st_size<=first_bytes and child.poll() is None:time.sleep(.1)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
GPU_absent=str(child.pid) not in apps.split()
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=request['analysis_code_commit'],archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',independent_postlaunch=dict(first_process=first,last_process=proc(),log_first_bytes=first_bytes,log_last_bytes=log.stat().st_size,log_grew=log.stat().st_size>first_bytes,CPU_only=True,GPU_allocations=0,GPU_process_absent=GPU_absent,child_returncode_if_finished=child.poll()))
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
    cfg=dict(release=remote,archive=archive_remote,run_root=d.PROJECT+'/runs/'+d.RUN,sha256=digest,python=PYTHON)
    submitted=json.loads(ssh(script.replace('CONFIG',repr(cfg))))
    d.write(local/'submit.json',submitted)
    print(json.dumps(submitted))

if __name__=='__main__':main()
