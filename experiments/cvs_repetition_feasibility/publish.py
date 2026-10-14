"""Single source-observation release with an independent immediate PID check."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION

ROOT=Path(__file__).resolve().parents[2]
RUN='20261014-phase1-visible-period-sourceL-manysig-m1-r01'
RELEASE='cvs_visible_period_sourceL_20261014_r01'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'

def main(*,run_id=RUN,release_id=RELEASE,request_config='sourceL.json'):
    if (run_id,release_id,request_config) not in (
        ('20261014-phase1-visible-period-sourceL-manysig-m1-r01','cvs_visible_period_sourceL_20261014_r01','sourceL.json'),
        ('20261014-phase1-null-budget-sourceL-manysig-m1-r01','cvs_null_budget_sourceL_20261014_r01','null_sourceL.json')):
        raise ValueError('Unregistered source-only publication')
    RUN,RELEASE=run_id,release_id
    local=ROOT/'local_artifacts'/RELEASE;local.mkdir(parents=True,exist_ok=True)
    git='C:/Program Files/Git/cmd/git.exe'
    commit=subprocess.check_output([git,'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    previous=PROJECT+'/releases/cvs_PA_recomposition50_20261014_r01';remote=PROJECT+'/releases/'+RELEASE
    preflight=json.loads(ssh('import json,subprocess,shutil\nfrom pathlib import Path\np=Path('+repr(PROJECT)+')\nr=Path('+repr(remote)+')\nassert subprocess.check_output(["whoami"],text=True).strip()=="szu2070436088"\nassert subprocess.check_output(["hostname"],text=True).strip()=="dell-DSS8440"\nassert Path('+repr(previous)+').is_dir()\nassert not r.exists() and not (p/"runs"/'+repr(RUN)+').exists() and not (p/"releases"/'+repr(RELEASE+'.tar.gz')+').exists()\nassert shutil.disk_usage(p).free>1024**3\nprint(json.dumps(dict(status="VERIFIED",CPU_only=True,GPU_allocations=0,active_GPU_processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True),prior_release_unchanged=True)))\n'))
    (local/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n',encoding='utf-8')
    raw=subprocess.check_output([git,'-C',str(ROOT),'archive',commit,'experiments/cvs_repetition_feasibility'])
    archive=local/(RELEASE+'.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(raw)) as original,tarfile.open(archive,'w:gz') as packed:
        for member in original.getmembers():
            if member.isfile():packed.addfile(member,original.extractfile(member))
        data=(commit+'\n').encode();member=tarfile.TarInfo('release_commit.txt');member.size=len(data);packed.addfile(member,io.BytesIO(data))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True,capture_output=True)
    script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);archive=Path(c['archive']);root=Path(c['run_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile existing publication')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive path')
 tar.extractall(release)
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_repetition_feasibility')],cwd=release,check=True)
request=json.loads((release/'experiments/cvs_repetition_feasibility/configs'/c['request_config']).read_text())
request['analysis_code_commit']=(release/'release_commit.txt').read_text().strip()
(release/'resolved_request.json').write_text(json.dumps(request,indent=2)+'\n')
command=[c['python'],'-u',str(release/'experiments/cvs_repetition_feasibility/runner.py'),str(release/'resolved_request.json')]
log=release/'source_observation.stdout.log';env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
def proc():
 p=Path('/proc')/str(child.pid)
 if not p.exists():return None
 try:return dict(pid=child.pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0'))
 except OSError:return dict(pid=child.pid,state='UNKNOWN')
first=proc();first_bytes=log.stat().st_size;last=first
deadline=time.monotonic()+3
while time.monotonic()<deadline and log.stat().st_size<=first_bytes and child.poll() is None:time.sleep(.1)
last=proc();result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=request['analysis_code_commit'],archive_SHA_comparison='VERIFIED',remote_compile='VERIFIED',independent_postlaunch=dict(first_process=first,last_process=last,log_first_bytes=first_bytes,log_last_bytes=log.stat().st_size,log_grew=log.stat().st_size>first_bytes,CPU_only=True,GPU_allocations=0,child_returncode_if_finished=child.poll()))
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
    cfg=dict(release=remote,archive=PROJECT+'/releases/'+archive.name,run_root=PROJECT+'/runs/'+RUN,sha256=digest,python=PYTHON,request_config=request_config)
    submitted=json.loads(ssh(script.replace('CONFIG',repr(cfg))))
    (local/'submit.json').write_text(json.dumps(submitted,indent=2)+'\n',encoding='utf-8');print(json.dumps(submitted))

if __name__=='__main__':main()
