"""Publish the approved resource-only 24-to-26 worker controller."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

RUN = '20261022-phase1-T1-DURA-multiprocess-control-r03'
WORKER_COMMIT = '3e8806900449e97f3e8fa62822322932168c6447'

REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;rt=c['runtime'];release=Path(rt['release_root'])
if release.exists() or Path(rt['control_root']).exists():raise FileExistsError('Existing release; reconcile, never retry blindly')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
for key in ('previous_state','previous_runtime','jobs','external_pipeline_state'):
 if not Path(rt[key]).is_file():raise FileNotFoundError(rt[key])
if not Path(rt['worker_release']).is_dir():raise FileNotFoundError('Immutable science release missing')
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive_remote']) as archive:
 for member in archive.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid archive path')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.cric_dura_comparison.multiprocess_control','--config',rt['runtime_file']]
with (release/'continuation.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
'''


def runtime_for(previous, commit):
    if previous['worker_commit'] != WORKER_COMMIT or previous['run_id'] != '20261022-phase1-T1-DURA-multiprocess-control-r02':
        raise ValueError('Unexpected prior owner or scientific release')
    project = previous['project']
    return dict(previous, run_id=RUN, release_root=project+'/releases/'+RUN,
                control_root=project+'/runs/'+RUN, runtime_file=project+'/releases/'+RUN+'/runtime.json',
                code_commit=commit, launch_owner='codex-load-balance-20261022',
                previous_state=previous['control_root']+'/pipeline_state.json',
                previous_runtime=previous['runtime_file'], previous_control_root=previous['control_root'],
                previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
                previous_controller_cwd=previous['release_root'], max_own_per_gpu=3, max_compute_per_gpu=3,
                load_balance=dict(boost_gpus=[0,6], max_gpu_workers=26))


def publish(repo, output, previous_file, ssh_config):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote branch OID differs')
    previous=json.loads(Path(previous_file).read_text(encoding='utf-8'))
    rt=runtime_for(previous,commit)
    paths=['experiments/cric_dura_comparison/'+name for name in
           ('__init__.py','io.py','pipeline.py','source_queue.py','multiprocess_control.py','resource_balance.py')]
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Uncommitted controller sources')
    archive=output/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive; reconcile submission')
    with tarfile.open(archive,'w:gz') as tar:
        for name,blob in [(p,git('show',commit+':'+p)) for p in paths]+[('release_commit.txt',(commit+'\n').encode())]:
            member=tarfile.TarInfo(name);member.size=len(blob);member.mode=0o644;tar.addfile(member,io.BytesIO(blob))
    package=dict(runtime=rt,expected_user=rt['project'].split('/')[2],archive_remote=rt['project']+'/releases/'+archive.name,
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp','-F',str(ssh_config),str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=subprocess.run(['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','N607',rt['python']+' -'],
                          input=REMOTE.replace('CONFIG',repr(package)),text=True,encoding='utf-8',capture_output=True,timeout=90)
    print(result.stdout);print(result.stderr)
    result.check_returncode()
    receipt=json.loads(result.stdout)
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for option in ('repo','output','previous','ssh-config'):parser.add_argument('--'+option,required=True)
    args=parser.parse_args();publish(args.repo,args.output,args.previous,args.ssh_config)
