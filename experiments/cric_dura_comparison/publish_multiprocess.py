"""Publish only resource orchestration; immutable scientific commands stay fixed."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from .publish import ROOT,CONNECTION,ssh

RUN='20261021-phase1-T1-DURA-multiprocess-control-r01'

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;rt=c['runtime'];release=Path(rt['release_root'])
if release.exists() or Path(rt['control_root']).exists():raise FileExistsError('Existing multiprocess owner; reconcile before retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
for key in ('previous_state','previous_runtime','jobs','external_pipeline_state'):
 if not Path(rt[key]).is_file():raise FileNotFoundError(rt[key])
if not Path(rt['worker_release']).is_dir():raise FileNotFoundError('Original scientific release missing')
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid release member')
 tar.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.cric_dura_comparison.multiprocess_control','--config',rt['runtime_file']]
with (release/'continuation.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo,output):
    repo=Path(repo);output=Path(output)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Upstream OID mismatch')
    previous=json.loads((output/'continuation_runtime.json').read_text(encoding='utf-8'));project=previous['project']
    rt=dict(previous,run_id=RUN,release_root=project+'/releases/'+RUN,control_root=project+'/runs/'+RUN,
        runtime_file=project+'/releases/'+RUN+'/runtime.json',code_commit=commit,launch_owner='codex-root-multiprocess-control-20261021',
        worker_release=previous['release_root'],worker_commit=previous['code_commit'],previous_state=previous['control_root']+'/pipeline_state.json',
        previous_runtime=previous['runtime_file'],previous_control_root=previous['control_root'],
        external_pipeline_state=project+'/runs/20261021-phase1-dg-sourcecount1to6-m432-r01/pipeline_state.json')
    if rt['worker_commit']!='3e8806900449e97f3e8fa62822322932168c6447':raise ValueError('Scientific worker release changed')
    names=('__init__.py','io.py','pipeline.py','source_queue.py','multiprocess_control.py')
    paths=['experiments/cric_dura_comparison/'+name for name in names]
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Uncommitted controller sources')
    archive=output/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing controller archive; reconcile submission')
    with tarfile.open(archive,'w:gz') as tar:
        for name,blob in [(p,git('show',commit+':'+p)) for p in paths]+[('release_commit.txt',(commit+'\n').encode())]:
            member=tarfile.TarInfo(name);member.size=len(blob);member.mode=0o644;tar.addfile(member,io.BytesIO(blob))
    package=dict(runtime=rt,expected_user=project.split('/')[2],archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'multiprocess_runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    receipt=ssh(REMOTE.replace('CONFIG',repr(package)))
    (output/'multiprocess_submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();publish(a.repo,a.output)
