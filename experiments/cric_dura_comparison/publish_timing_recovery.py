"""Recover only the failed timing diagnostic in a new immutable run."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from .publish import ROOT,CONNECTION,ssh

RUN='20261021-phase1-T1-DURA-latency-r02'
RELEASE='20261021-phase1-T1-DURA-timing-recovery-r01'

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;rt=c['runtime'];release=Path(rt['release_root'])
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Existing recovery; reconcile before retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
if not Path(rt['original_config']).is_file():raise FileNotFoundError(rt['original_config'])
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid release member')
 tar.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.cric_dura_comparison.timing_recovery','dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo,output):
    repo=Path(repo);output=Path(output)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Upstream OID differs')
    parent=json.loads((output/'continuation_runtime.json').read_text(encoding='utf-8'));p=parent['project']
    rt=dict(parent,run_id=RUN,release_root=p+'/releases/'+RELEASE,run_root=p+'/runs/'+RUN,
        runtime_file=p+'/releases/'+RELEASE+'/runtime.json',code_commit=commit,
        original_config=parent['run_root']+'/configs/20261021-phase1-T1-DURA-latency-r01.json',launch_owner='codex-root-timing-recovery-20261021')
    paths=[q.relative_to(ROOT).as_posix() for q in (ROOT/'experiments/cric_dura_comparison').glob('*.py')]
    paths +=[q.relative_to(ROOT).as_posix() for q in (ROOT/'experiments/cric_dura_comparison/vendor').rglob('*') if q.is_file() and (q.suffix in ('.py','.txt') or q.name=='LICENSE')]
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Uncommitted recovery sources')
    archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Recovery archive exists; reconcile')
    with tarfile.open(archive,'w:gz') as tar:
        for name,blob in [(path,git('show',commit+':'+path)) for path in paths]+[('release_commit.txt',(commit+'\n').encode())]:
            info=tarfile.TarInfo(name);info.size=len(blob);info.mode=0o644;tar.addfile(info,io.BytesIO(blob))
    c=dict(runtime=rt,expected_user=p.split('/')[2],archive_remote=p+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'timing_recovery_runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+c['archive_remote']],check=True,timeout=60)
    receipt=ssh(REMOTE.replace('CONFIG',repr(c)))
    (output/'timing_recovery_submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();publish(a.repo,a.output)
