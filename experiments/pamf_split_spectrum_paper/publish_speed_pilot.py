"""Land a separate controller without changing the live model release."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison import publish as transport

RUN='20261109-phase1-pamf-split-spectrum-paper-m226-r01'
CONTROL=RUN+'-speed-pilot-r01'
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['control_release']);root=Path(rt['control_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile existing pilot')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_split_spectrum_paper/speed_pilot.py')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_split_spectrum_paper.speed_pilot','--config',str(release/'runtime.json')]
with (release/'controller.log').open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,archive_transfer='VERIFIED',compile='PASS')))
'''


def publish(repo,out):
    repo,out=Path(repo),Path(out)
    transport.CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote branch differs')
    if git('status','--porcelain','--','experiments/pamf_split_spectrum_paper/speed_pilot.py').strip():raise ValueError('Commit controller')
    ctx=json.loads((out/'context.json').read_text(encoding='utf8'));project=ctx['runtime']['project']
    source=project+'/releases/'+RUN; release=project+'/releases/'+CONTROL
    rt=dict(project=project,python=ctx['runtime']['python'],source_release=source,release_root=source,
            control_release=release,run_root=project+'/runs/'+RUN,control_root=project+'/runs/'+RUN+'/controls/'+CONTROL,
            source_model_commit='0a94a2e0bef5936b68b31783ccd4c8ef8b7f8240',code_commit=commit,
            launch_owner='codex-pamf-split-spectrum-pilot')
    archive=out/(CONTROL+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior attempt')
    roots=('experiments','code','baselines','comparison_suite')
    tracked=set(git('ls-files','-z','--',*roots).decode().split('\0'));src=git('archive','--format=tar',commit,'--',*roots)
    with tarfile.open(fileobj=io.BytesIO(src)) as inp,tarfile.open(archive,'w:gz') as dest:
        for m in inp:
            if m.name in tracked and m.isfile() and Path(m.name).suffix in ('.py','.json','.txt','.yaml','.yml','.toml'):
                dest.addfile(m,inp.extractfile(m))
    pkg=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    subprocess.run(['scp',*transport.CONNECTION,str(archive),'N607:'+pkg['archive_remote']],check=True,timeout=120)
    result=transport.ssh(REMOTE.replace('PACKAGE',repr(pkg)))
    (out/(CONTROL+'-submit.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    publish(a.repo,a.output)
