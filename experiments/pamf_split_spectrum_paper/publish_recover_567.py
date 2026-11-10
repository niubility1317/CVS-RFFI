"""Publish the single checked recovery release for both registered runs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison import publish as transport

NAME='20261110-phase1-pamf-recovery-gpu567-r02'
CURRENT='20261109-phase1-pamf-split-spectrum-paper-m226-r01'
OLD='20261109-phase1-pamf-no-reference-completion-m190-r01'
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['source_release'])
if release.exists() or Path(rt['run_root']).exists():
 raise FileExistsError('Reconcile existing recovery before submitting')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:
 raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Unsafe archive member')
 tar.extractall(release)
for name,value in [('runtime.json',rt)]:
 (release/name).write_text(json.dumps(value,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_split_spectrum_paper'),str(release/'experiments/pamf_no_reference_completion/analyze.py')],cwd=release,env=env,check=True)
result={}
for action,config in [('dispatch','runtime.json')]:
 cmd=[rt['python'],'-u','-m',rt['worker_module'],action,'--config',str(release/config)]
 with (release/(action+'.log')).open('x') as f:
  child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 result[action]=dict(pid=child.pid,cmd=cmd)
print(json.dumps(dict(status='SUBMITTED_NOT_YET_VERIFIED',archive_transfer='VERIFIED',compile='PASS',processes=result)))
'''


def publish(repo,out):
    repo,out=Path(repo),Path(out)
    transport.CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote branch OID differs')
    if git('status','--porcelain','--','experiments/pamf_split_spectrum_paper','experiments/pamf_no_reference_completion/analyze.py').strip():raise ValueError('Commit recovery code')
    ctx=json.loads((out/'context.json').read_text(encoding='utf8'));project=ctx['runtime']['project'];release=project+'/releases/'+NAME
    run_root=project+'/runs/'+NAME;oldrun=project+'/runs/20261110-phase1-pamf-recovery-gpu567-r01'
    common=dict(project=project,python=ctx['runtime']['python'],frozen_sources=ctx['runtime']['frozen_sources'],
       code_commit=commit,source_release=release,release_root=release,worker_module='experiments.pamf_split_spectrum_paper.recover_567',
       launch_owner='codex-pamf-recovery-gpu567',gpu7_limit=1)
    rt=dict(common,run_root=run_root,old_run_root=oldrun,old_control=oldrun,
       context=release+'/context.json',runtime_file=release+'/runtime.json',analyze_module='experiments.pamf_split_spectrum_paper.analyze')
    archive=out/(NAME+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile existing submission')
    roots=('experiments','code','baselines','comparison_suite')
    tracked=set(git('ls-files','-z','--',*roots).decode().split('\0'));src=git('archive','--format=tar',commit,'--',*roots)
    with tarfile.open(fileobj=io.BytesIO(src)) as inp,tarfile.open(archive,'w:gz') as dest:
        for m in inp:
            if m.name in tracked and m.isfile() and Path(m.name).suffix in ('.py','.json','.txt','.yaml','.yml','.toml'):
                dest.addfile(m,inp.extractfile(m))
        for name,data in [('context.json',(out/'context.json').read_bytes()),('release_commit.txt',(commit+'\n').encode()),
          ('historical_architecture.csv',Path('E:/type10-7/exports/results/20261014-sigtime-purece-e200-eight-seed-clean-r01/comparison_per_seed.csv').read_bytes())]:
            m=tarfile.TarInfo(name);m.size=len(data);dest.addfile(m,io.BytesIO(data))
    pkg=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    subprocess.run(['scp',*transport.CONNECTION,str(archive),'N607:'+pkg['archive_remote']],check=True,timeout=120)
    result=transport.ssh(REMOTE.replace('PACKAGE',repr(pkg)))
    (out/(NAME+'-submit.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    publish(a.repo,a.output)
