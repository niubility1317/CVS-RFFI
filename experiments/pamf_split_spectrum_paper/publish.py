"""Publish the fixed missing-matrix release after local validation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison import publish as transport
from .plan import RUN

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);root=Path(rt['run_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile prior launch')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_split_spectrum_paper')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_split_spectrum_paper.runner','dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(release/'controller.log'),archive_transfer='VERIFIED',compile='PASS',code_commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''


def publish(repo,out):
    repo,out=Path(repo),Path(out)
    transport.CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote branch OID mismatch')
    if git('status','--porcelain','--','experiments/pamf_split_spectrum_paper','automation_reports/CV-SincNet/'+RUN).strip():raise ValueError('Commit experiment before publishing')
    ctx=json.loads((out/'context.json').read_text(encoding='utf-8')); project=ctx['runtime']['project'];release=project+'/releases/'+RUN
    rt=dict(run_id=RUN,project=project,python=ctx['runtime']['python'],frozen_sources=ctx['runtime']['frozen_sources'],
            code_commit=commit,release_root=release,run_root=project+'/runs/'+RUN,runtime_file=release+'/runtime.json',
            context=release+'/context.json',allowed_gpus=list(range(8)),launch_owner='codex-pamf-split-spectrum-20261109')
    archive=out/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile existing submission')
    roots=('experiments','code','baselines','comparison_suite')
    tracked=set(git('ls-files','-z','--',*roots).decode().split('\0'));src=git('archive','--format=tar',commit,'--',*roots)
    with tarfile.open(fileobj=io.BytesIO(src)) as source,tarfile.open(archive,'w:gz') as dest:
        for m in source:
            if m.name in tracked and m.isfile() and Path(m.name).suffix in ('.py','.json','.txt','.yaml','.yml','.toml'):
                dest.addfile(m,source.extractfile(m))
        for name,data in [('context.json',(out/'context.json').read_bytes()),
                          ('release_commit.txt',(commit+'\n').encode('utf-8')),
                          ('experiment.json',git('show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json')),
                          ('external_experiment.json',git('show',commit+':automation_reports/CV-SincNet/20261023-phase1-practical-residual-full2x2-m600-r01/experiment.json')),
                          ('historical_architecture.csv',Path('E:/type10-7/exports/results/20261014-sigtime-purece-e200-eight-seed-clean-r01/comparison_per_seed.csv').read_bytes())]:
            m=tarfile.TarInfo(name);m.size=len(data);dest.addfile(m,io.BytesIO(data))
    pkg=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,obj in [('runtime.json',rt),('package.json',pkg)]: (out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*transport.CONNECTION,str(archive),'N607:'+pkg['archive_remote']],check=True,timeout=120)
    result=transport.ssh(REMOTE.replace('PACKAGE',repr(pkg)));(out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.repo,a.output)
