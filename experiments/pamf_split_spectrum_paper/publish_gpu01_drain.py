"""Publish one reservation controller for each explicitly authorized run."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison import publish as transport

PROJECT_RUNS={
 'current':dict(run='20261109-phase1-pamf-split-spectrum-paper-m226-r01',
   controller='20261109-phase1-pamf-split-spectrum-paper-m226-r01-speed-pilot-r01',
   controller_module='experiments.pamf_split_spectrum_paper.speed_pilot',
   worker_module='experiments.pamf_split_spectrum_paper.runner',
   analyze_module='experiments.pamf_split_spectrum_paper.analyze',gpu7_limit=1,control_jobs=False,
   source_model_commit='0a94a2e0bef5936b68b31783ccd4c8ef8b7f8240'),
 'old':dict(run='20261109-phase1-pamf-no-reference-completion-m190-r01',
   controller='20261110-phase1-pamf-no-reference-reserve-gpu01-r01',
   controller_module='experiments.pamf_no_reference_completion.drain_control',
   worker_module='experiments.pamf_no_reference_completion.runner',
   analyze_module='experiments.pamf_no_reference_completion.analyze',gpu7_limit=2,control_jobs=True,
   source_model_commit='33e46b2ab2b726900a2f0d9e775b0a2778820f39')}

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['control_release']);root=Path(rt['control_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile existing control')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_split_spectrum_paper/gpu01_drain.py')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_split_spectrum_paper.gpu01_drain','--config',str(release/'runtime.json')]
with (release/'controller.log').open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,archive_transfer='VERIFIED',compile='PASS')))
'''


def publish(repo,out,mode):
    repo,out=Path(repo),Path(out);spec=PROJECT_RUNS[mode]
    transport.CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote branch differs')
    if git('status','--porcelain','--','experiments/pamf_split_spectrum_paper/gpu01_drain.py').strip():raise ValueError('Commit control code')
    ctx=json.loads((out/'context.json').read_text(encoding='utf8'));project=ctx['runtime']['project']
    run=spec['run'];source_control=project+'/runs/'+run+'/controls/'+spec['controller']
    control=run+'-gpu01-release-r01';release=project+'/releases/'+control
    rt=dict(project=project,python=ctx['runtime']['python'],run_root=project+'/runs/'+run,
        source_release=project+'/releases/'+run,source_control_release=project+'/releases/'+spec['controller'],
        source_control_state=source_control+'/pipeline_state.json',
        source_jobs=source_control+'/jobs.json' if spec['control_jobs'] else project+'/runs/'+run+'/jobs.json',
        source_control_module=spec['controller_module'],worker_module=spec['worker_module'],
        analyze_module=spec['analyze_module'],gpu7_limit=spec['gpu7_limit'],control_release=release,
        control_root=project+'/runs/'+run+'/controls/'+control,source_model_commit=spec['source_model_commit'],
        code_commit=commit,launch_owner='codex-reserve-gpu01-'+mode)
    archive=out/(control+'.tar.gz')
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
    (out/(control+'-submit.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=PROJECT_RUNS,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.mode)
