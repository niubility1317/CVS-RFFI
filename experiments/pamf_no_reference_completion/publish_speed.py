"""Publish a technical speed controller for the existing immutable run."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cric_dura_comparison import publish as transport

NAME='20261109-phase1-pamf-no-reference-speed-control-r03'
RUN='20261109-phase1-pamf-no-reference-completion-m190-r01'
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);control=Path(rt['control_root']);source=Path(rt['source_release'])
if release.exists() or control.exists() or (Path(rt['run_root'])/'handoff.json').exists():raise FileExistsError('Reconcile existing release or handoff')
if not (source/'historical_architecture.csv').is_file():raise FileNotFoundError('Original historical input missing')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_no_reference_completion')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_no_reference_completion.speed_control','--config',str(release/'runtime.json')]
with (release/'controller.log').open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,archive_transfer='VERIFIED',compile='PASS')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def publish(repo,out):
    repo,out=Path(repo),Path(out)
    transport.CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip()
    ctx=json.loads((out/'context.json').read_text(encoding='utf-8'));project=ctx['runtime']['project'];source=project+'/releases/'+RUN;root=project+'/runs/'+RUN
    release=project+'/releases/'+NAME
    rt=dict(project=project,python=ctx['runtime']['python'],frozen_sources=ctx['runtime']['frozen_sources'],
        source_release=source,source_model_commit='33e46b2ab2b726900a2f0d9e775b0a2778820f39',
        code_commit=commit,release_root=release,run_root=root,control_root=root+'/controls/'+NAME,
        context=release+'/context.json',launch_owner=NAME)
    archive=out/(NAME+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile existing submission')
    roots=('experiments','code','baselines','comparison_suite');tracked=set(git('ls-files','-z','--',*roots).decode().split('\0'))
    src=git('archive','--format=tar',commit,'--',*roots)
    with tarfile.open(fileobj=io.BytesIO(src)) as inp,tarfile.open(archive,'w:gz') as dest:
        for m in inp:
            if m.name in tracked and m.isfile() and Path(m.name).suffix in ('.py','.json','.txt','.yaml','.yml','.toml'):dest.addfile(m,inp.extractfile(m))
        data=(out/'context.json').read_bytes();m=tarfile.TarInfo('context.json');m.size=len(data);dest.addfile(m,io.BytesIO(data))
    pkg=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'speed_runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*transport.CONNECTION,str(archive),'N607:'+pkg['archive_remote']],check=True,timeout=120)
    result=transport.ssh(REMOTE.replace('PACKAGE',repr(pkg)))
    (out/'speed_submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.repo,a.output)
