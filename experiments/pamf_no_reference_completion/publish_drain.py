"""Release a controller-only change, preserving the scientific worker releases."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cric_dura_comparison import publish as transport

NAME='20261110-phase1-pamf-no-reference-reserve-gpu01-r01'
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(p['release']);control=Path(rt['control_root']);previous=Path(rt['previous_control'])
if release.exists() or control.exists() or (previous/'handoff.json').exists():raise FileExistsError('Reconcile previous release/handoff')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_no_reference_completion')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_no_reference_completion.drain_control','--config',str(release/'runtime.json')]
with (release/'controller.log').open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,archive_transfer='VERIFIED',compile='PASS')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def publish(repo,out):
    repo,out=Path(repo),Path(out);transport.CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();old=json.loads((out/'speed_runtime.json').read_text(encoding='utf-8'))
    project=old['project'];new_release=project+'/releases/'+NAME
    rt=dict(old,code_commit=commit,previous_release=old['release_root'],previous_control=old['control_root'],
        control_root=old['run_root']+'/controls/'+NAME,launch_owner=NAME,dispatch_gpus=[2,3,4,5,6,7])
    archive=out/(NAME+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile existing submission')
    roots=('experiments','code','baselines','comparison_suite');tracked=set(git('ls-files','-z','--',*roots).decode().split('\0'))
    with tarfile.open(fileobj=io.BytesIO(git('archive','--format=tar',commit,'--',*roots))) as inp,tarfile.open(archive,'w:gz') as dest:
        for member in inp:
            if member.name in tracked and member.isfile() and Path(member.name).suffix in ('.py','.json','.txt','.yaml','.yml','.toml'):
                dest.addfile(member,inp.extractfile(member))
    pkg=dict(runtime=rt,release=new_release,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'drain_runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*transport.CONNECTION,str(archive),'N607:'+pkg['archive_remote']],check=True,timeout=120)
    result=transport.ssh(REMOTE.replace('PACKAGE',repr(pkg)))
    (out/'drain_submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.repo,a.output)
