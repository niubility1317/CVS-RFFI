"""Publish one immutable CPU-only source diagnostic release on N607."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from .register_source_adamw import RUN


REMOTE = r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE
project=Path(p['project']); release=Path(p['release']); root=Path(p['root']); archive=Path(p['archive'])
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=str(project).split('/')[2]:
 raise ValueError('Ordinary account required')
if release.exists() or root.exists(): raise FileExistsError('Existing diagnostic owner')
if not Path(p['runtime']).is_file() or not Path(p['python']).is_file(): raise FileNotFoundError('Existing runtime')
if shutil.disk_usage(project).free < 2*1024**3: raise ValueError('Insufficient disk space')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=p['sha256']: raise ValueError('Archive transfer mismatch')
release.mkdir();root.mkdir()
with tarfile.open(archive) as bundle:
 for member in bundle.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:
   raise ValueError('Invalid archive member')
 bundle.extractall(release)
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([p['python'],'-m','compileall','-q',str(release/'experiments/pamf_repl_weight')],cwd=release,env=env,check=True)
with (root/'focused_checks.log').open('x') as checklog:
 for checks in ([p['python'],'tests/test_pamf_tri_adamw_diagnostic.py'],
                [p['python'],'tests/test_pamf_paper_completion.py','MatrixAndPermissions.test_no_target_truth_in_training']):
  subprocess.run(checks,cwd=release,env=env,stdout=checklog,stderr=subprocess.STDOUT,check=True)
cmd=[p['python'],'-u','-m','experiments.pamf_repl_weight.source_adamw_diagnostic',p['runtime']]
with (root/'diagnostic.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,
            log=str(root/'diagnostic.log'),archive_transfer='VERIFIED',remote_compile='PASS',
            focused_checks='PASS',gpu=None)
(root/'submit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''


def publish(repo, output, ssh_config):
    repo, out = Path(repo), Path(output)
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args])
    commit = git('rev-parse','HEAD').decode().strip()
    branch = git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0] != commit:
        raise ValueError('Current branch must be pushed before publication')
    prior = json.loads((Path('E:/type10-7/local_artifacts/pamf_repl_weight_clean_20261115')/'submit.json').read_text(encoding='utf-8'))
    python = prior['argv'][0]
    runtime = prior['argv'][-1]
    project = runtime.split('/releases/')[0]
    release = project+'/releases/'+RUN+'-release'
    root = project+'/runs/'+RUN
    archive = out/(RUN+'-release.tar.gz')
    if archive.exists():
        raise FileExistsError('Reconcile previous release before retry')
    paths=('experiments','code','baselines','comparison_suite','tests/test_pamf_tri_adamw_diagnostic.py',
           'tests/test_pamf_paper_completion.py')
    tracked = {p for p in git('ls-files','-z','--',*paths).decode().split('\0')
               if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    source = git('archive','--format=tar',commit,'--',*paths)
    with tarfile.open(fileobj=io.BytesIO(source)) as original, tarfile.open(archive,'w:gz') as target:
        for member in original:
            if member.name not in tracked:
                continue
            value=original.extractfile(member).read()
            info=tarfile.TarInfo(member.name);info.size=len(value);info.mode=0o644
            target.addfile(info,io.BytesIO(value))
    package=dict(project=project,release=release,root=root,python=python,runtime=runtime,
                 archive=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    connection=['-F',str(ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*connection,str(archive),'N607:'+package['archive']],check=True,timeout=90)
    proc=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],
                        input=REMOTE.replace('PACKAGE',repr(package)).encode(),capture_output=True,timeout=60)
    if proc.returncode:
        raise RuntimeError((proc.stderr+proc.stdout).decode(errors='replace'))
    result=json.loads(proc.stdout)
    result['code_commit']=commit
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--ssh-config',required=True)
    args=parser.parse_args()
    publish(args.repo,args.output,args.ssh_config)
