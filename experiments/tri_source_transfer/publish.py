"""Immutable, CPU-only release using the established ordinary-account route."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from .register import RUN


REMOTE = r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE
project=Path(p['project']);release=Path(p['release']);root=Path(p['root']);archive=Path(p['archive'])
if os.geteuid()==0 or Path.home()!=project.parent.parent:raise ValueError('Wrong ordinary account')
if release.exists() or root.exists():raise FileExistsError('Existing run; reconcile before retry')
if shutil.disk_usage(project).free<2*1024**3:raise ValueError('Insufficient disk')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Transfer mismatch')
release.mkdir();root.mkdir()
with tarfile.open(archive) as b:
 for m in b.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Unsafe release member')
 b.extractall(release)
old=json.loads(Path(p['runtime']).read_text())
runtime={'project':old['project'],'frozen_sources':old['frozen_sources']}
(root/'runtime.json').write_text(json.dumps(runtime,indent=2))
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([p['python'],'-m','compileall','-q',str(release/'experiments/tri_source_transfer')],cwd=release,env=env,check=True)
with (root/'focused_checks.log').open('x') as log:
 subprocess.run([p['python'],'-m','unittest','discover','-s','tests','-p','test_tri_source_transfer.py'],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
cmd=[p['python'],'-u','-m','experiments.tri_source_transfer.diagnostic','--runtime',str(root/'runtime.json'),'--output',str(root)]
with (root/'diagnostic.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(root/'diagnostic.log'),archive_transfer='VERIFIED',remote_compile='PASS',focused_checks='PASS',gpu=None)
(root/'submit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


def publish(repo,out,ssh_config):
    repo,out=Path(repo),Path(out);out.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Unpublished commit')
    prior=json.loads(Path('E:/type10-7/local_artifacts/pamf_repl_weight_clean_20261115/submit.json').read_text(encoding='utf-8'))
    runtime=prior['argv'][-1];project=runtime.split('/releases/')[0]
    archive=out/(RUN+'-release.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile existing release first')
    paths=('experiments','code','baselines','comparison_suite','tests/test_tri_source_transfer.py')
    tracked={p for p in git('ls-files','-z','--',*paths).decode().split('\0') if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    source=git('archive','--format=tar',commit,'--',*paths)
    with tarfile.open(fileobj=io.BytesIO(source)) as original,tarfile.open(archive,'w:gz') as target:
        for member in original:
            if member.name not in tracked:continue
            data=original.extractfile(member).read();info=tarfile.TarInfo(member.name);info.size=len(data);info.mode=0o644
            target.addfile(info,io.BytesIO(data))
    package=dict(project=project,release=project+'/releases/'+RUN+'-release',root=project+'/runs/'+RUN,
        python=prior['argv'][0],runtime=runtime,archive=project+'/releases/'+archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    conn=['-F',ssh_config,'-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*conn,str(archive),'N607:'+package['archive']],check=True,timeout=90)
    result=subprocess.run(['ssh',*conn,'-T','N607','python3 -'],input=REMOTE.replace('PACKAGE',repr(package)).encode(),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError((result.stderr+result.stdout).decode(errors='replace'))
    value=json.loads(result.stdout);value['code_commit']=commit
    (out/'submit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--ssh-config',required=True);a=p.parse_args()
    publish(a.repo,a.output,a.ssh_config)
