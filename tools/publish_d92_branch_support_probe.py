"""Publish the pushed support-only probe and start its sole supervisor once."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from run_d92_branch_support_probe import validate_spec

ROOT=Path(__file__).resolve().parents[1]
FLAGS=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
PATHS=['code','tools/cvs_native_artifacts.py','tools/export_d92_branch_support_features.py',
       'tools/evaluate_d92_branch_support_probe.py','tools/run_d92_branch_support_probe.py',
       'configs/d92_branch_support_probe_frozen_20260929.json',
       'configs/d92_branch_support_probe_rx3_20260929.json','configs/d92_branch_support_probe_rx1_20260929.json']
REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
c=CONFIG
release=Path(c['release']);archive=Path(c['archive']);root=Path(c['root'])
if release.exists() or root.exists():raise FileExistsError('Existing output: reconcile, no repeated launch')
if shutil.disk_usage(release.parent).free<2*1024**3:raise RuntimeError('Insufficient disk')
used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not (release.parent/m.name).resolve().is_relative_to(release.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe archive')
 tar.extractall(release.parent)
py=c['python'];env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
subprocess.run([py,'-m','compileall','-q',str(release/'tools'),str(release/'code')],env=env,check=True)
argv=[py,'-u',str(release/'tools/run_d92_branch_support_probe.py'),'--spec',str(release/c['spec']),'--commit',c['commit']]
with (release/'run.log').open('x') as f:
 p=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=p.pid,argv=argv,cwd=str(release),commit=c['commit'],run=str(root),owner=c['owner'])
(release/'launch.json').write_text(json.dumps(launch,indent=2))
print(json.dumps(launch))
'''


def remote_archive_path(release,name):
    if '\\' in release or not release.startswith('/') or '/' in name or '\\' in name:
        raise ValueError('Remote paths must be POSIX paths')
    return str(PurePosixPath(release).parent/name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',default='configs/d92_branch_support_probe_20260929.json')
    p.add_argument('--resume-staged-commit',help='Explicit recovery only: original committed archive, verified no launch/output yet')
    a=p.parse_args();spec=json.loads((ROOT/a.spec).read_text(encoding='utf-8'));validate_spec(spec)
    release=spec['code']['cwd'];name=Path(release).name;remote_root=spec['execution']['remote_run_root']
    paths=PATHS+[a.spec]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=commit:
        raise ValueError('Release not pushed')
    if subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT,text=True).strip():
        raise ValueError('Uncommitted release paths')
    if a.resume_staged_commit:
        original=subprocess.check_output(['git','rev-parse',a.resume_staged_commit],cwd=ROOT,text=True).strip()
        if original!=a.resume_staged_commit:raise ValueError('Recovery requires exact original commit OID')
        subprocess.run(['git','merge-base','--is-ancestor',original,commit],cwd=ROOT,check=True)
        subprocess.run(['git','diff','--exit-code',original,commit,'--',*paths],cwd=ROOT,check=True)
        commit=original
    folder=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928')/name;folder.mkdir(parents=True,exist_ok=True)
    archive=folder/(name+'.tar');remote_archive=remote_archive_path(release,archive.name)
    if archive.exists() and not a.resume_staged_commit:raise FileExistsError('Local archive exists; reconcile')
    if a.resume_staged_commit and not archive.exists():raise FileNotFoundError('Original local archive missing')
    checked=[release,remote_root] if a.resume_staged_commit else [remote_archive,release,remote_root]
    probe='from pathlib import Path\nassert not any(Path(p).exists() for p in '+repr(checked)+'), "Output collision"\n'
    if a.resume_staged_commit:
        probe+='import hashlib\nassert hashlib.sha256(Path('+repr(remote_archive)+').read_bytes()).hexdigest()=='+repr(hashlib.sha256(archive.read_bytes()).hexdigest())+', "Staged archive mismatch"\n'
    subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=probe.encode(),check=True)
    if not a.resume_staged_commit:
        subprocess.run(['git','archive','--format=tar','--prefix='+name+'/','--output='+str(archive),commit,*paths],cwd=ROOT,check=True)
        subprocess.run(['scp',*FLAGS,str(archive),'N607:'+remote_archive],check=True)
    cfg=dict(release=release,archive=remote_archive,root=remote_root,spec=a.spec,commit=commit,
        python=spec['code']['environment'],owner=spec['execution']['launch_owner'],sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    script=REMOTE.replace('CONFIG',repr(cfg));compile(script,'remote','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True)
    prefix='recovery_landing' if a.resume_staged_commit else 'landing'
    with (folder/(prefix+'.stdout')).open('xb') as f:f.write(result.stdout)
    with (folder/(prefix+'.stderr')).open('xb') as f:f.write(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()


if __name__=='__main__':main()
