"""Publish pushed, registered paired matrix without overwriting any prior run."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RELEASE='d92_scv_confirmation_20260928_r01'
FLAGS=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
PATHS=['code','tools/cvs_native_artifacts.py','tools/cvs_d92_matched.py','tools/cvs_d92_covariance_matched.py',
    'tools/build_d92_confirmation_data.py','tools/predict_d92_support_cv.py','tools/score_d92_confirmation.py','tools/run_d92_confirmation.py',
    'configs/d92_confirmation_20260928.json','configs/d92_confirmation_data_20260928.json','configs/d92_scv_frozen_20260928.json']
REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
c=CONFIG
base=Path('/home/szu2070436088/2510044040/CV-SincNet/releases')
release=base/c['release'];archive=base/c['archive']
if any(Path(p).exists() for p in [str(release),c['run'],c['data_run']]):raise FileExistsError('Existing release/run/data; reconcile instead of retry')
if shutil.disk_usage(base).free<2*1024**3:raise RuntimeError('Insufficient disk')
used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>1024:raise RuntimeError('GPU0 currently occupied; do not disturb existing work')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not (base/m.name).resolve().is_relative_to(release.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe archive')
 tar.extractall(base)
py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([py,'-m','compileall','-q',str(release/'tools'),str(release/'code')],env=env,check=True)
argv=[py,'-u',str(release/'tools/run_d92_confirmation.py'),'--spec',str(release/'configs/d92_confirmation_20260928.json'),'--commit',c['commit']]
with (release/'run.log').open('x') as stream:
 p=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=p.pid,argv=argv,cwd=str(release),commit=c['commit'],run=c['run'],data_run=c['data_run'],owner='codex/root/d92-upgrade-20260928')
(release/'launch.json').write_text(json.dumps(launch,indent=2))
print(json.dumps(launch))
'''


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=commit:raise ValueError('Release commit not pushed')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release paths')
    spec=json.loads((ROOT/'configs/d92_confirmation_20260928.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'configs/d92_confirmation_data_20260928.json').read_text(encoding='utf-8'))
    out=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/confirmation_r01');out.mkdir(parents=True,exist_ok=True)
    archive=out/(RELEASE+'.tar')
    if archive.exists():raise FileExistsError('Local archive already exists; reconcile')
    remote_archive='/home/szu2070436088/2510044040/CV-SincNet/releases/'+archive.name
    paths=[remote_archive,spec['code']['cwd'],spec['execution']['remote_run_root'],cfg['output_root']]
    probe='from pathlib import Path\npaths='+repr(paths)+'\nassert not any(Path(p).exists() for p in paths), "Output collision"\n'
    subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=probe.encode(),check=True)
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),commit,*PATHS],cwd=ROOT,check=True)
    subprocess.run(['scp',*FLAGS,str(archive),'N607:'+remote_archive],check=True)
    config=dict(commit=commit,release=RELEASE,archive=archive.name,run=paths[2],data_run=paths[3],sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    payload=REMOTE.replace('CONFIG',repr(config));compile(payload,'remote','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=payload.encode(),capture_output=True)
    (out/'landing.stdout').write_bytes(result.stdout);(out/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()


if __name__=='__main__':main()
