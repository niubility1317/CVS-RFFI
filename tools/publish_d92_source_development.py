"""Publish pushed code and launch one registered source-only diagnostic."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RELEASE='d92_scv_source_20260928_r01'
PATHS=['code','tools/cvs_native_artifacts.py','tools/cvs_d92_matched.py',
       'tools/cvs_d92_covariance_matched.py','tools/export_d92_source_development.py',
       'tools/evaluate_d92_source_development.py','tools/run_d92_source_development.py',
       'tools/publish_d92_source_development.py','configs/d92_upgrade_source_20260928.json']
REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
c=CONFIG
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=project/'releases'/c['release'];archive=project/'releases'/c['archive']
if release.exists():raise FileExistsError('Existing release; reconcile instead of relaunch')
if shutil.disk_usage(project).free<2*1024**3:raise RuntimeError('Insufficient disk for bounded source export')
gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
if int(gpu)>1024:raise RuntimeError('Preselected GPU0 no longer free; do not disturb it')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not (project/'releases'/member.name).resolve().is_relative_to(release.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe archive')
 tar.extractall(project/'releases')
specpath=release/'configs/d92_upgrade_source_20260928.json';spec=json.loads(specpath.read_text())
root=Path(spec['execution']['remote_run_root'])
if root.exists():raise FileExistsError('Run already exists; reconcile')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release/'tools'),str(release/'code')],env=env,check=True)
argv=[python,str(release/'tools/run_d92_source_development.py'),'--spec',str(specpath),'--commit',c['commit']]
with (release/'launch.log').open('x') as log:
 proc=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=proc.pid,argv=argv,cwd=str(release),commit=c['commit'],run_root=str(root),owner=spec['execution']['launch_owner'])
(release/'launch.json').write_text(json.dumps(launch,indent=2))
print(json.dumps(launch))
'''


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()
    if not remote or remote[0]!=commit:raise ValueError('Release commit not independently confirmed remote')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release paths')
    output=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/source_r01');output.mkdir(parents=True,exist_ok=True)
    archive=output/(RELEASE+'.tar')
    if archive.exists():raise FileExistsError('Archive exists; reconcile')
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),commit,*PATHS],cwd=ROOT,check=True)
    flags=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    spec=json.loads((ROOT/'configs/d92_upgrade_source_20260928.json').read_text(encoding='utf-8'))
    remote_paths=['/home/szu2070436088/2510044040/CV-SincNet/releases/'+archive.name,
                  '/home/szu2070436088/2510044040/CV-SincNet/releases/'+RELEASE,
                  spec['execution']['remote_run_root']]
    probe=('from pathlib import Path\npaths='+repr(remote_paths)+
           '\nexisting=[p for p in paths if Path(p).exists()]\n'
           'if existing: raise FileExistsError(existing)\nprint("OUTPUT_PATHS_ABSENT")\n')
    subprocess.run(['ssh',*flags,'-T','N607','python3 -'],input=probe.encode('utf-8'),check=True)
    subprocess.run(['scp',*flags,str(archive),'N607:/home/szu2070436088/2510044040/CV-SincNet/releases/'+archive.name],check=True)
    config=dict(commit=commit,release=RELEASE,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    payload=REMOTE.replace('CONFIG',repr(config));compile(payload,'remote','exec')
    result=subprocess.run(['ssh',*flags,'-T','N607','python3 -'],input=payload.encode('utf-8'),capture_output=True)
    (output/'landing.stdout').write_bytes(result.stdout);(output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode('utf-8'));print(result.stderr.decode('utf-8'));result.check_returncode()


if __name__=='__main__':main()
