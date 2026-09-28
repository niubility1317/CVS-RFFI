"""Publish the registered source-only calibration; no target access or overwrite."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RELEASE='d92_registration_proxy_20260928_r01'
PATHS=['code/cvsrffi/stage2_d92_support_cv.py','tools/evaluate_d92_registration_proxy.py','configs/d92_registration_proxy_20260928.json']
FLAGS=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
base=Path('/home/szu2070436088/2510044040/CV-SincNet/releases')
release=base/c['release'];archive=base/c['archive']
if release.exists() or Path(c['run']).exists():raise FileExistsError('Existing release/run')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not (base/m.name).resolve().is_relative_to(release.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe archive')
 tar.extractall(base)
py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([py,'-m','compileall','-q',str(release)],env=env,check=True)
argv=[py,'-u',str(release/'tools/evaluate_d92_registration_proxy.py'),'--spec',str(release/'configs/d92_registration_proxy_20260928.json'),'--commit',c['commit']]
with (release/'run.log').open('x') as f:
 p=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=p.pid,argv=argv,cwd=str(release),commit=c['commit'],run=c['run'],owner='codex/root/d92-upgrade-20260928')
(release/'launch.json').write_text(json.dumps(launch,indent=2))
print(json.dumps(launch))
'''


def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=head:raise ValueError('Unpushed code')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release paths')
    spec=json.loads((ROOT/PATHS[-1]).read_text(encoding='utf-8'))
    out=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/proxy_r01');out.mkdir(parents=True,exist_ok=True)
    archive=out/(RELEASE+'.tar')
    paths=['/home/szu2070436088/2510044040/CV-SincNet/releases/'+RELEASE,
        '/home/szu2070436088/2510044040/CV-SincNet/releases/'+archive.name,spec['execution']['remote_run_root']]
    probe='from pathlib import Path\npaths='+repr(paths)+'\nassert not any(Path(p).exists() for p in paths), "Output collision"\n'
    subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=probe.encode(),check=True)
    if archive.exists():raise FileExistsError(archive)
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),head,*PATHS],cwd=ROOT,check=True)
    subprocess.run(['scp',*FLAGS,str(archive),'N607:'+paths[1]],check=True)
    c=dict(release=RELEASE,archive=archive.name,run=paths[2],commit=head,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    payload=REMOTE.replace('CONFIG',repr(c));compile(payload,'remote','exec')
    r=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=payload.encode(),capture_output=True)
    (out/'landing.stdout').write_bytes(r.stdout);(out/'landing.stderr').write_bytes(r.stderr)
    print(r.stdout.decode());print(r.stderr.decode());r.check_returncode()


if __name__=='__main__':main()
