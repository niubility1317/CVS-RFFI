"""Land one committed D92/final-evaluation release and submit dependencies once."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RELEASE='cvs_d92_matched_20260927_r01'
PATHS=['code','tools/cvs_d92_matched.py','tools/cvs_d92_covariance_matched.py','tools/cvs_native_artifacts.py',
       'tools/cvs_matched_pipeline.py','tools/publish_cvs_matched_pipeline.py',
       'tests/test_cvs_d92_matched.py','tests/test_cvs_native_artifacts.py','configs/cvs_d92_matched_20260927.json']
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];archive=project/'releases'/c['archive']
if release.exists():raise FileExistsError('Existing release; reconcile, do not retry')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not (project/'releases'/member.name).resolve().is_relative_to(release.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe archive')
 tar.extractall(project/'releases')
specpath=release/'configs/cvs_d92_matched_20260927.json';spec=json.loads(specpath.read_text())
for path in [Path(spec['source_run_root'])/'launch.json',Path(spec['final_capsule'])/'manifest.json',Path(spec['phase2_capsule'])/'manifest.json',Path(spec['phase2_scorer']),Path(spec['native_code'])/'scripts/train_rc4_matched.py']:
 if not path.is_file():raise FileNotFoundError(path)
if Path(spec['execution']['remote_run_root']).exists():raise FileExistsError('Run already exists')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release/'code'),str(release/'tools')],env=env,cwd=release,check=True)
with (release/'runtime_smoke.log').open('x') as log:
 subprocess.run([python,str(release/'tools/cvs_d92_matched.py'),'smoke'],env=env,cwd=release,stdout=log,stderr=subprocess.STDOUT,check=True)
result=subprocess.run([python,str(release/'tools/cvs_matched_pipeline.py'),'launch','--spec',str(specpath),'--commit',c['commit']],env=env,cwd=release,capture_output=True,text=True,check=True)
print(result.stdout)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=commit:
        raise ValueError('Unpushed release')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():
        raise ValueError('Uncommitted release paths')
    a.output.mkdir(parents=True,exist_ok=True);archive=a.output/(RELEASE+'.tar')
    if archive.exists():raise FileExistsError('Archive exists; reconcile before retry')
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),commit,*PATHS],cwd=ROOT,check=True)
    project='/home/szu2070436088/2510044040/CV-SincNet'
    c=dict(project=project,release=RELEASE,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit)
    flags=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*flags,str(archive),'N607:'+project+'/releases/'+archive.name],check=True)
    script=REMOTE.replace('CONFIG',repr(c));compile(script,'remote','exec')
    result=subprocess.run(['ssh',*flags,'-T','N607','python3 -'],input=script.encode(),capture_output=True)
    (a.output/'landing.stdout').write_bytes(result.stdout);(a.output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()


if __name__=='__main__':main()
