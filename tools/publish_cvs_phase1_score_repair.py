"""Publish committed Phase1 scoring repair and complete the existing frozen evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'cvs_phase1_score_repair_20260928_r01'
PATHS = ['tools/cvs_matched_pipeline.py', 'tests/test_cvs_matched_pipeline.py',
         'tools/publish_cvs_phase1_score_repair.py', 'configs/cvs_d92_matched_20260927.json']
REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=project/'releases'/c['release'];archive=project/'releases'/c['archive']
if release.exists():raise FileExistsError('Existing release; reconcile first')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not (project/'releases'/member.name).resolve().is_relative_to(release.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe archive')
 tar.extractall(project/'releases')
specpath=release/'configs/cvs_d92_matched_20260927.json'
spec=json.loads(specpath.read_text());run=Path(spec['execution']['remote_run_root'])
if (run/'phase1_final_results.json').exists():raise FileExistsError('Already scored')
launch=json.loads((run/'launch.json').read_text())
if (Path('/proc')/str(launch['pid'])/'cmdline').exists():raise RuntimeError('Dispatcher still active')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'tools')],env=env,check=True)
subprocess.run([python,str(release/'tests/test_cvs_matched_pipeline.py')],env=env,check=True)
argv=[python,str(release/'tools/cvs_matched_pipeline.py'),'final-score','--spec',str(specpath)]
with (run/'phase1_score_repair_startup.json').open('x') as f:
 json.dump(dict(commit=c['commit'],argv=argv,cwd=str(release),owner='codex/root/phase1-score-repair',training_changed=False,predictions_changed=False),f)
with (run/'phase1_score_repair.log').open('x') as log:
 subprocess.run(argv,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
result=json.loads((run/'phase1_final_results.json').read_text())
print(json.dumps(dict(status=result['status'],records=len(result['results']),commit=c['commit'])))
'''


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=commit:
        raise ValueError('Unpushed release')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():
        raise ValueError('Uncommitted release paths')
    output=Path('E:/type10-7/local_artifacts/cvs_matched_20260927')
    archive=output/(RELEASE+'.tar')
    if archive.exists():raise FileExistsError('Archive exists; reconcile first')
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),commit,*PATHS],cwd=ROOT,check=True)
    flags=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    project='/home/szu2070436088/2510044040/CV-SincNet'
    subprocess.run(['scp',*flags,str(archive),'N607:'+project+'/releases/'+archive.name],check=True)
    c=dict(commit=commit,release=RELEASE,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    script=REMOTE.replace('CONFIG',repr(c));compile(script,'remote','exec')
    result=subprocess.run(['ssh',*flags,'-T','N607','python3 -'],input=script.encode(),capture_output=True)
    (output/'phase1_score_repair.stdout').write_bytes(result.stdout)
    (output/'phase1_score_repair.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()


if __name__=='__main__':main()
