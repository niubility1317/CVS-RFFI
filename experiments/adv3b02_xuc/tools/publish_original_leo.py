"""Publish one committed registered experiment and launch once after GPU smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];archive=project/'releases'/c['archive']
run=project/'runs'/c['run_id'];logs=project/'logs'/c['run_id']
if release.exists() or run.exists() or logs.exists():raise FileExistsError('Existing landing/run; reconcile before any retry')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not (project/'releases'/member.name).resolve().is_relative_to(release.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe member')
    tar.extractall(project/'releases')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release/'code')+os.pathsep+str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release/'code')],cwd=release,env=env,check=True)
sys.path.insert(0,str(release/'code'))
from scripts.dispatch_xuc_full import available_gpu,occupancy
gpu=available_gpu({})
if gpu is None:raise RuntimeError('No legal GPU slot; release retained')
env['CUDA_VISIBLE_DEVICES']=str(gpu)
source=project/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
target=project/'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'
for path in [source,target/'target_inputs/iq.npy',target/'target_inputs/manifest.json',target/'target_truth/truth_sidecar.json',project/'Dataset_WigSig/ManySig.pkl']:
    if not path.is_file():raise FileNotFoundError(path)
run.mkdir();logs.mkdir()
config=release/'configs/rc4_original_leo_20260918.json'
with (logs/'smoke.log').open('x') as log:
    subprocess.run([python,'-u',str(release/'code/scripts/check_rc4_original_leo.py'),'--config',str(config),'--output',str(run/'startup_smoke'),'--device','cuda:0'],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
acceptance=json.loads((run/'startup_smoke/acceptance.json').read_text())
if acceptance['status']!='PASS' or not acceptance['daot'] or not acceptance['rc4']:raise ValueError('Bad smoke')
slots=occupancy({})
if len(slots[gpu]['pids'])>=2 or slots[gpu]['free_mb']<20000:raise RuntimeError('GPU changed during smoke; retained, no launch')
row=json.loads(config.read_text())['id']
command=[python,'-u',str(release/'code/scripts/train_rc4_original_leo.py'),'--config',str(config),
    '--dataset',str(project/'Dataset_WigSig/ManySig.pkl'),'--output',str(run/row),'--source-contract',str(source),
    '--target-inputs',str(target/'target_inputs'),'--target-truth',str(target/'target_truth/truth_sidecar.json'),'--run-id',c['run_id']]
log_path=logs/(row+'.train.log')
with log_path.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='LAUNCH_SUBMITTED',pid=child.pid,gpu=gpu,argv=command,cwd=str(release),run_id=c['run_id'],
    row_id=row,run_root=str(run),output=str(run/row),log=str(log_path),commit=c['commit'],source_contract=str(source),timestamp=time.time())
(run/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]!=commit:
        raise ValueError('Remote branch differs from HEAD')
    if subprocess.check_output(['git','status','--porcelain','--','experiments/adv3b02_xuc'],cwd=repo,text=True).strip():
        raise ValueError('Uncommitted release files')
    a.output.mkdir(parents=True,exist_ok=True)
    release='daot_rc4_original_leo_20260918_'+commit[:10];archive=a.output/(release+'.tar')
    subprocess.run(['git','archive','--format=tar','--prefix='+release+'/','--output='+str(archive),commit+':experiments/adv3b02_xuc'],cwd=repo,check=True)
    project='/home/szu2070436088/2510044040/CV-SincNet'
    c=dict(project=project,release=release,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),run_id=a.run_id,commit=commit)
    connection=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*connection,str(archive),'N607:'+project+'/releases/'+archive.name],check=True)
    script=REMOTE.replace('CONFIG',repr(c))
    result=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True)
    (a.output/'landing.stdout').write_bytes(result.stdout);(a.output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()
    (a.output/'launch.json').write_text(json.dumps(json.loads(result.stdout.decode().splitlines()[-1]),indent=2),encoding='utf-8')

if __name__=='__main__':main()
