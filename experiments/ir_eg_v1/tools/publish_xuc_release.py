"""Local committed archive -> N607 landing, one smoke, one dispatcher launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REMOTE_SCRIPT=r'''
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];archive=project/'releases'/c['archive']
if release.exists():raise FileExistsError('release already exists; reconcile previous landing')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('archive transfer mismatch')
with tarfile.open(archive) as t:
    for m in t.getmembers():
        target=(project/'releases'/m.name).resolve()
        if not target.is_relative_to(release.resolve()) or m.issym() or m.islnk():raise ValueError('archive path escapes release')
    t.extractall(project/'releases')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release/'code')+os.pathsep+str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'code')],env=env,cwd=release,check=True)
# This launcher first exercises its own scratch checkpoint without query access.
sys.path.insert(0,str(release/'code'))
from scripts.dispatch_xuc15 import available_gpu
gpu=available_gpu({})
if gpu is None:raise RuntimeError('no legal GPU slot; landing retained, no formal launch')
env['CUDA_VISIBLE_DEVICES']=str(gpu)
check=project/'logs'/(c['run_id']+'_preflight')
with (project/'logs'/(c['run_id']+'.smoke.log')).open('x') as log:
    subprocess.run([python,str(release/'code/scripts/check_xuc_execution.py'),'--rows','M12','--output',str(check),
        '--device','cuda:0'],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if json.loads((check/'acceptance.json').read_text())['status']!='PASS':raise ValueError('checkpoint smoke not PASS')
command=[python,'-u',str(release/'code/scripts/dispatch_xuc15.py'),'--project',str(project),'--run-id',c['run_id'],'--commit',c['commit']]
log_path=project/'logs'/(c['run_id']+'.dispatcher.log')
with log_path.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='LAUNCH_SUBMITTED',pid=child.pid,release=str(release),run_id=c['run_id'],commit=c['commit'],
    sha256=c['sha256'],argv=command,log=str(log_path),smoke=str(check/'acceptance.json'))
(release/'launch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    # Path is repo/experiments/adv3b02_xuc/tools/this_file.py.
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]
    if remote!=commit:raise ValueError('local HEAD is not independently verified on origin')
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments/adv3b02_xuc'],cwd=repo,text=True)
    if dirty:raise ValueError('release code/config has uncommitted changes')
    a.output.mkdir(parents=True,exist_ok=True)
    release='adv3b02_xuc15_'+commit[:10];archive=a.output/(release+'.tar')
    subprocess.run(['git','archive','--format=tar','--prefix='+release+'/','--output='+str(archive),commit+':experiments/adv3b02_xuc'],cwd=repo,check=True)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    project='/home/szu2070436088/2510044040/CV-SincNet'
    connection=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*connection,str(archive),'N607:'+project+'/releases/'+archive.name],check=True)
    config=dict(project=project,release=release,archive=archive.name,sha256=sha,run_id=a.run_id,commit=commit)
    script=REMOTE_SCRIPT.replace('CONFIG',repr(config))
    result=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True)
    (a.output/'landing.stdout').write_bytes(result.stdout);(a.output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()
if __name__=='__main__':main()
