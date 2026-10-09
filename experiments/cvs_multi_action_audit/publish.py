"""Publish only committed source-audit files using the existing checked transport."""
import argparse
from pathlib import Path
from experiments.cvs_multi_action_audit import design as d
from experiments.cvs_multi_disentangle import publish as transport

REMOTE=r'''
from pathlib import Path
import json,os,subprocess,tarfile,hashlib,shutil
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
for path in [release,p/'runs'/c['run'],p/'logs'/c['run']]:
    if path.exists():raise FileExistsError('Existing diagnostic path; reconcile '+str(path))
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
if shutil.disk_usage(p).free<5*1024**3:raise RuntimeError('Insufficient space')
release.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
    t.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release)],env=env,check=True)
check="from experiments.cvs_multi_action_audit.runner import run; from experiments.cvs_multi_action_audit.checks import channel_bridge_check; from experiments.cvs_multi_action_audit import design as d; d.write('checkpoint_preflight.json',dict(status='VERIFIED',channel_bridge=channel_bridge_check(),rows=[run(d.config(r),smoke=True) for r in d.rows()],target_access=False))"
with (release/'preflight.log').open('x') as log:
    subprocess.run([python,'-c',check],cwd=release,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
cmd=[python,'-u','-m','experiments.cvs_multi_action_audit.dispatch'];logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,log=str(logpath),commit=c['commit'],run_id=c['run'])
with (release/'submit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''

def publish(output):
    transport.RUN=d.RUN;transport.RELEASE=d.RELEASE;transport.REMOTE=REMOTE
    transport.PATHS=[*transport.PATHS,'experiments/cvs_multi_action_audit']
    transport.publish(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.output)
