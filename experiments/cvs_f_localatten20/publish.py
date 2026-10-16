"""Immutable source-only E20 release; one archive transfer comparison."""
import os
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from . import design as d
SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']

SSH_EXE='C:/Program Files/Git/usr/bin/ssh.exe' if os.name=='nt' else 'ssh'
SCP_EXE='C:/Program Files/Git/usr/bin/scp.exe' if os.name=='nt' else 'scp'
FOLDERS=['experiments/cvs_ordinary_fold20','experiments/cvs_raw_anchor20',
 'experiments/cvs_urban_sigtime/__init__.py','experiments/cvs_urban_sigtime/channel.py',
 'experiments/cvs_f_crossrx_practical20','experiments/cvs_f_localatten20']


def remote(script):
    compile(script,'ordinary_remote','exec')
    result=subprocess.run([SSH_EXE,*SSH,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+result.stdout.decode('utf-8',errors='replace'))
    return json.loads(result.stdout)


REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];root=project/'runs'/c['run']
if release.exists() or root.exists():raise FileExistsError('Reconcile existing immutable release/output')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary account/host differs')
if shutil.disk_usage(project).free<5*1024**3:raise ValueError('Insufficient output space')
for path in (c['source'],project/'releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02',project/'releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01',project/'releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'):
 if not Path(path).exists():raise FileNotFoundError(path)
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as f:
 for m in f.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive destination')
 f.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',*[str(release/v) for v in c['folders']]],cwd=release,env=env,check=True)
preflight=dict(status='VERIFIED',archive_transfer='VERIFIED',remote_compile='PASS',ordinary_account=True,commit=c['commit'],read_at=time.time(),
 GPU_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True),disk_free_bytes=shutil.disk_usage(project).free)
(release/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
command=[python,'-u',str(release/c['package']/'runner.py')];log=release/'dispatcher.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),run_root=str(root),commit=c['commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''


def publish(output,design=None,package='experiments/cvs_f_localatten20',folders=None):
    d=globals()['d'] if design is None else design
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    folder=package;folders=FOLDERS if folders is None else folders
    if subprocess.check_output(['git','status','--porcelain','--',*folders],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual code/config first')
    spec=d.read(d.ROOT/folder/'registration.json')
    if [v['config'] for v in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered native configurations differ')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    archive=output/(d.RELEASE+'.tar.gz');names=subprocess.check_output(['git','ls-files',*folders],cwd=d.ROOT,text=True).splitlines()
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name in names:
                if Path(name).suffix in ('.py','.json'):tar.add(d.ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();m=tarfile.TarInfo('release_commit.txt');m.size=len(blob);tar.addfile(m,io.BytesIO(blob))
    settings=dict(project=d.PROJECT,release=d.RELEASE,run=d.RUN,source=d.SOURCE,archive=archive.name,package=package,folders=folders,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit)
    d.write(output/'package.json',settings)
    subprocess.run([SCP_EXE,'-S',SSH_EXE,*SSH,archive.name,'N607:'+d.PROJECT+'/releases/'+archive.name],cwd=output,check=True,timeout=45)
    result=remote(REMOTE.replace('CONFIG',repr(settings)));d.write(output/'submission.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);publish(parser.parse_args().output)
