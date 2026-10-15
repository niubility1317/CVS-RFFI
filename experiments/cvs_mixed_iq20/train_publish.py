"""Thin immutable release, one archive comparison and CPU dispatcher."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from . import train_design as d

# The operation helper is loaded locally without importing another experiment's
# design or runtime. Its SSH route stays the project's ordinary direct route.
SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']


def remote(script,timeout=60):
    compile(script,'mixedIQ_remote','exec')
    r=subprocess.run(['ssh',*SSH,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf-8',errors='replace')+'\n'+r.stdout.decode('utf-8',errors='replace'))
    return json.loads(r.stdout)


REMOTE=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run']
if release.exists() or run.exists():raise FileExistsError('Existing run/release must be reconciled before resubmission')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer comparison failed')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary account/host differs')
release.mkdir()
with tarfile.open(archive) as f:
 for m in f.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe release path')
 f.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_mixed_iq20')],cwd=release,env=env,check=True)
spec=json.loads((release/'experiments/cvs_mixed_iq20/train_registration.json').read_text())
for path in (spec['data']['contract_ref'],spec['data']['validation_ref'],spec['data']['leo_config_ref']):
 if not Path(path).is_file():raise FileNotFoundError(path)
preflight=dict(status='VERIFIED',archive_transfer='VERIFIED',remote_compile='PASS',ordinary_account=True,
 GPU_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True),
 disk_free_bytes=__import__('shutil').disk_usage(project).free,commit=c['commit'],read_at=time.time())
(release/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
command=[python,'-u',str(release/'experiments/cvs_mixed_iq20/train_runner.py')]
log=release/'dispatcher.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
submit=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),run_root=str(run),commit=c['commit'])
(release/'submit.json').write_text(json.dumps(submit,indent=2)+'\n')
print(json.dumps(submit))
'''


def publish(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments/cvs_mixed_iq20'],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Commit intended code/config first')
    names=subprocess.check_output(['git','ls-files','experiments/cvs_mixed_iq20'],cwd=d.ROOT,text=True).splitlines()
    names=[n for n in names if Path(n).suffix in ('.py','.json')]
    archive=output/(d.RELEASE+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name in names:tar.add(d.ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();m=tarfile.TarInfo('release_commit.txt');m.size=len(blob);tar.addfile(m,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    value=remote(REMOTE.replace('CONFIG',repr(dict(project=d.PROJECT,run=d.RUN,release=d.RELEASE,archive=archive.name,sha256=sha,commit=commit))))
    d.write(output/'submission.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();publish(a.output)
