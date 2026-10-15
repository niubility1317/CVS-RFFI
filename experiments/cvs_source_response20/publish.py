"""Single immutable source-only release, one transfer comparison, ordinary N607."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from . import probe_design as d
from experiments.cvs_urban_sigtime.operations import remote,SSH

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile,time,shutil
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run']
if release.exists() or run.exists():raise FileExistsError('Reconcile existing release/output before resubmitting')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary account/host mismatch')
if shutil.disk_usage(project).free<2*1024**3:raise ValueError('Insufficient source output space')
release.mkdir()
with tarfile.open(archive) as f:
 for member in f.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe release destination')
 f.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_source_response20'),str(release/'experiments/cvs_mixed_iq20')],cwd=release,env=env,check=True)
spec=json.loads((release/'experiments/cvs_source_response20/probe_registration.json').read_text())
for path in (spec['data']['contract_ref'],spec['data']['validation_ref'],spec['checkpoint']['sources'][0]):
 if not Path(path).is_file():raise FileNotFoundError(path)
preflight=dict(status='VERIFIED',archive_transfer='VERIFIED',remote_compile='PASS',ordinary_account=True,
 disk_free_bytes=shutil.disk_usage(project).free,commit=c['commit'],read_at=time.time(),CPU_threads=2,GPU='none',cwd=str(release),python=python)
(release/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
command=[python,'-u',str(release/'experiments/cvs_source_response20/probe_runner.py')];log=release/'source_probe.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
submit=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),run_root=str(run),commit=c['commit'])
(release/'submit.json').write_text(json.dumps(submit,indent=2)+'\n');print(json.dumps(submit))
'''


def publish(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    folders=['experiments/cvs_source_response20','experiments/cvs_mixed_iq20']
    dirty=subprocess.check_output(['git','status','--porcelain','--',*folders],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Commit actual operator/probe code/config first')
    spec=d.read(d.ROOT/'experiments/cvs_source_response20/probe_registration.json');d.validate(spec['rows'][0]['config'])
    from experiments.cvs_response_coverage.core import source
    from experiments.cvs_response_coverage.design import EVIDENCE
    evidence=d.read(EVIDENCE);predicted,_,_=source(evidence)
    payload=dict(source_run=d.config()['source_bank_run'],primary='rectangular/DCT1to8',L_count=6300,
        U_used=False,V_used=False,target_access=False,predicted_contrasts=predicted.tolist())
    d.write(output/'source_bank_input.json',payload)
    names=subprocess.check_output(['git','ls-files',*folders],cwd=d.ROOT,text=True).splitlines()
    archive=output/(d.RELEASE+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name in names:
                if Path(name).suffix in ('.py','.json'):tar.add(d.ROOT/name,arcname=name,recursive=False)
            for name,blob in [('release_commit.txt',(commit+'\n').encode()),('source_bank_input.json',(json.dumps(payload)+'\n').encode())]:
                member=tarfile.TarInfo(name);member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    settings=dict(project=d.PROJECT,release=d.RELEASE,run=d.RUN,archive=archive.name,sha256=digest,commit=commit)
    result=remote(REMOTE.replace('CONFIG',repr(settings)),timeout=60)
    d.write(output/'submission.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();publish(a.output)
