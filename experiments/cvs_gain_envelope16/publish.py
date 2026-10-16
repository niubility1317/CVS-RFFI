"""Release committed CPU-only source statistics once; no process intervention."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from . import design as d

SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'


def remote(script):
    completed=subprocess.run(['ssh',*SSH,'-T','N607',PYTHON+' -'],input=script.encode('utf8'),capture_output=True,timeout=60)
    if completed.returncode:raise RuntimeError(completed.stderr.decode('utf8',errors='replace'))
    return json.loads(completed.stdout)


def main():
    package='experiments/cvs_gain_envelope16';local=d.ROOT/'local_artifacts/F_gain_envelope_20261016';local.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual source diagnostic before release')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    release=d.PROJECT+'/releases/'+d.RELEASE
    c=dict(project=d.PROJECT,release=release,archive=release+'.tar.gz',run=d.BASE,config=d.config(),python=PYTHON,commit=commit)
    script=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Reconcile existing immutable experiment; no resubmission')
for p in (c['config']['source_contract'],c['config']['preparation']+'/source_cache.pt',c['config']['preparation']+'/preparation_complete.json'):
 if not Path(p).is_file():raise FileNotFoundError(p)
if shutil.disk_usage(c['project']).free<1024**3:raise ValueError('Insufficient output space')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,source_cache_present=True,CPU_threads=2,GPU='none',checkpoint_or_target_access=False)))
'''
    d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c))))
    raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT)
    record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT)
    archive=local/(d.RELEASE+'.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
        for member in source.getmembers():
            if member.isfile():target.addfile(member,source.extractfile(member))
        for name,data in (('release_commit.txt',(commit+'\n').encode()),('registration/experiment.json',record)):
            item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
    c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
    script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);run=Path(c['run']);archive=Path(c['archive'])
if release.exists() or run.exists():raise FileExistsError('Reconcile prior mutation before any relaunch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_gain_envelope16')],cwd=release,env=env,check=True)
command=[c['python'],'-u','-m','experiments.cvs_gain_envelope16.run'];log=release/'source_gain.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',CPU_threads=2,GPU='none')
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
proc=Path('/proc')/str(child.pid);deadline=time.monotonic()+2.
while time.monotonic()<deadline and child.poll() is None:
 try:
  actual=dict(cwd=os.readlink(proc/'cwd'),argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1])
  if actual['cwd']==str(release) and actual['argv']==command:
   receipt.update(status='OBSERVED',actual=actual);break
 except FileNotFoundError:pass
 time.sleep(.05)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
    result=remote(script.replace('CONFIG',repr(c)));d.write(local/'submit.json',result)
    print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile','GPU')}))


if __name__=='__main__':main()
