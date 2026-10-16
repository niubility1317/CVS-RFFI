"""One committed CPU release, one archive comparison and no retries."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'

def remote(script):
    r=subprocess.run(['ssh',*SSH,'-T','N607',PYTHON+' -'],input=script.encode('utf8'),capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
    return json.loads(r.stdout)

def main():
    package='experiments/cvs_same_tx_iq20';local=d.ROOT/d.LOCAL;local.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit source diagnostic before release')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();release=d.PROJECT+'/releases/'+d.RELEASE
    c=dict(release=release,archive=release+'.tar.gz',run=d.BASE.as_posix(),backend=d.BACKEND,backend_commit=d.BACKEND_COMMIT,source=d.SMOKE_SOURCE.as_posix(),python=PYTHON,commit=commit,project=d.PROJECT)
    script=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Immutable run exists;reconcile instead of resubmit')
if (Path(c['backend'])/'release_commit.txt').read_text().strip()!=c['backend_commit']:raise ValueError('Wrong frozen model code dependency')
if not (Path(c['source'])/'final_ssdg.pth').is_file() or shutil.disk_usage(c['project']).free<1024**3:raise ValueError('Missing checkpoint or output space')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,immutable_model_dependency=True,CPU_dispatcher_threads=2,GPU_workers='shared capacity limit3',target_access=False)))
'''
    d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c))))
    raw=subprocess.check_output(['git','archive',commit,package,'experiments/cvs_same_tx_iq16/__init__.py','experiments/cvs_same_tx_iq16/iqmix.py'],cwd=d.ROOT);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT)
    archive=local/(d.RELEASE+'.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
        for member in source.getmembers():
            if member.isfile():target.addfile(member,source.extractfile(member))
        for name,data in (('release_commit.txt',(commit+'\n').encode()),('runtime.json',json.dumps(dict(backend=d.BACKEND)).encode()),('registration/experiment.json',record)):
            item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
    c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp',*SSH,str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
    script=r'''
import hashlib,importlib.util,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release']);archive=Path(c['archive'])
if release.exists() or Path(c['run']).exists():raise FileExistsError('Reconcile prior mutation before relaunch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_same_tx_iq20')],cwd=release,env=env,check=True)
spec=importlib.util.spec_from_file_location('sameTX_IQmix_bootstrap',release/'experiments/cvs_same_tx_iq20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
from experiments.cvs_rdrop_sinc_units.startup import observe_started_child
command=[c['python'],'-u',str(release/'experiments/cvs_same_tx_iq20/runner.py')];log=release/'diagnostic.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',GPU='dispatcher_CPU;workers_shared_capacity',CPU_threads=2)
(release/'submit.json').write_text(json.dumps(value,indent=2)+'\n')
try:value.update(status='OBSERVED',actual=observe_started_child(child,cwd=str(release),argv=command,gpu='',timeout=2.))
except Exception as error:value.update(status='OBSERVATION_FAILED',observation_error=str(error))
(release/'submit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
    result=remote(script.replace('CONFIG',repr(c)));d.write(local/'submit.json',result)
    print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile','GPU')}))

if __name__=='__main__':main()
