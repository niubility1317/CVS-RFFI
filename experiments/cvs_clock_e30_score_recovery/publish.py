"""One committed release archive and one transfer comparison; no automatic resubmission."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
rt=d.runtime();SSH=['-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10'];PY=rt['python']
def remote(script):
 r=subprocess.run(['ssh',*SSH,'-T','N607',PY+' -'],input=script.encode('utf8'),capture_output=True,timeout=55)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return json.loads(r.stdout)
def main():
 package='experiments/cvs_clock_e30_score_recovery';local=d.ROOT/d.LOCAL
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual code/config before release')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();release=d.PROJECT+'/releases/'+d.RELEASE
 raw=subprocess.check_output(['git','archive',commit,package,'configs/cvs_clock_e30_score_recovery.json'],cwd=d.ROOT);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT)
 def expand(v):
  if isinstance(v,str):return v.replace('PROJECT',d.PROJECT)
  if isinstance(v,list):return [expand(x) for x in v]
  if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
  return v
 record_value=expand(json.loads(record));record_value['code']['commit']=commit
 record=json.dumps(record_value,ensure_ascii=False).encode('utf8')
 archive=local/(d.RELEASE+'.tar.gz')
 with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
  for member in source.getmembers():
   if member.isfile():target.addfile(member,source.extractfile(member))
  for name,data in (('release_commit.txt',(commit+'\n').encode()),('runtime.json',json.dumps(dict(AF=d.AF_RELEASE,AF_commit=rt['AF_commit'],project=d.PROJECT,reference_release=rt['reference_release'],reference_commit=rt['reference_commit'],clock_release=rt['clock_release'],clock_commit=rt['clock_commit'],training_release=rt['training_release'],training_commit=rt['training_commit'])).encode()),('registration/experiment.json',record)):
   item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
 c=dict(release=release,archive=release+'.tar.gz',run=d.BASE.as_posix(),project=d.PROJECT,python=PY,commit=commit,AF=d.AF_RELEASE,AF_commit=rt['AF_commit'],reference_release=rt['reference_release'],reference_commit=rt['reference_commit'],clock_release=rt['clock_release'],clock_commit=rt['clock_commit'],training_release=rt['training_release'],training_commit=rt['training_commit'],sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 guard="import json,os,shutil\nfrom pathlib import Path\nc="+repr(c)+"\nif os.geteuid()==0:raise ValueError('Ordinary account only')\nif any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Reconcile existing run/release before retry')\nif (Path(c['AF'])/'release_commit.txt').read_text().strip()!=c['AF_commit']:raise ValueError('Pinned existing F dependency differs')\nif (Path(c['reference_release'])/'release_commit.txt').read_text().strip()!=c['reference_commit']:raise ValueError('Pinned fullsource reference runtime differs')\nif (Path(c['clock_release'])/'release_commit.txt').read_text().strip()!=c['clock_commit']:raise ValueError('Pinned clock20 implementation differs')\nif (Path(c['training_release'])/'release_commit.txt').read_text().strip()!=c['training_commit']:raise ValueError('Trained E30 runtime differs')\nif shutil.disk_usage(c['project']).free<1024**3:raise ValueError('Insufficient output space')\nprint(json.dumps(dict(status='VERIFIED',ordinary_account=True,immutable_output_free=True)))\n"
 d.write(local/'release_precondition.json',remote(guard));subprocess.run(['scp',*SSH,str(archive),'N607:'+c['archive']],check=True,capture_output=True,timeout=55)
 script=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release']);archive=Path(c['archive'])
if release.exists() or Path(c['run']).exists():raise FileExistsError('Reconcile possibly completed prior mutation')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_clock_e30_score_recovery')],cwd=release,env=env,check=True)
check=[c['python'],'-u',str(release/'experiments/cvs_clock_e30_score_recovery/runner.py'),'--checks']
with (release/'checks.stdout.log').open('x') as stream:subprocess.run(check,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
command=[c['python'],'-u',str(release/'experiments/cvs_clock_e30_score_recovery/runner.py')];log=release/'dispatcher.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',synthetic_checks='PASS',GPU='CPUdispatcher/sharedcapacityworkers')
(release/'submit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
 result=remote(script.replace('CONFIG',repr(c)));d.write(local/'submit.json',result);print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile','synthetic_checks','GPU')}))
if __name__=='__main__':main()
