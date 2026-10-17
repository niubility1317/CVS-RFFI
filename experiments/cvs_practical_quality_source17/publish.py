"""One ordinary-account CPU source release;private runtime stays outside Git."""
import hashlib,io,json,subprocess,tarfile
from . import design as d

def remote(script,runtime):
 args=['ssh','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',runtime['python']+' -'];r=subprocess.run(args,input=script.encode('utf8'),capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return json.loads(r.stdout)
def main():
 package='experiments/cvs_practical_quality_source17';local=d.ROOT/'local_artifacts/F_practical_quality_source_20261017';runtime=d.read(local/'runtime.json')
 if runtime['run_id']!=d.RUN:raise ValueError('Private runtime for another run')
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual diagnostic before release')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();project=runtime['project'];release=project+'/releases/'+d.RELEASE
 c=dict(project=project,release=release,archive=release+'.tar.gz',run=project+'/'+d.BASE[len('PROJECT/'):],config=d.config(),python=runtime['python'],user=runtime['user'],host=runtime['host'],commit=commit)
 script=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Existing immutable release/run;reconcile,no resubmission')
for alias in (c['config']['source_contract'],c['config']['preparation']+'/source_cache.pt',c['config']['preparation']+'/preparation_complete.json'):
 p=Path(c['project'])/alias[len('PROJECT/'):]
 if not p.is_file():raise FileNotFoundError(p)
if shutil.disk_usage(c['project']).free<1024**3:raise ValueError('Insufficient output space')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,existing_source_cache_present=True,CPU_threads=2,GPU='none',checkpoint_or_target_access=False)))
'''
 d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c)),runtime))
 raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT);archive=local/(d.RELEASE+'.tar.gz')
 runtime_bytes=json.dumps(dict(run_id=d.RUN,project=project,training_release=runtime['training_release'],training_commit=runtime['training_commit']),indent=2).encode()
 with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
  for member in source.getmembers():
   if member.isfile():target.addfile(member,source.extractfile(member))
  for name,data in [('release_commit.txt',(commit+chr(10)).encode()),('registration/experiment.json',record),('runtime.json',runtime_bytes)]:
   item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
 script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);run=Path(c['run']);archive=Path(c['archive'])
if release.exists() or run.exists():raise FileExistsError('Existing mutation;reconcile before relaunch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_practical_quality_source17')],cwd=release,env=env,check=True)
check=[c['python'],'-u',str(release/'experiments/cvs_practical_quality_source17/runner.py'),'--checks']
with (release/'checks.stdout.log').open('x') as stream:subprocess.run(check,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
command=[c['python'],'-u',str(release/'experiments/cvs_practical_quality_source17/runner.py')];log=release/'source_quality.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',CPU_threads=2,GPU='none')
(release/'submit.json').write_text(json.dumps(result,indent=2)+chr(10));proc=Path('/proc')/str(child.pid);deadline=time.monotonic()+2
while time.monotonic()<deadline and child.poll() is None:
 try:
  actual=dict(cwd=os.readlink(proc/'cwd'),argv=(proc/'cmdline').read_bytes().decode().split(chr(0))[:-1],cuda=[v for v in (proc/'environ').read_bytes().decode().split(chr(0)) if v.startswith('CUDA_VISIBLE_DEVICES=')])
  if actual['cwd']==str(release) and actual['argv']==command and actual['cuda']==['CUDA_VISIBLE_DEVICES=']:result.update(status='OBSERVED',actual=actual);break
 except FileNotFoundError:pass
 time.sleep(.05)
(release/'submit.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result))
'''
 result=remote(script.replace('CONFIG',repr(c)),runtime);d.write(local/'submit.json',result);print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile','GPU')}))
if __name__=='__main__':main()
