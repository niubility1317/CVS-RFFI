"""One committed immutable release;reuse existing ordinary source runtime."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
def remote(script):
 rt=d.runtime();ssh=['-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10'];p=subprocess.run(['ssh',*ssh,'-T','N607',rt['python']+' -'],input=script.encode('utf8'),capture_output=True,timeout=55)
 if p.returncode:raise RuntimeError(p.stderr.decode('utf8',errors='replace'))
 return json.loads(p.stdout)
def expand(v):
 if isinstance(v,str):return v.replace('PROJECT',d.PROJECT)
 if isinstance(v,list):return [expand(x) for x in v]
 if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
 return v
def main():
 package='experiments/cvs_shadow_lhs30';rt=d.runtime();local=d.ROOT/d.LOCAL;local.mkdir(parents=True,exist_ok=True)
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual code/config before release')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();release=d.PROJECT+'/releases/'+d.RELEASE;c=dict(project=d.PROJECT,release=release,archive=release+'.tar.gz',run=d.BASE.as_posix(),python=rt['python'],commit=commit,parent_runtime=rt['parent_runtime_release'],parent_runtime_commit=rt['parent_runtime_commit'],source=d.SOURCE,parent=(d.parent_source()/'final_ssdg.pth').as_posix())
 script=r'''
import json,os,pwd,shutil
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or pwd.getpwuid(os.geteuid()).pw_name!=Path(c['python']).parts[2]:raise ValueError('Configured ordinary N607 account only')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Reconcile existing immutable output,never duplicate')
for p in (c['source'],c['parent'],str(Path(c['parent_runtime'])/'runtime.json')):
 if not Path(p).is_file():raise FileNotFoundError(p)
if (Path(c['parent_runtime'])/'release_commit.txt').read_text().strip()!=c['parent_runtime_commit']:raise ValueError('Consumed source runtime Gitidentity differs')
if shutil.disk_usage(c['project']).free<1024**3:raise ValueError('Insufficient output space')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,immutable_output_free=True,common_NoAs20_parent_present=True,source_runtime_present=True)))
'''
 d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c))))
 raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT);record=expand(json.loads(subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT)));record['code']['commit']=commit;archive=local/(d.RELEASE+'.tar.gz');runtime={k:v for k,v in rt.items() if k!='ssh_config'}
 with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
  for member in source.getmembers():
   if member.isfile():target.addfile(member,source.extractfile(member))
  for name,data in [('release_commit.txt',(commit+'\n').encode()),('runtime.json',json.dumps(runtime).encode('utf8')),('registration/experiment.json',json.dumps(record,ensure_ascii=False).encode('utf8'))]:
   member=tarfile.TarInfo(name);member.size=len(data);target.addfile(member,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();ssh=['-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10'];subprocess.run(['scp',*ssh,str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=55)
 script=r'''
import hashlib,json,os,subprocess,time,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release']);archive=Path(c['archive'])
if release.exists() or Path(c['run']).exists():raise FileExistsError('Reconcile prior mutation,never relaunch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid release archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_shadow_lhs30')],cwd=release,env=env,check=True)
with (release/'checks.stdout.log').open('x') as stream:subprocess.run([c['python'],'-u',str(release/'experiments/cvs_shadow_lhs30/runner.py'),'--checks'],cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
command=[c['python'],'-u',str(release/'experiments/cvs_shadow_lhs30/runner.py')];log=release/'dispatcher.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',synthetic_checks='PASS',GPU='CPUdispatcher/sharedcapacity4workers')
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n');deadline=time.monotonic()+2
while time.monotonic()<deadline and child.poll() is None:
 try:
  proc=Path('/proc')/str(child.pid);actual=dict(cwd=os.readlink(proc/'cwd'),argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1])
  if actual['cwd']==str(release) and actual['argv']==command:receipt.update(status='OBSERVED',actual=actual);break
 except FileNotFoundError:pass
 time.sleep(.05)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
 receipt=remote(script.replace('CONFIG',repr(c)));d.write(local/'submit.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','pid','commit','archive_transfer','remote_compile','synthetic_checks','GPU')}))
if __name__=='__main__':main()
