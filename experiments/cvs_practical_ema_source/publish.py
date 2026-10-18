"""One ordinary-account immutable source-only release; no formal evaluator."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d

def remote(script,rt):
 args=['ssh','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',rt['python']+' -']
 result=subprocess.run(args,input=script.encode('utf8'),capture_output=True,timeout=60)
 if result.returncode:raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
 return json.loads(result.stdout)

def main():
 package='experiments/cvs_practical_ema_source';local=d.ROOT/d.LOCAL;rt=d.read(local/'runtime.json')
 if rt.get('run_id')!=d.RUN or rt.get('parent_commit')!=d.PARENT_COMMIT or Path(rt['parent_release'])!=Path(rt['project'])/'releases'/d.PARENT_RELEASE:raise ValueError('Private runtime for another fixed source run')
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual code before release')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();project=rt['project'];gpu=rt.get('gpu',0)
 if type(gpu) is not int or gpu not in range(4):raise ValueError('One physical GPU index 0..3 required')
 c=dict(project=project,release=project+'/releases/'+d.RELEASE,archive=project+'/releases/'+d.RELEASE+'.tar.gz',run=project+'/runs/'+d.RUN,parent_release=rt['parent_release'],parent_commit=d.PARENT_COMMIT,parent_run=d.PARENT_RUN,run_id=d.RUN,python=rt['python'],user=rt['user'],host=rt['host'],gpu=gpu,commit=commit)
 script=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Immutable release/run exists; reconcile,no resubmission')
if (Path(c['parent_release'])/'release_commit.txt').read_text().strip()!=c['parent_commit']:raise ValueError('Only original fixed E50 release')
for arm in ('B00','B01'):
 root=Path(c['project'])/'runs'/c['parent_run']/(arm+'-s2026092701')
 for name in ('source/final_ssdg.pth','source/completion.json','source/source_frozen.json','source/source_contract.json','source_qualification/predictions.npz','source_qualification/complete.json'):
  if not (root/name).is_file():raise FileNotFoundError(root/name)
if shutil.disk_usage(c['project']).free<1024**3:raise ValueError('Insufficient source output space')
gpu=str(c['gpu']);pids=[v for v in subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if v.strip().isdigit()]
free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if len(pids)>=3 or free<12000:raise ValueError('Original total GPU process cap3/free12GB not available')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,parent_release_present=True,source_artifacts_present=True,existing_GPU_processes=len(pids),free_GPU_MB=free,GPU=c['gpu'],checkpoint_or_IQ_read=False)))
'''
 d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c)),rt))
 raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT)
 record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT)
 runtime_bytes=json.dumps({k:rt[k] for k in ('run_id','project','parent_release','parent_commit')},indent=2).encode('utf8');archive=local/(d.RELEASE+'.tar.gz')
 with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
  for member in source.getmembers():
   if member.isfile():target.addfile(member,source.extractfile(member))
  for name,data in [('release_commit.txt',(commit+'\n').encode()),('registration/experiment.json',record),('runtime.json',runtime_bytes)]:
   item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
 subprocess.run(['scp','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
 script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);run=Path(c['run']);archive=Path(c['archive'])
if release.exists() or run.exists():raise FileExistsError('Existing mutation; reconcile,no duplicate')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer comparison differs')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_practical_ema_source')],cwd=release,env=env,check=True)
command=[c['python'],'-u',str(release/'experiments/cvs_practical_ema_source/runner.py')]
with (release/'checks.stdout.log').open('x') as stream:subprocess.run(command+['--checks'],cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
env['CUDA_VISIBLE_DEVICES']=str(c['gpu']);log=release/'ownEMA_source.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',single_archive_SHA=c['sha256'],GPU=c['gpu'],source_only=True)
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');proc=Path('/proc')/str(child.pid);deadline=time.monotonic()+2
while time.monotonic()<deadline and child.poll() is None:
 try:
  actual=dict(cwd=os.readlink(proc/'cwd'),argv=(proc/'cmdline').read_bytes().decode().split(chr(0))[:-1],cuda=[v for v in (proc/'environ').read_bytes().decode().split(chr(0)) if v.startswith('CUDA_VISIBLE_DEVICES=')])
  if actual['cwd']==str(release) and actual['argv']==command and actual['cuda']==['CUDA_VISIBLE_DEVICES='+str(c['gpu'])]:result.update(status='OBSERVED',actual=actual);break
 except FileNotFoundError:pass
 time.sleep(.05)
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
 result=remote(script.replace('CONFIG',repr(c)),rt);d.write(local/'submit.json',result)
 print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile','GPU','source_only')}))

if __name__=='__main__':main()
