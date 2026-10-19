"""One archive, ordinary account, one compile/check, immutable source run."""
import hashlib,io,json,subprocess,tarfile
from . import design as d
from experiments.cvs_sameiq_experts.publish import remote
def main():
 package='experiments/cvs_raw_coadapt30';local=d.ROOT/d.LOCAL;rt=d.read(local/'runtime.json')
 if rt['run_id']!=d.RUN:raise ValueError('Wrong runtime/run')
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual code/config first')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();release=rt['project']+'/releases/'+d.RELEASE
 c=dict(rt,commit=commit,release=release,archive=release+'.tar.gz',run=rt['project']+'/runs/'+d.RUN)
 code=r'''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Existing output; reconcile before any resubmission')
if (Path(c['reference_release'])/'release_commit.txt').read_text().strip()!=c['reference_commit']:raise ValueError('Pinned original runtime differs')
if shutil.disk_usage(c['project']).free<10*1024**3:raise ValueError('Insufficient space')
devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
states=[]
for line in devices:
 g,u,f=[v.strip() for v in line.split(',')];states.append(dict(gpu=int(g),free_MB=int(f),processes=sum(a.split(',')[-1].strip()==u for a in apps)))
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,immutable_paths_absent=True,disk_free_bytes=shutil.disk_usage(c['project']).free,resources=states)))
'''
 d.write(local/'preflight.json',remote(code.replace('CONFIG',repr(c)),rt))
 raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT);archive=local/(d.RELEASE+'.tar.gz')
 runtime=json.dumps({k:rt[k] for k in ('run_id','project','reference_release','reference_commit')},indent=2).encode()
 with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
  for member in source.getmembers():
   if member.isfile():target.addfile(member,source.extractfile(member))
  for name,data in [('release_commit.txt',(commit+'\n').encode()),('registration/experiment.json',record),('runtime.json',runtime)]:
   item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
 code=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release']);run=Path(c['run']);archive=Path(c['archive'])
if release.exists() or run.exists():raise FileExistsError('Possibly completed mutation; reconcile,no overwrite')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive path')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_raw_coadapt30')],cwd=release,env=env,check=True)
runner=release/'experiments/cvs_raw_coadapt30/runner.py'
cmd=[c['python'],'-u',str(runner)];log=release/'dispatcher.stdout.log'
with log.open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',CPU_threads=2,GPU='queue allocated')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
 result=remote(code.replace('CONFIG',repr(c)),rt);d.write(local/'submit.json',result);print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile')}))
if __name__=='__main__':main()
