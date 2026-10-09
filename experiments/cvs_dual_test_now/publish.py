"""Publish the reviewed evaluation-only controller from committed local code."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_dual_evidence.publish import PATHS as ORIGINAL_PATHS,ROOT,CONNECTION,ssh
from experiments.cvs_dual_test_now.driver import RUN,RELEASE,source
PATHS=[*ORIGINAL_PATHS,'experiments/cvs_dual_test_now']
REMOTE=r'''
from pathlib import Path
import hashlib,json,os,subprocess,tarfile
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host identity differs')
for q in [release,p/'runs'/c['run'],p/'logs'/c['run']]:
 if q.exists():raise FileExistsError('Existing path; reconcile rather than retry '+str(q))
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer differs')
release.mkdir()
with tarfile.open(archive) as t:
 for m in t.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
 t.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_dual_test_now')],env=env,check=True)
cmd=[python,'-u','-m','experiments.cvs_dual_test_now.driver']
with (release/'dispatcher.stdout.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,commit=c['commit'],run_id=c['run'])
with (release/'submit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
def publish(output):
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
 if remote!=commit:raise ValueError('Unpushed release')
 if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted code')
 output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
 if archive.exists():raise FileExistsError('Delivery exists; reconcile')
 blob=subprocess.check_output(['git','archive','--format=tar',commit,'--',*PATHS],cwd=ROOT)
 with tarfile.open(fileobj=io.BytesIO(blob)) as src,tarfile.open(archive,'w:gz') as dst:
  for m in src.getmembers():
   if m.isfile() and Path(m.name).suffix in ('.py','.json'):dst.addfile(m,src.extractfile(m))
  b=(commit+'\n').encode();m=tarfile.TarInfo('release_commit.txt');m.size=len(b);dst.addfile(m,io.BytesIO(b))
 cfg=dict(project=source.PROJECT,run=RUN,release=RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 source.write(output/'package.json',cfg)
 subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+source.PROJECT+'/releases/'+archive.name],check=True)
 result=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))));source.write(output/'submit.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.output)
