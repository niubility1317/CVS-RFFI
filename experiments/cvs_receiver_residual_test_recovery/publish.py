"""Committed evaluation-only release; preserve all original training artifacts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_receiver_residual_test_recovery import design as d
from experiments.cvs_receiver_residual_v2.publish import ssh, CONNECTION

REMOTE=r'''
import hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host/user differs')
if release.exists() or any((p/'runs'/r['run_id']).exists() or (p/'logs'/r['run_id']).exists() for r in c['runs']):raise FileExistsError('Existing delivery: reconcile before resubmit')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer differs')
if shutil.disk_usage(p).free<2*1024**3:raise RuntimeError('Insufficient new prediction storage')
release.mkdir()
with tarfile.open(archive) as tar:
    for m in tar.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';env=dict(os.environ,PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
env.pop('CUDA_VISIBLE_DEVICES',None)
subprocess.run([python,'-m','compileall','-q',str(release/'experiments')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_receiver_residual_test_recovery.contract_smoke'],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
sys.path.insert(0,str(release))
from experiments.cvs_receiver_residual_test_recovery import evaluate as ev
for r in c['runs']:
    ev.configure(r['run_id'])
    base=p/'runs'/r['parent_run_id']
    if json.loads((base/'failure.json').read_text())['error']!="KeyError('classes')" or (base/'source_matrix_frozen.json').exists() or (base/'launch_predict.json').exists():raise ValueError('Original failure/target state differs')
    for cfg in ev.d.rows():
        ev.source_provenance(cfg)
    ev.manifest() # existing VALIDATED_ONCE only; no IQ builder or truth read
command=[python,'-u','-m','experiments.cvs_receiver_residual_test_recovery.dispatch']
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],runs=c['runs'],training=False)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(output):
    prefixes=['experiments/cvs_receiver_residual_test_recovery/','experiments/cvs_receiver_residual_capacity4/']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=d.ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=d.ROOT,text=True).split()[0]
    if remote!=head:raise ValueError('Remote OID differs')
    if subprocess.check_output(['git','status','--porcelain','--',*prefixes],cwd=d.ROOT,text=True).strip():raise ValueError('Uncommitted evaluation code')
    names=subprocess.check_output(['git','ls-files','--',*prefixes],cwd=d.ROOT,text=True).splitlines()
    output.mkdir(parents=True,exist_ok=True);archive=output/(d.RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive: reconcile before resubmit')
    with tarfile.open(archive,'w:gz') as tar:
        for n in names:
            if not n.endswith('.py'):continue
            blob=subprocess.check_output(['git','show',head+':'+n],cwd=d.ROOT);item=tarfile.TarInfo(n);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
        blob=(head+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    config=dict(project=d.PROJECT,release=d.RELEASE,commit=head,runs=d.RUNS,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(config))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);publish(p.parse_args().output)
