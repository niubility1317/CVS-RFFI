"""Deliver a new immutable diagnostic; never touch previous model/test releases."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_equivariant_identity.publish import ssh, CONNECTION
from experiments.cvs_source_sensitivity.audit import RUN, PROJECT, BASE, BASE_COMMIT

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 'cvs_received_sensitivity_source_20261002_r01'
REMOTE = r'''
import ast,hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run']
if release.exists() or run.exists():raise FileExistsError('Reconcile existing diagnostic; never resubmit')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host/user mismatch')
base=Path(c['base'])
if (base/'release_commit.txt').read_text().strip()!=c['base_commit']:raise ValueError('Immutable base mismatch')
tree=ast.parse((base/'experiments/cvs_clean_design/dispatch.py').read_text())
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('occupancy','choose_gpu')]
if len(functions)!=2:raise ValueError('Immutable GPU preflight helpers absent')
exec(compile(ast.Module(body=functions,type_ignores=[]),'immutable_gpu_helpers','exec'))
gpu=choose_gpu({})
if gpu is None:raise RuntimeError('No authorized free GPU capacity; no launch')
preflight=dict(read_at=time.time(),gpu=gpu,occupancy={str(k):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for k,v in occupancy({}).items()})
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive member')
    tar.extractall(release)
(release/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(base)+os.pathsep+str(base/'code'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'audit.py')],env=env,check=True)
command=[python,'-u',str(release/'audit.py'),'--output',str(run),'--device','cuda:0']
with (release/'stdout.log').open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,gpu=gpu,log=str(release/'stdout.log'),commit=c['commit'],run_id=c['run'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(output):
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote = subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote != commit:
        raise ValueError('Remote branch differs from HEAD')
    prefix = 'experiments/cvs_source_sensitivity/'
    dirty = subprocess.check_output(['git','status','--porcelain','--',prefix],cwd=ROOT,text=True)
    if dirty.strip():
        raise ValueError('Uncommitted diagnostic code/config')
    output.mkdir(parents=True,exist_ok=False)
    archive = output / (RELEASE+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for source, target in [('audit.py','audit.py'),('configs/experiment_spec.json','experiment_spec.json')]:
            blob = subprocess.check_output(['git','show',commit+':'+prefix+source],cwd=ROOT)
            item = tarfile.TarInfo(target);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
        blob = (commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    config = dict(project=PROJECT,run=RUN,release=RELEASE,archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit,base=BASE,base_commit=BASE_COMMIT)
    (output/'package.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt = json.loads(ssh(REMOTE.replace('CONFIG',repr(config))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    publish(parser.parse_args().output)
