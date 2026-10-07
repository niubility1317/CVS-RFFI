"""Publish fixed scratch repair controls and automatic seven-view tests."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish_recovery import PATHS,ROOT,CONNECTION,ssh
from experiments.cvs_phase1_stack.design import PROJECT
PATHS = [*PATHS, 'experiments/cvs_phase1_repair']
RUN='20261008-phase1-reference-repair-manysig-m32-r01'
RELEASE='cvs_reference_repair_20261008_r01'

REMOTE=r'''
from pathlib import Path
import json,os,subprocess,tarfile,hashlib
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
for path in [release,p/'runs'/c['run'],p/'logs'/c['run']]:
    if path.exists():raise FileExistsError('Existing repair path; reconcile before retry '+str(path))
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
    t.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release)],env=env,check=True)
check="from experiments.cvs_phase1_repair.smoke import smoke; from experiments.cvs_phase1_repair.design import *; assert Path(SOURCE).is_file(); assert Path(PROJECT+'/Dataset_WigSig/ManySig.pkl').is_file(); assert (Path(CAPSULE)/'manifest.json').is_file(); assert Path(TRUTH).is_file(); smoke('smoke-remote','cpu')"
with (release/'preflight.log').open('x') as log:subprocess.run([python,'-c',check],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if __import__('shutil').disk_usage(p).free < 100*1024**3:raise RuntimeError('Insufficient output space')
cmd=[python,'-u','-m','experiments.cvs_phase1_repair.dispatch'];logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=cmd,log=str(logpath),commit=c['commit'],run_id=c['run'])
with (release/'submit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''

def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote OID differs')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing delivery; reconcile')
    blob=subprocess.check_output(['git','archive','--format=tar',commit,'--',*PATHS],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(blob)) as src,tarfile.open(archive,'w:gz') as dst:
        for item in src.getmembers():
            if item.isfile() and Path(item.name).suffix in {'.py','.json'}:dst.addfile(item,src.extractfile(item))
        content=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(content);dst.addfile(item,io.BytesIO(content))
    cfg=dict(project=PROJECT,run=RUN,release=RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))));(output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();publish(a.output)
