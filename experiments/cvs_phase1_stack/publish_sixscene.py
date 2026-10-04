"""Publish the authorized deferred evaluation without touching training owners."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_phase1_stack import sixscene_after as s
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh

REMOTE=r'''
import hashlib,importlib.util,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run'];logs=project/'logs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if release.exists() or run.exists() or logs.exists():raise FileExistsError('Reconcile existing eval delivery; no duplicate')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if m.name not in ['sixscene_after.py','experiment_spec.json','release_commit.txt'] or not m.isfile():raise ValueError('Unexpected archive contents')
    t.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';worker=Path(c['worker_root'])
env=dict(os.environ,PYTHONPATH=str(worker)+os.pathsep+str(worker/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','py_compile',str(release/'sixscene_after.py')],check=True)
check="import importlib.util; from pathlib import Path; p=Path("+repr(str(release/'sixscene_after.py'))+"); q=importlib.util.spec_from_file_location('e',p); m=importlib.util.module_from_spec(q); q.loader.exec_module(m); d=m.original(); v=m.manifest(d); assert len(d.rows())==136; assert len(m.VIEWS)==7; assert all((m.VIEWS_ROOT/(x+'.npy')).is_file() for x in m.SCENES); assert Path(v['clean_ref']).is_file(); m.parent_ready(d)"
subprocess.run([python,'-c',check],cwd=worker,env=env,stdout=subprocess.DEVNULL,check=True)
command=[python,'-u',str(release/'sixscene_after.py'),'--mode','dispatch']
with (release/'dispatcher.stdout.log').open('x') as log:
    child=subprocess.Popen(command,cwd=worker,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='SUBMITTED',pid=child.pid,cwd=str(worker),argv=command,commit=c['commit'],log=str(release/'dispatcher.stdout.log'),run_root=str(run))
(release/'submit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    oid=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if oid!=commit:raise ValueError('Remote HEAD differs')
    paths=['experiments/cvs_phase1_stack/'+v for v in ['sixscene_after.py','publish_sixscene.py','sixscene_configs/experiment_spec.json']]
    if subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted evaluation code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(s.RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive; reconcile before retry')
    with tarfile.open(archive,'w:gz') as t:
        for name,data in [('sixscene_after.py',subprocess.check_output(['git','show',commit+':'+paths[0]],cwd=ROOT)),('experiment_spec.json',subprocess.check_output(['git','show',commit+':'+paths[2]],cwd=ROOT)),('release_commit.txt',(commit+'\n').encode())]:
            m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o644;t.addfile(m,io.BytesIO(data))
    c=dict(project=s.PROJECT,run=s.RUN,release=s.RELEASE,worker_root=s.WORKER_ROOT.as_posix(),archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit)
    (output/'package.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+s.PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(c))));(output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.output)
