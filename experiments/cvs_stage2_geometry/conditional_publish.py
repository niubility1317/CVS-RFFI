"""Single CPU source diagnostic; committed release, no CUDA or training queue."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from . import design as d
from .conditional_audit import RUN,RELEASE
from .publish import PATHS,CONNECTION,ssh


REMOTE = r'''
from pathlib import Path
import os,json,subprocess,tarfile,hashlib
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];base=p/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if release.exists() or base.exists():raise FileExistsError('Existing release/run; reconcile')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transport mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe member')
 tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (release/'checks.log').open('x') as log:
 subprocess.run([python,'-m',c['checks_module']],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
command=[python,'-u','-m',c['module']]
logpath=release/'stdout.log'
with logpath.open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],run_id=c['run'],cuda_visible_devices='')
with (release/'submit.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
'''

INSPECT = r'''
from pathlib import Path
import json,time,os
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];base=p/'runs'/c['run']
def read(path):return json.loads(path.read_text()) if path.is_file() else None
submit=read(release/'submit.json');proc=None
if submit:
 path=Path('/proc')/str(submit['pid'])
 try:proc=dict(cwd=str((path/'cwd').resolve(strict=True)),argv=(path/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
result=dict(at=time.time(),submit=submit,process=proc,completion=read(base/'completion.json'),failure=read(base/'failure.json'),
 summary=read(base/'summary.json'),rows=[dict(seed=s,completion=read(base/('s'+str(s))/'completion.json')) for s in c['seeds']],
 log_tail=(release/'stdout.log').read_text()[-4000:] if (release/'stdout.log').is_file() else None,
 checks=(release/'checks.log').read_text() if (release/'checks.log').is_file() else None)
result['status']='UNKNOWN'
if result['failure']:result['status']='FAILED'
elif result['completion']:
 done=result['completion']
 if done['rows_completed']==3 and done['identity_updates']==0 and not done['target_access'] and all(r['completion'] for r in result['rows']):result['status']='VERIFIED'
elif proc and proc['cwd']==submit['cwd'] and proc['argv']==submit['argv']:result['status']='VERIFIED'
print(json.dumps(result))
'''


def inspect(output):
    out=Path(output);config=d.read(out/'package.json')
    result=json.loads(ssh(INSPECT.replace('CONFIG',repr(config))))
    d.write(out/'readback.json',result)
    print(json.dumps(result));return result


def publish(output,run_id=RUN,release_name=RELEASE,
            module='experiments.cvs_stage2_geometry.conditional_audit',
            checks_module='experiments.cvs_stage2_geometry.conditional_checks',extra_paths=()):
    paths=list(dict.fromkeys([*PATHS,*extra_paths]))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=d.ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1',
        'ls-remote','origin','refs/heads/'+branch],cwd=d.ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote differs from HEAD')
    record='automation_reports/CV-SincNet/'+run_id
    dirty=subprocess.check_output(['git','status','--porcelain','--',*paths,record],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Uncommitted release paths')
    archive=out/(release_name+'.tar.gz')
    blob=subprocess.check_output(['git','archive','--format=tar',commit,'--',*paths],cwd=d.ROOT)
    with tarfile.open(fileobj=io.BytesIO(blob)) as source,tarfile.open(archive,'w:gz') as target:
        for member in source.getmembers():
            if member.isfile() and Path(member.name).suffix in {'.py','.json','.md'}:
                target.addfile(member,source.extractfile(member))
        entries={'release_commit.txt':(commit+'\n').encode()}
        for name in ('experiment.json','report.md','events.jsonl'):
            entries['registration/'+name]=subprocess.check_output(['git','show',commit+':'+record+'/'+name],cwd=d.ROOT)
        for name,value in entries.items():
            member=tarfile.TarInfo(name);member.size=len(value);target.addfile(member,io.BytesIO(value))
    config=dict(project=d.PROJECT,run=run_id,release=release_name,commit=commit,archive=archive.name,
        module=module,checks_module=checks_module,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),seeds=list(d.SEEDS))
    d.write(out/'package.json',config)
    before=json.loads(ssh("from pathlib import Path\nimport json\nc="+repr(config)+
        "\np=Path(c['project']);paths=[p/'releases'/c['release'],p/'runs'/c['run'],p/'releases'/c['archive']]\nprint(json.dumps([str(v) for v in paths if v.exists()]))\n"))
    d.write(out/'before.json',before)
    if before:raise FileExistsError('Reconcile existing paths: '+repr(before))
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True)
    try:d.write(out/'submit.json',json.loads(ssh(REMOTE.replace('CONFIG',repr(config)))))
    except Exception as exc:
        d.write(out/'unknown.json',dict(status='UNKNOWN',error=repr(exc),automatic_retry=False));raise
    return inspect(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    inspect(a.output) if a.inspect else publish(a.output)
