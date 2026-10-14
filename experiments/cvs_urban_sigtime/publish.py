"""One immutable Git code/config release and independently observable launch."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
from .operations import remote,SSH

REMOTE=r'''
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
c=CONFIG;project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run']
analysis=c.get('analysis',False)
if release.exists() or (run.exists() and not analysis):raise FileExistsError('Existing release/run; reconcile before resubmit')
if analysis and not (run/'complete.json').is_file():raise ValueError('Post-freeze analysis requires completed registered rows')
if analysis and (run/'source_mechanism').exists():raise FileExistsError('Existing source mechanism output; reconcile before retry')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer comparison failed')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary account/host differs')
release.mkdir()
with tarfile.open(archive) as f:
 for member in f.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive destination')
 f.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_urban_sigtime'),str(release/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py')],cwd=release,env=env,check=True)
spec=json.loads((release/'experiments/cvs_urban_sigtime/registration.json').read_text())
for path in [spec['data']['contract_ref'],spec['data']['validation_ref']]:
 if not Path(path).is_file():raise FileNotFoundError(path)
preflight=dict(status='VERIFIED',archive_transfer='VERIFIED',remote_compile='PASS',ordinary_account=True,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True),
 cwd=str(release),python=python,commit=c['commit'],read_at=time.time())
(release/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
command=[python,'-u','-m','experiments.cvs_urban_sigtime.mechanism' if analysis else 'experiments.cvs_urban_sigtime.dispatch']
logpath=release/'dispatcher.log'
with logpath.open('x') as f:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
submitted=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],run_root=str(run))
(release/'submit.json').write_text(json.dumps(submitted,indent=2)+'\n')
print(json.dumps(submitted))
'''

INSPECT=r'''
import json,os,time,subprocess
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run']
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def process(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,cwd=os.readlink(p/'cwd'),argv=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
submit=read(release/'submit.json');main=process(submit['pid']) if submit else None
workers=[]
for row in (read(base/'launch.json') or {}).get('rows',[]):
 src=base/row['row_id']/'source';ep=src/'epoch_metrics.jsonl'
 count=sum(1 for _ in ep.open()) if ep.is_file() else 0
 log=Path(row['log']);workers.append(dict(row,process=process(row['pid']),epoch_count=count,
  log_bytes=log.stat().st_size if log.is_file() else 0,log_mtime=log.stat().st_mtime if log.is_file() else None,
  resolved=read(src/'resolved_config.json'),source_complete=read(src/'completion.json') is not None,
  prediction_complete=(base/row['row_id']/'prediction/complete.json').is_file(),scored=(base/row['row_id']/'scoring/complete.json').is_file()))
failure=read(base/'failure.json');done=read(base/'complete.json')
diagnostic=read(base/'source_diagnostic/results.json')
valid=bool(main and main['cwd']==str(release) and main['argv']==submit['argv'])
if done:valid=all(w['scored'] for w in workers) and len(workers)==done['actual_model_count']
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=main,
 workers=workers,queue=read(base/'queue_state.json'),completion=done,failure=failure,
 source_diagnostic=diagnostic,source_qualification=read(base/'source_qualification.json'),
 source_mechanism_progress=read(base/'source_mechanism/progress.json'),source_mechanism_complete=(base/'source_mechanism/results.json').is_file(),
 dispatcher_log_tail=(release/'dispatcher.log').read_text(errors='replace')[-4000:] if (release/'dispatcher.log').is_file() else None,
 preflight=read(release/'preflight.json'),read_at=time.time(),
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True))))
'''

def configuration(analysis=False):return dict(project=d.PROJECT,release=d.RELEASE+('_analysis' if analysis else ''),run=d.RUN,analysis=analysis)

def inspect(output,analysis=False):
    value=remote(INSPECT.replace('CONFIG',repr(configuration(analysis))))
    d.write(Path(output)/'readback.json',value)
    print(json.dumps(dict(status=value['status'],workers=[dict(row_id=w['row_id'],pid=w['pid'],live=w['process'] is not None,epochs=w['epoch_count'],source_complete=w['source_complete'],scored=w['scored']) for w in value['workers']],failure=value['failure'],completion=value['completion'],dispatcher_log_tail=value['dispatcher_log_tail'] if not value['workers'] else None),ensure_ascii=False))
    return value

def publish(output,analysis=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    names=subprocess.check_output(['git','ls-files'],cwd=d.ROOT,text=True).splitlines()
    names=[n for n in names if n.startswith(('experiments/','code/','baselines/','comparison_suite/')) and Path(n).suffix in ('.py','.json')
        and '/snapshots/' not in n and '/local_artifacts/' not in n and '/tests/' not in n]
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments','code','baselines','comparison_suite'],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Uncommitted release code/config; preserve unrelated changes')
    archive=output/(configuration(analysis)['release']+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name in names:tar.add(d.ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();entry=tarfile.TarInfo('release_commit.txt');entry.size=len(blob);tar.addfile(entry,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    value=remote(REMOTE.replace('CONFIG',repr(dict(configuration(analysis),archive=archive.name,sha256=sha,commit=commit))))
    d.write(output/'submission.json',value);print(json.dumps(value));return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');p.add_argument('--analysis',action='store_true');a=p.parse_args()
    inspect(a.output,a.analysis) if a.inspect else publish(a.output,a.analysis)
