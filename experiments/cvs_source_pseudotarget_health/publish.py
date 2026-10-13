"""Single ordinary-account immutable source diagnostic release."""
import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport

REMOTE=r'''
from pathlib import Path
import json,os,subprocess,tarfile,hashlib,shutil,sys,fcntl
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];base=p/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong ordinary account/host')
for path in [release,base]:
 if path.exists():raise FileExistsError('Existing path; reconcile: '+str(path))
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer differs')
if shutil.disk_usage(p).free<10*1024**3:raise RuntimeError('Insufficient diagnostic output space')
release.mkdir()
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive member')
 tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
with (release/'preflight.log').open('x') as log:
 subprocess.run([python,'-u','-m','experiments.cvs_source_pseudotarget_health.run','--remote-preflight','--device','cpu','--output',str(release/'preflight.json')],cwd=release,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
# Resource inspection runs in the project's torch environment and accounts every owner.
probe="import sys,json;sys.path[:0]="+repr([str(release),str(release/'code')])+";from experiments.cvs_multi_state_action.dispatch import capacity;print(json.dumps({str(g):{'pids':sorted(v['pids']),'free_mb':v['free_mb']} for g,v in capacity().items()}))"
with (p/'runs/receiver_residual_capacity4.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX)
 caps=json.loads(subprocess.check_output([python,'-c',probe],cwd=release,env=env,text=True));ready=[(len(v['pids']),-v['free_mb'],int(g)) for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
 if not ready:raise RuntimeError('No legal GPU capacity; preserve release without run')
 gpu=min(ready)[2];command=[python,'-u','-m','experiments.cvs_source_pseudotarget_health.run','--device','cuda:0','--output',str(base)];logpath=release/'dispatcher.stdout.log'
 with logpath.open('x') as log:child=subprocess.Popen(command,cwd=release,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],run_id=c['run'],gpu=gpu)
with (release/'submit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''

INSPECT=r'''
import sys,json,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run'];sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_source_pseudotarget_health import design as d
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');actual=process(submit['pid']) if submit else None;done=read(base/'completion.json');failure=read(base/'failure.json');log=release/'dispatcher.stdout.log';rows=[]
for row in d.rows():
 p=base/row['row_id'];rows.append(dict(row,summary_exists=(p/'source_summary.json').is_file(),model_contract=read(p/'model_contract.json'),probe_exists=(p/'rx_probes.json').is_file()))
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'] and actual['gpu']==str(submit['gpu']))
if done:
 if done.get('rows')!=3 or done.get('source_packets')!=270000 or done.get('identity_updates')!=0 or done.get('target_query_access') or done.get('U_truth_read') or done.get('V_updates') or not all(r['summary_exists'] and r['probe_exists'] and r['model_contract'] for r in rows):raise ValueError('Source diagnostic incomplete or unauthorized')
 valid=True
print(json.dumps(dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,completion=done,failure=failure,rows=rows,log_bytes=log.stat().st_size if log.is_file() else 0,preflight=read(release/'preflight.json'),capacity={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()},read_at=time.time())))
'''

def configure():
 from experiments.cvs_repeated_rx_margin.publish import configure as parent
 parent();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_pseudotarget_health']));transport.REMOTE=REMOTE;transport.INSPECT=INSPECT

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],completion=v['completion'],failure=v['failure'],completed_rows=sum(r['summary_exists'] and r['probe_exists'] for r in v['rows']))))
