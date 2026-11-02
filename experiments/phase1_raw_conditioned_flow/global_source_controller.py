"""Source-only identity flow smoke and fixed three-epoch training."""
import json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-global-flow-source-s2026092701-r01';SMOKE='20261102-phase1-global-flow-smoke-s2026092701-r01';C='a11a61dbd2dee6d484daf78773b613127cf9435a';P=R/'runs'/(N+'-controller')
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
def wait_capacity():
 while os.getloadavg()[0]>os.cpu_count()-24:status('WAITING_CPU_CAPACITY',load=os.getloadavg()[0]);time.sleep(60)
try:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(R/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(R/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';entry=R/'releases'/(N+'-release')/'experiments/phase1_raw_conditioned_flow/train_global_source.py'
 for mode,name in [('SMOKE',SMOKE),('TRAIN',N)]:
  wait_capacity();status('RUNNING_'+mode);args=[py,'-X','utf8','-u',str(entry),'--root',str(R),'--out',str(R/'runs'/name),'--commit',C]+(['--smoke'] if mode=='SMOKE' else [])
  with (P/(mode.lower()+'.log')).open('x') as f:q=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(mode+' exited '+str(q.returncode))
  result=json.loads((R/'runs'/name/'result.json').read_text());expected='SOURCE_GLOBAL_FLOW_SMOKE_PASS' if mode=='SMOKE' else 'SOURCE_GLOBAL_FLOW_TRAINING_COMPLETE'
  if result.get('status')!=expected or result.get('commit')!=C or (result.get('L_s_truth_reads')!=6300 or any(result.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads'))):raise ValueError('Fixed source output contract failed')
 status('SOURCE_TRAINING_COMPLETE',source_prediction_sha256=result['prediction_sha256'])
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
