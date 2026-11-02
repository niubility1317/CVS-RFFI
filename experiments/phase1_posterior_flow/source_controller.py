"""Run source smoke and fixed nonlocal posterior synthesis training."""
import json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-posterior-nonlocal-source-s2026092701-r01';SMOKE='20261102-phase1-posterior-nonlocal-smoke-s2026092701-r01';C='8c97af0696c0e143158f8d6bf7cc53a996a8cf09';P=R/'runs'/(N+'-controller');REL=R/'releases/20261102-phase1-posterior-nonlocal-s2026092701-r01-release'
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
def wait_capacity():
 while os.getloadavg()[0]>os.cpu_count()-24:status('WAITING_CPU_CAPACITY',load=os.getloadavg()[0]);time.sleep(60)
try:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(R/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(R/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';entry=REL/'experiments/phase1_posterior_flow/train_posterior_nonlocal.py'
 for mode,name in [('SMOKE',SMOKE),('TRAIN',N)]:
  wait_capacity();status('RUNNING_'+mode);args=[py,'-X','utf8','-u',str(entry),'--root',str(R),'--out',str(R/'runs'/name),'--commit',C,'--fit-size','8192']+(['--smoke'] if mode=='SMOKE' else [])
  with (P/(mode.lower()+'.log')).open('x') as f:q=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(mode+' exited '+str(q.returncode))
  result=json.loads((R/'runs'/name/('result.json' if mode=='SMOKE' else 'prediction_complete.json')).read_text())
  if mode=='SMOKE':
   if result.get('status')!='PASS' or result.get('source_fit_ids')!=32 or result.get('F_frozen') is not True or result.get('blind_frozen') is not True or result.get('DURA_loaded') is not False or any(result.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Source smoke contract failed')
  elif result.get('status')!='SOURCE_PREDICTIONS_FIXED' or result.get('commit')!=C or result.get('source_fit_ids')!=8192 or result.get('count')!=672 or not result.get('checkpoint_sha256') or any(result.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads','model_updates_after_prediction')):raise ValueError('Fixed source prediction contract failed')
 status('SOURCE_TRAINING_COMPLETE',source_prediction_sha256=result['prediction_sha256'],checkpoint_sha256=result['checkpoint_sha256'])
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
