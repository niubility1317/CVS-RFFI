"""Wait for source audit and prior prediction, then verify ground routes before truth."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-gate-aware-router-ground-test-s2026092701-r01';S='20261102-phase1-gate-aware-router-source-score-s2026092701-r01';C='9c031ca56c1b701abe95e1fc32aea9161d992540';PREV='20261102-phase1-observation-unroll-ground-test-s2026092701-r02';P=R/'runs'/(N+'-controller')
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
def proc_has(pid,token):
 p=Path('/proc')/str(pid);return token in ((p/'cmdline').read_bytes().decode(errors='replace') if p.exists() else '')
def capacity():
 while os.getloadavg()[0]>os.cpu_count()-24:status('WAITING_CPU_CAPACITY',load=os.getloadavg()[0]);time.sleep(60)
try:
 ap=argparse.ArgumentParser();ap.add_argument('--score-pid',type=int,required=True);ap.add_argument('--prior-pid',type=int,required=True);a=ap.parse_args();up=R/'runs'/(S+'-controller')/'status.json'
 while True:
  state=json.loads(up.read_text()) if up.exists() else {}
  if state.get('state')=='SOURCE_SCORE_AND_AUDIT_COMPLETE':break
  if state.get('state')=='FAILED':raise RuntimeError('Source integrity audit failed: '+str(state))
  if not proc_has(a.score_pid,S):raise RuntimeError('Source audit stopped without completion')
  status('WAITING_SOURCE_AUDIT',source_state=state.get('state'));time.sleep(60)
 audit=json.loads((R/'runs'/S/'independent_audit.json').read_text())
 if audit.get('status')!='PASS' or audit.get('records_replayed')!=11761 or audit.get('model_replayed')!=3 or any(audit.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Source model integrity audit required')
 previous=R/'runs'/PREV/'flow/test_prediction/complete.json'
 while not previous.exists() and proc_has(a.prior_pid,PREV):status('WAITING_PRIOR_PREDICTION');time.sleep(60)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONPATH=str(R/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(R/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';rel=R/'releases'/(N+'-release')/'experiments/phase1_joint_mixture';entry=rel/'phase1_gate_router_cpu.py';out=R/'runs'/N
 for mode in ('smoke','predict'):
  capacity();status('RUNNING_SOURCE_REPLAY' if mode=='smoke' else 'RUNNING_SEVEN_VIEW_PREDICTION')
  with (P/(mode+'.log')).open('x') as f:q=subprocess.run([py,'-X','utf8','-u',str(entry),mode,'--root',str(R),'--out',str(out),'--commit',C,'--device','cpu'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(mode+' exited '+str(q.returncode))
 status('RUNNING_PRETRUTH_ROUTE_AUDIT')
 with (P/'route_audit.log').open('x') as f:q=subprocess.run([py,'-X','utf8','-u',str(rel/'audit_gate_router_ground.py'),'--root',str(R),'--out',str(out),'--commit',C],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
 if q.returncode:raise RuntimeError('Independent route audit exited '+str(q.returncode))
 result=json.loads((out/'pretruth_route_audit.json').read_text())
 if result.get('status')!='PASS' or result.get('count')!=168000 or any(result.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_IQ_reads','model_updates')):raise ValueError('Independent pretruth route audit required')
 dst=out/'flow/test_prediction';pending=dst/'pending_complete.json';complete=dst/'complete.json'
 if complete.exists() or not pending.is_file():raise FileExistsError('Pending prediction manifest required')
 os.replace(pending,complete);status('PREDICTIONS_COMPLETE_AWAITING_AUTO_AUDIT')
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
