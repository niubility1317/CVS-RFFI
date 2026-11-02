"""Run source replay and seven-view prediction for fixed raw IQ flow."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-spectral-synthesis-pilot-ground-test-s2026092701-r01';C='be052d41090b38e3017f611c577f07a0a9c85a55';P=R/'runs'/(N+'-controller')
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
def wait_capacity():
 while os.getloadavg()[0]>os.cpu_count()-24:status('WAITING_CPU_CAPACITY',load=os.getloadavg()[0]);time.sleep(60)
try:
 ap=argparse.ArgumentParser();ap.add_argument('--prior-pid',type=int,required=True);a=ap.parse_args();previous='20261102-phase1-spectral-synthesis-scaled-ground-test-s2026092701-r01';up=R/'runs'/(previous+'-controller')/'status.json'
 while True:
  state=json.loads(up.read_text()) if up.exists() else {}
  if state.get('state') in ('PREDICTIONS_COMPLETE_AWAITING_AUTO_AUDIT','FAILED'):break
  proc=Path('/proc')/str(a.prior_pid);cmd=(proc/'cmdline').read_bytes().decode(errors='replace') if proc.exists() else ''
  if previous not in cmd:raise RuntimeError('Prior prediction stopped without terminal status')
  status('WAITING_PRIOR_PREDICTION',prior_state=state.get('state'));time.sleep(60)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(R/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(R/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';entry=R/'releases'/(N+'-release')/'experiments/phase1_spectral_synthesis/phase1_pilot_synthesis_cpu.py';out=R/'runs'/N
 for mode in ('smoke','predict'):
  wait_capacity();status('RUNNING_SOURCE_REPLAY' if mode=='smoke' else 'RUNNING_SEVEN_VIEW_PREDICTION')
  with (P/(mode+'.log')).open('x') as f:q=subprocess.run([py,'-X','utf8','-u',str(entry),mode,'--root',str(R),'--out',str(out),'--commit',C,'--device','cpu'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(mode+' failed with exit '+str(q.returncode))
 status('PREDICTIONS_COMPLETE_AWAITING_AUTO_AUDIT')
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
