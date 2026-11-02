"""Wait for source selector then independently score and replay all folds."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-gate-aware-router-source-score-s2026092701-r01';S='20261102-phase1-gate-aware-router-source-s2026092701-r01';P=R/'runs'/(N+'-controller')
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
try:
 ap=argparse.ArgumentParser();ap.add_argument('--source-pid',type=int,required=True);a=ap.parse_args();up=R/'runs'/(S+'-controller')/'status.json'
 while True:
  state=json.loads(up.read_text()) if up.exists() else {}
  if state.get('state')=='SOURCE_TRAINING_COMPLETE':break
  if state.get('state')=='FAILED':raise RuntimeError('Source training failed: '+str(state))
  proc=Path('/proc')/str(a.source_pid);cmd=(proc/'cmdline').read_bytes().decode(errors='replace') if proc.exists() else ''
  if S not in cmd:raise RuntimeError('Source controller stopped before completion')
  status('WAITING_SOURCE_TRAINING',source_state=state.get('state'));time.sleep(60)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python';rel=R/'releases'/(N+'-release')/'experiments/phase1_joint_mixture';out=R/'runs'/N;out.mkdir(exist_ok=False);source=R/'runs'/S
 for label,entry,args in [('SCORE','score_gate_aware_source.py',['--root',str(R),'--source',str(source),'--out',str(out/'source_score.json')]),('AUDIT','audit_gate_aware_source.py',['--root',str(R),'--source',str(source),'--score',str(out/'source_score.json'),'--out',str(out/'independent_audit.json')])]:
  status('RUNNING_'+label)
  with (out/(label.lower()+'.log')).open('x') as f:q=subprocess.run([py,str(rel/entry)]+args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(label+' exited '+str(q.returncode))
 if json.loads((out/'independent_audit.json').read_text()).get('status')!='PASS':raise ValueError('Independent source audit failed')
 status('SOURCE_SCORE_AND_AUDIT_COMPLETE')
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
