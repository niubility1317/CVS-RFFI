"""Wait for source model, replay the complete source cohort, score and audit."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-posterior-nonlocal-full-source-s2026092701-r02';SMOKE='20261102-phase1-posterior-nonlocal-full-smoke-s2026092701-r02';S='20261102-phase1-posterior-nonlocal-source-s2026092701-r02';SCORE='20261102-phase1-posterior-nonlocal-full-score-s2026092701-r02';C='10b1f8cdaa3eb340c2a29e60456547290320ef6a';P=R/'runs'/(N+'-controller');REL=R/'releases'/(N+'-release')/'experiments/phase1_posterior_flow'
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
def wait_capacity():
 while os.getloadavg()[0]>os.cpu_count()-24:status('WAITING_CPU_CAPACITY',load=os.getloadavg()[0]);time.sleep(60)
try:
 ap=argparse.ArgumentParser();ap.add_argument('--source-pid',type=int,required=True);a=ap.parse_args();up=R/'runs'/(S+'-controller')/'status.json'
 while True:
  state=json.loads(up.read_text()) if up.exists() else {}
  if state.get('state')=='SOURCE_TRAINING_COMPLETE':break
  if state.get('state')=='FAILED':raise RuntimeError('Source training failed: '+str(state))
  proc=Path('/proc')/str(a.source_pid);cmd=(proc/'cmdline').read_bytes().decode(errors='replace') if proc.exists() else ''
  if 'phase1_posterior_flow/source_controller.py' not in cmd:raise RuntimeError('Source controller stopped before completion')
  status('WAITING_FIXED_SOURCE',source_state=state.get('state'));time.sleep(60)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(R/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(R/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
 for mode,name in [('SMOKE',SMOKE),('REPLAY',N)]:
  wait_capacity();status('RUNNING_'+mode);args=[py,'-X','utf8','-u',str(REL/'replay_nonlocal_full_source.py'),'--root',str(R),'--out',str(R/'runs'/name),'--commit',C]+(['--smoke'] if mode=='SMOKE' else [])
  with (P/(mode.lower()+'.log')).open('x') as f:q=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(mode+' exited '+str(q.returncode))
  result=json.loads((R/'runs'/name/('result.json' if mode=='SMOKE' else 'prediction_complete.json')).read_text())
  if mode=='SMOKE':
   if result.get('status')!='PASS' or result.get('count')!=64 or result.get('raw_gate_physical_match') is not True or result.get('batch_decisions_match') is not True:raise ValueError('Source smoke contract failed')
  elif result.get('status')!='SOURCE_PREDICTIONS_FIXED' or result.get('commit')!=C or result.get('count')!=11761 or any(result.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Fixed source replay contract failed')
 out=R/'runs'/SCORE;out.mkdir(exist_ok=False);source=R/'runs'/N
 for label,entry,args in [('SCORE','score_nonlocal_full_source.py',['--root',str(R),'--source',str(source),'--output',str(out/'source_score.json')]),('AUDIT','audit_nonlocal_full_source.py',['--root',str(R),'--source',str(source),'--score',str(out/'source_score.json'),'--output',str(out/'independent_audit.json')])]:
  status('RUNNING_'+label)
  with (out/(label.lower()+'.log')).open('x') as f:q=subprocess.run([py,str(REL/entry)]+args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(label+' exited '+str(q.returncode))
 audit=json.loads((out/'independent_audit.json').read_text())
 if audit.get('status')!='PASS' or audit.get('records')!=11761 or audit.get('L_s_scored')!=1047 or audit.get('fields_recomputed')!=496:raise ValueError('Independent source audit failed')
 status('FULL_SOURCE_SCORE_AND_AUDIT_COMPLETE',prediction_sha256=audit['prediction_sha256'])
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
