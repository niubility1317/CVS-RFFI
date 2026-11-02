"""Source posterior sampling, fixed selection and truth-last audit."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);C=ap.parse_args().commit
R=Path('/home/szu2070436088/2510044040/CV-SincNet');N='20261102-phase1-posterior-sampling-source-s2026092701-r03';SMOKE='20261102-phase1-posterior-sampling-smoke-s2026092701-r03';SCORE='20261102-phase1-posterior-sampling-score-s2026092701-r03';P=R/'runs'/(N+'-controller');REL=R/'releases'/(N+'-release')/'experiments/phase1_posterior_sampling'
def status(state,**kw):
 q=P/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,P/'status.json')
def wait_capacity():
 while os.getloadavg()[0]>os.cpu_count()-24:status('WAITING_CPU_CAPACITY',load=os.getloadavg()[0]);time.sleep(60)
try:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(R/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(R/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
 for mode,name in [('SMOKE',SMOKE),('PREDICT',N)]:
  wait_capacity();status('RUNNING_'+mode);args=[py,'-X','utf8','-u',str(REL/'sample_source_pilot.py'),'--root',str(R),'--out',str(R/'runs'/name),'--commit',C]+(['--smoke'] if mode=='SMOKE' else [])
  with (P/(mode.lower()+'.log')).open('x') as f:q=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(mode+' exited '+str(q.returncode))
  result=json.loads((R/'runs'/name/('result.json' if mode=='SMOKE' else 'prediction_complete.json')).read_text())
  if mode=='SMOKE':
   if result.get('status')!='PASS' or result.get('records')!=28 or result.get('posterior_mean_and_weights_match') is not True or result.get('selected_waveform_predictions_match') is not True:raise ValueError('Sampling source smoke failed')
  elif result.get('status')!='SOURCE_PREDICTIONS_FIXED' or result.get('commit')!=C or result.get('count')!=672 or any(result.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Fixed source sampling output required')
 out=R/'runs'/SCORE;out.mkdir(exist_ok=False);source=R/'runs'/N
 for label,entry,args in [('SCORE','score_sample_source.py',['--root',str(R),'--source',str(source),'--out',str(out/'source_score.json')]),('AUDIT','audit_sample_source.py',['--root',str(R),'--source',str(source),'--score',str(out/'source_score.json'),'--out',str(out/'independent_audit.json')])]:
  status('RUNNING_'+label)
  with (out/(label.lower()+'.log')).open('x') as f:q=subprocess.run([py,str(REL/entry)]+args+['--commit',C],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  if q.returncode:raise RuntimeError(label+' exited '+str(q.returncode))
 audit=json.loads((out/'independent_audit.json').read_text())
 if audit.get('status')!='PASS' or audit.get('records')!=672 or audit.get('selected_iq_predictions_replayed')!=1344 or audit.get('fields_recomputed')!=544:raise ValueError('Independent source sampling audit failed')
 status('SOURCE_SAMPLE_SCORE_AND_AUDIT_COMPLETE',prediction_sha256=audit['prediction_sha256'])
except Exception as e:
 status('FAILED',error=str(e),traceback=traceback.format_exc());raise
