"""Wait for fixed Phase1 ground predictions, independently verify, score, and audit."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from pretruth_audit import check
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',required=True);ap.add_argument('--commit',required=True);ap.add_argument('--entry',type=Path,required=True);ap.add_argument('--pid',type=int,required=True);ap.add_argument('--process-token',required=True);ap.add_argument('--upstream-status',type=Path);a=ap.parse_args();out=a.root/'runs'/(a.run+'-auto-audit')
 if out.exists():raise FileExistsError('Immutable ground audit controller required')
 out.mkdir(parents=True)
 def status(state,**kw):
  q=out/'status.tmp';q.write_text(json.dumps(dict(state=state,**kw),indent=2),encoding='utf8');os.replace(q,out/'status.json')
 try:
  candidate=a.root/'runs'/a.run/'flow/test_prediction';baseline=a.root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';manifest=candidate/'complete.json';summary=candidate.parent/'test_summary.json'
  if summary.exists():raise FileExistsError('Existing ground score requires separate reconciliation')
  while not manifest.exists():
   if a.upstream_status and a.upstream_status.exists():
    state=json.loads(a.upstream_status.read_text())
    if state.get('state')=='FAILED':raise RuntimeError('Prediction controller failed: '+str(state))
   proc=Path('/proc')/str(a.pid);command=(proc/'cmdline').read_bytes().decode(errors='replace') if proc.exists() else ''
   if a.process_token not in command and not manifest.exists():raise RuntimeError('Prediction process stopped without completed manifest')
   status('WAITING_SEVEN_FIXED_VIEWS',prediction_pid=a.pid);time.sleep(60)
  if summary.exists():raise FileExistsError('Ground score appeared before independent pre-truth verification')
  status('CHECKING_ALL_VIEWS_BEFORE_TRUTH');evidence=check(candidate,baseline,168000,a.commit,VIEWS)
  with (out/'pretruth_audit.json').open('x',encoding='utf8') as f:json.dump(evidence,f,indent=2);f.write('\n')
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(a.root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release')+':'+str(a.root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));py='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
  if not a.entry.is_file():raise FileNotFoundError('Fixed scoring interface missing')
  status('SCORING_FIXED_PREDICTIONS');args=[py,'-X','utf8','-u',str(a.entry),'score','--root',str(a.root),'--out',str(a.root/'runs'/a.run),'--commit',a.commit,'--device','cpu']
  with (out/'score.log').open('x') as f:q=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
  if q.returncode!=0:raise RuntimeError('Ground scoring failed with exit '+str(q.returncode))
  score=json.loads(summary.read_text())
  if score.get('status')!='ANALYZED' or score.get('all_predictions_before_truth') is not True or score.get('target_feedback') is not False:raise ValueError('Ground score provenance invalid')
  if check(candidate,baseline,168000,a.commit,VIEWS)['sha256']!=evidence['sha256']:raise ValueError('Prediction files changed during scoring')
  status('INDEPENDENTLY_AUDITING_GROUPS');audit=Path(__file__).with_name('audit_ground_and_subgroups.py')
  with (out/'independent_audit.json').open('x') as f:q=subprocess.run([py,str(audit),a.run],stdout=f,stderr=subprocess.PIPE,stdin=subprocess.DEVNULL,env=env,text=True)
  if q.returncode!=0:raise RuntimeError('Independent ground audit failed: '+q.stderr[-3000:])
  result=json.loads((out/'independent_audit.json').read_text())
  if result.get('status')!='PASS' or result.get('group_checks')!=119:raise ValueError('Independent group audit did not pass')
  status('GROUND_SCORE_AND_INDEPENDENT_AUDIT_COMPLETE',pretruth_manifest_sha256=evidence['sha256']['candidate/physical_ids.npy'],all_six_within_relative_3percent=result['all_six_within_relative_3percent'],all_six_within_3pp=result['all_six_within_3pp'])
 except Exception as exc:
  status('FAILED',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
