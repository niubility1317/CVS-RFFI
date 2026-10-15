"""Only finish missing CPU scores of original, already-started models."""
import os,sys,subprocess,time
from pathlib import Path
from . import design as d
from experiments.cvs_rdrop_fold20.resources import process
def command(rid,mode):
 if rid not in d.OLD_SCORE_ROWS:raise ValueError('Not an allowed original closure row')
 release=Path(d.ORIGINAL_RELEASE)
 return [sys.executable,'-u',str(release/'experiments/cvs_sigtime_rdrop_legacy/runner.py'),str(release/'experiments/cvs_sigtime_rdrop_legacy/configs/matrix.json'),mode,rid]
def find(cmd):
 values=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  value=process(int(p.name))
  if value and value['argv']==cmd:values.append(value)
 if len(values)>1:raise RuntimeError('Duplicate original row process; preserve and reconcile')
 return values[0] if values else None
def poll(children,receipts,cpu):
 complete=[]
 for rid in d.OLD_SCORE_ROWS:
  root=d.ORIGINAL_BASE/rid;done=root/'completion.json'
  if done.is_file():
   value=d.read(done)
   if value['status']!='MODEL_PREDICTED_SCORED' or value['count']!=168000 or value['metric_records']!=18 or not value['truth_last'] or value['head_fit_updates']!=0:raise ValueError('Original row closure is incomplete')
   complete.append(rid);continue
  if rid in children:
   if children[rid].poll() is not None:raise RuntimeError('Original CPU score exited without completion '+rid)
   continue
  cmd=command(rid,'--score');existing=find(cmd)
  if existing:
   if existing['gpu']!='':raise ValueError('Original independent scorer must be CPU-only')
   continue
  pred=root/'prediction/complete.json'
  if not pred.is_file():
   if find(command(rid,'--worker')) is None:raise RuntimeError('Original worker absent without fixed prediction '+rid)
   continue
  p=d.read(pred)
  if p['status']!='PREDICTIONS_COMPLETE' or p['count']!=168000 or p['truth_read'] is not False:raise ValueError('Original full frozen predictions required before truth')
  log=d.ORIGINAL_BASE/'logs'/(rid+'-score.log')
  if (root/'scores').exists() or log.exists():raise FileExistsError('Preserve existing partial original score '+rid)
  with log.open('x') as stream:child=subprocess.Popen(cmd,cwd=d.ORIGINAL_RELEASE,env=cpu,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  children[rid]=child;receipts.append(dict(row_id=rid,pid=child.pid,argv=cmd,cwd=d.ORIGINAL_RELEASE,CUDA_VISIBLE_DEVICES='',owner=d.OWNER,read_at=time.time(),no_model_restart=True));d.write(d.BASE/'old_score_launch.json',dict(rows=receipts))
 return complete
