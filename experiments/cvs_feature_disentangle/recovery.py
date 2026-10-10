"""New output root: recover completed diagnostics, retain live original training."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from . import evaluate as e
from . import dispatch as original
from .capacity4 import patched_worker,expected_status
from experiments.cvs_receiver_residual_capacity4.control import process,lock

OLD_RUN='20261010-phase1-feature-disentangle-manysig-m48-r01'
RUN='20261010-phase1-feature-disentangle-manysig-m48-r02'
RELEASE='cvs_feature_disentangle_20261010_r02'
OLD_BASE=Path(d.PROJECT)/'runs'/OLD_RUN
OLD_COMMIT='d4723b1aa227fce6368fd25680bb74b1190557d7'
ERROR="ValueError('Expected exactly one downstream time t_proj for dependency probe')"
MANIFEST=d.ROOT/'experiments/cvs_feature_disentangle/recovery_manifest.json'
ADOPT={}

def configure():
 global ADOPT
 if ADOPT:return
 manifest=d.read(MANIFEST);ADOPT={r['row_id']:r for r in manifest['adopt']}
 if len(ADOPT)!=26:raise ValueError('Recovery adoption matrix differs')
 config=d.config
 d.RUN=RUN;d.BASE=Path(d.PROJECT)/'runs'/RUN;d.RELEASE=RELEASE
 d.OWNER='codex/root/feature-diagnostic-recovery-20261010'
 def recovery_config(row,selected_ratio=None):
  c=config(row,selected_ratio)
  if row['row_id'] in ADOPT:
   c['checkpoint_origin']=dict(config=ADOPT[row['row_id']]['config'],commit=OLD_COMMIT,mode='post_training_diagnostic_recovery')
  return c
 d.config=recovery_config
 e.RUN=RUN;e.RELEASE=RELEASE;e.BASE=d.BASE
 provenance=e.checkpoint_provenance
 def checkpoint_provenance(design,c,ck):
  origin=c.get('checkpoint_origin')
  if origin:
   if origin!=recovery_config(next(r for r in d.rows() if r['row_id']==c['row_id']))['checkpoint_origin']:
    raise ValueError('Changed checkpoint origin')
   return provenance(design,origin['config'],ck)
  return provenance(design,c,ck)
 e.checkpoint_provenance=checkpoint_provenance
 original.test_row=test_row

def old_ready(rid):
 if rid not in ADOPT:return True
 receipt=ADOPT[rid]['launch'];actual=process(receipt['pid'])
 if actual:
  if any(actual[k]!=receipt[k] for k in ('cwd','argv')) or ('start_ticks' in receipt and actual['start_ticks']!=receipt['start_ticks']):
   raise ValueError('Original training PID identity changed '+rid)
  return False
 failure=d.read(OLD_BASE/rid/'failure.json')
 if failure.get('error')!=ERROR:raise ValueError('Different original failure; no automatic recovery '+rid)
 if (OLD_BASE/rid/'prediction').exists():raise ValueError('Unexpected target contact '+rid)
 return True

def test_row(rid):
 e.EVAL_ROW=rid;e.snapshot(d);e.predict(rid)
 original.event('PREDICTIONS_COMPLETE',row_id=rid)
 subprocess.run([sys.executable,'-m','experiments.cvs_feature_disentangle.recovery','--mode','score','--row',rid],
  cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 scored=d.read(d.BASE/'row_scoring'/rid/'scoring_complete.json')
 if (scored['models'],scored['views'],scored['metric_records'],scored['day_metric_records'])!=(1,7,98,28):raise ValueError('Incomplete scoring')
 d.write(d.BASE/rid/'completion.json',dict(status='ANALYZED',views=7,independent_recount='VERIFIED'))
 original.event('ANALYZED',row_id=rid)

def worker(kind,rid):
 if kind=='train' and not old_ready(rid):raise ValueError('Original worker still running')
 from . import source
 train=source.train
 def execute(c):
  if rid in ADOPT:
   from .recover_source import recover
   recover(c,ADOPT[rid]['config'])
  else:train(c)
 source.train=execute
 patched_worker(original);original.worker(kind,rid)

def phase(kind,items,receipts):
 pending=list(items);active={};done=[];failures=[]
 while pending or active:
  for rid,(child,receipt) in list(active.items()):
   rc=child.poll()
   if rc is None:continue
   path=d.BASE/rid/'completion.json'
   if rc or not path.is_file() or d.read(path)['status']!=expected_status(d,rid,kind):failures.append(dict(row_id=rid,exit_code=rc))
   else:done.append(rid)
   del active[rid]
  while pending and not failures:
   eligible=next((rid for rid in pending if kind=='test' or old_ready(rid)),None)
   if eligible is None:break
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    choices=[(len(x['pids']),-x['free_mb'],gpu) for gpu,x in original.capacity().items() if len(x['pids'])<4 and x['free_mb']>=12000]
    if not choices:break
    gpu=min(choices)[2];rid=eligible;pending.remove(rid)
    cmd=[sys.executable,'-u','-m','experiments.cvs_feature_disentangle.recovery','--mode','worker','--kind',kind,'--row',rid]
    log=d.BASE/'logs'/(rid+'-'+kind+'.log')
    with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    actual=process(child.pid)
    if not actual:raise ValueError('New recovery worker missing')
    receipt=dict(row_id=rid,kind=kind,pid=child.pid,start_ticks=actual['start_ticks'],cwd=str(d.ROOT),argv=cmd,gpu=gpu,log=str(log),mode='recover_diagnostics' if rid in ADOPT else 'scratch_train')
    receipts.append(receipt);d.write(d.BASE/'launch.json',dict(rows=receipts));active[rid]=(child,receipt)
   original.event('RUNNING',**receipt)
  d.write(d.BASE/'queue_state.json',dict(phase=kind,active=list(active),pending=pending,completed=done,failures=failures,
   waiting_original=[rid for rid in pending if rid in ADOPT and process(ADOPT[rid]['launch']['pid'])],per_gpu_limit=4,
   trained_identity_rows=48,reused_training_rows=26,new_training_rows=22,total_test_rows=36,updated_at=time.time()))
  if failures and not active:raise RuntimeError('Failed recovery retained; no retry '+repr(failures))
  if pending or active:time.sleep(10)

def dispatch():
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
  for row in d.rows():d.write(d.BASE/'configs'/(row['row_id']+'.json'),d.config(row))
  d.write(d.BASE/'experiment.json',d.read(d.ROOT/'experiments/cvs_feature_disentangle/recovery_experiment.json'))
  d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=str(d.ROOT),owner=d.OWNER))
  receipts=[]
  try:
   phase('train',[r['row_id'] for r in d.rows() if r['arm'] in d.FIXED_ARMS],receipts)
   from .validation import source_select
   selection=source_select();allowed=d.test_rows(selection)
   spec=d.read(d.BASE/'experiment.json')
   for row in spec['rows']:
    r=next(r for r in d.rows() if r['row_id']==row['row_id'])
    if r['arm'] in d.SELECTED_ARMS:
     row['config']=d.resolved_config(r);d.write(d.BASE/'configs'/(r['row_id']+'.json'),row['config'])
    if r['arm'] in d.SEARCH_ARMS and r['row_id'] not in {q['row_id'] for q in allowed}:
     row['status']='SOURCE_ONLY_NOT_SELECTED';d.write(d.BASE/r['row_id']/'completion.json',dict(status='SOURCE_ONLY_NOT_SELECTED',target_access=False))
   spec['source_selection']=selection;d.write(d.BASE/'experiment.json',spec)
   original.event('SOURCE_SELECTION_FROZEN',selected_ratio=selection['selected_ratio'],target_scores_consumed=False)
   phase('test',[r['row_id'] for r in allowed if r['arm'] in d.SEARCH_ARMS],receipts)
   phase('train',[r['row_id'] for r in d.rows() if r['arm'] in d.SELECTED_ARMS],receipts)
   subprocess.run([sys.executable,'-m','experiments.cvs_feature_disentangle.recovery','--mode','score'],cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
   result=d.read(d.BASE/'scoring_complete.json')
   if (result['models'],result['views'],result['metric_records'],result['day_metric_records'])!=(36,7,3528,1008):raise ValueError('Incomplete matrix')
   d.write(d.BASE/'completion.json',dict(status='ANALYZED',identity_models=48,reused_training=26,new_training=22,tested_models=36,source_only_models=12,views=7))
   spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED';d.write(d.BASE/'experiment.json',spec);original.event('ANALYZED')
  except Exception as exc:
   d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc)));original.event('FAILED',error=repr(exc));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['dispatch','worker','score'],required=True);p.add_argument('--row');p.add_argument('--kind',choices=['train','test']);a=p.parse_args()
 configure()
 if a.mode=='dispatch':dispatch()
 elif a.mode=='worker':worker(a.kind,a.row)
 else:e.EVAL_ROW=a.row;e.score()
