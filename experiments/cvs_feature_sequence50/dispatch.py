import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def replace(mode,row,package='experiments.cvs_feature_sequence50'):os.execv(sys.executable,[sys.executable,'-u','-m',package+'.dispatch',mode,row])
def worker(row,design=d,package="experiments.cvs_feature_sequence50"):
 d=design
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  while True:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    state=capacity()[gpu]
    if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:break
   time.sleep(3)
  from importlib import import_module
  train=import_module(package+".source").train
  source_proof=import_module(package+".provenance").source_proof
  diagnose=import_module(package+".diagnostics").run
  c=d.config(next(r for r in d.rows() if r['row_id']==row))
  if not c['baseline_reuse']:train(c)
  source_proof(c);diagnose(row)
 replace('--predict',row,package)

def eligible(row,design=d,proof=None):
 d=design
 if row['model_seed']==d.SEEDS[0]:return True
 first=[d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]]
 paths=[Path(c['baseline_reference']['output_root'] if c['baseline_reuse'] else c['output_root'])/'completion.json' for c in first]
 if any(not p.is_file() for p in paths):return False
 from .provenance import source_proof
 if proof is not None:source_proof=proof
 evidence={}
 for c in first:
  source=source_proof(c);v=d.read(source/'completion.json');evidence[c['arm']]=v['final_source_metrics']
  if not 0<=v['final_training_L_CE']<=d.SOURCE_GATE['final_L_CE_max']:raise ValueError('Source health falsification failed: '+c['arm'])
 base=evidence['BASE_REF']
 for arm in d.ARMS[1:]:
  v=evidence[arm]
  for key,margin in [('source_val_accuracy',d.SOURCE_GATE['V_mean_drop_max']),('source_val_worst_rx',d.SOURCE_GATE['V_worst_RX_drop_max'])]:
   if v[key]<base[key]-margin:raise ValueError('Source default-head noninferiority failed: '+arm+'/'+key)
 return True

def dispatch(design=d,package="experiments.cvs_feature_sequence50"):
 d=design
 from importlib import import_module
 proof=import_module(package+".provenance").source_proof
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir();spec=d.read(d.ROOT/'registration/experiment.json')
  if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Sequence-feature preregistration differs')
  spec.update(status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip());d.write(d.BASE/'experiment.json',spec);d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
  pending=d.rows();active={};launches=[];completed=[];failures=[];scientific_failure=None
  while pending or active:
   for rowid,child in list(active.items()):
    code=child.poll()
    if code is None:continue
    p=d.BASE/rowid/'completion.json';v=d.read(p) if p.is_file() else {}
    if code or v.get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED' or v.get('views')!=list(d.VIEWS) or v.get('count_per_readout')!=168000 or v.get('readouts')!=['NATIVE_HEAD'] or not v.get('truth_last') or v.get('independent_recount')!='VERIFIED':failures.append(dict(row_id=rowid,exit_code=code,error='Training/default prediction/score did not close'))
    else:completed.append(rowid)
    del active[rowid]
   if pending and not failures and scientific_failure is None:
    try:ready_rows=[r for r in pending if eligible(r,design=d,proof=proof)]
    except ValueError as exc:scientific_failure=str(exc);ready_rows=[]
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
     states=capacity();reserved={r['pid']:r['gpu'] for r in launches if r['row_id'] in active};ready=[]
     for gpu,state in states.items():
      pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
      if len(pids)<3 and state['free_mb']>=12000:ready.append((len(pids),-state['free_mb'],gpu))
     if ready and ready_rows:
      gpu=min(ready)[2];row=ready_rows[0];pending.remove(row);cmd=[sys.executable,'-u','-m',package+'.dispatch','--worker',row['row_id']];log=d.BASE/'logs'/f"{row['row_id']}.log"
      with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[row['row_id']]=child;launches.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=launches))
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,source_screen_not_passed=scientific_failure,max_total_processes_per_GPU=3,updated_at=time.time()))
   if (failures or scientific_failure is not None) and not active:break
   if pending or active:time.sleep(3)
  if failures:
   d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,no_automatic_retry=True));raise RuntimeError('Sequence technical failure; preserve all rows')
  if scientific_failure is not None:
   d.write(d.BASE/'completion.json',dict(status='SOURCE_SCREEN_NOT_PASSED',source_reason=scientific_failure,completed_rows=completed,pending_rows=pending,technical_failure=False,no_target_feedback=True,goal_84=None));return
  if len(completed)!=len(d.rows()):raise ValueError('Incomplete sequence comparison')
  finish=import_module(package+'.score').finish
  result=finish();d.write(d.BASE/'completion.json',dict(status='EXPERIMENT_COMPLETE',new_backbones=result['new_backbones'],existing_backbone_references=result['existing_backbone_references'],rows=result['rows'],readouts=result['readouts'],views=1,prediction_count=result['prediction_count'],default_classifier_only=True,head_fit_updates=0,truth_last=True,goal_84=result['goal_84']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker)
 elif a.predict:
  from .predict import predict
  predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';replace('--score',a.predict)
 elif a.score:
  from .score import score
  score(a.score)
 else:dispatch()
