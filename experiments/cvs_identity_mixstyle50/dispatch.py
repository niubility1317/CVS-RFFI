import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as default_design
d=default_design
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
def replace(mode,row,module='experiments.cvs_identity_mixstyle50.dispatch'):os.execv(sys.executable,[sys.executable,'-u','-m',module,mode,row])
def worker(row,*,design=None,train_fn=None,proof_fn=None,module="experiments.cvs_identity_mixstyle50.dispatch"):
 d=design or default_design
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  while True:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    state=capacity()[gpu]
    if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:break
   time.sleep(3)
  from .source import train
  from .predict import source_proof
  c=d.config(next(r for r in d.rows() if r['row_id']==row))
  if c.get('source_config') is None:(train_fn or train)(c)
  (proof_fn or source_proof)(c)
 replace('--predict',row,module)
def dispatch(*,design=None,eligible_fn=None,finish_fn=None,module="experiments.cvs_identity_mixstyle50.dispatch"):
 d=design or default_design
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir();spec=d.read(d.ROOT/'registration/experiment.json')
  if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('PA/default-head preregistration differs')
  spec.update(status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip());spec['code']['commit']=spec['execution_commit'];d.write(d.BASE/'experiment.json',spec);d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
  pending=d.rows();active={};receipts=[];completed=[];failures=[]
  while pending or active:
   for rowid,child in list(active.items()):
    code=child.poll()
    if code is None:continue
    p=d.BASE/rowid/'completion.json';v=d.read(p) if p.is_file() else {}
    if code or v.get('status')!='TRAINED_PREDICTED_SCORED' or v.get('readouts')!=list(d.READOUTS) or v.get('count_per_readout')!=d.QUERY_COUNT or not v.get('truth_last') or v.get('independent_recount')!='VERIFIED':failures.append(dict(row_id=rowid,exit_code=code,error='Training/prediction/score did not close'))
    else:completed.append(rowid)
    del active[rowid]
   if pending and not failures:
    try:ready_rows=[r for r in pending if eligible_fn is None or eligible_fn(r)]
    except ValueError as exc:failures.append(dict(error=str(exc),kind='SOURCE_FALSIFICATION'));ready_rows=[]
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
     states=capacity();reserved={r['pid']:r['gpu'] for r in receipts if r['row_id'] in active};ready=[]
     for gpu,state in states.items():
      pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
      if len(pids)<3 and state['free_mb']>=12000:ready.append((len(pids),-state['free_mb'],gpu))
     if ready and ready_rows:
      gpu=min(ready)[2];row=ready_rows[0];pending.remove(row);cmd=[sys.executable,'-u','-m',module,'--worker',row['row_id']];log=d.BASE/'logs'/f"{row['row_id']}.log"
      with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=receipts))
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,max_total_processes_per_GPU=3,updated_at=time.time()))
   if failures and not active:break
   if pending or active:time.sleep(3)
  if failures:
   d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,no_automatic_retry=True));raise RuntimeError('PA/default-head technical failure; preserve all rows')
  if len(completed)!=len(d.rows()):raise ValueError('Incomplete PA/default-head comparison')
  from .score import finish
  result=(finish_fn or finish)();d.write(d.BASE/'completion.json',dict(status='EXPERIMENT_COMPLETE',models=len(d.rows()),new_trained_models=len(d.training_rows()),reused_baseline_models=len(d.rows())-len(d.training_rows()),views=len(d.VIEWS),readouts=list(d.READOUTS),prediction_count=len(d.rows())*len(d.READOUTS)*d.QUERY_COUNT,truth_last=True,goal_84=result['goal_84']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker)
 elif a.predict:
  from .predict import predict
  predict(a.predict);replace('--score',a.predict)
 elif a.score:
  from .score import score
  score(a.score)
 else:dispatch()
