import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
def replace(mode,row):os.execv(sys.executable,[sys.executable,'-u','-m','experiments.cvs_feature_restitution.dispatch',mode,row])
def worker(row):
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  while True:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    state=capacity()[gpu]
    if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:break
   time.sleep(3)
  from .source import train
  from .provenance import source_proof
  from .probe import run as source_probes
  c=d.config(next(r for r in d.rows() if r['row_id']==row))
  if not c['baseline_reuse']:train(c)
  source_proof(c);source_probes(row)
 replace('--predict',row)

def eligible(row):
 if row['model_seed']==d.SEEDS[0]:return True
 first=[d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]]
 if any(not (d.BASE/c['row_id']/'probe/summary.json').is_file() for c in first):return False
 from .provenance import source_proof
 from .probe import proof
 evidence={}
 for c in first:
  source=source_proof(c);_,evidence[c['arm']]=proof(c)
  if not 0<=d.read(source/'completion.json')['final_training_L_CE']<=d.SOURCE_GATE['final_L_CE_max']:raise ValueError('Source health falsification failed: '+c['arm'])
 base=evidence['BASE_REF']
 for arm in d.ARMS[1:]:
  v=evidence[arm]
  for key,margin in [('leave_RX_mean',.01),('leave_RX_worst',.02)]:
   if v[key]<base[key]-margin:raise ValueError('Source frozen leave-RX falsification failed: '+arm+'/'+key)
  if v['full_TX']['accuracy']<base['full_TX']['accuracy']-.01:raise ValueError('Source frozen TX falsification failed: '+arm)
 return True

def dispatch():
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir();spec=d.read(d.ROOT/'registration/experiment.json')
  if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Restitution preregistration differs')
  spec.update(status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip());d.write(d.BASE/'experiment.json',spec);d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
  pending=d.rows();active={};receipts=[];completed=[];failures=[]
  while pending or active:
   for rowid,child in list(active.items()):
    code=child.poll()
    if code is None:continue
    p=d.BASE/rowid/'completion.json';v=d.read(p) if p.is_file() else {}
    if code or v.get('status')!='TRAINED_PROBED_PREDICTED_SCORED' or v.get('views')!=list(d.VIEWS) or v.get('count_per_readout')!=168000 or v.get('readouts')!=list(d.READOUTS) or not v.get('truth_last') or v.get('independent_recount')!='VERIFIED':failures.append(dict(row_id=rowid,exit_code=code,error='Training/prediction/score did not close'))
    else:completed.append(rowid)
    del active[rowid]
   if pending and not failures:
    try:ready_rows=[r for r in pending if eligible(r)]
    except ValueError as exc:failures.append(dict(error=str(exc),kind='SOURCE_FALSIFICATION'));ready_rows=[]
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
     states=capacity();reserved={r['pid']:r['gpu'] for r in receipts if r['row_id'] in active};ready=[]
     for gpu,state in states.items():
      pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
      if len(pids)<3 and state['free_mb']>=12000:ready.append((len(pids),-state['free_mb'],gpu))
     if ready and ready_rows:
      gpu=min(ready)[2];row=ready_rows[0];pending.remove(row);cmd=[sys.executable,'-u','-m','experiments.cvs_feature_restitution.dispatch','--worker',row['row_id']];log=d.BASE/'logs'/f"{row['row_id']}.log"
      with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=receipts))
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,max_total_processes_per_GPU=3,updated_at=time.time()))
   if failures and not active:break
   if pending or active:time.sleep(3)
  if failures:
   d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,no_automatic_retry=True));raise RuntimeError('Restitution technical failure; preserve all rows')
  if len(completed)!=len(d.rows()):raise ValueError('Incomplete restitution comparison')
  from .score import finish
  result=finish();d.write(d.BASE/'completion.json',dict(status='EXPERIMENT_COMPLETE',new_backbones=12,existing_backbone_references=3,rows=15,readouts=30,views=1,prediction_count=5040000,truth_last=True,goal_84=result['goal_84']))
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
