import argparse,os,subprocess,sys,time,traceback,statistics
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def execute(mode,row):
 env=dict(os.environ)
 if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
 os.execve(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_t1_e50_confirm/runner.py'),'--mode',mode,'--row',row],env)

def reused_complete():
 row=d.rows()[0];root=d.row_root(row['row_id']);c=d.config(row)
 done=d.read(root/'complete.json');trained=d.read(root/'source/completion.json')
 pred=d.read(root/'prediction/complete.json');scored=d.read(root/'scoring/complete.json')
 if done['status']!='TRAINED_AND_SCORED' or trained['config']!=c or trained['epoch']!=20 or trained['optimizer_steps']!=4440 or trained['target_access']:
  raise ValueError('Completed seed01 exact source artifact differs')
 if pred['status']!='PREDICTIONS_COMPLETE' or pred['config']!=c or pred['views']!=list(d.VIEWS) or pred['count']!=168000 or pred['truth_read'] or pred['query_fit'] or scored['status']!='SCORED_COMPLETE' or scored['views']!=list(d.VIEWS) or not scored['truth_last']:
  raise ValueError('Seed01 seven-view prediction/scoring reuse incomplete')
 return dict(row_id=row['row_id'],root=str(root),training_repeated=False,prediction_repeated=False,scoring_repeated=False)

def worker(row):
 r=next(r for r in d.rows() if r['row_id']==row)
 if d.is_reused(r):raise ValueError('Never train completed seed01 again')
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
  if gpu not in d.GPU_IDS:raise ValueError('GPU outside registered pool')
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   if len(capacity()[gpu]['pids']|{os.getpid()})>d.MAX_TOTAL_GPU_PROCESSES:raise ValueError('All-owner GPU process cap exceeded')
  from .observer import train
  train(d.config(r))
 execute('predict',row)

def collect():
 reused_complete();allrows=[];cells={};paired={}
 for row in d.rows():
  root=d.row_root(row['row_id']);done=d.read(root/'complete.json');pred=d.read(root/'prediction/complete.json');scored=d.read(root/'scoring/complete.json')
  if done['status']!='TRAINED_AND_SCORED' or pred['views']!=list(d.VIEWS) or scored['views']!=list(d.VIEWS) or not scored['truth_last']:raise ValueError('All three rows require complete seven-view scoring')
  records=d.read(root/'scoring/scores.json')['results']
  # Parent metrics are read only after the full new matrix is trained and scored.
  parent={ (x['view'],x['dimension'],x['stratum']):x for x in d.read(d.parent_source(row['model_seed']).parent/'scoring/scores.json')['results'] }
  if len(records)!=126 or len(parent)!=126:raise ValueError('Missing registered metric strata')
  allrows.append(dict(row=row,root=str(root),reused=d.is_reused(row),source=d.read(root/'source/result.json'),scores=records))
  for r in records:
   key=(r['view'],r['dimension'],r['stratum']);cells.setdefault(key,[]).append(dict(seed=row['model_seed'],accuracy=r['accuracy'],macro_f1=r['macro_f1'],query_count=r['query_count']))
   p=parent[key];paired.setdefault(key,[]).append(dict(seed=row['model_seed'],accuracy_delta_pp=100*(r['accuracy']-p['accuracy']),macro_f1_delta_pp=100*(r['macro_f1']-p['macro_f1'])))
 metrics=[]
 for key,values in cells.items():
  if len(values)!=3 or len({v['query_count'] for v in values})!=1:raise ValueError('Three comparable seeds required')
  metrics.append(dict(view=key[0],dimension=key[1],stratum=key[2],model_seed_count=3,query_count_per_seed=values[0]['query_count'],
   accuracy_mean=statistics.mean(x['accuracy'] for x in values),accuracy_std=statistics.stdev(x['accuracy'] for x in values),
   macro_f1_mean=statistics.mean(x['macro_f1'] for x in values),macro_f1_std=statistics.stdev(x['macro_f1'] for x in values),
   values=values,paired_E30_deltas=paired[key]))
 d.write(d.BASE/'results.json',dict(rows=allrows,metrics=metrics,std_ddof=1,model_seed_count=3,target_feedback_forbidden=True,
   benchmark_context='User-selected exposed-benchmark seed extension;not a new blind test'))
 d.write(d.BASE/'complete.json',dict(status='TRAINED_AND_SCORED',model_seed_count=3,rows=[r['row_id'] for r in d.rows()],views=list(d.VIEWS),
   newly_trained_epochs=40,newly_trained_updates=8880,seed01_reused=True,cumulative_epochs_per_model=50))

def dispatch():
 if d.BASE.exists():raise FileExistsError('Reconcile existing immutable run before any retry')
 d.BASE.mkdir();(d.BASE/'logs').mkdir()
 d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  from .preflight import smoke
  smoke();d.write(d.BASE/'reused_seed01.json',reused_complete())
  pending=[r for r in d.rows() if not d.is_reused(r)];active={};launched=[];completed=[d.rows()[0]['row_id']]
  while pending or active:
   for row,info in list(active.items()):
    code=info['child'].poll()
    if code is None:continue
    done=d.row_root(row)/'complete.json'
    if code or not done.is_file() or d.read(done)['status']!='TRAINED_AND_SCORED':raise RuntimeError('Own technical failure '+row+';preserve healthy workers and outputs;no retry')
    completed.append(row);del active[row]
   if pending and len(active)<2:
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
     states=capacity();used={x['gpu'] for x in active.values()}
     slots=sorted((len(states[g]['pids']),-states[g]['free_mb'],g) for g in d.GPU_IDS if g not in used and len(states[g]['pids'])<d.MAX_TOTAL_GPU_PROCESSES and states[g]['free_mb']>=12000)
     while slots and pending and len(active)<2:
      gpu=slots.pop(0)[2];row=pending.pop(0);rid=row['row_id'];cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_t1_e50_confirm/runner.py'),'--mode','worker','--row',rid]
      log=d.BASE/'logs'/(rid+'.log')
      with log.open('x',encoding='utf-8') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[rid]=dict(child=child,gpu=gpu);launched.append(dict(row_id=rid,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
      d.write(d.BASE/'launch.json',dict(rows=launched))
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,updated_at=time.time(),maximum_own_workers=2,views=list(d.VIEWS)))
   if pending or active:time.sleep(5)
  collect()
 except BaseException as error:
  d.write(d.BASE/'failure.json',dict(status='SYSTEMIC_TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc(),preserve_outputs=True,no_automatic_retry=True));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
 allowed={r['row_id'] for r in d.rows() if not d.is_reused(r)}
 if a.mode and a.row not in allowed:raise ValueError('Only seed02/03 may execute training/prediction/scoring')
 if a.mode=='worker':worker(a.row)
 elif a.mode=='predict':
  from .evaluate import predict
  predict(a.row);execute('score',a.row)
 elif a.mode=='score':
  from .evaluate import score
  with lock(d.BASE/'CPU_scorer.lock'):score(a.row)
  d.write(d.row_root(a.row)/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=a.row))
 else:dispatch()
