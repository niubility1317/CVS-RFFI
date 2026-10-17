"""One owner; full-source fixed rows train then automatically predict and score."""
import json,os,subprocess,sys,time
from . import design as d
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose
from experiments.cvs_rdrop_sinc_units.startup import launch_once

def run():
 spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered matrix differs from actual full-source run')
 if d.BASE.exists():raise FileExistsError('Immutable full-source run exists;reconcile,never duplicate')
 d.BASE.mkdir();(d.BASE/'logs').mkdir()
 from .preflight import smoke
 smoke()
 d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=str(d.ROOT),argv=sys.argv,owner=d.OWNER,source_RX_holdout=None))
 pending=d.train_rows().copy();live={};launched=[];completed=[];failure=None;start=time.perf_counter()
 def record(value):
  i=next((i for i,v in enumerate(launched) if v['row_id']==value['row_id']),None)
  if i is None:launched.append(value.copy())
  else:launched[i]=value.copy()
  d.write(d.BASE/'launch.json',dict(rows=launched))
 while pending or live:
  for pid,(child,row) in list(live.items()):
   code=child.poll()
   if code is None:continue
   del live[pid];path=d.BASE/row['row_id']/'completion.json';v=d.read(path) if path.is_file() else {}
   if code!=0 or v.get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED' or v.get('views')!=list(d.VIEWS) or v.get('count')!=168000 or not v.get('truth_last'):
    failure=dict(row_id=row['row_id'],pid=pid,returncode=code,reason='Owned row technical failure;preserve artifacts/healthy rows;no automatic retry');d.write(d.BASE/'failure.json',failure);pending=[]
   else:completed.append(row['row_id'])
  if pending and failure is None:
   with capacity_lock():
    states=capacity();gpu=choose(states)
    if gpu is not None:
     ready=next((r for r in pending if all((d.parent_source(r['arm'])/name).is_file() for name in ('source_frozen.json','training_result.json'))),None)
     if ready is None:gpu=None
    if gpu is not None:
     row=ready;pending.remove(row);path=d.BASE/'logs'/(row['row_id']+'.stdout.log');command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_mmse_mix30/runner.py'),'--worker',row['row_id']];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
     with path.open('x') as stream:
      create=lambda:subprocess.Popen(command,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
      info=dict(row_id=row['row_id'],cwd=str(d.ROOT),argv=command,gpu=gpu,owner=d.OWNER,log=str(path),per_GPU_limit=3)
      child,receipt=launch_once(create,info,record,timeout=2.);live[child.pid]=(child,row)
  d.write(d.BASE/'queue_state.json',dict(pending=[r['row_id'] for r in pending],live=[r['row_id'] for child,r in live.values()],completed=completed,failure=failure,source_RX_holdout=None,score_feedback_to_other_rows=False))
  if pending or live:time.sleep(3.)
 if failure:raise RuntimeError('Preserved systemic technical failure')
 from .finish import run as finish
 value=finish();value['matrix_wall_seconds']=time.perf_counter()-start;d.write(d.BASE/'result.json',value)
 d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',rows=1,continued_models=1,reused_nominal_controls=1,parent_model_epochs=20,new_model_epochs=10,cumulative_model_epochs=30,new_optimizer_updates=2220,cumulative_epochs_each=30,prediction_count=7*168000,truth_last=True,head_fit_updates=0,joint_goal_complete=False))
 print('FULL_SOURCE_REFERENCE_COMPLETE',flush=True)
