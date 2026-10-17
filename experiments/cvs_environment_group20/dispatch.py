"""One owner;matching realCP smoke then four fixed sourceRX6 E20 rows."""
import os,subprocess,sys,time
from . import design as d
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose
from experiments.cvs_rdrop_sinc_units.startup import launch_once

def run():
 if d.BASE.exists():raise FileExistsError('Reconcile immutable own run,never duplicate')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();started=time.time()
 from .preflight import smoke
 smoke()
 d.write(d.BASE/'request.json',dict(rows=[d.config(r) for r in d.rows()],source_only=True,full_practical_no_confidence_filter=True))
 pending=d.rows().copy();live={};receipts=[];finished=[];failure=None
 def record(r):
  old=next((i for i,x in enumerate(receipts) if x['row_id']==r['row_id']),None)
  if old is None:receipts.append(r.copy())
  else:receipts[old]=r.copy()
  d.write(d.BASE/'launch.json',dict(rows=receipts))
 while pending or live:
  for pid,(child,row) in list(live.items()):
   code=child.poll()
   if code is None:continue
   del live[pid]
   if code or not (d.BASE/row['row_id']/'source/satmix_result.json').is_file():failure=dict(stage='training',row_id=row['row_id'],pid=pid,returncode=code,no_automatic_retry=True);d.write(d.BASE/'failure.json',failure);pending=[]
   else:finished.append(row['row_id'])
  if pending and not failure:
   with capacity_lock():
    caps=capacity();gpu=choose(caps)
    if gpu is not None:
     row=pending.pop(0);path=d.BASE/'logs'/(row['row_id']+'.stdout.log');cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_environment_group20/runner.py'),'--worker',row['row_id']];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
     with path.open('x') as stream:
      create=lambda:subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);child,receipt=launch_once(create,dict(row_id=row['row_id'],cwd=str(d.ROOT),argv=cmd,gpu=gpu,owner=d.OWNER,log=str(path),capacity_before=dict(process_count=len(caps[gpu]['pids']),free_MiB=caps[gpu]['free_MiB']),per_GPU_limit=3),record,timeout=2.);live[child.pid]=(child,row)
  d.write(d.BASE/'queue_state.json',dict(stage='training',pending=[r['row_id'] for r in pending],live=[r['row_id'] for _,r in live.values()],completed=finished,failure=failure,source_only=True))
  if live or pending:time.sleep(3.)
 if failure:raise RuntimeError('Preserved source training failure,nohealthyworkerstop')
 results={r['arm']:d.read(d.BASE/r['row_id']/'source/satmix_result.json') for r in d.rows()};q=d.qualify(results);d.write(d.BASE/'result.json',dict(status='ANALYZED_SOURCE_ONLY',results=results,qualification=q,target_access=False));d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',model_epochs=80,updates=17760,qualified=q['qualified'],source_only=True,target_access=False,wall_seconds=time.time()-started));print('TRAIN COMPLETE '+str(q['qualified']),flush=True)
