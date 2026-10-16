"""One owner: full-source F train -> clean predictions -> independent CPU score."""
import os,sys,time,subprocess,traceback
from . import design as d
from experiments.cvs_rdrop_sinc_units.startup import launch_once
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose,serial
def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU dispatcher required')
 if d.BASE.exists():raise FileExistsError('Immutable F root exists: reconcile, never repeat')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();pending=d.training_rows().copy();active={};scoring={};receipts=[];scores=[];complete=[];failures=[]
 cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def command(mode,rid=None):return [sys.executable,'-u',str(d.ROOT/'experiments/cvs_f_band50/runner.py'),mode]+([] if rid is None else [rid])
 try:
  subprocess.run(command('--preflight'),cwd=d.ROOT,env=cpu,check=True)
  d.write(d.BASE/'frozen_matrix.json',dict(rows=[d.config(r) for r in d.rows()],all_three_seeds_frozen=True,no_source_performance_gate=True,target_feedback=False))
  for row in d.rows():
   if row['arm']!='FULL':continue
   with (d.BASE/'logs'/(row['row_id']+'-reference.log')).open('x') as f:subprocess.run(command('--reference',row['row_id']),cwd=d.ROOT,env=cpu,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
   complete.append(row['row_id'])
  while pending or active or scoring:
   for rid,child in list(active.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'prediction/complete.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='PREDICTIONS_COMPLETE':failures.append(dict(row_id=rid,phase='train_predict',returncode=rc))
    else:
     cmd=command('--score',rid);log=d.BASE/'logs'/(rid+'-score.log')
     with log.open('x') as f:p=subprocess.Popen(cmd,cwd=d.ROOT,env=cpu,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     scoring[rid]=p;scores.append(dict(row_id=rid,pid=p.pid,cwd=str(d.ROOT),argv=cmd,gpu='',log=str(log)));d.write(d.BASE/'scorer_launch.json',dict(rows=scores))
    del active[rid]
   for rid,child in list(scoring.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'completion.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='MODEL_PREDICTED_SCORED':failures.append(dict(row_id=rid,phase='CPU_score',returncode=rc))
    else:complete.append(rid)
    del scoring[rid]
   while pending and not failures:
    with capacity_lock():
     caps=capacity();gpu=choose(caps)
     if gpu is None:break
     row=pending.pop(0);rid=row['row_id'];cmd=command('--worker',rid);log=d.BASE/'logs'/(rid+'-train_predict.log')
     def create():
      with log.open('x') as f:p=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(cpu,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[rid]=p;return p
     def record(receipt):
      old=next((v for v in receipts if v['row_id']==rid),None)
      if old is None:receipts.append(receipt.copy())
      else:old.update(receipt)
      d.write(d.BASE/'launch.json',dict(rows=receipts))
     launch_once(create,dict(row_id=rid,owner=d.OWNER,epochs=50,cwd=str(d.ROOT),argv=cmd,gpu=gpu,log=str(log),capacity_before=serial(caps),per_GPU_limit=3),record,timeout=2.)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),CPU_scoring=list(scoring),pending=[r['row_id'] for r in pending],complete=complete,failures=failures,updated_at=time.time(),registered_rows=9,all_three_seeds_frozen=True))
   if failures and not active and not scoring:raise RuntimeError('Technical failure preserved; healthy children not stopped; no retry')
   if pending or active or scoring:time.sleep(5.)
  subprocess.run(command('--finish'),cwd=d.ROOT,env=cpu,check=True)
 except Exception as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),outputs_preserved=True,healthy_children_not_stopped=True));raise
