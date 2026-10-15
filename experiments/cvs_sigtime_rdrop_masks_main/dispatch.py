import os,sys,time,subprocess,traceback
from pathlib import Path
from . import design as d
from .history import preflight_all
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose,serial
from experiments.cvs_sigtime_rdrop_recovery.startup import launch_once
def run(request):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only dispatcher required')
 if d.BASE.exists():raise FileExistsError('Immutable main root exists; reconcile')
 history=preflight_all();d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'historical_reuse.json',history)
 pending=list(d.rows());active={};scoring={};receipts=[];scorers=[];complete=[];failures=[];started=time.time()
 cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def command(mode,rid):return [sys.executable,'-u',str(d.ROOT/'experiments/cvs_sigtime_rdrop_masks_main/runner.py'),str(request),mode,rid]
 def launch(cmd,env,base,items,name):
  def record(value):
   old=next((r for r in items if r['row_id']==value['row_id']),None)
   if old is None:items.append(dict(value))
   else:old.update(value)
   d.write(d.BASE/name,dict(rows=items))
  with Path(base['log']).open('x') as stream:return launch_once(lambda:subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True),base,record)[0]
 try:
  while pending or active or scoring:
   for rid,child in list(active.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'prediction/complete.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='PREDICTIONS_COMPLETE':failures.append(dict(row_id=rid,phase='train_predict',returncode=rc))
    else:
     cmd=command('--score',rid);base=dict(row_id=rid,owner=d.OWNER,gpu='',cwd=str(d.ROOT),argv=cmd,log=str(d.BASE/'logs'/(rid+'-score.log')),CPU_threads=2)
     scoring[rid]=launch(cmd,cpu,base,scorers,'scorer_launch.json')
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
     row=pending.pop(0);rid=row['row_id'];cmd=command('--worker',rid);base=dict(row_id=rid,epochs=row['epochs'],owner=d.OWNER,gpu=str(gpu),cwd=str(d.ROOT),argv=cmd,log=str(d.BASE/'logs'/(rid+'-train_predict.log')),capacity_before=serial(caps),per_GPU_limit=3,CPU_threads=2,read_at=time.time())
     active[rid]=launch(cmd,dict(cpu,CUDA_VISIBLE_DEVICES=str(gpu)),base,receipts,'launch.json');print('LAUNCH_OBSERVED '+rid+' '+str(active[rid].pid),flush=True)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),CPU_scoring=list(scoring),pending=[r['row_id'] for r in pending],complete=complete,failures=failures,registered_rows=12,owner=d.OWNER,updated_at=time.time()))
   if failures and not active and not scoring:raise RuntimeError('Technical failure; preserved artifacts,no automatic retry '+repr(failures))
   if pending or active or scoring:time.sleep(5)
  subprocess.run([sys.executable,'-u',str(d.ROOT/'experiments/cvs_sigtime_rdrop_masks_main/runner.py'),str(request),'--finish'],cwd=d.ROOT,env=cpu,check=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True,healthy_children_not_stopped=True));raise
