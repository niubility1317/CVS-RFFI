import os,sys,time,subprocess,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose,serial
from experiments.cvs_sigtime_rdrop_recovery.startup import launch_once
def run(request):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source dispatcher required')
 if d.BASE.exists():raise FileExistsError('Immutable source run exists; reconcile')
 if (Path(d.MODEL_RELEASE)/'release_commit.txt').read_text().strip()!=d.MODEL_COMMIT:raise ValueError('Original model-code dependency differs')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();pending=list(d.rows());active={};complete={};receipts=[];failures=[];started=time.time()
 try:
  while pending or active:
   for rid,p in list(active.items()):
    rc=p.poll()
    if rc is None:continue
    row=next(r for r in d.rows() if r['row_id']==rid);path=Path(d.config(row)['output_root'])/'rdrop_completion_result.json'
    if rc or not path.is_file() or d.read(path)['status']!='SOURCE_RDROP_COMPLETION_COMPLETE':failures.append(dict(row_id=rid,returncode=rc))
    else:complete[row['arm']]=d.read(path)
    del active[rid]
   if failures:raise RuntimeError('Source technical failure; preserve healthy children, no retry '+repr(failures))
   while pending:
    with capacity_lock():
     caps=capacity();gpu=choose(caps)
     if gpu is None:break
     row=pending.pop(0);rid=row['row_id'];log=d.BASE/'logs'/(rid+'.log')
     cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_sigtime_rdrop_completion20/runner.py'),str(request),'--worker',rid]
     base=dict(row_id=rid,owner=d.OWNER,cwd=str(d.ROOT),argv=cmd,gpu=str(gpu),log=str(log),capacity_before=serial(caps),per_GPU_limit=3,CPU_threads=2,read_at=time.time())
     def record(v):
      old=next((r for r in receipts if r['row_id']==rid),None)
      if old is None:receipts.append(dict(v))
      else:old.update(v)
      d.write(d.BASE/'launch.json',dict(rows=receipts))
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
     with log.open('x') as stream:child,value=launch_once(lambda:subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True),base,record)
     active[rid]=child;print('LAUNCH_OBSERVED '+rid+' '+str(child.pid),flush=True)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=[r['row_id'] for r in pending],complete=list(complete),failures=failures,owner=d.OWNER,updated_at=time.time()))
   if pending or active:time.sleep(5)
  result=dict(status='SOURCE_MATRIX_COMPLETE',run_id=d.RUN,results=complete,qualification=d.qualify(complete),actual_epochs=80,actual_optimizer_steps=17760,source_only=True,target_access=False,head_fit_updates=0,goal_completed=False,elapsed_seconds=time.time()-started)
  d.write(d.BASE/'matrix_result.json',result);d.write(d.BASE/'completion.json',result);print('SOURCE_RDROP_ROUTE_MATRIX_COMPLETE',flush=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,healthy_children_not_stopped=True,no_automatic_retry=True));raise
