import os,sys,time,subprocess,traceback
from pathlib import Path
from . import design as d
from .resources import capacity_lock,capacity,serial,choose,process
def run(request):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Dispatcher must remain CPU only')
 if d.BASE.exists():raise FileExistsError('Existing immutable source run; reconcile original')
 signal=d.read(Path(d.PROJECT)/'runs'/d.SIGNAL_RUN/'matrix_result.json')
 if not signal['all_three_source_confirmed'] or signal['actual_model_seeds']!=3:raise ValueError('Fixed E20 source signal prerequisite did not pass')
 if (Path(d.DEPENDENCY_RELEASE)/'release_commit.txt').read_text().strip()!=d.DEPENDENCY_COMMIT:raise ValueError('Original source dependency differs')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();pending=list(d.ARMS);active={};complete={};receipts=[];failures=[];started=time.time()
 try:
  while pending or active:
   for arm,item in list(active.items()):
    rc=item['child'].poll()
    if rc is None:continue
    c=d.config(arm);p=Path(c['output_root'])/'rdrop_result.json'
    if rc or not p.is_file():failures.append(dict(arm=arm,returncode=rc,reason='Worker ended without valid source completion'))
    else:
     result=d.read(p)
     if result['status']!='SOURCE_RDROP_TRAINED' or result['actual_steps']!=4440:raise ValueError('Source artifact budget differs')
     complete[arm]=result
    del active[arm]
   while pending and not failures:
    with capacity_lock():
     caps=capacity();gpu=choose(caps)
     if gpu is None:break
     arm=pending.pop(0);c=d.config(arm);cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_rdrop_fold20/runner.py'),str(request),'--row',c['row_id']]
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
     log=d.BASE/'logs'/(arm+'.log')
     with log.open('x') as stream:child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
     actual=process(child.pid)
     if actual is None or actual['gpu']!=str(gpu):raise RuntimeError('New source worker not observable')
     receipt=dict(row_id=c['row_id'],arm=arm,owner=d.OWNER,pid=child.pid,start_ticks=actual['start_ticks'],gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log),capacity_before=serial(caps),per_GPU_limit=3,CPU_threads=2,read_at=time.time())
     receipts.append(receipt);d.write(d.BASE/'launch.json',dict(rows=receipts));active[arm]=dict(child=child,receipt=receipt)
     print('LAUNCH '+str(receipt),flush=True)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,complete=list(complete),failures=failures,updated_at=time.time(),owner=d.OWNER,source_only=True))
   if failures and not active:raise RuntimeError('Technical failure preserved; no automatic retry '+repr(failures))
   if pending or active:time.sleep(5)
  qualification=d.qualify(complete)
  final=dict(status='SOURCE_MATRIX_COMPLETE',run_id=d.RUN,results=complete,qualification=qualification,actual_optimizer_steps=8880,actual_epochs=40,source_only=True,target_access=False,head_fit_updates=0,goal_completed=False,elapsed_seconds=time.time()-started)
  d.write(d.BASE/'matrix_result.json',final);d.write(d.BASE/'completion.json',final);print('RDROP_SOURCE_TRAINING_COMPLETE '+str(final),flush=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
