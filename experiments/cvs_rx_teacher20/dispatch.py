import os,sys,time,subprocess,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose,serial,process
def run(request):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU dispatcher required')
 if d.BASE.exists():raise FileExistsError('Immutable teacher run exists; reconcile')
 if (Path(d.DEPENDENCY_RELEASE)/'release_commit.txt').read_text().strip()!=d.DEPENDENCY_COMMIT:raise ValueError('Actual original code dependency differs')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();pending=list(d.rows());active={};receipts=[];complete=[];failures=[]
 cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 try:
  while pending or active:
   for rid,child in list(active.items()):
    rc=child.poll()
    if rc is None:continue
    result=Path(d.config(next(r for r in d.rows() if r['row_id']==rid))['output_root'])/'held_RX1_V.json'
    if rc or not result.is_file() or d.read(result).get('status')!='SOURCE_TEACHER_COMPLETE':failures.append(dict(row_id=rid,returncode=rc))
    else:complete.append(rid)
    del active[rid]
   while pending and not failures:
    with capacity_lock():
     caps=capacity();gpu=choose(caps)
     if gpu is None:break
     row=pending.pop(0);rid=row['row_id'];log=d.BASE/'logs'/(rid+'.log');cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_rx_teacher20/runner.py'),str(request),'--worker',rid]
     with log.open('x') as stream:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(cpu,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
     actual=process(child.pid)
     if actual is None or actual['gpu']!=str(gpu):raise RuntimeError('Teacher child not observable')
     active[rid]=child;receipts.append(dict(row_id=rid,owner=d.OWNER,pid=child.pid,start_ticks=actual['start_ticks'],gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log),capacity_before=serial(caps),per_GPU_limit=3,CPU_threads=2,read_at=time.time()));d.write(d.BASE/'launch.json',dict(rows=receipts));print('LAUNCH '+str(receipts[-1]),flush=True)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=[r['row_id'] for r in pending],complete=complete,failures=failures,owner=d.OWNER,updated_at=time.time()))
   if failures and not active:raise RuntimeError('Teacher technical failure: preserve outputs; no pending launch/retry '+repr(failures))
   if pending or active:time.sleep(5)
  log=d.BASE/'logs/source_compare.log';cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_rx_teacher20/runner.py'),str(request),'--compare']
  with log.open('x') as stream:
   child=subprocess.Popen(cmd,cwd=d.ROOT,env=cpu,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   d.write(d.BASE/'compare_launch.json',dict(pid=child.pid,cwd=str(d.ROOT),argv=cmd,CUDA_VISIBLE_DEVICES='',CPU_threads=2));rc=child.wait()
  if rc:raise RuntimeError('Independent source comparison failed '+str(rc))
  if d.read(d.BASE/'completion.json').get('status')!='SOURCE_MATRIX_COMPLETE':raise ValueError('Source completion missing')
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
