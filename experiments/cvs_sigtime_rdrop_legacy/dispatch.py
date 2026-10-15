import os,sys,time,subprocess,traceback
from pathlib import Path
from . import design as d
from .gate import evaluate_source_budget
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,serial,choose,process
def run(request):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only dispatcher required')
 if d.BASE.exists():raise FileExistsError('Immutable original SIG_TIME run exists; reconcile')
 if (Path(d.DEPENDENCY_RELEASE)/'release_commit.txt').read_text().strip()!=d.DEPENDENCY_COMMIT:raise ValueError('Original code dependency differs')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();pending=list(d.rows());active={};scoring={};receipts=[];scorer_receipts=[];complete=[];failures=[];gates={};skipped=[]
 cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def command(mode,rid):return [sys.executable,'-u',str(d.ROOT/'experiments/cvs_sigtime_rdrop_legacy/runner.py'),str(request),mode,rid]
 try:
  while pending or active or scoring:
   for rid,child in list(active.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'prediction/complete.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='PREDICTIONS_COMPLETE':failures.append(dict(row_id=rid,phase='train_predict',returncode=rc))
    else:
     cmd=command('--score',rid);log=d.BASE/'logs'/(rid+'-score.log')
     with log.open('x') as stream:p=subprocess.Popen(cmd,cwd=d.ROOT,env=cpu,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
     scoring[rid]=p;scorer_receipts.append(dict(row_id=rid,pid=p.pid,cwd=str(d.ROOT),argv=cmd,CUDA_VISIBLE_DEVICES='',log=str(log)));d.write(d.BASE/'scorer_launch.json',dict(rows=scorer_receipts))
    del active[rid]
   for rid,child in list(scoring.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'completion.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='MODEL_PREDICTED_SCORED':failures.append(dict(row_id=rid,phase='CPU_score',returncode=rc))
    else:complete.append(rid)
    del scoring[rid]
   if not failures:
    for e in d.BUDGETS:
     if str(e) in gates:continue
     gate=evaluate_source_budget(e)
     if gate is None:continue
     gates[str(e)]=gate;d.write(d.BASE/'source_gates.json',gates)
     if not gate['qualified']:
      omitted=[r for r in pending if r['epochs']==e and r['model_seed']!=d.SEEDS[0]]
      skipped.extend(dict(row_id=r['row_id'],reason='Fixed per-budget source qualification failed',epochs=e,model_seed=r['model_seed']) for r in omitted)
      pending=[r for r in pending if r not in omitted]
   while pending and not failures:
    row=next((r for r in pending if r['model_seed']==d.SEEDS[0] or gates.get(str(r['epochs']),{}).get('qualified')),None)
    if row is None:break
    with capacity_lock():
     caps=capacity();gpu=choose(caps)
     if gpu is None:break
     pending.remove(row);rid=row['row_id'];cmd=command('--worker',rid);log=d.BASE/'logs'/(rid+'-train_predict.log')
     with log.open('x') as stream:p=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(cpu,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
     actual=process(p.pid)
     if actual is None or actual['gpu']!=str(gpu):raise RuntimeError('Original SIG_TIME worker not observable')
     active[rid]=p;receipts.append(dict(row_id=rid,epochs=row['epochs'],owner=d.OWNER,pid=p.pid,start_ticks=actual['start_ticks'],gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log),capacity_before=serial(caps),per_GPU_limit=3,CPU_threads=2,read_at=time.time()));d.write(d.BASE/'launch.json',dict(rows=receipts));print('LAUNCH '+str(receipts[-1]),flush=True)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),CPU_scoring=list(scoring),pending=[r['row_id'] for r in pending],complete=complete,failures=failures,source_gates=gates,source_skipped=skipped,updated_at=time.time(),owner=d.OWNER,per_GPU_limit=3,registered_rows=18))
   if failures and not active and not scoring:raise RuntimeError('Technical failure preserved; no pending launch/retry '+repr(failures))
   if pending or active or scoring:time.sleep(5)
  subprocess.run([sys.executable,'-u',str(d.ROOT/'experiments/cvs_sigtime_rdrop_legacy/runner.py'),str(request),'--finish'],cwd=d.ROOT,env=cpu,check=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
