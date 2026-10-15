import os,sys,time,subprocess,traceback
from pathlib import Path
from . import design as d
from .startup import launch_once
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,serial,choose,process

def run(request):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only dispatcher required')
 if d.BASE.exists():raise FileExistsError('Existing immutable combination run; reconcile,never restart')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();pending=list(d.training_rows());active={};scoring={};receipts=[];scorer_receipts=[];complete=[];failures=[];skipped=[]
 cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def command(mode,rid=None):return [sys.executable,'-u',str(d.ROOT/'experiments/cvs_rdrop_sinc_units/runner.py'),str(request),mode]+([] if rid is None else [rid])
 try:
  subprocess.run(command('--preflight'),cwd=d.ROOT,env=cpu,check=True)
  for row in d.rows():
   if row['arm']!=d.BASELINE_ARM:continue
   rid=row['row_id']
   with (d.BASE/'logs'/(rid+'-reference.log')).open('x') as f:subprocess.run(command('--reference',rid),cwd=d.ROOT,env=cpu,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
   complete.append(rid)
  while pending or active or scoring:
   for rid,child in list(active.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'prediction/complete.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='PREDICTIONS_COMPLETE':failures.append(dict(row_id=rid,phase='train_predict',returncode=rc))
    else:
     cmd=command('--score',rid);log=d.BASE/'logs'/(rid+'-score.log')
     with log.open('x') as f:p=subprocess.Popen(cmd,cwd=d.ROOT,env=cpu,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     scoring[rid]=p;scorer_receipts.append(dict(row_id=rid,pid=p.pid,cwd=str(d.ROOT),argv=cmd,gpu='',log=str(log)));d.write(d.BASE/'scorer_launch.json',dict(rows=scorer_receipts))
    del active[rid]
   for rid,child in list(scoring.items()):
    rc=child.poll()
    if rc is None:continue
    artifact=d.BASE/rid/'completion.json'
    if rc or not artifact.is_file() or d.read(artifact)['status']!='MODEL_PREDICTED_SCORED':failures.append(dict(row_id=rid,phase='CPU_score',returncode=rc))
    else:complete.append(rid)
    del scoring[rid]
   qualification=d.read(d.BASE/'source_qualification.json') if (d.BASE/'source_qualification.json').is_file() else None
   if qualification is None:
    from .gate import evaluate_source
    qualification=evaluate_source()
    if qualification is not None:d.write(d.BASE/'source_qualification.json',qualification)
   from .gate import expansion_released
   released=qualification is not None and expansion_released(qualification)
   if qualification is not None and not qualification['qualified']:
    omitted=[r for r in pending if r['model_seed']!=d.SEEDS[0]];skipped.extend(dict(row_id=r['row_id'],reason='Fixed first-source mechanism/health falsified',model_seed=r['model_seed']) for r in omitted);pending=[r for r in pending if r not in omitted]
   while pending and not failures:
    row=next((r for r in pending if r['model_seed']==d.SEEDS[0] or released),None)
    if row is None:break
    with capacity_lock():
     caps=capacity();gpu=choose(caps)
     if gpu is None:break
     rid=row['row_id'];cmd=command('--worker',rid);log=d.BASE/'logs'/(rid+'-train_predict.log');pending.remove(row)
     def create():
      with log.open('x') as f:p=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(cpu,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[rid]=p;return p
     def record(receipt):
      existing=next((x for x in receipts if x['row_id']==rid),None)
      if existing is None:receipts.append(receipt)
      else:existing.update(receipt)
      d.write(d.BASE/'launch.json',dict(rows=receipts))
     launch_once(create,dict(row_id=rid,epochs=50,owner=d.OWNER,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log),capacity_before=serial(caps),per_GPU_limit=3,CPU_threads=2,read_at=time.time()),record)
   d.write(d.BASE/'queue_state.json',dict(active=list(active),CPU_scoring=list(scoring),pending=[r['row_id'] for r in pending],complete=complete,failures=failures,source_qualification=qualification,source_skipped=skipped,updated_at=time.time(),owner=d.OWNER,per_GPU_limit=3,registered_rows=18))
   if failures and not active and not scoring:raise RuntimeError('Technical failure preserved; no pending launch/retry '+repr(failures))
   if pending or active or scoring:time.sleep(5)
  from .gate import decision_uploaded
  qualification=d.read(d.BASE/'source_qualification.json')
  while not decision_uploaded(qualification):
   d.write(d.BASE/'aggregation_wait.json',dict(status='AWAITING_SOURCE_DECISION_GIT_UPLOAD',qualified=qualification['qualified'],target_summary_read=False))
   time.sleep(5)
  subprocess.run(command('--finish'),cwd=d.ROOT,env=cpu,check=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,healthy_children_not_stopped=True,no_launch_retry=True));raise
