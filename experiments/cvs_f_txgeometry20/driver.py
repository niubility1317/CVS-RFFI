"""One local GPU worker at a time; source-only expansion, automatic row tests."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d

def update(p,v):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)

def run(args):
 profile=d.read(args.profile);base=Path(args.base).resolve()
 if base.name!=d.RUN or profile['max_gpu_workers']!=1 or profile['physical_gpu']!=0 or profile['source_data_status']!='READY':raise ValueError('Fixed local single GPU/data/output profile required')
 if d.ROOT.resolve()!=Path.cwd().resolve() or (d.ROOT/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Run from the immutable committed local release')
 lock=Path(args.profile).parent/'gpu0.worker.lock'
 with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),run_id=d.RUN,cwd=str(d.ROOT)),f)
 try:
  base.mkdir(parents=True,exist_ok=False);d.write(base/'execution.json',dict(run_id=d.RUN,commit=args.commit,pid=os.getpid(),cwd=str(d.ROOT),python=sys.executable,profile=args.profile,GPU=0,max_GPU_workers=1,source_selection_before_target=True))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
  env['PYTHONPATH']=os.pathsep.join((str(d.ROOT),str(d.ROOT/'code')))
  d.write(base/'registered_rows.json',dict(initial=[d.config(r) for r in d.rows(d.SEEDS[:1])],expansion_candidates=[d.config(r) for r in d.rows(d.SEEDS[1:])]))
  def child(role,row):
   command=[sys.executable,'-X','utf8','-u','-m','experiments.cvs_f_txgeometry20.'+role,'--row',row['row_id'],'--profile',args.profile,'--base',str(base)]
   if role=='source':command+=['--commit',args.commit]
   child_env=dict(env,CUDA_VISIBLE_DEVICES='' if role=='score' else '0');logpath=base/(row['row_id']+'-'+role+'.log')
   with logpath.open('x',encoding='utf-8') as stream:
    p=subprocess.Popen(command,cwd=d.ROOT,env=child_env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    evidence=dict(pid=p.pid,role=role,row_id=row['row_id'],cwd=str(d.ROOT),argv=command,gpu=None if role=='score' else 0,log=str(logpath))
    d.write(base/(row['row_id']+'-'+role+'-start.json'),evidence);update(base/'progress.json',dict(status='RUNNING',**evidence));print(json.dumps(dict(event='CHILD_STARTED',**evidence)),flush=True)
    code=p.wait()
   d.write(base/(row['row_id']+'-'+role+'-exit.json'),dict(pid=p.pid,returncode=code,role=role))
   if code:raise RuntimeError('Preserved failed '+role+' row '+row['row_id']+' exit '+str(code)+'; no automatic retry')
  initial=d.rows(d.SEEDS[:1]);source={}
  for row in initial:
   child('source',row);source[row['arm']]=d.read(base/row['row_id']/'source/completion.json')['sourceV']
  decision=d.expansion(source);d.write(base/'source_expansion.json',decision);print('SOURCE_EXPANSION_FIXED '+json.dumps(decision),flush=True)
  # The entire expansion rule and actual selected rows are now fixed. Neither
  # this controller nor training ever reads scores or target ranking feedback.
  remaining=[r for r in d.rows(d.SEEDS[1:]) if r['arm'] in decision['expand_arms']]
  d.write(base/'frozen_matrix.json',dict(rows=initial+remaining,source_only_decision=decision,target_feedback=False))
  for row in initial:
   child('predict',row);child('score',row)
  for row in remaining:
   child('source',row);child('predict',row);child('score',row)
  from .analysis import aggregate
  aggregate(base);update(base/'progress.json',dict(status='ARTIFACTS_COMPLETE',rows=len(initial+remaining),commit=args.commit));print('LOCAL_FACTORIAL_COMPLETE',flush=True)
 finally:
  # Remove only our own ordinary local GPU reservation, never another process.
  if lock.exists() and d.read(lock).get('pid')==os.getpid():lock.unlink()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--base',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
