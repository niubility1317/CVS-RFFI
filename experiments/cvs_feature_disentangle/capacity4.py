"""Adopt existing workers; change only scheduling to four total owners/GPU."""
import argparse,inspect,os,subprocess,sys,time
from pathlib import Path

LIMIT=4
FREE_MB=12000

def load(release):
 release=Path(release).resolve();sys.path[:0]=[str(release),str(release/'code')]
 from experiments.cvs_feature_disentangle import design as d,dispatch as original
 from experiments.cvs_receiver_residual_capacity4.control import process,running,lock
 if d.ROOT.resolve()!=release:raise ValueError('Wrong scientific release')
 return d,original,process,running,lock

def expected_status(d,rid,kind):
 r=next(r for r in d.rows() if r['row_id']==rid)
 return 'SOURCE_CANDIDATE_FROZEN' if kind=='train' and r['arm'] in d.SEARCH_ARMS else 'ANALYZED'

def reconcile(d,kind,items,receipts,process):
 active={};pending=[];done=[]
 for rid in items:
  matches=[r for r in receipts if r['row_id']==rid and r['kind']==kind]
  if len(matches)>1:raise ValueError('Duplicate launch receipt '+rid)
  if matches:
   receipt=matches[0];actual=process(receipt['pid'])
   if actual:
    if actual['cwd']!=receipt['cwd'] or actual['argv']!=receipt['argv']:
     raise ValueError('Worker identity mismatch '+rid)
    if 'start_ticks' in receipt and actual['start_ticks']!=receipt['start_ticks']:raise ValueError('PID reused '+rid)
    receipt['start_ticks']=actual['start_ticks'];active[rid]=receipt
   else:
    p=d.BASE/rid/'completion.json'
    if not p.is_file() or d.read(p)['status']!=expected_status(d,rid,kind):
     raise ValueError('Ended without completion; no relaunch '+rid)
    done.append(rid)
  else:
   p=d.BASE/rid/('source' if kind=='train' else 'prediction')
   if p.exists():raise ValueError('Unreceipted output; refusing overwrite '+rid)
   pending.append(rid)
 return active,pending,done

def phase(d,original,process,lock,kind,items,receipts):
 active,pending,done=reconcile(d,kind,items,receipts,process);children={};failures=[]
 while pending or active:
  for rid,receipt in list(active.items()):
   if rid in children:children[rid].poll()
   actual=process(receipt['pid'])
   if actual:
    if actual['start_ticks']!=receipt['start_ticks'] or actual['cwd']!=receipt['cwd'] or actual['argv']!=receipt['argv']:
     raise ValueError('Active PID identity changed '+rid)
    continue
   p=d.BASE/rid/'completion.json'
   if not p.is_file() or d.read(p)['status']!=expected_status(d,rid,kind):failures.append(dict(row_id=rid,reason='ended without valid artifact'))
   else:done.append(rid)
   del active[rid]
  while pending and not failures:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    choices=[(len(x['pids']),-x['free_mb'],gpu) for gpu,x in original.capacity().items() if len(x['pids'])<LIMIT and x['free_mb']>=FREE_MB]
    if not choices:break
    gpu=min(choices)[2];rid=pending.pop(0);log=d.BASE/'logs'/(rid+'-'+kind+'.log')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--release',str(d.ROOT),'--worker',kind,'--row',rid]
    with log.open('x') as f:
     child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    actual=process(child.pid)
    if not actual:raise RuntimeError('New worker absent')
    receipt=dict(kind=kind,row_id=rid,pid=child.pid,start_ticks=actual['start_ticks'],gpu=gpu,log=str(log),argv=cmd,cwd=str(d.ROOT),per_gpu_limit=LIMIT)
    receipts.append(receipt);d.write(d.BASE/'launch.json',dict(rows=receipts));active[rid]=receipt;children[rid]=child
   original.event('RUNNING',row_id=rid,kind=kind,pid=child.pid,gpu=gpu,per_gpu_limit=LIMIT)
  d.write(d.BASE/'queue_state.json',dict(phase=kind,active=list(active),pending=pending,completed=done,failures=failures,
   total_identity_rows=48,total_test_rows=36,per_gpu_limit=LIMIT,max_active='all available GPU slots',updated_at=time.time()))
  if failures and not active:raise RuntimeError('Failed rows retained; no retry '+repr(failures))
  if pending or active:time.sleep(10)

def patched_worker(original):
 source=inspect.getsource(original.worker)
 old="len(cap['pids']|{os.getpid()})<=2"
 if source.count(old)!=1:raise ValueError('Unexpected worker admission source')
 source=source.replace(old,"len(cap['pids']|{os.getpid()})<=4")
 exec(compile(source,'capacity4-worker-only','exec'),original.__dict__)

def patched_dispatch(original):
 source=inspect.getsource(original.dispatch)
 start=source.index('  d.BASE.mkdir(');end=source.index('  try:',start)
 expected="  receipts=[]\n"
 if expected not in source[start:end]:raise ValueError('Unexpected dispatcher initialization')
 source=source[:start]+"  if not d.BASE.is_dir():raise ValueError('Missing existing run')\n  receipts=d.read(d.BASE/'launch.json')['rows']\n"+source[end:]
 exec(compile(source,'capacity4-adopting-dispatch','exec'),original.__dict__)

def main(a):
 d,original,process,running,lock=load(a.release)
 if a.worker:
  patched_worker(original);original.worker(a.worker,a.row);return
 # Handoff publisher keeps this lock until the new owner marker is durable.
 with lock(d.BASE/'capacity4_owner.lock'):
  marker=d.read(d.BASE/'dispatcher_active.json')
  if marker['pid']!=os.getpid() or marker['argv']!=process(os.getpid())['argv']:raise ValueError('Owner marker differs')
  if running(d.read(d.BASE/'dispatcher.json')['pid']):raise ValueError('Old controller still live')
  original.phase=lambda kind,items,receipts:phase(d,original,process,lock,kind,items,receipts)
  patched_dispatch(original);original.dispatch()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--worker',choices=['train','test']);p.add_argument('--row');main(p.parse_args())
