"""CPU-only checks for scheduler adoption and capacity-only worker changes."""
import contextlib,json,os,tempfile,types
from pathlib import Path
from unittest.mock import patch
from . import capacity4 as c,dispatch as original
from .publish_capacity4 import REMOTE

def run():
 checks=[]
 with tempfile.TemporaryDirectory() as tmp:
  base=Path(tmp)
  d=types.SimpleNamespace(BASE=base,SEARCH_ARMS={'search'},rows=lambda:[dict(row_id='r',arm='fixed'),dict(row_id='s',arm='search')],read=lambda p:json.loads(p.read_text()))
  receipt=dict(row_id='r',kind='train',pid=9,cwd='/release',argv=['python','worker'])
  actual=dict(pid=9,cwd='/release',argv=['python','worker'],start_ticks=123)
  active,pending,done=c.reconcile(d,'train',['r','s'],[receipt],lambda pid:actual)
  assert active['r']['start_ticks']==123 and pending==['s'] and not done
  checks.append('live worker adopted without relaunch')
  def rejects(receipts,proc):
   try:c.reconcile(d,'train',['r'],receipts,proc)
   except ValueError:return
   raise AssertionError('unsafe adoption accepted')
  for key,value in [('cwd','/other'),('argv',['other']),('start_ticks',124)]:
   rejects([receipt],lambda pid,key=key,value=value:dict(actual,**{key:value}))
  rejects([receipt,receipt],lambda pid:actual)
  rejects([receipt],lambda pid:None)
  (base/'r').mkdir();(base/'r/completion.json').write_text('{"status":"ANALYZED"}')
  assert c.reconcile(d,'train',['r'],[receipt],lambda pid:None)==({},[],['r'])
  (base/'r/source').mkdir();rejects([],lambda pid:None)
  assert c.expected_status(d,'s','train')=='SOURCE_CANDIDATE_FROZEN'
  assert c.expected_status(d,'s','test')=='ANALYZED'
  checks.append('PID reuse, duplicates, missing artifacts and unreceipted output rejected')
 module=types.ModuleType('test_worker');module.__dict__.update(original.__dict__)
 c.patched_worker(module)
 calls=[];sleeps=[];counts=iter([5,4,4,4]);pid=os.getpid()
 module.capacity=lambda:{0:dict(pids={pid}|set(range(next(counts)-1)),free_mb=13000)}
 module.lock=lambda *a,**k:contextlib.nullcontext()
 module.time=types.SimpleNamespace(sleep=sleeps.append)
 module.d=types.SimpleNamespace(PROJECT='/tmp',rows=lambda:[dict(row_id='r')])
 module.test_row=calls.append
 with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):module.worker('test','r')
 assert calls==['r'] and sleeps==[5,1,1]
 checks.append('worker rejects five owners and admits four after three stable observations')
 module=types.ModuleType('test_dispatch');module.__dict__.update(original.__dict__)
 c.patched_dispatch(module)
 assert module.dispatch.__code__.co_filename=='capacity4-adopting-dispatch'
 for name in ['source_select','test_rows','resolved_config','subprocess']:
  assert name in module.dispatch.__code__.co_names
 assert 'mkdir' not in module.dispatch.__code__.co_names
 checks.append('adopting dispatch preserves source selection and later pipeline')
 compile(REMOTE.replace('CONFIG','{}'),'remote-publish','exec')
 checks.append('remote publisher compiles')
 print(json.dumps(dict(status='PASS',checks=checks),indent=2))

if __name__=='__main__':run()
