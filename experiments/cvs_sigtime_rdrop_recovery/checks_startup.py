"""Regression at the observed /proc startup boundary; never launches models."""
import copy,json,sys
class Child:
 pid=1234
 def __init__(self,returncode=None):self.returncode=returncode
 def poll(self):return self.returncode
class Clock:
 def __init__(self):self.t=0.
 def now(self):return self.t
 def sleep(self,n):self.t+=n
def fixture():return dict(pid=1234,start_ticks=123,cwd='/release',argv=['python','runner','--worker','row'],gpu='2')
def legacy(child,*,reader,cwd,argv,gpu,**unused):
 actual=reader(child.pid)
 if actual is None or actual['gpu']!=gpu:raise RuntimeError('Original SIG_TIME worker not observable')
 return actual
def main():
 if '--legacy' in sys.argv:observe=legacy
 else:
  from .startup import observe_started_child as observe
 calls=[];ready=fixture();states=iter([None,dict(ready,gpu=None),ready]);clock=Clock();child=Child()
 def reader(pid):calls.append(pid);return next(states)
 actual=observe(child,reader=reader,cwd=ready['cwd'],argv=ready['argv'],gpu='2',now=clock.now,sleep=clock.sleep,timeout=.1,interval=.02)
 assert actual==ready and calls==[1234]*3 and child.poll() is None
 if '--legacy' in sys.argv:raise AssertionError('Legacy boundary unexpectedly passed the transient startup regression')
 for field,value in (('gpu','3'),('cwd','/other'),('argv',['other']),('pid',999)):
  bad=dict(ready,**{field:value});clock=Clock()
  try:observe(Child(),reader=lambda pid:bad,cwd=ready['cwd'],argv=ready['argv'],gpu='2',now=clock.now,sleep=clock.sleep,timeout=.04,interval=.01)
  except TimeoutError as error:assert '1234' in str(error)
  else:raise AssertionError('Wrong startup identity accepted: '+field)
 clock=Clock()
 try:observe(Child(7),reader=lambda pid:ready,cwd=ready['cwd'],argv=ready['argv'],gpu='2',now=clock.now,sleep=clock.sleep)
 except ChildProcessError as error:assert '7' in str(error) and '1234' in str(error)
 else:raise AssertionError('Exited child accepted')
 from .startup import launch_once
 for success in (True,False):
  receipts=[];launched=[];reads=[];clock=Clock();states=iter([None,dict(ready,gpu=None),ready])
  def create():launched.append(1234);return Child()
  def record(value):receipts.append(copy.deepcopy(value))
  def inspect(pid):
   assert receipts and receipts[0]['pid']==1234 and receipts[0]['status']=='SUBMITTED';reads.append(pid)
   return next(states) if success else None
  base=dict(row_id='row',cwd=ready['cwd'],argv=ready['argv'],gpu='2')
  try:child,receipt=launch_once(create,base,record,reader=inspect,now=clock.now,sleep=clock.sleep,timeout=.05,interval=.01)
  except TimeoutError:
   assert not success and receipts[-1]['status']=='OBSERVATION_FAILED'
  else:assert success and receipt['status']=='OBSERVED' and receipt['start_ticks']==123
  assert launched==[1234] and len(receipts)==2 and receipts[0]['pid']==receipts[-1]['pid']==1234
 print(json.dumps(dict(status='PASS',startup_regressions=8,model_launch_retries=0,experiment_processes_stopped=0,IQ_reads=0,checkpoint_reads=0)))
if __name__=='__main__':main()
