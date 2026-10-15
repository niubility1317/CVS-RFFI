"""Keep ownership before observing one already-started child; never relaunch."""
import time
def observe_started_child(child,*,cwd,argv,gpu,reader=None,timeout=2.,interval=.02,now=time.monotonic,sleep=time.sleep):
 if reader is None:
  from experiments.cvs_rdrop_fold20.resources import process
  reader=process
 deadline=now()+timeout
 while True:
  rc=child.poll()
  if rc is not None:raise ChildProcessError(f'Child pid={child.pid} exited during startup: {rc}')
  p=reader(child.pid)
  if p is not None and p['pid']==child.pid and p['cwd']==str(cwd) and p['argv']==list(argv) and p['gpu']==str(gpu):return p
  if now()>=deadline:raise TimeoutError(f'Startup identity observation timed out for existing pid={child.pid}; preserve child and artifacts, no launch retry')
  sleep(min(interval,max(0.,deadline-now())))
def launch_once(create,base,record,**observe_options):
 child=create();receipt=dict(base,pid=child.pid,start_ticks=None,status='SUBMITTED');record(receipt)
 try:actual=observe_started_child(child,cwd=base['cwd'],argv=base['argv'],gpu=base['gpu'],**observe_options)
 except Exception:
  receipt['status']='OBSERVATION_FAILED';record(receipt);raise
 receipt.update(start_ticks=actual['start_ticks'],status='OBSERVED');record(receipt)
 return child,receipt
