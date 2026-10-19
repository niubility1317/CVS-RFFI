"""Continue an immutable local factorial with a bounded GPU pool.

Only scheduling changes. Workers run from the original release and retain its
config, source-only expansion rule, checkpoint and truth-last scoring path.
"""
import argparse, ctypes, json, os, subprocess, sys, time
from pathlib import Path

ROLES=('source','predict','score')

def read(p):
 return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def replace(p,value):
 p=Path(p); tmp=p.with_name(p.name+'.parallel.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def write_new(p,value):
 with Path(p).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def command(python,role,row,profile,base,commit):
 value=[python,'-X','utf8','-u','-m','experiments.cvs_f_txgeometry20.'+role,'--row',row,'--profile',str(profile),'--base',str(base)]
 return value+(['--commit',commit] if role=='source' else [])

def split_windows(text):
 from ctypes import wintypes
 shell=ctypes.WinDLL('shell32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
 shell.CommandLineToArgvW.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_int)];shell.CommandLineToArgvW.restype=ctypes.POINTER(wintypes.LPWSTR)
 kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
 count=ctypes.c_int();ptr=shell.CommandLineToArgvW(text,ctypes.byref(count))
 if not ptr:raise ctypes.WinError(ctypes.get_last_error())
 try:return [ptr[i] for i in range(count.value)]
 finally:kernel.LocalFree(ptr)

class ProcessHandle:
 def __init__(self,pid,terminate=False):
  from ctypes import wintypes
  self.pid=pid;self.k=ctypes.WinDLL('kernel32',use_last_error=True)
  self.k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];self.k.OpenProcess.restype=wintypes.HANDLE
  self.k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];self.k.WaitForSingleObject.restype=wintypes.DWORD
  self.k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];self.k.GetExitCodeProcess.restype=wintypes.BOOL
  self.k.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];self.k.TerminateProcess.restype=wintypes.BOOL
  self.k.CloseHandle.argtypes=[wintypes.HANDLE];self.k.CloseHandle.restype=wintypes.BOOL
  self.handle=self.k.OpenProcess(0x100000|0x1000|(1 if terminate else 0),False,pid)
  if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
 def poll(self):
  from ctypes import wintypes
  wait=self.k.WaitForSingleObject(self.handle,0)
  if wait==258:return None
  if wait!=0:raise ctypes.WinError(ctypes.get_last_error())
  code=wintypes.DWORD()
  if not self.k.GetExitCodeProcess(self.handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
  return code.value
 def stop_controller(self):
  if self.poll() is None and not self.k.TerminateProcess(self.handle,0):raise ctypes.WinError(ctypes.get_last_error())
  if self.k.WaitForSingleObject(self.handle,10000)!=0:raise RuntimeError('Controller handoff termination unconfirmed')
 def close(self):
  if self.handle:self.k.CloseHandle(self.handle);self.handle=None

def inventory():
 script="Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
 result=subprocess.run([os.environ.get('SystemRoot','C:/Windows')+'/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',script],check=True,capture_output=True,text=True,encoding='utf-8')
 values=json.loads(result.stdout or '[]');return values if isinstance(values,list) else [values]

def complete(base,row,role):
 tail={'source':'source/completion.json','predict':'prediction/complete.json','score':'scores/metrics.json'}[role]
 return (base/row/tail).is_file()

def candidates(initial,remaining,frozen,base,active):
 """No target job is eligible before the complete firstseed source gate."""
 all_rows=initial+remaining if frozen else initial
 occupied={(j['row_id'],j['role']) for j in active}
 pending=[]
 for row in all_rows:
  rid=row['row_id']
  if not complete(base,rid,'source') and (rid,'source') not in occupied:pending.append(('source',rid))
 if frozen:
  for row in all_rows:
   rid=row['row_id']
   if complete(base,rid,'source') and not complete(base,rid,'predict') and (rid,'predict') not in occupied:pending.append(('predict',rid))
 return pending

def run(a):
 if os.name!='nt' or a.max_workers!=3:raise ValueError('This continuation is Windows GPU0, fixed max3 workers')
 worker=Path(a.worker_release).resolve();base=Path(a.base).resolve();profile_path=Path(a.profile).resolve();profile=read(profile_path)
 if not base.is_dir() or profile['physical_gpu']!=0 or profile['source_data_status']!='READY':raise ValueError('Existing ready local run required')
 if (worker/'release_commit.txt').read_text().strip()!=a.worker_commit:raise ValueError('Original worker commit differs')
 sys.path[:0]=[str(worker),str(worker/'code')]
 from experiments.cvs_f_txgeometry20 import design as d
 if d.ROOT.resolve()!=worker or base.name!=d.RUN:raise ValueError('Immutable original worker import/run mismatch')
 execution=read(base/'execution.json')
 if execution['commit']!=a.worker_commit or Path(execution['cwd']).resolve()!=worker:raise ValueError('Continuation provenance differs')
 lock=profile_path.parent/'gpu0.worker.lock';reservation=read(lock)
 if reservation.get('pid')!=a.old_controller or reservation.get('run_id')!=d.RUN or Path(reservation['cwd']).resolve()!=worker:raise ValueError('Reservation is not the named serial owner')
 expected=[profile['base_python'],'-X','utf8','-u','-m','experiments.cvs_f_txgeometry20.driver','--profile',str(profile_path),'--base',str(base),'--commit',a.worker_commit]
 # Exact process argv + original execution CWD/run/lock evidence. Open a stable
 # handle before mutation. Never terminate the training child or a process tree.
 old=ProcessHandle(a.old_controller,terminate=True)
 processes=inventory();match=next((p for p in processes if p['ProcessId']==a.old_controller),None)
 if match is None or split_windows(match['CommandLine'])!=expected:old.close();raise ValueError('Serial controller PID/argv ownership differs')
 if (base/'parallel_handoff.json').exists():old.close();raise FileExistsError('Handoff already attempted; inspect state instead of retrying')
 write_new(base/'parallel_handoff.json',dict(status='CONTROLLER_HANDOFF_PREPARED',old_controller=a.old_controller,new_controller=os.getpid(),worker_commit=a.worker_commit,controller_commit=a.controller_commit,max_GPU_workers=3,worker_release=str(worker),original_execution=execution))
 old.stop_controller();old.close()
 replace(lock,dict(pid=os.getpid(),run_id=d.RUN,cwd=str(worker),controller=str(Path(__file__).resolve()),max_GPU_workers=3))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=os.pathsep.join((str(worker),str(worker/'code'))),TORCHINDUCTOR_CACHE_DIR=str(base/'torch_cache'))
 initial=d.rows(d.SEEDS[:1]);active=[];popen_refs={};known={r['row_id'] for r in d.rows()}
 def progress(status,**extra):
  value=dict(status=status,controller_pid=os.getpid(),max_GPU_workers=3,worker_commit=a.worker_commit,controller_commit=a.controller_commit,active=[{k:v for k,v in j.items() if k!='handle'} for j in active],updated=time.time(),**extra)
  replace(base/'parallel_progress.json',value);replace(base/'progress.json',value)
 def launch(role,rid):
  out=base/rid/{'source':'source','predict':'prediction','score':'scores'}[role]
  if out.exists():raise FileExistsError('Preserve incomplete '+str(out)+'; no automatic restart')
  argv=command(profile['base_python'],role,rid,profile_path,base,a.worker_commit);logpath=base/(rid+'-'+role+'.log')
  with logpath.open('x',encoding='utf-8') as stream:
   p=subprocess.Popen(argv,cwd=worker,env=dict(env,CUDA_VISIBLE_DEVICES='' if role=='score' else '0'),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
  popen_refs[p.pid]=p
  evidence=dict(pid=p.pid,role=role,row_id=rid,cwd=str(worker),argv=argv,gpu=None if role=='score' else 0,log=str(logpath),adopted=False)
  active.append(dict(evidence,handle=p));write_new(base/(rid+'-'+role+'-start.json'),evidence)
  print(json.dumps(dict(event='CHILD_STARTED',**evidence)),flush=True)
 try:
  # Inventory AFTER controller exit also catches a child it launched during the
  # preflight. Adopt every exact matching worker, never duplicate its row.
  for proc in inventory():
   text=proc.get('CommandLine') or ''
   if str(base).lower() not in text.lower():continue
   argv=split_windows(text)
   role=next((r for r in ROLES if 'experiments.cvs_f_txgeometry20.'+r in argv),None)
   if role is None:continue
   if '--row' not in argv:raise ValueError('Unidentified run worker')
   rid=argv[argv.index('--row')+1]
   if rid not in known or argv!=command(profile['base_python'],role,rid,profile_path,base,a.worker_commit):raise ValueError('Run worker argv differs')
   saved=read(base/(rid+'-'+role+'-start.json'))
   if saved['pid']!=proc['ProcessId'] or Path(saved['cwd']).resolve()!=worker or saved['argv']!=argv:raise ValueError('Adopted worker launch evidence differs')
   handle=ProcessHandle(proc['ProcessId'])
   active.append(dict(saved,adopted=True,handle=handle))
  if len([j for j in active if j['role']!='score'])>3:raise ValueError('Existing GPU pool exceeds3')
  write_new(base/'parallel_adoption.json',dict(status='VERIFIED',old_controller_stopped=True,new_controller=os.getpid(),adopted=[{k:v for k,v in j.items() if k!='handle'} for j in active],worker_recipe_changed=False))
  while True:
   for job in list(active):
    code=job['handle'].poll()
    if code is None:continue
    if isinstance(job['handle'],ProcessHandle):job['handle'].close()
    active.remove(job);path=base/(job['row_id']+'-'+job['role']+'-exit.json')
    if not path.exists():write_new(path,dict(pid=job['pid'],returncode=code,role=job['role'],adopted=job['adopted']))
    if code or not complete(base,job['row_id'],job['role']):raise RuntimeError('Preserved technical failure '+job['row_id']+' '+job['role']+' exit '+str(code))
   frozen=(base/'frozen_matrix.json').exists()
   if not frozen and all(complete(base,r['row_id'],'source') for r in initial):
    source={r['arm']:d.read(base/r['row_id']/'source/completion.json')['sourceV'] for r in initial}
    decision=d.expansion(source)
    if (base/'source_expansion.json').exists():
     if d.read(base/'source_expansion.json')!=decision:raise ValueError('Existing source decision differs')
    else:d.write(base/'source_expansion.json',decision)
    remaining=[r for r in d.rows(d.SEEDS[1:]) if r['arm'] in decision['expand_arms']]
    d.write(base/'frozen_matrix.json',dict(rows=initial+remaining,source_only_decision=decision,target_feedback=False));frozen=True
    print('SOURCE_EXPANSION_FIXED '+json.dumps(decision),flush=True)
   matrix=d.read(base/'frozen_matrix.json')['rows'] if frozen else initial
   remaining=matrix[len(initial):] if frozen else []
   for role,rid in candidates(initial,remaining,frozen,base,active):
    if sum(j['role']!='score' for j in active)>=a.max_workers:break
    launch(role,rid)
   if frozen and not any(j['role']=='score' for j in active):
    for row in matrix:
     rid=row['row_id']
     if complete(base,rid,'predict') and not complete(base,rid,'score') and not any(j['row_id']==rid and j['role']=='predict' for j in active):launch('score',rid);break
   if frozen and not active and all(complete(base,r['row_id'],'score') for r in matrix):
    from experiments.cvs_f_txgeometry20.analysis import aggregate
    if not (base/'result.json').exists():aggregate(base)
    progress('ARTIFACTS_COMPLETE',rows=len(matrix));print('LOCAL_PARALLEL_FACTORIAL_COMPLETE',flush=True)
    if read(lock).get('pid')==os.getpid():lock.unlink()
    break
   progress('RUNNING',source_expansion_fixed=frozen);time.sleep(3)
 except Exception as error:
  # Do not stop healthy workers on any controller/row failure. Preserve live
  # process evidence and forbid silent relaunch/overwriting incomplete outputs.
  progress('FAILED_DISPATCH_STOPPED',error=repr(error),healthy_workers_preserved=True)
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--base',required=True);p.add_argument('--worker-release',required=True);p.add_argument('--worker-commit',required=True);p.add_argument('--controller-commit',required=True);p.add_argument('--old-controller',type=int,required=True);p.add_argument('--max-workers',type=int,default=3);run(p.parse_args())
