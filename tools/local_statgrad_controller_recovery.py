"""Control-only recovery for one stopped local dispatcher; never restarts live models.

Uses a new control root. Workers keep their original immutable scientific release,
six-row matrix, output paths, and commit. The failed progress file is preserved.
"""
import argparse,ctypes,json,os,subprocess,sys,time
from pathlib import Path
from ctypes import wintypes as W

RUN='20261020-recovery-FcrossRX-statGradient-controller-local-r01'
SCIENCE_RUN='20261020-phase1-FcrossRX-statGradient-e50-clean-local-m6-r01'
SCIENCE_COMMIT='7b5532e9ff3845c23c7c229385ff5763254d31c3'
MODULE='experiments.cvs_f_statgrad50'
ARTIFACT={'source':'source/local_source_complete.json','predict':'prediction/complete.json','score':'scores/metrics.json'}
DIRECTORY={'source':'source','predict':'prediction','score':'scores'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def exclusive(p,value):
 with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def atomic_write(p,value,attempts=40,delay=.05):
 p=Path(p);temp=p.with_name(p.name+'.recovery.tmp')
 temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
 for i in range(attempts):
  try:temp.replace(p);return
  except PermissionError as error:
   if getattr(error,'winerror',None) not in (5,32,33) or i+1==attempts:raise
   time.sleep(delay)

def process_snapshot():
 command="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress"
 r=subprocess.run([os.environ.get('SystemRoot','C:/Windows')+'/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,encoding='utf8',check=True,timeout=15)
 values=json.loads(r.stdout) or []
 return values if isinstance(values,list) else [values]

class ObservedProcess:
 """Read-only handles retain real exit status and prevent PID-reuse ambiguity."""
 def __init__(self,pid,pids):
  self.pid=pid;self.pids=list(pids);self.handles=[]
  self.k=ctypes.WinDLL('kernel32',use_last_error=True)
  self.k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];self.k.OpenProcess.restype=W.HANDLE
  self.k.GetExitCodeProcess.argtypes=[W.HANDLE,ctypes.POINTER(W.DWORD)];self.k.GetExitCodeProcess.restype=W.BOOL
  self.k.CloseHandle.argtypes=[W.HANDLE];self.k.CloseHandle.restype=W.BOOL
  try:
   for value in self.pids:
    handle=self.k.OpenProcess(0x1000,False,value)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    self.handles.append(handle)
  except BaseException:self.close();raise
 def poll(self):
  codes=[]
  for handle in self.handles:
   code=W.DWORD()
   if not self.k.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
   codes.append(code.value)
  if any(v==259 for v in codes):return None
  return next((v for v in codes if v),0)
 def close(self):
  for handle in self.handles:self.k.CloseHandle(handle)
  self.handles=[]

def same_job(process,job,base):
 command=process.get('CommandLine') or ''
 return MODULE+'.'+job['role'] in command and '--row '+job['row_id'] in command and Path(base).name in command

def assert_no_dispatcher(processes,base,ignore=()):
 for p in processes:
  command=p.get('CommandLine') or ''
  if p['ProcessId'] not in ignore and Path(base).name in command and (MODULE+'.driver' in command or 'local_statgrad_controller_recovery.py' in command):
   raise RuntimeError('An existing dispatcher still owns the scientific run')

def check_pending(rows,base,active):
 """Never silently restart an output that already exists."""
 base=Path(base);result=[]
 for row in rows:
  name=row['row_id'];owned={(j['row_id'],j['role']) for j in active}
  for role in ARTIFACT:
   out=base/name/DIRECTORY[role];artifact=base/name/ARTIFACT[role]
   if out.exists() and not artifact.is_file() and (name,role) not in owned:
    raise RuntimeError('Preserve unowned partial output '+name+'/'+role)
   if artifact.is_file() and (name,role) not in owned:
    end=read(base/(name+'-'+role+'-exit.json'))
    if end['returncode']!=0:raise RuntimeError('Completed artifact has failed worker '+name+'/'+role)
  if not (base/name/ARTIFACT['source']).is_file() and (name,'source') not in owned:result.append(row)
 return result

def run(a):
 control=Path(__file__).resolve().parents[1];base=Path(a.base).resolve();recovery=Path(a.recovery_base).resolve();release=Path(a.worker_release).resolve()
 if os.name!='nt' or Path.cwd().resolve()!=control or recovery.name!=RUN or recovery.parent!=base.parent or base.name!=SCIENCE_RUN or a.commit!=SCIENCE_COMMIT or a.scientific_module!=MODULE:raise ValueError('Fixed control/scientific roots only')
 if (control/'control_commit.txt').read_text().strip()!=a.control_commit or (release/'release_commit.txt').read_text().strip()!=SCIENCE_COMMIT:raise ValueError('Control or original scientific commit differs')
 if recovery.exists() or (base/'control_recovery.json').exists():raise FileExistsError('Preserve recovery and original roots')
 sys.path.insert(0,str(release));sys.path.insert(1,str(release/'code'))
 from experiments.cvs_f_statgrad50 import design as d
 from experiments.cvs_f_statgrad50 import driver as original
 from experiments.cvs_f_crossrx50_local.provenance import source_proof
 from tools.local_phase1_env import child_env
 if d.ROOT.resolve()!=release or read(base/'frozen_matrix.json')!=dict(rows=d.rows(),user_fixed_extension=d.fixed_authorization(),target_feedback=False) or read(base/'source_expansion.json')!=d.fixed_authorization() or read(base/'execution.json')['commit']!=SCIENCE_COMMIT:raise ValueError('Original fixed matrix or scientific identity differs')
 profile=Path(a.profile).resolve();metadata=Path(a.metadata).resolve();reference=Path(a.reference_base).resolve();p=read(profile)
 if p['physical_gpu']!=0 or p['source_data_status']!='READY' or reference.name!=original.REFERENCE_RUN:raise ValueError('Original local data/reference/profile required')
 for seed in d.SEEDS:
  ref=read(reference/('BASE-s'+str(seed))/'source/local_source_complete.json');source_proof(ref['config'])
  if ref['source_commit']!=original.REFERENCE_COMMIT or Path(ref['config']['runtime_profile']).resolve()!=profile or Path(ref['config']['runtime_metadata']).resolve()!=metadata:raise ValueError('Matched reused BASE differs')
 before=read(base/'progress.json')
 if before['status']!='FAILED_DISPATCH_STOPPED' or not before.get('healthy_workers_preserved'):raise ValueError('Only proven stopped dispatcher recovery')
 processes=process_snapshot();by_pid={v['ProcessId']:v for v in processes};ancestors={os.getpid()};parent=by_pid.get(os.getpid(),{}).get('ParentProcessId')
 while parent in by_pid and parent not in ancestors:
  ancestors.add(parent);parent=by_pid[parent]['ParentProcessId']
 assert_no_dispatcher(processes,base,ancestors)
 active=[];observed=[]
 for job in before['active']:
  if job['row_id'] not in {r['row_id'] for r in d.rows()} or job['role'] not in ARTIFACT:raise ValueError('Unexpected adopted row')
  start=read(base/(job['row_id']+'-'+job['role']+'-start.json'))
  if start['pid']!=job['pid'] or start['argv']!=job['argv'] or Path(start['cwd']).resolve()!=release:raise ValueError('Original owned worker record differs')
  process=by_pid.get(job['pid'])
  if process is None:
   evidence=read(a.exit_observation)
   if evidence['status']!='OBSERVED_PROCESS_EXIT' or evidence['pids'][0]!=job['pid'] or evidence['terminated_by_observer'] or any(evidence['returncodes']) or not (base/job['row_id']/ARTIFACT[job['role']]).is_file():raise RuntimeError('Missing live worker or verified exit; never retrain it')
   exitpath=base/(job['row_id']+'-'+job['role']+'-exit.json')
   if not exitpath.exists():exclusive(exitpath,dict(pid=job['pid'],returncode=0,read_only_observation=evidence))
   observed.append(dict(row_id=job['row_id'],role=job['role'],already_exited=True));continue
  if not same_job(process,job,base):raise ValueError('Refuse unrelated PID adoption')
  children=[v['ProcessId'] for v in processes if v['ParentProcessId']==job['pid'] and same_job(v,job,base)]
  pinned=ObservedProcess(job['pid'],[job['pid'],*children])
  active.append(dict(**job,process=pinned));observed.append(dict(row_id=job['row_id'],role=job['role'],pid=job['pid'],backend_pids=children,already_exited=False))
 queue=check_pending(d.rows(),base,active);recovery.mkdir(exist_ok=False)
 exclusive(recovery/'original_failed_progress.json',before)
 exclusive(recovery/'execution.json',dict(run_id=RUN,scientific_run_id=d.RUN,control_commit=a.control_commit,scientific_commit=SCIENCE_COMMIT,controller_pid=os.getpid(),cwd=str(control),worker_cwd=str(release),adopted=observed,fixed_rows=d.rows(),model_training_restarted=False,healthy_workers_terminated=False))
 exclusive(base/'control_recovery.json',dict(recovery_run_id=RUN,recovery_base=str(recovery),scientific_run_id=d.RUN,scientific_commit=SCIENCE_COMMIT,control_commit=a.control_commit,original_failed_progress_preserved=True))
 env=child_env(p);env.update(PYTHONPATH=os.pathsep.join((str(release),str(release/'code'))),TORCHINDUCTOR_CACHE_DIR=str(base/'torch_cache'))
 last_resource=None
 def path(row,role):return base/row['row_id']/ARTIFACT[role]
 def finished(row,role):return path(row,role).is_file() and not any(j['row_id']==row['row_id'] and j['role']==role for j in active)
 def progress(status,**extra):
  atomic_write(recovery/'progress.json',dict(status=status,run_id=RUN,scientific_run_id=d.RUN,scientific_commit=SCIENCE_COMMIT,control_commit=a.control_commit,controller_pid=os.getpid(),max_GPU_workers=3,source_expansion_fixed=True,active=[{k:v for k,v in j.items() if k!='process'} for j in active],last_resource=last_resource,**extra))
 def launch(role,row):
  out=base/row['row_id']/DIRECTORY[role];start=base/(row['row_id']+'-'+role+'-start.json');logfile=base/(row['row_id']+'-'+role+'.log')
  if out.exists() or start.exists() or logfile.exists():raise FileExistsError('Preserve prior role, no repeat '+row['row_id']+'/'+role)
  argv=[p['python'],'-X','utf8','-u','-m',MODULE+'.'+role,'--row',row['row_id'],'--profile',str(profile),'--metadata',str(metadata),'--base',str(base)]
  with logfile.open('x',encoding='utf8') as log:
   child=subprocess.Popen(argv,cwd=release,env=dict(env,CUDA_VISIBLE_DEVICES='' if role=='score' else '0'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
  job=dict(pid=child.pid,role=role,row_id=row['row_id'],cwd=str(release),argv=argv,log=str(logfile));exclusive(start,job);active.append(dict(process=child,**job));print(json.dumps(dict(event='STARTED',**job)),flush=True)
 try:
  progress('RUNNING',adopted_without_restart=observed)
  while True:
   for job in list(active):
    code=job['process'].poll()
    if code is None:continue
    exitpath=base/(job['row_id']+'-'+job['role']+'-exit.json')
    if exitpath.exists():raise FileExistsError('Preserve existing exit record')
    exclusive(exitpath,dict(pid=job['pid'],returncode=code))
    if code or not path(job,job['role']).exists():raise RuntimeError('Preserved failed worker '+job['row_id']+'/'+job['role'])
    active.remove(job)
    if isinstance(job['process'],ObservedProcess):job['process'].close()
   try:
    # Preserve original conservative launcher/child accounting and resource budget.
    gpu,ram=original.resources();own={j['pid'] for j in active if j['role']!='score'};foreign=gpu-own;slots=3-len(own)-len(foreign);last_resource=dict(foreign_GPU_workers=len(foreign),own_GPU_workers=len(own),free_system_GiB=ram)
   except Exception as error:slots=0;last_resource=dict(observation='UNKNOWN',error=repr(error))
   if slots>0 and last_resource.get('free_system_GiB',0)>=4.5:
    pending=next((r for r in d.rows() if finished(r,'source') and not path(r,'predict').exists() and not any(j['row_id']==r['row_id'] and j['role']=='predict' for j in active)),None)
    if pending is not None:launch('predict',pending)
    elif queue:launch('source',queue.pop(0))
   if not any(j['role']=='score' for j in active) and not original.cpu_scorers():
    ready=next((r for r in d.rows() if finished(r,'predict') and not path(r,'score').exists()),None)
    if ready is not None:launch('score',ready)
   if not active and all(finished(r,'score') for r in d.rows()):
    from experiments.cvs_f_statgrad50.analysis import aggregate
    from experiments.cvs_f_statgrad50.report_results import publish
    if not (base/'result.json').exists():aggregate(base,reference)
    if not (base/'analysis_report').exists():publish(base,base/'analysis_report')
    progress('ARTIFACTS_COMPLETE',rows=len(d.rows()));print('RECOVERY_FIXED_MATRIX_COMPLETE',flush=True);return
   progress('RUNNING');time.sleep(3)
 except BaseException as error:
  try:progress('FAILED_DISPATCH_STOPPED',error=repr(error),healthy_workers_preserved=True)
  except Exception as progress_error:print(json.dumps(dict(progress_failure=repr(progress_error))),flush=True)
  raise
 finally:
  for job in active:
   if isinstance(job['process'],ObservedProcess):job['process'].close()

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for key in ('base','recovery-base','worker-release','reference-base','profile','metadata','commit','control-commit','scientific-module','exit-observation'):parser.add_argument('--'+key,required=True)
 run(parser.parse_args())
