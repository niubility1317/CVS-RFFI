"""Local max3 GPU workers across owners, original50 source freeze then clean."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d

def replace(p,value):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)

def resources():
 # NVML is unavailable on this host; authoritative live Python+CIM GPU counters.
 script="$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | Select-Object ProcessId,CommandLine); $g = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction Stop | Select-Object Name,DedicatedUsage); $o = Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory; [pscustomobject]@{processes=$p;gpu=$g;memory=$o} | ConvertTo-Json -Depth 4 -Compress"
 result=subprocess.run([os.environ.get('SystemRoot','C:/Windows')+'/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',script],capture_output=True,text=True,encoding='utf-8',check=True,timeout=10)
 values=json.loads(result.stdout);live={int(p['ProcessId']) for p in values['processes']};gpu={int(g['Name'].split('_')[1]) for g in values['gpu'] if '_phys_0' in g['Name'] and int(g['DedicatedUsage'])>=64*1024**2};return gpu&live,int(values['memory']['FreePhysicalMemory'])/1024**2

def run(a):
 base=Path(a.base).resolve();profile=Path(a.profile).resolve();metadata=Path(a.metadata).resolve()
 if os.name!='nt' or base.name!=d.RUN or d.ROOT.resolve()!=Path.cwd().resolve() or (d.ROOT/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Use fixed local immutable release/run')
 p=d.read(profile)
 if p['physical_gpu']!=0 or p['source_data_status']!='READY' or a.max_workers!=3:raise ValueError('Verified local GPU0, max3 including other owners required')
 base.mkdir(parents=True,exist_ok=False);active=[];queue=d.rows(d.SEEDS[:1]);initial=list(queue);all_rows=list(queue);frozen=False;last_resource=None
 d.write(base/'execution.json',dict(run_id=d.RUN,commit=a.commit,pid=os.getpid(),cwd=str(d.ROOT),max_GPU_workers=3,global_other_owners_counted=True,satellite_training=False,test_views=['clean'],source_recipe='original50=33L+17U/oldgate-ownEMA-cos/RDrop1/defaultheadjoint'))
 d.write(base/'registered_rows.json',dict(initial=initial,conditional_remaining=d.rows(d.SEEDS[1:]),source_rule='Fixed weak2>=+0.2pp,overall>=-0.2pp,worstRX>=-0.5pp,anyTX>=-1pp before any target'))
 env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(d.ROOT),str(d.ROOT/'code'))),TORCHINDUCTOR_CACHE_DIR=str(base/'torch_cache'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def path(row,role):return base/row['row_id']/{'source':'source/local_source_complete.json','predict':'prediction/complete.json','score':'scores/metrics.json'}[role]
 def finished(row,role):return path(row,role).is_file() and not any(j['row_id']==row['row_id'] and j['role']==role for j in active)
 def progress(status,**extra):
  replace(base/'progress.json',dict(status=status,controller_pid=os.getpid(),max_GPU_workers=3,active=[{k:v for k,v in j.items() if k!='process'} for j in active],source_expansion_fixed=frozen,last_resource=last_resource,**extra))
 def launch(role,row):
  argv=[sys.executable,'-X','utf8','-u','-m','experiments.cvs_f_crossrx50_local.'+role,'--row',row['row_id'],'--profile',str(profile),'--metadata',str(metadata),'--base',str(base)]
  c=d.config(row,str(base),str(profile),str(metadata));out=base/row['row_id']/{'source':'source','predict':'prediction','score':'scores'}[role]
  if out.exists():raise FileExistsError('Preserve existing output; no automatic restart '+str(out))
  logfile=base/(row['row_id']+'-'+role+'.log')
  with logfile.open('x',encoding='utf-8') as log:child=subprocess.Popen(argv,cwd=d.ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='' if role=='score' else '0'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
  evidence=dict(pid=child.pid,role=role,row_id=row['row_id'],cwd=str(d.ROOT),argv=argv,log=str(logfile));d.write(base/(row['row_id']+'-'+role+'-start.json'),evidence);active.append(dict(process=child,**evidence));print(json.dumps(dict(event='STARTED',**evidence)),flush=True)
 try:
  while True:
   for job in list(active):
    code=job['process'].poll()
    if code is None:continue
    active.remove(job);d.write(base/(job['row_id']+'-'+job['role']+'-exit.json'),dict(pid=job['pid'],returncode=code));row=next(r for r in all_rows if r['row_id']==job['row_id'])
    if code or not path(row,job['role']).exists():raise RuntimeError('Preserved '+job['role']+' failure '+job['row_id']+' exit '+str(code))
   if not frozen and all(finished(row,'source') for row in initial):
    source={r['treatment']:d.read(path(r,'source'))['sourceV'] for r in initial};decision=d.expansion(source);d.write(base/'source_expansion.json',decision)
    remaining=[r for r in d.rows(d.SEEDS[1:]) if r['treatment'] in decision['expand_arms']];all_rows=initial+remaining;queue.extend(remaining);d.write(base/'frozen_matrix.json',dict(rows=all_rows,source_only_decision=decision,target_feedback=False));frozen=True;print('SOURCE_EXPANSION_FIXED '+json.dumps(decision),flush=True)
   # New GPU jobs are limited by actual foreign GPU PIDs, plus our own live
   # workers including imports/preCUDA. Unknown resource observation => wait.
   try:
    live_gpu,free_GiB=resources();own={j['pid'] for j in active if j['role']!='score'};foreign=live_gpu-own;slots=3-len(own)-len(foreign);last_resource=dict(foreign_GPU_workers=len(foreign),own_GPU_workers=len(own),free_system_GiB=free_GiB)
   except Exception as error:slots=0;last_resource=dict(observation='UNKNOWN',error=repr(error))
   if slots>0 and last_resource.get('free_system_GiB',0)>=4.5:
    if queue:launch('source',queue.pop(0))
    elif frozen:
     pending=next((r for r in all_rows if finished(r,'source') and not path(r,'predict').exists() and not any(j['row_id']==r['row_id'] and j['role']=='predict' for j in active)),None)
     if pending is not None:launch('predict',pending)
   if frozen and not any(j['role']=='score' for j in active):
    ready=next((r for r in all_rows if finished(r,'predict') and not path(r,'score').exists()),None)
    if ready is not None:launch('score',ready)
   if frozen and not active and all(finished(r,'score') for r in all_rows):
    from .analysis import aggregate
    aggregate(base);progress('ARTIFACTS_COMPLETE',rows=len(all_rows));print('ORIGINAL50_WEAK_TX_COMPLETE',flush=True);return
   progress('RUNNING');time.sleep(3)
 except Exception as error:
  progress('FAILED_DISPATCH_STOPPED',error=repr(error),healthy_workers_preserved=True);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--metadata',required=True);p.add_argument('--base',required=True);p.add_argument('--commit',required=True);p.add_argument('--max-workers',type=int,default=3);run(p.parse_args())
