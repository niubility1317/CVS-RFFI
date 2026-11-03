"""Wait for both fixed physical models and independent audits before paired replay."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261103-phase1-iterative-scale-fullsource-s2026092701-r01'
RELEASE=ROOT/'releases'/(RUN+'-release')
OUT=ROOT/'runs'/(RUN+'-dispatcher')
CONTROL=ROOT/'runs'/(RUN+'-controller')
UPSTREAM='20261103-phase1-iterative-large-source-s2026092701-r01'
ORIGINAL='20261103-phase1-iterative-timefreq-source-s2026092701-r01'
ORIGINAL_AUDIT='20261103-phase1-iterative-timefreq-source-independent-audit-s2026092701-r01'
ALIGNED_AUDIT='20261103-phase1-iterative-large-source-independent-audit-s2026092701-r01'
UPSTREAM_DISPATCH_PID=2062493

def status(state,**kw):
 p=OUT/'status.tmp';p.write_text(json.dumps(dict(state=state,**kw)),encoding='utf8');os.replace(p,OUT/'status.json')

def alive(pid,token):
 try:return token in (Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
 except (FileNotFoundError,ProcessLookupError):return False

def read(path):return json.loads(path.read_text()) if path.exists() else {}

def registered_processes(info):
 if info.get('state')=='FAILED':raise RuntimeError('Large-source experiment dispatcher failed: '+str(info.get('error')))
 if info.get('state')!='LAUNCHED':return None
 pids=info.get('pids',{});source=pids.get('source');audit=pids.get('source_audit')
 if type(source) is not int or type(audit) is not int or source<=0 or audit<=0 or source==audit:
  raise ValueError('Distinct registered source and auditor PIDs required')
 return source,audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);a=ap.parse_args()
 try:
  if (RELEASE/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Exact fixed release required')
  while True:
   upstream_path=ROOT/'runs'/(UPSTREAM+'-dispatcher')/'status.json';upstream=read(upstream_path)
   pids=registered_processes(upstream)
   if pids is not None:break
   if not alive(UPSTREAM_DISPATCH_PID,UPSTREAM+'-dispatcher'):
    upstream=read(upstream_path);pids=registered_processes(upstream)
    if pids is not None:break
    raise RuntimeError('Large-source dispatcher missing before launch')
   status('WAITING_UPSTREAM_LAUNCH',upstream=upstream.get('state'));time.sleep(30)
  items=((ORIGINAL,2050189,ORIGINAL_AUDIT,2050190,'phase1_iterative_timefreq.source_controller'),
         (UPSTREAM,pids[0],ALIGNED_AUDIT,pids[1],'phase1_iterative_large.source_controller'))
  while True:
   waiting=[]
   for name,pid,audit,apid,token in items:
    path=ROOT/'runs'/(name+'-controller')/'status.json';source=read(path)
    v=read(ROOT/'runs'/audit/'status.json')
    if source.get('state')=='FAILED':raise RuntimeError('Source failed: '+name)
    if source.get('state')!='SOURCE_SCORE_COMPLETE':
     if not alive(pid,token) and read(path).get('state')!='SOURCE_SCORE_COMPLETE':raise RuntimeError('Source controller missing: '+name)
     waiting.append(name+':source')
    if v.get('state')=='FAILED':raise RuntimeError('Source audit failed: '+name)
    if v.get('state')=='COMPLETE':
     if v.get('results',{}).get(name,{}).get('state')!='PASS':raise RuntimeError('Source audit not passed: '+name)
    else:
     if not alive(apid,'source-independent-audit') and read(ROOT/'runs'/audit/'status.json').get('state')!='COMPLETE':raise RuntimeError('Source auditor missing: '+name)
     waiting.append(name+':audit')
   if not waiting:break
   status('WAITING_FIXED_AUDITED_SOURCES',waiting=waiting);time.sleep(30)
  while os.getloadavg()[0]>72:status('WAITING_CPU_CAPACITY',phase='tests',load=os.getloadavg()[0]);time.sleep(30)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
   PYTHONPATH=':'.join(map(str,(RELEASE,ROOT/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',
        ROOT/'releases/pamf_stage2_inference_20261021_r01',ROOT/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'))))
  status('RUNNING_TESTS')
  with (OUT/'tests.log').open('x') as log:
   result=subprocess.run([sys.executable,'-m','unittest','experiments.phase1_iterative_scale_fullsource.test_logic',
                          'experiments.phase1_iterative_scale_fullsource.test_fit',
                          'experiments.phase1_iterative_scale_fullsource.test_dispatcher'],cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('Physical range pair tests failed')
  while os.getloadavg()[0]>72:status('WAITING_CPU_CAPACITY',phase='launch',load=os.getloadavg()[0]);time.sleep(30)
  if CONTROL.exists() or (ROOT/'runs'/RUN).exists():raise FileExistsError('Unique source pair paths required')
  CONTROL.mkdir()
  with (OUT/'controller.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-u','-m','experiments.phase1_iterative_scale_fullsource.controller','--commit',a.commit],cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status('LAUNCHED',pid=child.pid,load=os.getloadavg()[0])
 except Exception as exc:
  status('FAILED',error=str(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
