"""Wait for both fixed physical models and independent audits before paired replay."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261103-phase1-physical-range-fullsource-s2026092701-r01'
RELEASE=ROOT/'releases'/(RUN+'-release')
OUT=ROOT/'runs'/(RUN+'-dispatcher')
CONTROL=ROOT/'runs'/(RUN+'-controller')
ITEMS=(
 ('20261103-phase1-dura-physical-fusion-source-s2026092701-r02',2001041,
  '20261103-phase1-dura-physical-fusion-source-independent-audit-s2026092701-r02',2001227,'phase1_dura_physical_fusion.source_controller'),
 ('20261103-phase1-dura-physical-wide-source-s2026092701-r01',2015695,
  '20261103-phase1-dura-physical-wide-source-independent-audit-s2026092701-r01',2016342,'phase1_dura_physical_wide.source_controller'))

def status(state,**kw):
 p=OUT/'status.tmp';p.write_text(json.dumps(dict(state=state,**kw)),encoding='utf8');os.replace(p,OUT/'status.json')

def alive(pid,token):
 try:return token in (Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
 except (FileNotFoundError,ProcessLookupError):return False

def read(path):return json.loads(path.read_text()) if path.exists() else {}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);a=ap.parse_args()
 try:
  if (RELEASE/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Exact fixed release required')
  while True:
   waiting=[]
   for name,pid,audit,apid,token in ITEMS:
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
   result=subprocess.run([sys.executable,'-m','unittest','experiments.phase1_physical_range_fullsource.test_logic',
                          'experiments.phase1_physical_range_fullsource.test_factory'],cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('Physical range pair tests failed')
  while os.getloadavg()[0]>72:status('WAITING_CPU_CAPACITY',phase='launch',load=os.getloadavg()[0]);time.sleep(30)
  if CONTROL.exists() or (ROOT/'runs'/RUN).exists():raise FileExistsError('Unique source pair paths required')
  CONTROL.mkdir()
  with (OUT/'controller.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-u','-m','experiments.phase1_physical_range_fullsource.controller','--commit',a.commit],cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status('LAUNCHED',pid=child.pid,load=os.getloadavg()[0])
 except Exception as exc:
  status('FAILED',error=str(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
