"""Wait for fixed independent source model and audit, then start source pair replay."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261103-phase1-projector-multiscale-fullsource-s2026092701-r01'
RELEASE=ROOT/'releases'/(RUN+'-release')
OUT=ROOT/'runs'/(RUN+'-dispatcher')
CONTROL=ROOT/'runs'/(RUN+'-controller')
SOURCE='20261103-phase1-frozen-feature-projector-source-s2026092701-r01'
AUDIT=ROOT/'runs/20261103-phase1-frozen-feature-projector-source-independent-audit-s2026092701-r01'

def status(state,**kw):
 p=OUT/'status.tmp';p.write_text(json.dumps(dict(state=state,**kw)),encoding='utf8');os.replace(p,OUT/'status.json')

def alive(pid,token):
 p=Path('/proc')/str(pid)/'cmdline'
 return p.exists() and token in p.read_bytes().decode(errors='replace')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--source-pid',type=int,required=True);ap.add_argument('--audit-pid',type=int,required=True);a=ap.parse_args()
 try:
  if (RELEASE/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Exact fixed release required')
  while True:
   p=ROOT/'runs'/(SOURCE+'-controller')/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
   q=AUDIT/'status.json';v=json.loads(q.read_text()) if q.exists() else {}
   if s.get('state')=='FAILED':raise RuntimeError('Projector source failed')
   if s.get('state')!='SOURCE_SCORE_COMPLETE' and not alive(a.source_pid,'phase1_frozen_feature_projector.source_controller'):
    raise RuntimeError('Projector source controller missing before scoring')
   if v.get('state')=='FAILED':raise RuntimeError('Projector independent source audit failed')
   if v.get('state')=='COMPLETE':
    if v.get('results',{}).get(SOURCE,{}).get('state')!='PASS':raise RuntimeError('Projector independent source audit did not pass')
    if s.get('state')=='SOURCE_SCORE_COMPLETE':break
   elif not alive(a.audit_pid,'source-independent-audit'):raise RuntimeError('Projector source auditor missing before completion')
   status('WAITING_FIXED_AUDITED_SOURCE',source=s.get('state'),audit=v.get('state'));time.sleep(30)
  while os.getloadavg()[0]>72:status('WAITING_CPU_CAPACITY',phase='tests',load=os.getloadavg()[0]);time.sleep(30)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
   PYTHONPATH=':'.join(map(str,(RELEASE,ROOT/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',
        ROOT/'releases/pamf_stage2_inference_20261021_r01',ROOT/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'))))
  status('RUNNING_TESTS')
  with (OUT/'tests.log').open('x') as log:
   result=subprocess.run([sys.executable,'-m','unittest','experiments.phase1_projector_multiscale_fullsource.test_logic',
                          'experiments.phase1_projector_multiscale_fullsource.test_factory'],cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('Source pair tests failed')
  while os.getloadavg()[0]>72:status('WAITING_CPU_CAPACITY',phase='launch',load=os.getloadavg()[0]);time.sleep(30)
  if CONTROL.exists() or (ROOT/'runs'/RUN).exists():raise FileExistsError('Unique source pair paths required')
  CONTROL.mkdir()
  with (OUT/'controller.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-u','-m','experiments.phase1_projector_multiscale_fullsource.controller','--commit',a.commit],cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status('LAUNCHED',pid=child.pid,load=os.getloadavg()[0])
 except Exception as exc:
  status('FAILED',error=str(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
