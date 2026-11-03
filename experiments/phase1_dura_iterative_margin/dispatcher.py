
from pathlib import Path
import os,sys,time,json,subprocess,traceback,argparse
r=Path('/home/szu2070436088/2510044040/CV-SincNet');run='20261103-phase1-dura-iterative-margin-source-s2026092701-r01';ground='20261103-phase1-dura-iterative-margin-ground-test-s2026092701-r01'
release=r/'releases'/(run+'-release');parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);commit=parser.parse_args().commit
dispatch=r/'runs'/(run+'-dispatcher');source_ctl=r/'runs'/(run+'-controller');ground_ctl=r/'runs'/(ground+'-controller')
auto=r/'releases/20261102-phase1-posterior-nonlocal-s2026092701-r02-release/experiments/phase1_ground_auto/auto_score.py'
entry=release/'experiments/phase1_dura_iterative_margin/ground_score_entry.py'
def status(state,**kw):
 tmp=dispatch/'status.tmp';tmp.write_text(json.dumps(dict(state=state,**kw)),encoding='utf8');os.replace(tmp,dispatch/'status.json')
pids={}
try:
 while True:
  load=os.getloadavg()[0]
  if load<=72:break
  status('WAITING_CPU_LOAD',load=load);time.sleep(30)
 paths=[source_ctl,ground_ctl,r/'runs'/run,r/'runs'/(run+'-training-smoke'),r/'runs'/ground,r/'runs'/(ground+'-auto-audit'),r/'runs'/(run+'-launcher.log'),r/'runs'/(ground+'-launcher.log'),r/'runs'/(ground+'-auto-audit-launcher.log')]
 if any(p.exists() for p in paths) or (r/'runs'/'20261103-phase1-dura-iterative-margin-source-independent-audit-s2026092701-r01').exists():raise ValueError('Unique run paths required')
 if not auto.is_file() or not entry.is_file() or (release/'release_commit.txt').read_text().strip()!=commit:raise ValueError('Exact release or score interface missing')
 while os.getloadavg()[0]>72:status('WAITING_CPU_LOAD',load=os.getloadavg()[0]);time.sleep(30)
 env=os.environ.copy();env.update(PYTHONPATH=':'.join(map(str,[release,r/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',r/'releases/pamf_stage2_inference_20261021_r01',r/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'])),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 while os.getloadavg()[0]>72:
  status('WAITING_CPU_LOAD_FOR_TEST',load=os.getloadavg()[0]);time.sleep(30)
 status('RUNNING_UNIT_TESTS')
 with (dispatch/'tests.log').open('x') as log:
  test=subprocess.run([sys.executable,'-m','unittest','experiments.phase1_dura_iterative_margin.test_objective','experiments.phase1_dura_iterative_margin.test_integration','experiments.phase1_dura_iterative_timefreq.test_integration','experiments.phase1_iterative_timefreq.test_model','experiments.phase1_iterative_timefreq.test_adapter','experiments.phase1_dura_volterra.test_pipeline'],cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 if test.returncode:raise RuntimeError('Model and adapter tests failed')
 while os.getloadavg()[0]>72:
  status('WAITING_CPU_LOAD_AFTER_TEST',load=os.getloadavg()[0]);time.sleep(30)
 source_ctl.mkdir();ground_ctl.mkdir()
 def start(args,name):
  f=(r/'runs'/name).open('x');return subprocess.Popen(args,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 s=start([sys.executable,'-X','utf8','-u','-m','experiments.phase1_dura_iterative_margin.source_controller','--commit',commit],run+'-launcher.log');pids['source']=s.pid
 audit_dir=r/'runs'/'20261103-phase1-dura-iterative-margin-source-independent-audit-s2026092701-r01'
 audit_dir.mkdir()
 watcher=(r/'runs'/'20261103-phase1-dura-physical-wide-source-independent-audit-s2026092701-r01'/'watcher.py').read_text()
 watcher=watcher.replace('20261103-phase1-dura-physical-wide-source-s2026092701-r01',run).replace('20261103-phase1-dura-physical-wide-source-independent-audit-s2026092701-r01','20261103-phase1-dura-iterative-margin-source-independent-audit-s2026092701-r01').replace('2015695',str(s.pid)).replace('phase1_dura_physical_wide.source_controller','phase1_dura_iterative_margin.source_controller')
 (audit_dir/'watcher.py').write_text(watcher,encoding='utf8')
 sa=start([sys.executable,'-X','utf8','-u',str(audit_dir/'watcher.py')],'20261103-phase1-dura-iterative-margin-source-independent-audit-s2026092701-r01'+'-launcher.log');pids['source_audit']=sa.pid
 g=start([sys.executable,'-X','utf8','-u','-m','experiments.phase1_dura_iterative_margin.ground_controller','--commit',commit,'--source-pid',str(s.pid)],ground+'-launcher.log');pids['ground']=g.pid
 a=start([sys.executable,'-X','utf8','-u',str(auto),'--root',str(r),'--run',ground,'--commit',commit,'--entry',str(entry),'--pid',str(g.pid),'--process-token','phase1_dura_iterative_margin.ground_controller','--upstream-status',str(ground_ctl/'status.json')],ground+'-auto-audit-launcher.log');pids['audit']=a.pid
 status('LAUNCHED',pids=pids,load=os.getloadavg()[0])
except Exception as e:
 status('FAILED',pids=pids,error=str(e),traceback=traceback.format_exc());raise

