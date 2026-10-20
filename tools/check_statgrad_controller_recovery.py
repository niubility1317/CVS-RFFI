"""Focused Windows regression checks. Fixtures only; no IQ, truth, models or CUDA."""
import argparse,ctypes,json,os,subprocess,sys,tempfile,threading,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.local_statgrad_controller_recovery import atomic_write,ObservedProcess,check_pending,assert_no_dispatcher,MODULE

def run(workspace):
 parent=Path(workspace).resolve();root=Path(tempfile.mkdtemp(prefix='recovery_fixture_',dir=parent)).resolve();assert root.parent==parent
 files=[];directories=[]
 try:
  p=root/'progress.json';p.write_text('{"old":true}',encoding='utf8')
  k=ctypes.WinDLL('kernel32',use_last_error=True)
  from ctypes import wintypes as W
  k.CreateFileW.argtypes=[W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE];k.CreateFileW.restype=W.HANDLE
  k.CloseHandle.argtypes=[W.HANDLE];k.CloseHandle.restype=W.BOOL
  h=k.CreateFileW(str(p),0x80000000,3,None,3,0x80,None)
  if h==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
  denied=threading.Event();original=Path.replace;attempts=[]
  def observed_replace(source,target):
   attempts.append(1)
   try:return original(source,target)
   except PermissionError:denied.set();raise
  def close_reader():
   denied.wait(5);time.sleep(.05);k.CloseHandle(h)
  closer=threading.Thread(target=close_reader);closer.start()
  with patch.object(Path,'replace',observed_replace):atomic_write(p,dict(new=True))
  closer.join(5);assert denied.is_set() and len(attempts)>=2 and json.loads(p.read_text())==dict(new=True)
  with patch.object(Path,'replace',side_effect=ctypes.WinError(5)) as blocked:
   try:atomic_write(p,dict(never=True),attempts=3,delay=0)
   except PermissionError:pass
   else:raise AssertionError('Permanent denial silently ignored')
   assert blocked.call_count==3 and json.loads(p.read_text())==dict(new=True)
  with patch.object(Path,'replace',side_effect=ctypes.WinError(87)) as other:
   try:atomic_write(p,dict(never=True))
   except OSError:pass
   else:raise AssertionError('Unexpected filesystem error ignored')
   assert other.call_count==1
  fixture=subprocess.Popen([sys._base_executable,'-c','import time,sys;time.sleep(.5);sys.exit(7)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
  observed=ObservedProcess(fixture.pid,[fixture.pid]);assert observed.poll() is None
  until=time.monotonic()+5
  while observed.poll() is None and time.monotonic()<until:time.sleep(.02)
  assert observed.poll()==fixture.wait(timeout=5)==7;observed.close()
  rows=[dict(row_id=v) for v in ('A','B','C')]
  src=root/'A/source';src.mkdir(parents=True);(src/'local_source_complete.json').write_text('{}',encoding='utf8');(root/'A-source-exit.json').write_text('{"returncode":0}',encoding='utf8')
  (root/'B/source').mkdir(parents=True)
  active=[dict(row_id='B',role='source')]
  assert check_pending(rows,root,active)==[dict(row_id='C')]
  (root/'C/source').mkdir(parents=True)
  try:check_pending(rows,root,active)
  except RuntimeError:pass
  else:raise AssertionError('Unowned partial output could be retrained')
  dispatcher=[dict(ProcessId=9,CommandLine='python -m '+MODULE+'.driver --base '+str(root))]
  try:assert_no_dispatcher(dispatcher,root)
  except RuntimeError:pass
  else:raise AssertionError('Live dispatcher duplicated')
  assert_no_dispatcher(dispatcher,root,{9})
  from experiments.cvs_f_statgrad50.audit import check
  from tools.local_statgrad_controller_recovery import SCIENCE_RUN,SCIENCE_COMMIT,RUN
  scientific=root/SCIENCE_RUN;scientific.mkdir();control=root/RUN;control.mkdir()
  link=dict(scientific_run_id=SCIENCE_RUN,scientific_commit=SCIENCE_COMMIT,recovery_base=str(control),recovery_run_id=RUN,control_commit='control-fixture')
  (scientific/'control_recovery.json').write_text(json.dumps(link),encoding='utf8')
  (scientific/'progress.json').write_text(json.dumps(dict(status='FAILED_DISPATCH_STOPPED',active=[])),encoding='utf8')
  (control/'execution.json').write_text(json.dumps(dict(scientific_commit=SCIENCE_COMMIT,scientific_run_id=SCIENCE_RUN,control_commit='control-fixture',model_training_restarted=False,healthy_workers_terminated=False)),encoding='utf8')
  (control/'progress.json').write_text(json.dumps(dict(status='RUNNING',active=[])),encoding='utf8')
  audit=check(scientific,root/'20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01',SCIENCE_COMMIT)
  assert audit['status']=='WAIT_FIXED_MATRIX_RUNNING' and audit['checkpoint_loads']==audit['IQ_reads']==audit['truth_reads']==0
  link['scientific_commit']='wrong';(scientific/'control_recovery.json').write_text(json.dumps(link),encoding='utf8')
  try:check(scientific,root/'20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01',SCIENCE_COMMIT)
  except AssertionError:pass
  else:raise AssertionError('Foreign scientific recovery cursor accepted')
  return dict(status='PASS',recovered_progress_cursor_and_wrong_scientific_identity_guard_verified=True,real_Windows_reader_denial_retried=True,permanent_denial_bounded_and_target_preserved=True,other_errors_not_swallowed=True,read_only_real_exit_code7_verified=True,completed_and_live_source_not_queued_again=True,unowned_partial_output_rejected=True,existing_dispatcher_rejected=True,production_outputs_touched=False,IQ_reads=0,truth_reads=0,model_forwards=0,CUDA_contexts=0)
 finally:
  # Only this resolved, newly created fixture is removed, deepest paths first.
  assert root.parent==parent
  for v in sorted(root.rglob('*'),key=lambda x:len(x.parts),reverse=True):
   if v.is_dir():v.rmdir()
   else:v.unlink()
  root.rmdir()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);args=parser.parse_args();print(json.dumps(run(args.workspace)))
