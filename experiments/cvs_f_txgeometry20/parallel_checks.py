"""Focused queue isolation tests; no IQ, truth or GPU access."""
import importlib.util,json,tempfile,subprocess,sys,time
from pathlib import Path
p=Path(__file__).with_name('parallel.py');sp=importlib.util.spec_from_file_location('parallel_queue',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);initial=[{'row_id':'A'},{'row_id':'B'}];remaining=[{'row_id':'C'}]
 def finish(row,role):
  path=root/row/{'source':'source/completion.json','predict':'prediction/complete.json','score':'scores/metrics.json'}[role];path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
 assert m.candidates(initial,remaining,False,root,[])==[('source','A'),('source','B')]
 assert m.candidates(initial,remaining,False,root,[{'row_id':'A','role':'source'}])==[('source','B')]
 finish('A','source');assert m.candidates(initial,remaining,False,root,[])==[('source','B')]
 finish('B','source');assert m.candidates(initial,remaining,False,root,[])==[]
 assert m.candidates(initial,remaining,True,root,[])==[('source','C'),('predict','A'),('predict','B')]
 finish('A','predict');assert m.candidates(initial,remaining,True,root,[{'row_id':'B','role':'predict'}])==[('source','C')]
 assert m.command('python','score','A','profile','base','commit')==['python','-X','utf8','-u','-m','experiments.cvs_f_txgeometry20.score','--row','A','--profile','profile','--base','base']
 # Stable Windows handle and quoted argv without touching a real experiment.
 argv=[sys.executable,'-c','import time; time.sleep(30)']
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
 handle=m.ProcessHandle(child.pid,terminate=True)
 try:
  assert handle.poll() is None
  proc=next(x for x in m.inventory() if x['ProcessId']==child.pid)
  assert m.split_windows(proc['CommandLine'])==argv
  handle.stop_controller();assert handle.poll()==0;assert child.wait(timeout=5)==0
 finally:handle.close()
 print(json.dumps(dict(status='PASS',source_gate=True,adopted_row_not_duplicated=True,completed_row_not_retrained=True,source_expansion_before_prediction=True,registered_remaining_only=True,Windows_handle=True,quoted_argv=True,IQ_access=False,truth_access=False,GPU_used=False)))
