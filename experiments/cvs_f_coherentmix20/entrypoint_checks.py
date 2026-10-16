"""Direct production script entry: invalid argv must reject before backend/data."""
import json,subprocess,sys
from pathlib import Path

def check():
 root=Path(__file__).resolve().parents[2]
 p=subprocess.run([sys.executable,str(root/'experiments/cvs_f_coherentmix20/runner.py'),'--invalid-entry-test'],cwd=root,text=True,capture_output=True,timeout=20)
 if p.returncode==0 or 'No data/budget/recipe overrides permitted' not in p.stderr or 'partially initialized' in p.stderr or 'namedtuple' in p.stderr:
  raise AssertionError('Direct-script entry failed before intended argv guard: '+p.stderr[-3000:])
 return dict(status='PASS',direct_file_entry=True,intended_invalid_argument_guard=True,backend_or_data_open=False,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(check()))
