"""Launch registered local diagnostic, preserving exclusive stdout and PID evidence."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--method',choices=['issl','lora'],required=True)
p.add_argument('--output',required=True); args=p.parse_args()
out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
if out.exists(): raise FileExistsError(out)
log_path=out.with_suffix('.stdout.log'); launch_path=out.with_suffix('.launch.json')
if log_path.exists() or launch_path.exists(): raise FileExistsError('Launch evidence already exists')
command=[sys.executable,'-X','utf8',str(Path(__file__).with_name('acceptance.py').resolve()),
         '--method',args.method,'--data',str(Path('local_artifacts/newclass_registration_20260930/wisig_diagnostic.npz').resolve()),
         '--output',str(out)]
with log_path.open('x',encoding='utf-8') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=Path.cwd())
    launch_path.write_text(json.dumps({'pid':process.pid,'command':command,'cwd':str(Path.cwd()),'log':str(log_path)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pid':process.pid,'log':str(log_path)}),flush=True)
    result=process.wait()
if result: raise RuntimeError(f'Execution failed ({result}); inspect {log_path}')
# Independent output readback, not exit-code-only success.
resolved=json.loads((out/'resolved_config.json').read_text(encoding='utf-8'))
acceptance=json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
if resolved['pid']!=process.pid or acceptance['status']!='VERIFIED' or not (out/'predictions.npz').exists():
    raise RuntimeError('Post-state evidence mismatch')
print(json.dumps({'status':'VERIFIED','method':args.method,'pid':process.pid,'steps':acceptance['steps'],'output':str(out)}),flush=True)
