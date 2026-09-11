"""Short read-only SSH polls; stop when target artifacts complete or process exits."""
import json
from pathlib import Path
import subprocess
import time

REMOTE=r'''
import json,time
from pathlib import Path
root=Path('/home/szu2070436088/2510044040/CV-SincNet');run='core90_game_v2_target10_20260912_r1'
out=root/'runs'/run;log=root/'logs'/run/'eval.stdout.log'
text=log.read_text();lines=text.splitlines()
row=out/'V2_A_seed392005_predictions.jsonl'
with row.open('rb') as f:
 f.seek(max(0,row.stat().st_size-4096));tail=f.read().decode(errors='replace').splitlines()
last=None
for line in reversed(tail):
 try:last=json.loads(line);break
 except ValueError:pass
print(json.dumps(dict(time=time.time(),alive=Path('/proc/2493804').exists(),complete=(out/'complete.json').exists(),
 prediction_bytes=row.stat().st_size,last_scene=last.get('scene') if last else None,log_tail=lines[-2:])))
'''

def main():
    while True:
        r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607',
            '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=25)
        if r.returncode:
            print(json.dumps(dict(state='UNKNOWN_READ_FAILURE',error=r.stderr)),flush=True);return
        state=json.loads(r.stdout);print(json.dumps(state),flush=True)
        if state['complete'] or not state['alive']:return
        time.sleep(55)

if __name__=='__main__':main()
