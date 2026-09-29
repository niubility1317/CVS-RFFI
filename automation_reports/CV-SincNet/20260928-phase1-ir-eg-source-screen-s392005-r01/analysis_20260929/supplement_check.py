import subprocess,json
from pathlib import Path
source=r'''
import json,copy,time,statistics,sys
from pathlib import Path
base=Path('/home/szu2070436088/2510044040/CV-SincNet');run=base/'runs/20260928-phase1-ir-eg-source-screen-s392005-r01'
sys.path.insert(0,str(base/'releases/ir_eg_source_screen_20260928_r01/code'))
import torch
torch.set_num_threads(1)
payload=torch.load(run/'IR_s392005/latest_ssdg.pth',map_location='cpu')
history=payload['daot_rc4']['route_history'];times=[]
for _ in range(7):
 t=time.perf_counter();v=copy.deepcopy(history);times.append(time.perf_counter()-t)
out=dict(cpu_only=True,training_started=False,benchmark=dict(operation='deepcopy actual IR route_history on N607 CPU',entries=len(history),seconds=times,median_seconds=statistics.median(times),caveat='isolated Python copy only; not a full training profile; shared CPU load'))
out['panels']={}
for name in ['IR_s392005','IR_G0_s392005','OR_EG_s392005','IR_ENCODER_OFF_s392005']:
 with (run/name/'ir_steps.jsonl').open() as f:
  for line in f:
   try:r=json.loads(line)
   except ValueError:continue
   h=r.get('heavy_diagnostics')
   if h:
    entries=h['source_identity']['rx_tx'];out['panels'][name]=dict(epoch=r['epoch'],records=h['source_panel']['records'],TX=sorted({x['tx'] for x in entries}),RX=sorted({x['rx'] for x in entries}),cells=entries,policy=h['source_panel']['policy']);break
print(json.dumps(out,indent=2))
'''
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=source,text=True,encoding='utf-8',capture_output=True,check=True)
p=Path('E:/type10-7/code/worktrees/ir_eg_v1_20260927/automation_reports/CV-SincNet/20260928-phase1-ir-eg-source-screen-s392005-r01/analysis_20260929/supplement.json');p.write_text(r.stdout,encoding='utf-8');print(r.stdout)
