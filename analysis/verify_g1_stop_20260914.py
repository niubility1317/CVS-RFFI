"""Independent read-only post-state check after the scoped stop."""
import json,subprocess
from pathlib import Path
source=r'''
import datetime,json,subprocess
from pathlib import Path
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/a1_mechanism_periodic_s392005_20260910_r1')
f=root/'G1_FISHER_GATE';receipt=json.loads((f/'user_stop_20260914.json').read_text())
state=json.loads((root/'pipeline_state.json').read_text())
survivors=[]
for item in receipt['owned']:
 p=Path('/proc')/str(item['pid'])/'stat'
 if p.exists():
  parts=p.read_text().rsplit(')',1)[1].split()
  if parts[19]==item['start'] and parts[0]!='Z':survivors.append(item['pid'])
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
gpu_pids={int(s.split(',')[1]) for s in gpu.splitlines() if ',' in s}
owned={x['pid'] for x in receipt['owned']}
dp=Path('/proc')/str(state['pid'])/'cmdline'
out={'at':datetime.datetime.now().astimezone().isoformat(),'survivors':survivors,'owned_cuda_pids':sorted(owned&gpu_pids),
 'dispatcher_alive':dp.exists() and bool(dp.read_bytes()),'state':state,'gpu_processes':gpu,
 'last_completed_epoch':len((f/'metrics_epoch.jsonl').read_text().splitlines()),
 'test_epochs':[p.parent.name for p in sorted(f.glob('target_epochs/E*/score.json'))],
 'checkpoint_files':[p.name for p in f.glob('*.pth')],'stop_receipt':receipt}
assert not survivors and not out['owned_cuda_pids']
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=source,text=True,encoding='utf-8',capture_output=True,check=True,timeout=30)
d=json.loads(r.stdout);out=Path(__file__).parent/'g1_user_stop_20260914'
(out/'verification.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:d[k] for k in ['at','survivors','owned_cuda_pids','dispatcher_alive','last_completed_epoch','test_epochs','checkpoint_files']}))
