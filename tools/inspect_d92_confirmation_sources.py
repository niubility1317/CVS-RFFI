"""Read actual source initialization/completion and available inference capacity."""
import json
from pathlib import Path
import subprocess
from read_d92_run import FLAGS

REMOTE=r'''
import json,subprocess,shutil
from pathlib import Path
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
rows=[]
for seed in range(2026092701,2026092705):
 root=base/'runs/20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'/f'cvs-daot-rc4-s{seed}'
 rows.append(dict(seed=seed,initialization=json.loads((root/'initialization.json').read_text()),completion=json.loads((root/'completion.json').read_text())))
print(json.dumps(dict(rows=rows,disk_free=shutil.disk_usage(base).free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],text=True))))
'''

result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=REMOTE.encode(),capture_output=True,check=True)
data=json.loads(result.stdout)
path=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/confirmation_source_readback.json')
path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
for row in data['rows']:
    row['initialization'].pop('argv',None)
print(json.dumps(data,indent=2))
