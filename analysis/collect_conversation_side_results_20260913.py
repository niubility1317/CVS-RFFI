import json,subprocess
from pathlib import Path
source=r'''
import json
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
f=p/'runs/tweak_config2_portability_20260909_v5/official_config2_full/results.json'
print(json.dumps({'path':str(f),'result':json.loads(f.read_text())}))
'''
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=source,text=True,encoding='utf-8',capture_output=True,check=True,timeout=30)
d=json.loads(r.stdout)
dest=Path(__file__).parent/'all_exploration_20260913/lora_v5_live.json'
with dest.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
print(d['result']['checkpoint'],len(d['result']['training']['epoch_rows']))
