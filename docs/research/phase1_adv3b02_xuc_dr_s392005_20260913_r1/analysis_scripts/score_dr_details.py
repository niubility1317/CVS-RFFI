import json,subprocess
from pathlib import Path
CODE=r'''
import hashlib,hmac,json,time
from pathlib import Path
from collections import defaultdict
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
base=p/'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'
truth=json.loads((base/'target_truth/truth_sidecar.json').read_text())
t={r['sample_id']:r['label'] for r in truth['records']}
key=hashlib.sha256(('cvs.phase1.truth-last|'+truth['split_binding']).encode()).digest()
meta={}
for tx in range(6):
 for rx in [0,2,5,7,9,10,11]:
  for day in range(4):
   for sig in range(1000):
    physical=f'tx{tx}:rx{rx}:day{day}:eq1:sig{sig}'
    sid='sample:'+hmac.new(key,physical.encode(),hashlib.sha256).hexdigest()
    meta[sid]=(tx,rx,day)
assert set(meta)==set(t),'physical metadata reconstruction must exactly match every sealed ID'
assert all(t[sid]==v[0] for sid,v in meta.items())
out=dict(metadata_mapping_exact=True,records=len(t),rows={})
roots=['phase1_adv3b02_xuc_dr_s392005_20260913_r1','phase1_adv3b02_xuc15_s392005_20260913_r1']
for rid in roots:
 root=p/'runs'/rid;state=json.loads((root/'pipeline_state.json').read_text())
 for row in state['rows']:
  f=root/row/'target_prediction';prediction=json.loads((f/'predictions.json').read_text());score=json.loads((f/'score.json').read_text())
  counts=defaultdict(lambda:[0,0]);conf=defaultdict(lambda:[[0]*6 for _ in range(6)]);seen=set()
  for rec in prediction['records']:
   sid=rec['sample_id'];scene=rec['scenario'];pred=rec['predicted_class'];tx,rx,day=meta[sid]
   assert (scene,sid) not in seen;seen.add((scene,sid))
   ok=int(pred==tx);conf[scene][tx][pred]+=1
   for scope,k in [('scene','all'),('rx',str(rx)),('day',str(day)),('tx',str(tx)),('rx_day',f'{rx}:{day}'),('rx_tx',f'{rx}:{tx}')]:
    c=counts[(scene,scope,k)];c[0]+=ok;c[1]+=1
  assert len(seen)==672000
  for scene,m in score['metrics'].items():assert counts[(scene,'scene','all')]==[m['correct'],m['total']]
  out['rows'][row]=dict(record_count=len(seen),source_run=rid,confusion=dict(conf),details=[dict(scene=s,scope=sc,group=g,correct=v[0],total=v[1],accuracy=v[0]/v[1]) for (s,sc,g),v in sorted(counts.items())])
  del prediction,seen
out['created']=time.time()
print(json.dumps(out))
'''
compile(CODE,'score_details','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=CODE.encode(),capture_output=True,check=True)
d=json.loads(r.stdout)
p=Path('E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_xuc_dr_s392005_20260913_r1/receiver_class_details.json')
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(rows=len(d['rows']),metadata_mapping_exact=d['metadata_mapping_exact'],predictions=sum(v['record_count'] for v in d['rows'].values()))))
