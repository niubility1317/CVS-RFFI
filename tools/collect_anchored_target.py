"""Collect completed target evidence and independently recount fixed predictions."""
import json,subprocess
from pathlib import Path
PAYLOAD=r'''
import json,hashlib,time
from pathlib import Path
import torch
torch.set_num_threads(4)
b=Path('/home/szu2070436088/2510044040/CV-SincNet');name='core90_anchored_target_all_s392005_20260912_r1';r=b/'runs'/name
assert (r/'completed.json').exists()
files={str(p.relative_to(r)):p.read_text() for p in r.rglob('*') if p.suffix in {'.json','.csv'} and p.is_file()}
logs={p.name:p.read_text() for p in (b/'logs'/name).glob('*.log')};logs['coordinator.log']=(b/'logs'/(name+'.coordinator.log')).read_text()
contract=json.loads((b/'runs/core90_evidence_frozen_392005_20260911/data_contract.json').read_text())
truth={hashlib.sha256(k.encode()).hexdigest():int(k.split(':')[0]) for k in contract['roles']['target']}
reported=json.loads(files['scores.json'])['records'];paired=[]
for scene in ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'):
    base=torch.load(r/scene/'A0.pt',map_location='cpu',weights_only=True);ids=[bytes(x.tolist()).hex() for x in base['ids']];y=torch.tensor([truth[x] for x in ids]);bc=base['top_class']==y
    for p in sorted((r/scene).glob('*.pt')):
        d=torch.load(p,map_location='cpu',weights_only=True);assert torch.equal(d['ids'],base['ids'])
        c=d['top_class']==y;a=d['accepted'];n=len(y)
        old=next(x for x in reported if x['candidate']==p.stem and x['scene']==scene and x['group']=='all')
        assert old['correct']==int(c.sum()) and old['accepted']==int(a.sum()) and old['accepted_correct']==int((c&a).sum())
        paired.append(dict(candidate=p.stem,scene=scene,rows=n,rescue=int((c&~bc).sum()),harm=int((~c&bc).sum()),net=int(c.sum()-bc.sum()),changed=int((d['top_class']!=base['top_class']).sum())))
print(json.dumps(dict(files=files,logs=logs,paired=paired,recount_verified=True,time=time.time(),tx_mapping=contract['tx_mapping'])))
'''
if __name__=='__main__':
    r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=PAYLOAD.encode(),capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    d=json.loads(r.stdout);out=Path('analysis/core90_anchored_target_all_20260912');out.mkdir(parents=True,exist_ok=True)
    for name,s in d.pop('files').items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:f.write(s)
    for name,s in d.pop('logs').items():
        p=out/'logs'/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:f.write(s)
    with (out/'independent_recount.json').open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
    print(json.dumps(dict(paired=len(d['paired']),verified=d['recount_verified'])))
