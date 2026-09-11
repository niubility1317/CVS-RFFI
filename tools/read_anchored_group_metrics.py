"""Read-only CPU scoring of saved source predictions and frozen V systems."""
import json,subprocess
from pathlib import Path
PAYLOAD=r'''
import sys,json
from pathlib import Path
release=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/core90_anchored_392005_1afa43a7')
sys.path[:0]=[str(release/'code'),str(release/'code/scripts')]
import torch
torch.set_num_threads(4)
from cvsrffi.anchored_cache import CacheIdentity,load_source_cache
from cvsrffi.anchored_pipeline import FrozenAnchoredSystem
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/core90_anchored_s392005_20260911_r1')
def cache(role):
    p=root/'cache'/('joint_'+role)
    return load_source_cache(p,CacheIdentity(**json.loads((p/'manifest.json').read_text())['identity']))
records=[];confusions=[];checks=[]
def score(name,role,rows,pred,accepted=None):
    y=rows.labels;before=rows.baseline_inference_logits.argmax(-1)
    groups=[('all','all',torch.ones(len(rows),dtype=torch.bool))]
    for key,t in [('RX',rows.receiver),('TX',y),('day',rows.day)]:
        groups += [(key,str(int(v)),t==v) for v in t.unique(sorted=True)]
    groups += [('view',v,torch.tensor([z==v for z in rows.view_ids])) for v in sorted(set(rows.view_ids))]
    for group,value,m in groups:
        n=int(m.sum());correct=pred==y;bc=before==y
        rec=dict(candidate=name,role=role,group=group,value=value,rows=n,correct=int((correct&m).sum()),baseline_correct=int((bc&m).sum()),rescue=int((correct&~bc&m).sum()),harm=int((~correct&bc&m).sum()))
        rec['accuracy']=rec['correct']/n;rec['net_correct']=rec['rescue']-rec['harm']
        if accepted is not None:
            a=int((accepted&m).sum());e=int((accepted&~correct&m).sum());rec.update(accepted=a,accepted_errors=e,coverage=a/n,risk=e/a if a else None)
        records.append(rec)
    confusions.append(dict(candidate=name,role=role,matrix=torch.bincount(y*6+pred,minlength=36).reshape(6,6).tolist()))
ls=cache('L_s')
score('H0','head_only_OOF',ls,ls.baseline_inference_logits.argmax(-1))
for name in ['A2','A3','A4','C_angle','C_angle_keep']:
    p=torch.load(root/'oof'/name/'oof_predictions.pt',map_location='cpu',weights_only=True)
    assert p['physical_ids']==list(ls.physical_ids) and p['view_ids']==list(ls.view_ids)
    assert torch.isfinite(p['scores']).all()
    assert torch.equal(p['s0'],ls.baseline_inference_logits)
    score(name,'head_only_OOF',ls,p['scores'].argmax(-1))
    checks.append(dict(candidate=name,complete_rows=len(ls),finite=True,identity_order_match=True))
p=torch.load(root/'candidates/A6/fuse/nested/nested_fusion.pt',map_location='cpu',weights_only=True)
score('A6','nested_head_only_OOF',ls,p['nested_predictions'])
score('A5_inner_selected','nested_head_only_OOF',ls,p['nested_fixed_predictions'])
gate=dict(action_counts=torch.bincount(p['selected_actions'],minlength=5).tolist(),final_fixed_action=p['final_fixed_action'],final_oof_action_utilities=p['final_oof_action_utilities'].tolist(),training_set_count=p['training_set_count'])
v=cache('V')
with torch.no_grad():
    for name in ['A0','A1','A2','A3','A4','C_angle','C_angle_keep','A5-0','A5-25','A5-50','A5-75','A5-100','A6']:
        s=FrozenAnchoredSystem.from_state(torch.load(root/'candidates'/name/'export/frozen_system.pt',map_location='cpu',weights_only=True))
        pred=s.predict_features(v.h,v.baseline_inference_logits,v.quality,v.valid)
        score(name,'source_V_calibration_descriptive',v,pred['top_class'],pred['accepted'])
print(json.dumps(dict(records=records,confusions=confusions,checks=checks,gate=gate,target_access=False,fit_performed=False)))
'''
if __name__=='__main__':
    r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=PAYLOAD.encode(),capture_output=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    d=json.loads(r.stdout);p=Path('analysis/core90_anchored_results_20260912/group_metrics.json')
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
    print(json.dumps(dict(groups=len(d['records']),checks=d['checks'],gate=d['gate'])))
