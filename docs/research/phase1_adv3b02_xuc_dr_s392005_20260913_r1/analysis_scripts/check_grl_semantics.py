import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('experiments/adv3b02_xuc/code').resolve()))
import torch
from cvsrffi.game_tracking.legacy.model_dual_cvsincnet import grad_reverse
torch.manual_seed(392005)
z=torch.randn(8,160,requires_grad=True)
head=torch.nn.Linear(160,15)
labels=torch.arange(8)
result={}
for value in [0.,1.]:
    loss=.16*torch.nn.functional.cross_entropy(head(grad_reverse(z,value)),labels)
    dz,dh=torch.autograd.grad(loss,[z,head.weight])
    result[str(value)]=dict(loss=float(loss),identity_gradient_norm=float(dz.norm()),head_gradient_norm=float(dh.norm()))
assert result['0.0']['loss']==result['1.0']['loss']
assert result['0.0']['identity_gradient_norm']==0 and result['1.0']['identity_gradient_norm']>0
assert result['0.0']['head_gradient_norm']==result['1.0']['head_gradient_norm']>0
out=dict(status='PASS',synthetic_source_only=True,scope='isolated U adversarial CE; not a target performance attribution',results=result)
p=Path('E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_xuc_dr_s392005_20260913_r1/grl_semantics_check.json')
p.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
