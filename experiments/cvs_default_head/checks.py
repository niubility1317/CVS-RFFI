import copy,json
from types import SimpleNamespace
import torch
from torch import nn
from .predict import logits,READOUTS
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
class Fixture(nn.Module):
 def __init__(self):
  super().__init__();self.feature=nn.Linear(256,160);self.head=ResidualPhysicalHead(160)
  self.id_backbone=SimpleNamespace(encoder=SimpleNamespace(core=SimpleNamespace(id_backbone=SimpleNamespace(cls_head=self.head))))
 def forward_identity_only(self,x):return dict(tx_logits=self.head.classify(self.feature(x[:,0])))
def checks():
 torch.set_num_threads(2);torch.manual_seed(73019);m=Fixture().eval().requires_grad_(False);x=torch.randn(4,2,256);before={k:v.clone() for k,v in m.state_dict().items()}
 value=logits(m,x)
 with torch.no_grad():expected=m.forward_identity_only(x)['tx_logits']
 assert torch.equal(value,expected) and value.requires_grad is False and READOUTS==('ORIGINAL_HEAD',)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
 assert all(p.grad is None for p in m.parameters())
 assert torch.allclose(logits(m,x[:1]),value[:1],atol=2e-5,rtol=2e-5)
 negative=0
 for bad in [copy.deepcopy(m).train(),copy.deepcopy(m).requires_grad_(True)]:
  try:logits(bad,x)
  except ValueError:negative+=1
  else:raise AssertionError('Unfrozen joint model accepted')
 for q in [x.double(),x.clone().requires_grad_(True),torch.zeros(0,2,256),torch.zeros(4,2,128)]:
  try:logits(m,q)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid inference tensor accepted')
 bad=x.clone();bad[0,0,0]=float('nan')
 try:logits(m,bad)
 except ValueError:negative+=1
 else:raise AssertionError('Nonfinite input accepted')
 bad=copy.deepcopy(m);bad.head.scale=20.
 try:logits(bad,x)
 except ValueError:negative+=1
 else:raise AssertionError('Modified classifier accepted')
 return dict(status='PASS',default_classifier_exact=True,model_parameters_unchanged=True,optimizer_steps=0,classifier_only_updates=0,feature_updates=0,alternative_readouts=0,negative_cases=negative,synthetic_only=True)
if __name__=='__main__':print(json.dumps(checks()))
