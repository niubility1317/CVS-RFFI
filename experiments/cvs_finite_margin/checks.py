import argparse,json
from pathlib import Path
from copy import deepcopy
import torch
from . import design as d
from .objective import robust_loss,objective

def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(32);scores=torch.randn(5,12,6,device=device,requires_grad=True);y=torch.arange(12,device=device)%6
 loss,m=robust_loss(scores,y);worst=torch.stack([torch.nn.functional.cross_entropy(scores[i],y,reduction='none') for i in range(5)]).max(0).values.mean()
 if loss+1e-6<worst:raise AssertionError('Finite margin loss must upper-bound all actual-view CE')
 p=torch.tensor([2,0,5,1,4,3],device=device);inv=p.argsort();other,_=robust_loss(scores[:,:,p],inv[y])
 if not torch.allclose(loss,other,atol=1e-6):raise AssertionError('Class permutation changes finite margin rule')
 loss.backward()
 if scores.grad is None or not torch.isfinite(scores.grad).all():raise AssertionError('Actual margin gradient missing')
 identical=scores.detach()[0:1].repeat(5,1,1);same,_=robust_loss(identical,y)
 if not torch.allclose(same,torch.nn.functional.cross_entropy(identical[0],y)):raise AssertionError('Zero-action limit is CE')
 class Toy(torch.nn.Module):
  def __init__(self):super().__init__();self.bn=torch.nn.BatchNorm1d(2);self.fc=torch.nn.Linear(512,6)
  def forward(self,x):return self.fc(self.bn(x).flatten(1))
 x=torch.randn(128,2,256,device=device);ys=torch.arange(128,device=device)%6
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));toy=Toy().to(device).train();before=toy.bn.num_batches_tracked.clone();value,_=objective(toy,x,ys,c);value.backward()
  if not toy.training or toy.bn.num_batches_tracked-before!=1:raise AssertionError('Extra branch updated normalization or lost train mode')
  if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in toy.parameters()):raise AssertionError('Native objective path gradient')
 for key,value in [('target_access',True),('source_U',True),('epochs',21),('checkpoint_sources',['unknown'])]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Illegal input config')
 return dict(status='PASS',device=device,finite_loss=float(loss),max_actual_view_CE=float(worst),permutation_invariant=True,identical_view_limit_CE=True,extra_BN_updates=0,negative_config_cases=4)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
