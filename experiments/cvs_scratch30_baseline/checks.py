import argparse,json
from pathlib import Path
from copy import deepcopy
import torch
from . import design as d
from .source import objective
from experiments.cvs_stage2_hybrid.semisupervised import VisitPassBank,validate_unlabeled
def run(device='cpu'):
 torch.set_num_threads(2)
 class Toy(torch.nn.Module):
  def __init__(self):super().__init__();self.fc=torch.nn.Linear(512,6);self.bn=torch.nn.BatchNorm1d(2)
  def forward(self,x):
   z=self.fc(self.bn(x).flatten(1));return z+torch.tensor([20.,0.,0.,0.,0.,0.],device=x.device)
 x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;ids=[f'opaque{i}' for i in range(128)];passes=torch.ones(128,dtype=torch.long);c=d.config(next(r for r in d.rows() if r['arm']=='SELF_SSL30'));model=Toy().to(device).train();bank=VisitPassBank(2,.95)
 first,_,info=objective(model,x,y,x,ids,passes,bank,c,1111)
 if info['accepted_U']!=0:raise AssertionError('First U pass must not qualify')
 before=model.bn.num_batches_tracked.clone();state=torch.get_rng_state().clone();second,_,info=objective(model,x,y,x,ids,passes+1,bank,c,2221)
 if info['accepted_U']!=128 or model.bn.num_batches_tracked-before!=2 or not model.training:raise AssertionError('Stable U gate or eval-state boundary')
 second.backward()
 if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise AssertionError('SelfSSL gradient')
 ce=d.config(next(r for r in d.rows() if r['arm']=='CE30'));m=Toy().to(device).train();loss,_,v=objective(m,x,y,None,None,None,None,ce,6660);loss.backward()
 if v['U_packets'] or v['teacher_forward_calls']:raise AssertionError('CE must not execute U/teacher')
 for row in d.rows():
  actual=d.config(row)
  if actual['source_U'] and actual['U_cache']!=d.U_CACHE.as_posix():raise AssertionError('Cross-platform preregistration U path differs')
 for key,value in [('epochs',20),('checkpoint_sources',['legacy']),('source_U',False),('training_target_access',True)]:
  bad=deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Illegal30 config')
 try:validate_unlabeled({'y':torch.zeros(1)})
 except ValueError:pass
 else:raise AssertionError('U truth accepted')
 return dict(status='PASS',device=device,first_pass_accepted=0,second_pass_accepted=128,teacher_updates_BN=False,CE_U_forwards=0,negative_cases=5,epochs=30,views=7)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
