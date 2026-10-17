"""Focused local algebra/autograd checks;no IQ/checkpoint/training experiment."""
import json
import numpy as np
import torch
from torch import nn
from .operator import EPS,XI,PEAK,complex_rms,initial_probe,bound,permuted_control,make_direction

def run():
 torch.set_num_threads(2);negative=0;packets=0;backwards=0
 def reject(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1
  else:raise AssertionError('Invalid shared IQ direction accepted')
 class Toy(nn.Module):
  def __init__(self):
   super().__init__();self.conv=nn.Conv1d(2,4,3,padding=1);self.bn=nn.BatchNorm1d(4);self.drop=nn.Dropout(.3);self.head=nn.Linear(4,6)
  def forward(self,x):return self.head(self.drop(self.bn(self.conv(x))).square().mean(-1))
 torch.manual_seed(202610178101);m=Toy().train();m.bn.eval();x=torch.randn(16,2,256,generator=torch.Generator().manual_seed(202610178102));x=x/complex_rms(x).float().view(-1,1,1)
 state={k:v.clone() for k,v in m.state_dict().items()};flags=[z.training for z in m.modules()];rng=torch.get_rng_state().clone();first=next(m.parameters());first.grad=torch.ones_like(first);grad0=first.grad.clone()
 d,e=make_direction(m,[x],m,202610178001);packets+=e['eval_forward_packets'];backwards+=e['input_backward_packets']
 assert e['identifiable'] and e['one_shared_direction'] and e['eval_forward_packets']==32 and e['input_backward_packets']==16 and not e['labels_consumed'] and e['new_training_loss_terms']==0 and torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()) and flags==[z.training for z in m.modules()] and torch.equal(first.grad,grad0) and all(p.grad is None for p in list(m.parameters())[1:])
 assert float(complex_rms(d)[0])<=EPS*(1+1e-6) and float(d.abs().max())<=PEAK*(1+1e-6) and float(d.mean(-1).abs().max())<1e-9
 c=permuted_control(d,202610178002);assert torch.allclose(complex_rms(c),complex_rms(d),rtol=1e-7,atol=1e-12) and torch.equal(c.abs().max(),d.abs().max()) and torch.allclose(c.mean(-1),d.mean(-1),atol=1e-10) and not torch.equal(c,d)
 d2,e2=make_direction(m,[x[:3],x[3:9],x[9:]],m,202610178001);packets+=e2['eval_forward_packets'];backwards+=e2['input_backward_packets'];assert torch.allclose(d,d2,rtol=2e-4,atol=2e-8)
 z,ez=bound(torch.zeros(1,2,256));assert not ez['identifiable'] and torch.count_nonzero(z)==0
 z,ez=bound(torch.ones(1,2,256));assert not ez['identifiable'] and torch.count_nonzero(z)==0
 for bad in (torch.zeros(2,2,256),torch.zeros(1,2,255),torch.full((1,2,256),float('nan'))):reject(lambda bad=bad:bound(bad));reject(lambda bad=bad:permuted_control(bad,1))
 for bad in (torch.zeros(2,256),torch.zeros(0,2,256),torch.zeros(2,2,255),torch.zeros(2,2,256,dtype=torch.float64),torch.full((2,2,256),float('nan'))):reject(lambda bad=bad:make_direction(m,[bad],m,1))
 reject(lambda:make_direction(m,[],m,1));reject(lambda:make_direction(m,[x],m,1,labels=torch.zeros(16)))
 for bad in (lambda v:torch.zeros(len(v),7),lambda v:torch.full((len(v),6),float('nan'))):reject(lambda bad=bad:make_direction(m,[x],bad,1))
 assert flags==[z.training for z in m.modules()] and torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
 return dict(status='PASS',negative_checks=negative,synthetic_forward_packets=packets,synthetic_input_backward_packets=backwards,optimizer_updates=0,real_IQ=0,checkpoint_loads=0,target_access=False,common_direction_bound_and_same_energy_control=True,mode_RNG_parameters_and_existing_grad_preserved=True,batch_partition_invariant=True,zero_gradient_is_scientific_degeneracy_no_random_fallback=True)
if __name__=='__main__':print(json.dumps(run()))
