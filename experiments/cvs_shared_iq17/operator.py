"""One shared,label-free local direction;no training loss,optimizer or new model.

Every reference observation contributes to one [1,2,256] additive vector.
The source question transfers that same vector to every TX in a held RX/day.
A source sensitivity result is not evidence of identity preservation or DG.
"""
from contextlib import nullcontext
import math
import numpy as np
import torch
from torch.nn import functional as F
EPS=1e-3
XI=1e-4
PEAK=3e-3

def complex_rms(z):return z.double().square().sum(dim=-2).mean(dim=-1).sqrt()
def initial_probe(seed,device):
 z=np.random.default_rng(seed).choice((-1.,1.),size=(1,2,256));v=torch.tensor(z.tolist(),dtype=torch.float32,device=device);v=v-v.mean(-1,keepdim=True);r=complex_rms(v)
 if not bool((r>0).all()):raise ValueError('Nonzero private synthetic probe required')
 return (v.double()*(XI/r).view(-1,1,1)).float()
def bound(g):
 if g.shape!=(1,2,256) or not bool(torch.isfinite(g).all()):raise ValueError('Finite one shared IQ direction required')
 z=g.double()-g.double().mean(-1,keepdim=True);r=float(complex_rms(z)[0]);peak=float(z.abs().max())
 if r<=1e-24:return torch.zeros_like(g,dtype=torch.float32),dict(identifiable=False,reason='zero_or_unresolvable_search_gradient',gradient_RMS=r)
 v=(z*min(EPS/r,PEAK/peak)).float()
 return v,dict(identifiable=True,gradient_RMS=r,direction_RMS=float(complex_rms(v)[0]),direction_peak=float(v.abs().max()),removed_DC=True)
def permuted_control(delta,seed):
 if delta.shape!=(1,2,256) or not bool(torch.isfinite(delta).all()):raise ValueError('One finite shared direction required')
 rng=np.random.default_rng(seed);v=torch.empty_like(delta)
 for channel in range(2):v[0,channel]=delta[0,channel].index_select(0,torch.tensor(rng.permutation(256).tolist(),device=delta.device,dtype=torch.long))
 return v

def make_direction(model,batches,forward_logits,seed,cache_context=None):
 """Two reference eval forwards and one input autograd.grad per reference batch.

The probability reference is the same current frozen model,no EMA or teacher
copy. No labels,RX,day,quality or decisions enter this function. Mode flags are
restored exactly;autograd.grad does not populate parameter .grad fields.
"""
 modes=[(m,m.training) for m in model.modules()];count=forwards=backwards=0;g=None;probe=None;kl_sum=0.
 def logits(x):
  with (cache_context() if cache_context is not None else nullcontext()):z=forward_logits(x)
  if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite original six-class default logits required')
  return z
 try:
  model.eval()
  for x in batches:
   if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('Finite native FP32 IQ batch [N,2,256] required')
   if probe is None:probe=initial_probe(seed,x.device);g=torch.zeros_like(probe,dtype=torch.float64)
   if x.device!=probe.device:raise ValueError('One fixed model/input device required')
   with torch.no_grad():logp=F.log_softmax(logits(x).double(),dim=1);prob=logp.exp()
   with torch.enable_grad():
    d=probe.detach().clone().requires_grad_(True);logq=F.log_softmax(logits(x+d).double(),dim=1);loss=(prob*(logp-logq)).sum()
    grad,=torch.autograd.grad(loss,d,only_inputs=True)
   if not bool(torch.isfinite(grad).all()) or not math.isfinite(float(loss.detach())):raise ValueError('Finite input derivative required')
   g+=grad.detach().double();count+=len(x);forwards+=2*len(x);backwards+=len(x);kl_sum+=float(loss.detach())
  if not count:raise ValueError('Nonempty legal reference observations required')
  direction,evidence=bound(g/count)
  evidence.update(reference_packets=count,eval_forward_packets=forwards,input_backward_packets=backwards,probe_KL_mean=kl_sum/count,parameter_optimizer_updates=0,labels_consumed=False,teacher_copies=0,new_training_loss_terms=0,one_shared_direction=True)
  return direction,evidence
 finally:
  for module,flag in modes:module.training=flag
