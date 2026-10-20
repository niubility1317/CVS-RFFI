"""Partial derivative with live mean and stopped current standard deviation."""
from .loss import standardize,penalty
import torch
from experiments.cvs_f_supcon50.mechanism import penalty as native
def run(device='cpu'):
 torch.manual_seed(7419);oracles=[]
 for dtype in (torch.float64,torch.float32):
  for scale in (1.,1e-7,0.):
   x=(torch.randn(18,8,device=device,dtype=dtype)*scale+2).requires_grad_();up=torch.randn_like(x);sigma=x.detach().std(0,unbiased=False).clamp_min(1e-4)
   reference=(x-x.mean(0))/sigma;old=(x-x.detach().mean(0))/sigma;info={};z=standardize(x,'CENTER_GRAD',info)
   assert torch.equal(z,old)
   gc=torch.autograd.grad((z*up).sum(),x,retain_graph=True)[0];truth=torch.autograd.grad((reference*up).sum(),x,retain_graph=True)[0];gn=torch.autograd.grad((old*up).sum(),x)[0];tol=1e-9 if dtype==torch.float64 else 3e-3
   assert torch.allclose(gc,truth,atol=tol,rtol=tol) and torch.allclose(gc,gn-gn.mean(0),atol=tol,rtol=tol)
   assert gc.sum(0).abs().max()<=tol*max(1.,float(gc.norm()))
   if scale==1.:
    response=(gc*(x-x.mean(0))).sum(0);before=(gn*(x-x.mean(0))).sum(0);assert torch.allclose(response,before,atol=tol,rtol=tol) and response.norm()>1e-8
   assert info['backward_calls']==1 and info['scale_removed_norm']==0
   oracles.append(dict(dtype=str(dtype),scale=scale,live_mean_stopped_std_reference=True))
 y=torch.tensor([0,1,2,3,4,5]*3,device=device);rx=torch.tensor([1]*6+[3]*6+[4]*6,device=device);maps={k:torch.randn(18,8,12,device=device,requires_grad=True) for k in ('time_down','t2.pool')};rng=torch.get_rng_state().clone();old,_=native(maps,y,rx,'CROSS_RX');value,detail=penalty(maps,y,rx,'CENTER_GRAD');assert torch.equal(value,old);grad=torch.autograd.grad(value,list(maps.values()));assert all(torch.isfinite(g).all() and g.norm()>0 for g in grad) and all(v['backward']['backward_calls']==1 for v in detail) and torch.equal(rng,torch.get_rng_state())
 negatives=0
 for yy,rr,mode in [(y-6,rx,'CENTER_GRAD'),(y.float(),rx,'CENTER_GRAD'),(y,rx*0,'CENTER_GRAD'),(y,rx,'STAT_GRAD')]:
  try:penalty(maps,yy,rr,mode)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid source/mode accepted')
 return dict(status='PASS',device=device,forward_loss_exact_native=True,partial_mean_derivative_reference=True,translation_gradient_removed=True,original_scale_response_retained=True,clamped_constant_channels_finite=True,private_RNG_unchanged=True,protocol_negative_cases=negatives,oracles=oracles,limitation='Partial derivative with current std stopped; no claim of total derivative/finite-difference gradcheck when current std is recomputed.')
if __name__=='__main__':
 import json;print(json.dumps(run()))
