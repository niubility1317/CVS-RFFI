"""Mathematical oracles for exact forward and full-statistic/size-only gradients."""
import torch
from .loss import standardize,penalty
from experiments.cvs_f_supcon50.mechanism import penalty as native

def run(device='cpu'):
 torch.manual_seed(7419);results=[]
 for dtype in (torch.float64,torch.float32):
  for scale in (1.,1e-7,0.):
   x=(torch.randn(18,8,device=device,dtype=dtype)*scale+2).requires_grad_();up=torch.randn_like(x);mean=x.mean(0);sigma=x.std(0,unbiased=False).clamp_min(1e-4)
   reference=(x-mean)/sigma;native_z=(x-mean.detach())/sigma.detach();full_info={};norm_info={}
   full=standardize(x,'STAT_GRAD',full_info);norm=standardize(x,'NORM_ONLY',norm_info)
   assert torch.equal(full,native_z) and torch.equal(norm,native_z)
   gf=torch.autograd.grad((full*up).sum(),x,retain_graph=True)[0];gn=torch.autograd.grad((norm*up).sum(),x,retain_graph=True)[0];truth=torch.autograd.grad((reference*up).sum(),x,retain_graph=True)[0];old=torch.autograd.grad((native_z*up).sum(),x)[0]
   tol=1e-9 if dtype==torch.float64 else 3e-3
   assert torch.allclose(gf,truth,atol=tol,rtol=tol) and torch.allclose(gn.norm(),gf.norm(),atol=tol,rtol=tol)
   assert torch.allclose(gn,old*(gf.norm()/old.norm()),atol=tol,rtol=tol)
   assert gf.sum(0).abs().max()<=tol*max(1.,float(gf.norm()))
   if scale==1.:assert (gf*(x-x.mean(0))).sum(0).abs().max()<=tol*max(1.,float(gf.norm()))
   assert full_info['backward_calls']==norm_info['backward_calls']==1
   results.append(dict(dtype=str(dtype),scale=scale,full_reference=True,size_control=True))
 x=torch.randn(5,3,dtype=torch.float64,device=device,requires_grad=True)
 assert torch.autograd.gradcheck(lambda v:standardize(v,'STAT_GRAD',{}),(x,),eps=1e-6,atol=1e-5)
 y=torch.tensor([0,1,2,3,4,5]*3,device=device);rx=torch.tensor([1]*6+[3]*6+[4]*6,device=device);maps={k:torch.randn(18,8,12,device=device,requires_grad=True) for k in ('time_down','t2.pool')};rng=torch.get_rng_state().clone()
 old,_=native(maps,y,rx,'CROSS_RX')
 for mode in ('STAT_GRAD','NORM_ONLY'):
  loss,detail=penalty(maps,y,rx,mode);assert torch.equal(loss,old);g=torch.autograd.grad(loss,list(maps.values()),retain_graph=True);assert all(torch.isfinite(t).all() and t.norm()>0 for t in g) and all(v['backward']['backward_calls']==1 for v in detail)
 assert torch.equal(rng,torch.get_rng_state())
 negatives=0
 for yy,rr,mode in [(y-6,rx,'STAT_GRAD'),(y.float(),rx,'STAT_GRAD'),(y,rx*0,'NORM_ONLY'),(y,rx,'MEM4')]:
  try:penalty(maps,yy,rr,mode)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid role/labels/mode accepted')
 return dict(status='PASS',device=device,forward_exact_native=True,true_statistic_gradient_autograd_reference=True,translation_and_unclamped_scale_orthogonal=True,size_control_norm_matches_same_descriptor_full_gradient=True,clamped_and_constant_channels=True,gradcheck=True,private_RNG_unchanged=True,protocol_negative_cases=negatives,oracles=results)
if __name__=='__main__':
 import json;print(json.dumps(run()))
