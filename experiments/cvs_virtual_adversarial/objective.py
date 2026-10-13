from contextlib import contextmanager
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_feature_challenge.objective import captured,layer_info
from experiments.cvs_e20_cross_style.style import _paths
from experiments.cvs_finite_margin.objective import deterministic

def packet_norm(t):return t.flatten(1).double().norm(dim=1)
def unit(t):
 n=packet_norm(t);shape=[len(t)]+[1]*(t.ndim-1)
 return (t.double()/n.clamp_min(1e-300).reshape(shape)).to(t.dtype)
def scaled(direction,center,ratio):
 shape=[len(center)]+[1]*(center.ndim-1)
 return direction*(ratio*packet_norm(center)).to(center.dtype).reshape(shape)
def divergence(logits,q):
 lp=logits.double().log_softmax(1)
 return (q*(q.clamp_min(1e-300).log()-lp)).sum(1).mean()
@contextmanager
def perturbed(model,level,delta):
 modules,paths=_paths(model.base);calls=[]
 def hook(module,inputs,out):
  if calls or out.shape!=delta.shape:raise ValueError('VAT feature routing differs')
  calls.append(tuple(out.shape));return out+delta
 h=modules[paths[level]].register_forward_hook(hook)
 try:
  yield
  if len(calls)!=1:raise ValueError('VAT feature was bypassed')
 finally:h.remove()

def consistency(model,ux,c,step):
 if not model.training or not c['source_U'] or step<=1110 or ux.ndim!=3 or ux.shape[1:]!=(2,256):raise ValueError('VAT expects legal U training batch')
 feature=c['arm']=='VAT_FEATURE';level=d.LEVELS[(step-1)%2] if feature else None
 devices=[ux.device.index or 0] if ux.is_cuda else []
 with torch.random.fork_rng(devices=devices),deterministic(model):
  torch.manual_seed((c['model_seed']+step*1979)%2**63)
  with torch.no_grad():
   if feature:
    with captured(model,level) as (v,path):weak=model(ux)
    center=v['feature'].detach()
   else:weak=model(ux);center=ux.detach()
   q=weak.double().softmax(1).detach()
  generator=torch.Generator(device=ux.device).manual_seed((c['model_seed']+step*9901)%2**63)
  original=unit(torch.randn(center.shape,device=center.device,dtype=center.dtype,generator=generator))
  probe=scaled(original,center,.001).detach().requires_grad_(True)
  if feature:
   with perturbed(model,level,probe):search_logits=model(ux)
  else:search_logits=model(ux+probe)
  search=divergence(search_logits,q);g=torch.autograd.grad(search,probe,create_graph=False,retain_graph=False)[0].detach()
  norms=packet_norm(g);fallback=norms==0
  direction=unit(g);direction=torch.where(fallback.reshape([len(center)]+[1]*(center.ndim-1)),original,direction)
  if c['arm']=='RANDOM_INPUT':direction=original
  delta=scaled(direction,center,.02).detach()
  if feature:
   with perturbed(model,level,delta):final_logits=model(ux)
  else:final_logits=model(ux+delta)
  kl=divergence(final_logits,q)
  if not torch.isfinite(kl) or not torch.isfinite(g).all():raise FloatingPointError('VAT gradient/consistency')
  ratio=packet_norm(delta)/packet_norm(center).clamp_min(1e-300)
  detail=dict(U_packets=len(ux),accepted_U=len(ux),teacher_forward_calls=1,search_forwards=1,search_grad_calls=1,consistency_forwards=1,U_confidence=float(q.max(1).values.mean()),U_entropy=float(-(q*q.clamp_min(1e-300).log()).sum(1).mean()),search_KL=float(search.detach()),power_gradient_norm=float(norms.mean()),zero_direction_fallbacks=int(fallback.sum()),perturbation_ratio_mean=float(ratio.mean()),perturbation_ratio_max=float(ratio.max()),layer_index=d.LEVELS.index(level) if feature else -1)
 return kl,detail

def objective(model,x,y,ux,ids,passes,bank,c,step):
 if bank is not None or (ux is not None and (not c['source_U'] or not isinstance(ids,list) or len(ids)!=len(ux))):raise ValueError('VAT U interface/role differs')
 if not model.training or y.dtype!=torch.long or y.shape!=(len(x),) or y.min()<0 or y.max()>=6:raise ValueError('Legal L CE only')
 logits=model(x);ce=F.cross_entropy(logits,y);kl=ce.detach()*0
 detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),U_packets=0,accepted_U=0,teacher_forward_calls=0,search_forwards=0,search_grad_calls=0,consistency_forwards=0,U_confidence=0.,U_entropy=0.,search_KL=0.,power_gradient_norm=0.,zero_direction_fallbacks=0,perturbation_ratio_mean=0.,perturbation_ratio_max=0.,layer_index=-1,ramp=0.)
 if ux is not None:
  kl,extra=consistency(model,ux,c,step);detail.update(extra);detail['ramp']=min(1.,(step-1110)/1110)
 return ce+.25*detail['ramp']*kl,dict(CE=float(ce.detach()),KL=float(kl.detach())),detail
