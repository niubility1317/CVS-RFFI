"""Detached source directions; differentiable real-feature symmetric readout."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

RX=(1,3,4,6,8)

def statistics(h):
 if h.ndim!=3 or not h.is_floating_point() or not torch.isfinite(h).all():raise ValueError('Finite real source map required')
 mu=h.mean(-1);sigma=(h.var(-1,unbiased=False)+1e-6).sqrt()
 scale=sigma.square().mean(0).sqrt().clamp_min(1e-6)
 return torch.cat([mu/scale,sigma.log()],1),scale

def direction(h,y,rx,day,mode,role='L_s'):
 if role!='L_s' or mode not in ('ZERO','RAW_MATCHED','IDSAFE'):raise ValueError('Only fixed L-source tangent modes permitted')
 if len(h)!=60 or len(y)!=60 or len(rx)!=60 or len(day)!=60 or set(y.tolist())!=set(range(6)) or set(rx.tolist())!=set(RX) or len(set(day.tolist()))!=1 or not set(day.tolist())<={1,2,3}:raise ValueError('Balanced6TXx5RXx2 same-day source batch required')
 with torch.no_grad():
  s,scale=statistics(h.detach());s=s.double();means=[]
  for tx in range(6):
   cells=[]
   for rr in RX:
    keep=(y==tx)&(rx==rr)
    if int(keep.sum())!=2:raise ValueError('Exactly2 physical L samples/cell required')
    cells.append(s[keep].mean(0))
   means.append(torch.stack(cells))
  cells=torch.stack(means);identity=cells.mean(1);identity-=identity.mean(0)
  _,singular,vh=torch.linalg.svd(identity,full_matrices=False)
  rank=int((singular>singular.max().clamp_min(1e-12)*1e-6).sum())
  basis=vh[:rank]
  raw=torch.empty_like(s)
  for tx in range(6):
   for j,rr in enumerate(RX):raw[(y==tx)&(rx==rr)]=cells[tx,(j+1)%5]-cells[tx,j]
  safe=raw-(raw@basis.T)@basis
  # Match the size of RAW to IDSAFE; compare directions rather than smaller noise.
  matched=raw*(safe.norm(dim=1)/raw.norm(dim=1).clamp_min(1e-12))[:,None]
  # One shared per-packet scalar bounds both arms while retaining orthogonality.
  bound=torch.maximum(safe.abs().amax(1),matched.abs().amax(1)).clamp_min(1.)[:,None]
  safe=safe/bound;matched=matched/bound
  value={'ZERO':torch.zeros_like(safe),'RAW_MATCHED':matched,'IDSAFE':safe}[mode]
  raw_energy=float(raw.square().sum());safe_unbounded=raw-(raw@basis.T)@basis
  detail=dict(identity_rank=rank,raw_energy=raw_energy,safe_retained_energy_fraction=float(safe_unbounded.square().sum())/raw_energy if raw_energy>0 else 0.,matched_energy=float(matched.square().sum()),safe_energy=float(safe.square().sum()),identity_overlap_RAW=float((matched@basis.T).square().sum()),identity_overlap_IDSAFE=float((safe@basis.T).square().sum()),maximum_coordinate=float(value.abs().max()),direction_requires_grad=value.requires_grad,role='L_s',class_centers_used_for_prediction=False)
 return value.to(h.dtype).detach(),scale.detach(),detail

def perturb(h,delta,scale,sign,epsilon=.1):
 if sign not in (-1,1) or epsilon!=.1 or delta.shape!=(len(h),2*h.shape[1]) or delta.requires_grad or scale.shape!=(h.shape[1],) or scale.requires_grad:raise ValueError('Fixed detached tangent shape/sign differs')
 mu=h.mean(-1,keepdim=True);sigma=(h.var(-1,unbiased=False,keepdim=True)+1e-6).sqrt();dm,ds=delta.chunk(2,1)
 # Algebraically exact zero intervention and live input derivative.
 return h+(h-mu)*(torch.exp(sign*epsilon*ds).unsqueeze(-1)-1)+sign*epsilon*(dm*scale).unsqueeze(-1)

def flatness(zplus,zminus,epsilon=.1):
 if zplus.shape!=zminus.shape or zplus.ndim!=2 or zplus.shape[1]!=160 or epsilon!=.1:raise ValueError('Actual joint160 symmetric pair required')
 if not torch.isfinite(zplus).all() or not torch.isfinite(zminus).all():raise FloatingPointError('Nonfinite symmetric source features')
 return (F.normalize(zplus,dim=1,eps=1e-12)-F.normalize(zminus,dim=1,eps=1e-12)).square().sum(1).mean()/(4*epsilon**2)

@contextmanager
def frozen_observation_mode(model):
 flags={m:m.training for m in model.modules()}
 model.eval()
 try:yield
 finally:
  for m,flag in flags.items():m.training=flag

def pair(model,x,y,rx,day,layer,mode):
 from experiments.cvs_feature_restitution.features import layout
 if layer not in ('t1','t2') or any(m.training for m in model.modules()):raise ValueError('Tangent replay requires eval actual t1/t2')
 core,_,_=layout(model);module=getattr(core,layer);captured=[]
 handle=module.register_forward_hook(lambda m,a,v:captured.append(v.detach()))
 try:
  with torch.no_grad():model.forward_identity_only(x)
 finally:handle.remove()
 if len(captured)!=1:raise ValueError('Actual shallow/middle map bypassed')
 delta,scale,info=direction(captured[0],y,rx,day,mode)
 zs=[]
 for sign in (1,-1):
  calls=[]
  def hook(m,a,v):calls.append(1);return perturb(v,delta,scale,sign)
  handle=module.register_forward_hook(hook)
  try:zs.append(model.forward_identity_only(x)['id_feat_joint'])
  finally:handle.remove()
  if len(calls)!=1:raise ValueError('Actual tangent intervention bypassed')
 loss=flatness(*zs)
 info.update(layer=layer,mode=mode,epsilon=.1,raw_flatness=float(loss.detach()),forward_packets=180,forward_calls=3,teacher_calls=0,U_calls=0,head_fit_updates=0)
 return loss,info

def cap(model,identity_loss,regularizer,ramp):
 # Class-specific W is absent from both R and its gradient-budget denominator.
 from experiments.cvs_feature_multiscale50.model import classifier
 w=classifier(model).weight
 params=[p for p in model.parameters() if p.requires_grad and p is not w]
 gi=torch.autograd.grad(identity_loss,params,retain_graph=True,allow_unused=True)
 gr=torch.autograd.grad(regularizer,params,retain_graph=True,allow_unused=True)
 norm=lambda gs:torch.stack([g.detach().double().square().sum() for g in gs if g is not None]).sum().sqrt()
 ni,nr=norm(gi),norm(gr)
 if not torch.isfinite(ni) or not torch.isfinite(nr):raise FloatingPointError('Nonfinite backbone tangent/CE gradients')
 factor=float(ramp)*min(1000.,.2*float(ni)/(float(nr)+1e-12)) if float(nr)>0 else 0.
 return regularizer*factor,dict(CE_feature_gradient_norm=float(ni),raw_feature_gradient_norm=float(nr),detached_coefficient=factor,actual_norm_ratio=factor*float(nr)/(float(ni)+1e-12),maximum_ratio=.2,class_W_in_budget=False)
