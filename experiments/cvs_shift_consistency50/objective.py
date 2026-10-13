from contextlib import contextmanager
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_e20_cross_style.style import _paths
from experiments.cvs_six_direction_mechanisms.sampling import shift,GEOMETRY
from experiments.cvs_finite_margin.objective import deterministic

@contextmanager
def capture(model):
 modules,paths=_paths(model);maps={};handles=[]
 for level in d.LEVELS:
  if level not in paths:raise ValueError('Source consistency layermissing')
  handles.append(modules[paths[level]].register_forward_hook(lambda m,i,o,k=level:maps.__setitem__(k,o)))
 try:yield maps
 finally:
  for h in handles:h.remove()

def common_statistics(features,level,delta,common_low,common_high):
 stride,offset,width=GEOMETRY[level]
 if features.ndim!=3 or not torch.isfinite(features).all():raise ValueError('Finite source time maps required')
 centers=torch.arange(features.shape[-1],device=features.device,dtype=features.dtype)*stride+offset-delta
 mask=(centers-(width-1)/2>=common_low)&(centers+(width-1)/2<=common_high)
 if int(mask.sum())<2:raise ValueError('Shared physical interior lacks statistics')
 values=features[...,mask];stats=torch.cat((values.mean(2),values.std(2,unbiased=False)),1)
 return F.normalize(stats,dim=1,eps=1e-5),int(mask.sum())

def statistics_penalty(original,moved,delta):
 low=max(0,-delta);high=min(255,255-delta);losses=[];detail=[]
 for level in d.LEVELS:
  a,na=common_statistics(original[level],level,0,low,high);b,nb=common_statistics(moved[level],level,delta,low,high)
  loss=(a-b).square().sum(1).mean();losses.append(loss);detail.append(dict(level=level,original_shape=list(original[level].shape),moved_shape=list(moved[level].shape),original_valid=na,moved_valid=nb,statistics_dim=a.shape[1],loss=float(loss.detach()),delta=delta,global_GN_boundary_dependence_not_removed=True))
 return torch.stack(losses).mean(),detail

def objective(model,x,y,ux,ids,passes,bank,c,step):
 if ux is not None and (not c['source_U'] or step<=d.WARMUP or bank is None):raise ValueError('Unauthorized Ubranch')
 multi=c['source_shift']['enabled'] and step>d.WARMUP;stats=c['shift_statistics']['enabled'] and step>d.WARMUP;delta=d.delta(step);ramp=d.ramp(step)
 if stats:
  with capture(model) as a:z=model(x)
 else:z=model(x)
 ce=F.cross_entropy(z,y);cons=z.new_zeros(());details=[];L_forwards=1
 if multi:
  moved=shift(x,delta)
  if stats:
   with capture(model) as b:zz=model(moved)
   cons,details=statistics_penalty(a,b,delta)
  else:zz=model(moved)
  ce=.5*(ce+F.cross_entropy(zz,y));L_forwards=2
 kl=z.new_zeros(());accepted=0;history=None;confidence_mean=0.;agreement=0.;teacher_forwards=0
 if ux is not None:
  with deterministic(model),torch.no_grad():
   q=model(ux).softmax(1).detach();q2=model(shift(ux,delta)).softmax(1).detach()
  confidence,pseudo=q.max(1);confidence2,pseudo2=q2.max(1);agree=pseudo==pseudo2
  confidence=torch.minimum(confidence,confidence2)*agree;mask,history=bank.update(ids,pseudo,confidence,passes)
  accepted=int(mask.sum());agreement=float(agree.float().mean());confidence_mean=float(confidence.mean());teacher_forwards=2
  generator=torch.Generator(device=ux.device).manual_seed(c['model_seed']+step*9901)
  strong=shift(ux,delta)+torch.randn(ux.shape,generator=generator,device=ux.device,dtype=ux.dtype)*.015
  with torch.random.fork_rng(devices=[ux.device.index or 0] if ux.is_cuda else []):
   torch.manual_seed(c['model_seed']+step*1979);strong_logits=model(strong)
  kl=(F.kl_div(strong_logits.log_softmax(1),q,reduction='none').sum(1)*mask).mean()
 loss=ce+.05*ramp*cons+.25*ramp*kl
 metrics=dict(CE=float(ce.detach()),shift_statistics=float(cons.detach()),weighted_shift_statistics=float(.05*ramp*cons.detach()),KL=float(kl.detach()),weighted_KL=float(.25*ramp*kl.detach()),ramp=ramp,L_accuracy=float((z.argmax(1)==y).float().mean()),L_model_forwards=L_forwards,U_packets=len(ux) if ux is not None else 0,accepted_U=accepted,teacher_forward_calls=teacher_forwards,U_student_forwards=int(ux is not None),shift_active=int(multi),statistics_active=int(stats),U_weak_agreement=agreement,U_min_confidence=confidence_mean,total_loss=float(loss.detach()))
 return loss,metrics,dict(layers=details,history=history)
