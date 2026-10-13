from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F
from experiments.cvs_e20_cross_style.style import _paths
from . import design as d

def mask_for(features,gradient,arm,seed):
 if features.ndim!=3 or features.shape!=gradient.shape or not torch.isfinite(features).all() or not torch.isfinite(gradient).all():raise ValueError('Invalid challenge feature/gradient')
 if arm not in ('RANDOM_MASK','GRAD_MASK'):raise ValueError('Invalid mask arm')
 score=(features.detach()*gradient.detach()).abs().mean(-1)
 count=math.ceil(features.shape[1]*.25)
 if not 0<count<features.shape[1]:raise ValueError('Challenge must preserve channels')
 if arm=='RANDOM_MASK':
  g=torch.Generator(device=features.device).manual_seed(seed)
  order=torch.rand(score.shape,device=features.device,generator=g)
 else:order=score
 ix=order.topk(count,dim=1,sorted=False).indices
 mask=torch.ones_like(score).scatter_(1,ix,0.).unsqueeze(-1).detach()
 return mask,score,count

@contextmanager
def captured(model,level):
 modules,paths=_paths(model.base);values={}
 def hook(module,inputs,out):
  if values:raise ValueError('Layer executed more than once')
  values['feature']=out
 h=modules[paths[level]].register_forward_hook(hook)
 try:yield values,paths[level]
 finally:h.remove()

@contextmanager
def masked(model,level,mask):
 modules,paths=_paths(model.base);calls=[]
 def hook(module,inputs,out):
  if calls or out.ndim!=3 or out.shape[:2]!=mask.shape[:2] or mask.shape[-1]!=1:raise ValueError('Mask routing differs')
  calls.append(tuple(out.shape));return out*mask
 h=modules[paths[level]].register_forward_hook(hook)
 try:
  yield calls
  if len(calls)!=1:raise ValueError('Challenge missed actual layer')
 finally:h.remove()

def layer_info(model,x):
 modules,paths=_paths(model.base);info=[];modes=[(m,m.training) for m in model.modules()]
 try:
  model.eval()
  with torch.no_grad():
   for level in d.LEVELS:
    with captured(model,level) as (v,path):model(x)
    out=v['feature']
    if out.ndim!=3:raise ValueError('Missing shallow/middle map')
    info.append(dict(level=level,module=path,shape=list(out.shape),channels=out.shape[1]))
 finally:
  for m,mode in modes:m.training=mode
 return info

def objective(model,x,y,ux,ids,passes,bank,c,step):
 if ux is not None or ids is not None or passes is not None or bank is not None or c['source_U'] or c['challenge']['role']!='L_s':raise ValueError('Challenge accepts only legal labeled source')
 if y.shape!=(len(x),) or y.dtype!=torch.long or y.min()<0 or y.max()>=6:raise ValueError('Invalid legal L labels')
 active=c['challenge']['enabled'] and step>1110
 if active and not model.training:raise ValueError('Training challenge cannot run in inference')
 level=d.LEVELS[(step-1)%2];before_cpu=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state(x.device) if x.is_cuda else None
 if active:
  with captured(model,level) as (v,path):logits=model(x)
  feature=v['feature'];gradient=torch.autograd.grad(logits.gather(1,y[:,None]).sum(),feature,retain_graph=True,create_graph=False)[0]
  mask,scores,count=mask_for(feature,gradient,c['arm'],(c['model_seed']+step*7919)%2**63)
 else:logits=model(x)
 ce=F.cross_entropy(logits,y);masked_ce=ce.detach()*0;ramp=min(1.,max(0.,(step-1110)/2220)) if active else 0.
 detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),U_packets=0,accepted_U=0,teacher_forward_calls=0,ramp=ramp,challenge_forwards=0,rank_grad_calls=0,layer_index=-1,suppressed_channel_fraction=0.,saliency_mean=0.,selected_saliency_fraction=0.)
 if active:
  devices=[x.device.index or 0] if x.is_cuda else []
  with torch.random.fork_rng(devices=devices):
   torch.set_rng_state(before_cpu)
   if x.is_cuda:torch.cuda.set_rng_state(before_cuda,x.device)
   with masked(model,level,mask):challenge_logits=model(x)
  masked_ce=F.cross_entropy(challenge_logits,y)
  frac=((scores*(1-mask.squeeze(-1))).sum(1)/scores.sum(1).clamp_min(1e-12)).mean()
  detail.update(challenge_forwards=1,rank_grad_calls=1,layer_index=d.LEVELS.index(level),suppressed_channel_fraction=count/feature.shape[1],saliency_mean=float(scores.mean()),selected_saliency_fraction=float(frac))
 return ce+.5*ramp*masked_ce,dict(CE=float(ce.detach()),masked_CE=float(masked_ce.detach())),detail
