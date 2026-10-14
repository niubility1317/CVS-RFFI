import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_feature_mean_teacher50.objective import consistency
from experiments.cvs_source_layer_order.probe import gradient_stats

def observe(logits,weak_logits,strong_feature,weak_feature,truth,params,role):
 if role!='L_s':raise ValueError('Only labeled source gradients allowed')
 n=len(logits)
 if not n or logits.shape!=(n,6) or weak_logits.shape!=(n,6) or strong_feature.shape!=(n,160) or weak_feature.shape!=(n,160) or truth.shape!=(n,) or truth.dtype!=torch.long or truth.min()<0 or truth.max()>=6:raise ValueError('Source graph shape/labels differ')
 if weak_feature.requires_grad or weak_logits.requires_grad or any(not torch.isfinite(v).all() for v in (logits,weak_logits,strong_feature,weak_feature)):raise ValueError('Detached finite actual EMA observations required')
 weak_right=weak_logits.argmax(1)==truth;strong_right=logits.detach().argmax(1)==truth
 masks=dict(zip(d.GROUPS,(weak_right&strong_right,weak_right&~strong_right,~weak_right&strong_right,~weak_right&~strong_right)))
 per_CE=F.cross_entropy(logits,truth,reduction='none',label_smoothing=.01)
 per_feature=.5*(F.normalize(strong_feature.float(),dim=1,eps=1e-8)-F.normalize(weak_feature.detach().float(),dim=1,eps=1e-8)).square().sum(1)
 feature=consistency(strong_feature,weak_feature)
 losses=dict(main=per_CE.mean(),feature=feature,effective=.1*feature)
 for name,mask in masks.items():losses['main_'+name]=(per_CE*mask).sum()/n;losses['feature_'+name]=(per_feature*mask).sum()/n
 vectors={}
 for name,value in losses.items():
  gg=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
  vectors[name]=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten().cpu() for p,g in zip(params,gg)]
 if any(not torch.isfinite(v).all() for vv in vectors.values() for v in vv):raise ValueError('Nonfinite source derivative')
 errors={}
 for key in ('main','feature'):
  errors[key]=max(float((vectors[key][k]-sum(vectors[key+'_'+name][k] for name in d.GROUPS)).abs().max()) for k in range(len(params)))
  scale=max(1.,max(float(v.abs().max()) for v in vectors[key]))
  if errors[key]>2e-5*scale:raise ValueError('Common-denominator source decomposition differs')
 return dict(count=n,weak_right=int(weak_right.sum()),strong_right=int(strong_right.sum()),groups={key:int(mask.sum()) for key,mask in masks.items()},losses={key:float(value.detach()) for key,value in losses.items()},decomposition_max_abs_error=errors,vectors=vectors)

def summarize_vectors(vectors,counts,parameters):
 result=[]
 for k,name in enumerate(parameters):
  groups={key:dict(count=counts[key],raw_vs_matched_true=gradient_stats(vectors['main_'+key][k],vectors['feature_'+key][k]),effective_vs_matched_true=gradient_stats(vectors['main_'+key][k],.1*vectors['feature_'+key][k])) for key in d.GROUPS}
  result.append(dict(parameter=name,raw=gradient_stats(vectors['main'][k],vectors['feature'][k]),effective=gradient_stats(vectors['main'][k],vectors['effective'][k]),groups=groups))
 return result
