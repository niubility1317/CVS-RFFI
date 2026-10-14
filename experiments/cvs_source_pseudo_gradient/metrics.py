import torch
from torch.nn import functional as F
from . import design as d

def observe_gradients(logits,pseudo,truth,selected,params,role):
 if role!='L_s':raise ValueError('Source V/U gradients are forbidden')
 if logits.ndim!=2 or logits.shape[1]!=6 or any(v.shape!=(len(logits),) for v in (pseudo,truth,selected)) or selected.dtype!=torch.bool:raise ValueError('Source gradient shapes differ')
 if not torch.isfinite(logits).all() or truth.min()<0 or truth.max()>=6:raise ValueError('Illegal source gradient inputs')
 n=int(selected.sum());correct=pseudo==truth
 pseudo_ce=F.cross_entropy(logits,pseudo,reduction='none')
 true_ce=F.cross_entropy(logits,truth,reduction='none')
 denominator=max(n,1)
 losses=dict(total=.16*(pseudo_ce*selected).sum()/denominator,matched_true=.16*(true_ce*selected).sum()/denominator,correct=.16*(pseudo_ce*(selected&correct)).sum()/denominator,wrong=.16*(pseudo_ce*(selected&~correct)).sum()/denominator)
 vectors={}
 for key,value in losses.items():
  gs=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
  vectors[key]=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(params,gs)]
 if any(not torch.isfinite(v).all() for values in vectors.values() for v in values):raise ValueError('Nonfinite source parameter gradient')
 error=max(float((a-b-c).abs().max()) for a,b,c in zip(vectors['total'],vectors['correct'],vectors['wrong']))
 scale=max(1.,max(float(v.abs().max()) for v in vectors['total']))
 if error>2e-5*scale:raise ValueError('Common-denominator gradient decomposition differs')
 return dict(accepted=n,accepted_wrong=int((selected&~correct).sum()),decomposition_max_abs_error=error,vectors=vectors)

def vector_stats(total,matched_true,wrong):
 a,b,w=(v.double() for v in (total,matched_true,wrong));an,bn,wn=(float(v.norm()) for v in (a,b,w));dot=float(a@b)
 return dict(pseudo_norm=an,matched_true_norm=bn,wrong_norm=wn,pseudo_vs_matched_true_cosine=dot/(an*bn) if an*bn else None,wrong_over_total_norm=wn/an if an else None,scope='Norm ratio may exceed1 due cancellation; not an additive contribution fraction or optimizer update')

def summary(values,correct,mask):
 result=dict(count=int(mask.sum()),wrong=int((mask&~correct).sum()),correct=int((mask&correct).sum()),groups={})
 for name,keep in [('correct',mask&correct),('wrong',mask&~correct)]:
  result['groups'][name]={}
  for metric,x in values.items():
   z=x[keep].double()
   result['groups'][name][metric]=dict(count=len(z),mean=float(z.mean()) if len(z) else None,quantiles={str(q):float(torch.quantile(z,q)) if len(z) else None for q in (.05,.25,.5,.75,.95)})
 return result
