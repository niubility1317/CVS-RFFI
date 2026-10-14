import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_source_feature_ce_gradient.metrics import observe as base_observe,summarize_vectors as base_summary
from experiments.cvs_source_layer_order.probe import gradient_stats

def policy(weak_logits,strong_logits):
 if weak_logits.ndim!=2 or weak_logits.shape!=strong_logits.shape or weak_logits.shape[1]!=6 or not len(weak_logits) or not torch.isfinite(weak_logits).all() or not torch.isfinite(strong_logits).all():raise ValueError('Finite default six-class logits required')
 with torch.no_grad():
  teacher=weak_logits.detach().float().softmax(1).max(1).values;student=strong_logits.detach().float().softmax(1).max(1).values
 return dict(keep=teacher>=student,teacher_confidence=teacher,student_confidence=student)

def policy_summary(v):
 keep=v['keep'];teacher=v['teacher_confidence'];student=v['student_confidence'];weak_right=v['weak_pred']==v['truth'];strong_right=v['strong_pred']==v['truth'];masks=dict(zip(d.GROUPS,(weak_right&strong_right,weak_right&~strong_right,~weak_right&strong_right,~weak_right&~strong_right)))
 if not torch.equal(keep,teacher>=student):raise ValueError('Saved gate differs from fixed confidence order')
 groups={key:dict(count=int(mask.sum()),retained=int((mask&keep).sum()),retention=float((mask&keep).sum())/int(mask.sum()) if int(mask.sum()) else None) for key,mask in masks.items()}
 return dict(count=len(keep),retained=int(keep.sum()),retention=float(keep.double().mean()),ties=int((teacher==student).sum()),weak_correct=int(weak_right.sum()),strong_correct=int(strong_right.sum()),groups=groups)

def observe(logits,weak_logits,strong_feature,weak_feature,truth,params,role):
 if role!='L_s':raise ValueError('V/U/query gradients forbidden')
 gate=policy(weak_logits,logits) # Rule is evaluated before source truth is consulted.
 r=base_observe(logits,weak_logits,strong_feature,weak_feature,truth,params,role);n=r['count'];weak_right=weak_logits.argmax(1)==truth;strong_right=logits.detach().argmax(1)==truth;masks=dict(zip(d.GROUPS,(weak_right&strong_right,weak_right&~strong_right,~weak_right&strong_right,~weak_right&~strong_right)))
 per=.5*(F.normalize(strong_feature.float(),dim=1,eps=1e-8)-F.normalize(weak_feature.detach().float(),dim=1,eps=1e-8)).square().sum(1)
 losses=dict(gated=.1*(per*gate['keep']).sum()/n)
 for key,mask in masks.items():losses['gated_'+key]=.1*(per*(mask&gate['keep'])).sum()/n
 for key,value in losses.items():
  gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);r['vectors'][key]=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten().cpu() for p,g in zip(params,gs)];r['losses'][key]=float(value.detach())
 if any(not torch.isfinite(x).all() for v in r['vectors'].values() for x in v):raise ValueError('Nonfinite fixed-gate source derivative')
 error=max(float((r['vectors']['gated'][k]-sum(r['vectors']['gated_'+key][k] for key in d.GROUPS)).abs().max()) for k in range(len(params)))
 if error>2e-5*max(1.,max(float(v.abs().max()) for v in r['vectors']['gated'])):raise ValueError('Gated common-denominator decomposition differs')
 observation=dict(**gate,weak_pred=weak_logits.argmax(1),strong_pred=logits.detach().argmax(1),truth=truth)
 r['retained_groups']={key:int((mask&gate['keep']).sum()) for key,mask in masks.items()};r['policy']=policy_summary(observation);r['observation']=observation;r['gated_decomposition_max_abs_error']=error;return r

def summarize_vectors(vectors,counts,retained,parameters):
 result=base_summary(vectors,counts,parameters)
 for k,r in enumerate(result):
  r['gated']=gradient_stats(vectors['main'][k],vectors['gated'][k]);r['gated_groups']={key:dict(count_full=counts[key],count_retained=retained[key],gated_feature_vs_full_group_CE=gradient_stats(vectors['main_'+key][k],vectors['gated_'+key][k])) for key in d.GROUPS}
 return result
