import torch
from torch.nn import functional as F

def terms(logits,features,base_logits,base_features):
 """Weak targets never receive gradients; logit shift is a softmax gauge."""
 target=base_logits.detach();q=target.softmax(1)
 dz=(logits-logits.mean(1,keepdim=True))-(target-target.mean(1,keepdim=True))
 feature=F.normalize(features,dim=1,eps=1e-4)
 base=F.normalize(base_features.detach(),dim=1,eps=1e-4)
 kl=F.kl_div(logits.log_softmax(1),q,reduction='none').sum(1)
 return dict(KL=kl*(q.max(1).values>=.95),margin=.5*dz.square().mean(1),feature=(feature-base).square().sum(1))

def record(logits,features,base_logits,base_features,head_weight,y=None):
 z=F.normalize(features,dim=1,eps=1e-4);b=F.normalize(base_features,dim=1,eps=1e-4)
 delta=z-b;w=F.normalize(head_weight,dim=1,eps=1e-4);wc=w-w.mean(0,keepdim=True)
 # Exact centered-logit row space in unit-feature coordinates; diagnostic only.
 _,s,vh=torch.linalg.svd(wc.double(),full_matrices=False)
 rank=int((s>s.max()*1e-7).sum()) if float(s.max())>0 else 0
 projected=delta.double()@vh[:rank].T
 row_energy=projected.square().sum(1).to(delta);total=delta.square().sum(1)
 q=base_logits.softmax(1);r=logits.softmax(1)
 result=dict(confidence=q.max(1).values,feature_energy=total,row_energy=row_energy,
  residual_energy=(total-row_energy).clamp_min(0),probability_L1=(q-r).abs().sum(1),
  disagreement=(q.argmax(1)!=r.argmax(1)).float(),**terms(logits,features,base_logits,base_features))
 if y is not None:
  old=q.argmax(1)==y;new=r.argmax(1)==y
  result.update(base_correct=old.float(),changed_correct=new.float(),fix=(~old&new).float(),broken=(old&~new).float())
 return result,rank

def aggregate(records,rx,day,bins):
 n=len(rx);masks={'all':torch.ones(n,dtype=torch.bool,device=rx.device)}
 for a,b in zip(bins,bins[1:]):masks[f'confidence[{a},{b})']=(records['confidence']>=a)&(records['confidence']<b)
 for key,t in [('RX',rx),('day',day)]:
  for value in t.unique().tolist():masks[f'{key}:{value}']=t==value
 out={}
 for name,m in masks.items():
  out[name]=dict(count=int(m.sum()),means={k:float(v[m].double().mean()) if m.any() else None for k,v in records.items()})
  total=float(records['feature_energy'][m].sum())
  out[name]['residual_energy_fraction']=float(records['residual_energy'][m].sum())/total if total>0 else None
 return out

def gradient_vectors(losses,params):
 result={}
 for name,loss in losses.items():
  gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
  result[name]=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])
 return result

def gradient_summary(vectors):
 out={};ce=vectors['L_CE'];cn=float(ce.norm())
 for name,g in vectors.items():
  norm=float(g.norm());out[name]=dict(norm=norm,relative_to_L_CE=norm/cn if cn>0 else None,
   cosine_to_L_CE=float(torch.dot(g,ce)/(g.norm()*ce.norm())) if norm>0 and cn>0 else None)
 return out
