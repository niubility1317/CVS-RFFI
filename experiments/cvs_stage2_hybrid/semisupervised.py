"""Truth-blind source pseudo labels, visit-pass stability, cache-safe EMA.

Not wired into the frozen H0-H10 release. A new registered run must own it.
"""
from copy import deepcopy
import torch
from torch.nn import functional as F
from experiments.cvs_stage2_geometry.features import identity

UNLABELED_KEYS={'x','ids','rx','day','domain','eq','sig','base_index','role','preprocessing'}


def validate_unlabeled(cache,expected_ids=None):
 if set(cache)!=UNLABELED_KEYS or cache['role']!='U_s':raise ValueError('Truth-blind U_s schema required; labels/features/margins forbidden')
 n=len(cache['ids'])
 if len(set(cache['ids']))!=n or cache['x'].shape!=(n,2,256):raise ValueError('Unique U_s IQ identities required')
 if cache['x'].dtype!=torch.float32 or cache['x'].device.type!='cpu' or not torch.isfinite(cache['x']).all():raise ValueError('Finite CPU float32 U_s IQ required')
 if any(cache[k].shape!=(n,) for k in ('rx','day','domain','eq','sig','base_index')):raise ValueError('U_s metadata shape differs')
 if expected_ids is not None and sorted(cache['ids'])!=sorted(expected_ids):raise ValueError('U_s physical role differs')
 return n


class UnlabeledStream:
 """One complete permutation before reuse; carry per-sample U traversal pass."""
 def __init__(self,n,seed):
  if n<1:raise ValueError('Empty U_s')
  self.n=n;self.generator=torch.Generator().manual_seed(seed+901337);self.pass_index=0;self.pool=torch.empty(0,dtype=torch.long)
 def batch(self,size):
  if size<1:raise ValueError('Positive U batch required')
  parts=[];passes=[];remaining=size
  while remaining:
   if not len(self.pool):self.pool=torch.randperm(self.n,generator=self.generator);self.pass_index+=1
   take=min(remaining,len(self.pool));parts.append(self.pool[:take]);passes.append(torch.full((take,),self.pass_index,dtype=torch.long))
   self.pool=self.pool[take:];remaining-=take
  return torch.cat(parts),torch.cat(passes)


class VisitPassBank:
 """Prediction history only, indexed by opaque physical ID, never TX truth."""
 def __init__(self,min_streak=2,min_conf=.80):
  if min_streak<2:raise ValueError('At least two U passes for stability')
  self.min_streak=min_streak;self.min_conf=min_conf;self.entries={}
 @torch.no_grad()
 def update(self,ids,prediction,confidence,passes):
  if not (len(ids)==len(prediction)==len(confidence)==len(passes)):raise ValueError('U history dimensions differ')
  mask=[];streaks=[]
  for uid,p,c,visit in zip(ids,prediction.cpu().tolist(),confidence.cpu().tolist(),passes.cpu().tolist()):
   old=self.entries.get(uid);streak=1
   if old and old[0]==p:
    if visit==old[1]+1:streak=old[2]+1
    elif visit==old[1]:streak=old[2]
   if c>=self.min_conf:self.entries[uid]=(p,visit,streak)
   else:self.entries.pop(uid,None)
   mask.append(c>=self.min_conf and streak>=self.min_streak);streaks.append(streak)
  return torch.tensor(mask,device=prediction.device),dict(history_size=len(self.entries),stable_count=sum(mask),mean_streak=sum(streaks)/max(len(streaks),1))


def confidence_mask(confidence,domains,tau_min=.92,tau_max=.97,quantile=.86):
 if not 0<=tau_min<=tau_max<=1 or not 0<=quantile<=1:raise ValueError('Invalid pseudo confidence contract')
 mask=torch.zeros_like(confidence,dtype=torch.bool)
 for domain in domains.unique():
  group=domains==domain;threshold=torch.quantile(confidence[group].float(),quantile).clamp(tau_min,tau_max)
  mask[group]=confidence[group]>=threshold
 return mask


class EMAIdentity:
 """Use in-place no_grad writes, preserving parameter-version cache invalidation."""
 def __init__(self,student,decay=.999):
  if not 0<=decay<1:raise ValueError('Invalid EMA decay')
  self.model=deepcopy(student).eval().requires_grad_(False);self.decay=decay;self.updates=0;self.calls=0
 @torch.no_grad()
 def logits(self,x):
  self.model.eval();self.calls+=1
  return identity(self.model,x)[0].detach()
 @torch.no_grad()
 def update(self,student):
  target=dict(self.model.named_parameters());source=dict(student.named_parameters())
  if target.keys()!=source.keys():raise ValueError('EMA/student parameter schema differs')
  for name,p in target.items():
   if name.startswith('id_backbone.'):
    # `.data` writes would leave _version unchanged and reuse stale Sinc/head caches.
    p.mul_(self.decay).add_(source[name].detach(),alpha=1-self.decay)
  target_buffers=dict(self.model.named_buffers());source_buffers=dict(student.named_buffers())
  if target_buffers.keys()!=source_buffers.keys():raise ValueError('EMA/student buffer schema differs')
  for name,b in target_buffers.items():
   if name.startswith('id_backbone.'):b.copy_(source_buffers[name])
  self.updates+=1


@torch.no_grad()
def make_plan(teacher,cache,indices,passes,step,seed,mode,bank=None):
 """Called exactly once before either SAM closure; no U label access exists."""
 x=cache['x'][indices].to(next(teacher.model.parameters()).device)
 scores=teacher.logits(x);confidence,pseudo=scores.softmax(1).max(1)
 domains=cache['domain'][indices].to(x.device);conf=confidence_mask(confidence,domains)
 metadata={k:cache[k][indices] for k in ('rx','day','eq','sig','base_index')}
 if mode=='visit_pass_bank':
  if bank is None:raise ValueError('Registered history required')
  stable,history=bank.update([cache['ids'][i] for i in indices.tolist()],pseudo,confidence,passes)
 elif mode=='batch_neighbor':
  from experiments.cvs_phase1_overlay.model import native_modules
  n=native_modules();meta={k+'_i' if k!='base_index' else k:v for k,v in metadata.items()}
  stable=torch.tensor(n.temporal_neighbor_agreement_mask(pseudo,confidence,meta,window=2,min_conf=.80),device=x.device);history={}
 else:raise ValueError('Unregistered temporal gate')
 generator=torch.Generator().manual_seed(seed+step*9901)
 noise=torch.randn(x.shape,generator=generator,dtype=x.dtype).to(x.device)*.015
 return dict(x_strong=(x+noise).detach(),pseudo=pseudo.detach(),base_mask=(conf&stable).detach(),confidence=confidence.detach(),rx=metadata['rx'].to(x.device),day=metadata['day'].to(x.device),
  diagnostics=dict(packets=len(x),confidence_pass=int(conf.sum()),temporal_pass=int(stable.sum()),history=history,
   min_U_pass=int(passes.min()),max_U_pass=int(passes.max()),truth_available=False,teacher_domain_forwards=0))


def student_objective(student,plan):
 logits,z=identity(student,plan['x_strong']);pseudo=plan['pseudo'];base=plan['base_mask']
 strong=logits.detach().argmax(1)==pseudo;mask=base&strong
 pseudo_loss=F.cross_entropy(logits[mask],pseudo[mask]) if mask.any() else logits.sum()*0.
 probabilities=logits.softmax(1);entropy=-(probabilities*probabilities.clamp_min(1e-8).log()).sum(1).mean()
 strata={f'{int(rx)}:{int(day)}':dict(seen=int(((plan['rx']==rx)&(plan['day']==day)).sum()),accepted=int((mask&(plan['rx']==rx)&(plan['day']==day)).sum())) for rx in plan['rx'].unique() for day in plan['day'].unique()}
 return pseudo_loss,entropy,dict(plan['diagnostics'],strong_pass=int(strong.sum()),accepted=int(mask.sum()),accept_rate=float(mask.float().mean()),RX_day=strata,pseudo_class_accepted=torch.bincount(pseudo[mask],minlength=6).cpu().tolist(),
  mean_confidence=float(plan['confidence'].mean()),pseudo_accuracy=None,pseudo_accuracy_reason='U_s TX truth hidden by contract'),z,mask
