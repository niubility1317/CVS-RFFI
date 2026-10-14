import torch
from torch.nn import functional as F

MIN_CONF=.99
WEAK_RMS=.003
MAX_CORRECTION_COUNT_RATIO=.1
MAX_CORRECTION_CE_RATIO=.25

def weak_view(x,seed):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point() or not torch.isfinite(x).all():raise ValueError('Finite original256 IQ required')
 gen=torch.Generator(device=x.device).manual_seed(int(seed))
 rms=x.detach().square().mean((1,2),keepdim=True).sqrt()
 noise=torch.randn(x.shape,generator=gen,device=x.device,dtype=x.dtype)
 return x+WEAK_RMS*rms*noise

def select(teacher_logits,second_weak_logits,student_logits,base_mask,enable_correction):
 if type(enable_correction) is not bool or teacher_logits.shape!=second_weak_logits.shape or teacher_logits.shape!=student_logits.shape or teacher_logits.ndim!=2 or teacher_logits.shape[1]!=6 or base_mask.shape!=(len(teacher_logits),) or base_mask.dtype!=torch.bool:raise ValueError('Six-class label-blind gate shape differs')
 if any(not torch.isfinite(v).all() for v in (teacher_logits,second_weak_logits,student_logits)):raise FloatingPointError('Nonfinite trust/correction logits')
 # No labels, RX-specific thresholds, true class counts or sample weights enter.
 with torch.no_grad():
  q=teacher_logits.detach().softmax(1);q2=second_weak_logits.detach().softmax(1)
  conf,pseudo=q.max(1);conf2,pseudo2=q2.max(1);p=student_logits.detach().softmax(1)
  agrees=p.argmax(1)==pseudo
  trusted=(conf>=MIN_CONF)&(conf2>=MIN_CONF)&(pseudo2==pseudo)
  original=base_mask&agrees
  candidate=base_mask&trusted&~agrees
  costs=F.cross_entropy(student_logits.detach(),pseudo,reduction='none')
  budget=MAX_CORRECTION_CE_RATIO*float(costs[original].sum())
  max_count=int(MAX_CORRECTION_COUNT_RATIO*int(original.sum()))
  correction=torch.zeros_like(original);used=0.;chosen=0
  # Teacher confidence orders eligible cases; no class quota or forced fill.
  for i in sorted(candidate.nonzero().flatten().tolist(),key=lambda j:(-float(conf[j]),j)):
   value=float(costs[i])
   if chosen>=max_count:break
   if used+value<=budget:
    correction[i]=True;used+=value;chosen+=1
  selected=original|(correction if enable_correction else torch.zeros_like(correction))
  masks=dict(base=base_mask.detach(),student_agreement=agrees,teacher_high_conf=conf>=MIN_CONF,weak_high_conf=conf2>=MIN_CONF,weak_stable=pseudo2==pseudo,trusted_teacher=trusted,original=original,reliable_already_agreed=base_mask&trusted&agrees,correction_candidate=candidate,correction_admitted=correction,selected=selected,teacher_rejected=~(base_mask&trusted))
  logit_gradient=p-F.one_hot(pseudo,6).to(p.dtype)
  facts=dict(original_count=int(original.sum()),candidate_count=int(candidate.sum()),admitted_count=int(correction.sum()),selected_count=int(selected.sum()),original_CE_sum=float(costs[original].sum()),admitted_CE_sum=used,CE_budget=budget,count_budget=max_count,hard_targets=True,normalization='mean over selected physical packets, unchanged native CE',uniform_selected_weights=True,teacher_views_correlated=True,teacher_probabilities_not_calibrated=True)
 return pseudo,masks,facts,dict(CE=costs,logit_gradient_norm=logit_gradient.norm(dim=1),teacher_confidence=conf)

def loss(student_logits,pseudo,mask):
 if mask.dtype!=torch.bool or mask.shape!=(len(pseudo),):raise ValueError('Invalid accepted mask')
 return F.cross_entropy(student_logits[mask],pseudo[mask]) if bool(mask.any()) else student_logits.sum()*0

def entropy(student_logits):
 p=student_logits.softmax(1)
 return -(p*p.clamp_min(1e-8).log()).sum(1)
