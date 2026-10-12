import torch
def probabilities(global_logits,expert_logits,kind):
 if global_logits.ndim!=2 or global_logits.shape[1]!=6 or expert_logits.ndim!=3 or expert_logits.shape[0]!=len(global_logits) or expert_logits.shape[2]!=6 or not torch.isfinite(global_logits).all() or not torch.isfinite(expert_logits).all():raise ValueError('Finite independent sixclass logits required')
 a=global_logits.softmax(1);b=expert_logits.softmax(-1)
 if kind=='GLOBAL_ONLY':p=a
 elif kind=='ENTROPY_GATE':
  entropy=-(b*b.clamp_min(1e-12).log()).sum(-1);w=(-4*entropy).softmax(1);p=.5*a+.5*(w[:,:,None]*b).sum(1)
 elif kind=='CONFIDENT_HEAD':
  index=b.max(-1).values.argmax(1);p=.5*a+.5*b[torch.arange(len(b),device=b.device),index]
 else:raise ValueError('Unregistered readout')
 return p
