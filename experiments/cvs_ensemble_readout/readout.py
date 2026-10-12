import torch
from torch.nn import functional as F
def scores(logits,kind):
 if logits.ndim!=3 or logits.shape[0]!=3 or logits.shape[2]!=6 or not torch.isfinite(logits).all():raise ValueError('Threefinite independent sixclass logits required')
 if kind=='MEAN_LOGIT':return logits.mean(0)
 if kind=='UNIT_MARGIN':
  centered=logits-logits.mean(-1,keepdim=True);return (centered/centered.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-8)).mean(0)
 if kind=='MAJORITY':return F.one_hot(logits.argmax(-1),num_classes=6).sum(0).float()+.001*logits.softmax(-1).mean(0)
 raise ValueError('Unregistered fusion')
