"""AND-inspired source gradient filter; no model, data loading, or updates."""
import torch
from .design import RX,TAU

def transform(per_rx,ce,segments,*,role='L_s',threshold=TAU,random_seed=2026101538):
 if role!='L_s' or threshold!=TAU:raise ValueError('Only fixed labelled-source consensus allowed')
 if per_rx.ndim!=2 or per_rx.shape[0]!=len(RX) or per_rx.shape[1]==0 or ce.shape!=(per_rx.shape[1],):raise ValueError('Five complete equal-width receiver gradients required')
 if per_rx.device.type!='cpu' or ce.device.type!='cpu' or per_rx.dtype!=torch.float32 or ce.dtype!=torch.float32 or not torch.isfinite(per_rx).all() or not torch.isfinite(ce).all():raise ValueError('Finite FP32 CPU vectors required')
 if not segments or any(n<=0 for n in segments) or sum(segments)!=ce.numel():raise ValueError('Actual parameter partition incomplete')
 eligible=(per_rx!=0).any(0);mask=(per_rx.sign().mean(0).abs()>=threshold)&eligible
 random=torch.zeros_like(mask);g=torch.Generator(device='cpu').manual_seed(random_seed);offset=0
 for n in segments:
  candidates=torch.where(eligible[offset:offset+n])[0];k=int(mask[offset:offset+n].sum())
  if k:random[offset+candidates[torch.randperm(len(candidates),generator=g)[:k]]]=True
  offset+=n
 return dict(mask=mask,random_mask=random,eligible=eligible,masked=ce*mask,random=ce*random)

def cosine(a,b):
 a,b=a.double(),b.double();den=float(a.norm()*b.norm())
 return float(torch.dot(a,b)/den) if den>1e-24 else None

def metrics(per_rx,ce,value):
 keep=int(value['mask'].sum());eligible=int(value['eligible'].sum());base=float(ce.double().norm())
 return dict(eligible=eligible,kept=keep,eligible_keep_fraction=keep/eligible if eligible else 0.,retained_gradient_norm_ratio=float(value['masked'].double().norm())/base if base else 0.,masked_CE_cosine=cosine(ce,value['masked']),random_CE_cosine=cosine(ce,value['random']),masked_RX_cosines=[cosine(value['masked'],g) for g in per_rx],random_RX_cosines=[cosine(value['random'],g) for g in per_rx])
