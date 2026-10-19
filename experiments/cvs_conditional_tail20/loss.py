import torch
from torch.nn import functional as F

def objective(first,second,labels,mode):
 if mode not in ('CE2','TAIL2') or first.shape!=second.shape or first.ndim!=2 or first.shape[1]!=6 or labels.shape!=(len(first),) or labels.dtype!=torch.long or not len(first) or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()) or bool(((labels<0)|(labels>=6)).any()):raise ValueError('Two finite six-class logit batches and legal true labels required')
 ce=torch.stack([F.cross_entropy(z,labels,label_smoothing=.01,reduction='none') for z in (first,second)])
 delta=ce[0].detach()-ce[1].detach();winner=torch.stack([(delta>0).float()+.5*(delta==0).float(),(delta<0).float()+.5*(delta==0).float()])
 weights=torch.ones_like(ce)*.5 if mode=='CE2' else .25+.5*winner
 risk=(ce*weights).sum(0).mean()
 return risk,dict(draw_CE=ce.detach().tolist(),draw_weights=weights.detach().tolist(),harder_draw=delta.sign().to(torch.int8).tolist(),conditional_mean_CE=float(ce.mean().detach()),conditional_max_CE=float(ce.max(0).values.mean().detach()),risk_CE=float(risk.detach()),weight_min=float(weights.min()),weight_max=float(weights.max()))
