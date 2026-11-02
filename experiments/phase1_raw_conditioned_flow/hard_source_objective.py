"""Source-only labeled and clean-F pseudo supervision for rollout recovery."""
import torch
from torch.nn import functional as F

def select_targets(labels,confidence,pseudo,raw):
 if labels.ndim!=1 or any(x.shape!=labels.shape for x in (confidence,pseudo,raw)):raise ValueError('Matching per-record target inputs required')
 labeled=labels>=0;target=torch.where(labeled,labels,pseudo);valid=labeled|(confidence>=.95);weight=1.+2.*(raw!=target).float();return target,valid,weight

def hard_source_loss(net,teacher,x,clean,labels):
 if x.shape!=clean.shape or x.ndim!=3 or x.shape[1:]!=(2,256) or labels.shape!=(len(x),):raise ValueError('Matching source IQ pairs and fit labels required')
 with torch.no_grad():
  logits=teacher.classify_features(teacher.features(clean));confidence,pseudo=F.softmax(logits,dim=1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);target,valid,weight=select_targets(labels,confidence,pseudo,raw)
 restored=net.restore(x,steps=4);out=teacher.classify_features(teacher.features(restored));per=F.cross_entropy(out,target.long(),reduction='none');loss=(per[valid]*weight[valid]).sum()/weight[valid].sum() if valid.any() else out.sum()*0
 stats=dict(labeled=int((labels>=0).sum()),pseudo=int(((labels<0)&valid).sum()),uncovered=int((~valid).sum()),hard=int((valid&(raw!=target)).sum()))
 return loss,stats
