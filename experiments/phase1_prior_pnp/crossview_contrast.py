"""Class-aware paired-view feature contrast for source restoration."""
import torch
from torch.nn import functional as F

def paired_class_contrast(first,second,labels,temperature=.2):
 if first.ndim!=2 or first.shape!=second.shape or labels.shape!=(len(first),) or len(first)==0 or temperature<=0 or not torch.isfinite(first).all() or not torch.isfinite(second).all():raise ValueError('Finite paired features, labels and positive temperature required')
 count=len(first);features=F.normalize(torch.cat((first,second),dim=0),dim=1,eps=1e-8);classes=labels.repeat(2);positions=torch.arange(count,device=first.device).repeat(2);self_mask=torch.eye(2*count,dtype=torch.bool,device=first.device);same_example=positions[:,None]==positions[None,:];known_same=(classes[:,None]>=0)&(classes[None,:]>=0)&(classes[:,None]==classes[None,:]);positive=(same_example|known_same)&~self_mask;similarity=(features@features.T/temperature).masked_fill(self_mask,-torch.inf);logprob=similarity-torch.logsumexp(similarity,dim=1,keepdim=True);selected=torch.where(positive,logprob,torch.zeros_like(logprob));return -(selected.sum(1)/positive.sum(1)).mean()
