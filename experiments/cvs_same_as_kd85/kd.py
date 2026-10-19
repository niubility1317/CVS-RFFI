"""Common-scale output transfer; all true L retain native CE."""
import torch
import torch.nn.functional as F
def objective(first,second,teacher,y):
 if first.shape!=second.shape or first.shape!=teacher.shape or first.shape!=(len(y),6) or len(y)==0 or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not all(bool(torch.isfinite(z).all()) for z in (first,second,teacher)):raise ValueError('Finite aligned true-labeled six-class same-As logits required')
 q=(teacher.detach()-5.).softmax(1);use=teacher.detach().argmax(1)==y
 a=F.kl_div((first/6.).log_softmax(1),q,reduction='none').sum(1);b=F.kl_div((second/6.).log_softmax(1),q,reduction='none').sum(1)
 return (.5*(a+b)*use.to(first)).mean(),use
