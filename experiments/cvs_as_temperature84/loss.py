"""Temperature changes only true-labeled satellite CE; default logits stay intact."""
import math
import torch
import torch.nn.functional as F
def pair(first,second,y,T):
 if not isinstance(T,(int,float)) or isinstance(T,bool) or not math.isfinite(T) or T<=0:raise ValueError('Positive finite T required')
 if first.shape!=second.shape or first.shape!=(len(y),6) or len(y)==0 or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not torch.isfinite(first).all() or not torch.isfinite(second).all():raise ValueError('Finite six-class true L logits required')
 a=T*F.cross_entropy(first/T,y,label_smoothing=.01);b=T*F.cross_entropy(second/T,y,label_smoothing=.01)
 return .5*(a+b),a,b
def mixed(first,second,y,mask,T,ordinary_terms):
 if mask.dtype!=torch.bool or mask.shape!=y.shape or int(mask.sum())!=64 or len(y)!=128:raise ValueError('Existing mixed 64+64 slots required')
 rd,oc,ok,o0,o1=ordinary_terms(first[~mask],second[~mask],y[~mask],1.)
 sc,s0,s1=pair(first[mask],second[mask],y[mask],T)
 ce=.5*(oc+sc);kl=.5*ok
 return ce+kl,ce,kl,.5*(o0+s0),.5*(o1+s1)
