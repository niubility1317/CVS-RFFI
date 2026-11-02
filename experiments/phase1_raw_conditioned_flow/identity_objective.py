"""Source-pair flow objective with clean identity and frozen-F preservation."""
import torch
from torch.nn import functional as F

def identity_objective(net,teacher,x,clean,aligned_clean,time):
 if x.shape!=clean.shape or x.shape!=aligned_clean.shape or x.ndim!=3 or x.shape[1:]!=(2,256) or time.shape!=(len(x),):raise ValueError('Matching source IQ pairs and per-record time required')
 state=(1-time[:,None,None])*x+time[:,None,None]*aligned_clean;velocity=net(state,x,time);target=aligned_clean-x;den=clean.square().sum((1,2)).clamp_min(1e-8);flow=((velocity-target).square().sum((1,2))/den).mean()
 clean_velocity=net(clean,clean,time);clean_loss=(clean_velocity.square().sum((1,2))/clean.square().sum((1,2)).clamp_min(1e-8)).mean()
 with torch.no_grad():clean_features=teacher.features(clean);clean_probs=F.softmax(teacher.classify_features(clean_features)/2,dim=1)
 estimate=x+net(x,x,torch.zeros_like(time));features=teacher.features(estimate);logits=teacher.classify_features(features);feature=(1-F.cosine_similarity(features,clean_features,dim=1,eps=1e-8)).mean();kl=F.kl_div(F.log_softmax(logits/2,dim=1),clean_probs,reduction='batchmean')*4
 loss=flow+.5*clean_loss+.2*feature+.1*kl
 return loss,dict(flow=flow,clean=clean_loss,feature=feature,kl=kl)
