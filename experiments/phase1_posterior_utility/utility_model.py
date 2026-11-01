"""Shared pairwise utility model for fixed conditional posterior IQ hypotheses."""
import torch
from torch import nn
from torch.nn import functional as F

FEATURES=347

def build_features(parent_feature,candidate_feature,parent_logits,candidate_logits,log_weight,log_noise,received,action,parent,states):
    n=len(received);c=6
    if parent_feature.shape!=(n,160) or candidate_feature.shape!=(n,c,160) or parent_logits.shape!=(n,c) or candidate_logits.shape!=(n,c,c) or log_weight.shape!=(n,c) or log_noise.shape!=(n,) or received.shape!=(n,512) or action.shape!=(n,512,512) or parent.shape!=(n,512) or states.shape!=(n,c,512):raise ValueError('Aligned posterior, classifier and IQ features required')
    pf=F.normalize(parent_feature,dim=1);cf=F.normalize(candidate_feature,dim=2);base=pf[:,None,:].expand(-1,c,-1);delta=cf-base;pbase=F.softmax(parent_logits,1)[:,None,:].expand(-1,c,-1);pcand=F.softmax(candidate_logits,2);weight=log_weight.exp()[:,None,:].expand(-1,c,-1);onehot=torch.eye(c,device=received.device,dtype=received.dtype)[None].expand(n,-1,-1);power=received.square().mean(1).clamp_min(1e-8);noise=((log_noise-power.log()).clamp(-12,4)/8)[:,None,None].expand(-1,c,-1);predicted=torch.einsum('bij,bkj->bki',action,states);residual=(predicted-received[:,None,:]).square().sum(2)/received.square().sum(1).clamp_min(1e-8)[:,None];change=(states-parent[:,None,:]).square().sum(2)/parent.square().sum(1).clamp_min(1e-8)[:,None];result=torch.cat((base,delta,pbase,pcand,weight,onehot,noise,torch.log1p(residual)[:,:,None],torch.log1p(change)[:,:,None]),2)
    if result.shape!=(n,c,FEATURES) or not torch.isfinite(result).all():raise ValueError('Finite pairwise features required')
    return result

class UtilityNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(FEATURES,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,features):
        if features.ndim!=3 or features.shape[1:]!=(6,FEATURES):raise ValueError('Six pairwise feature vectors required')
        return self.net(features).squeeze(-1)

def select_candidate(scores,parent_pred,candidate_pred,threshold):
    if scores.ndim!=2 or scores.shape[1]!=6 or parent_pred.shape!=(len(scores),) or candidate_pred.shape!=scores.shape or not 0<=threshold<=1:raise ValueError('Matched candidate scores and threshold required')
    eligible=candidate_pred!=parent_pred[:,None];prob=torch.sigmoid(scores).masked_fill(~eligible,-1);best,index=prob.max(1);return torch.where(best>=threshold,index,torch.full_like(index,-1))
