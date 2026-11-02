"""Learn data-driven mixture weights without anchoring to saturated posterior logits."""
import torch
from torch import nn
from torch.nn import functional as F
from mixture_controller import FEATURES, COMPONENTS

class FreeMixtureController(nn.Module):
 def __init__(self):
  super().__init__();self.network=nn.Sequential(nn.Linear(COMPONENTS*FEATURES,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,COMPONENTS));nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
 def forward(self,features,log_weights,states,scale=1.):
  n=len(features)
  if features.shape!=(n,COMPONENTS,FEATURES) or log_weights.shape!=(n,COMPONENTS) or states.shape!=(n,COMPONENTS,512) or not 0<=scale<=1:raise ValueError('Matched six-component posterior and blend in [0,1] required')
  logits=self.network(features.reshape(n,-1));weights=F.softmax(logits,dim=1);free=(states*weights.unsqueeze(-1)).sum(1);parent=(states*log_weights.exp().unsqueeze(-1)).sum(1);wave=((1-scale)*parent+scale*free).reshape(n,2,256);return wave,weights,logits
