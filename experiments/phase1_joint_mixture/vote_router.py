"""Choose among independent restoration waveforms using class-vote patterns."""
import torch
from torch import nn
from torch.nn import functional as F

class VoteRouter(nn.Module):
 def __init__(self,candidates=9):
  super().__init__()
  if candidates!=9:raise ValueError('Exactly nine fixed candidates required')
  self.candidates=candidates;self.network=nn.Sequential(nn.Linear(candidates+1,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,1));nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias);self.register_buffer('priority',torch.tensor([1,0,2,3,4,5,6,7,8],dtype=torch.long))
 def forward(self,predictions):
  if predictions.ndim!=2 or predictions.shape[1]!=self.candidates or predictions.dtype!=torch.long or (predictions<0).any() or (predictions>=6).any():raise ValueError('Nine valid candidate class predictions required')
  votes=F.one_hot(predictions,num_classes=6).transpose(1,2).to(self.network[0].weight.dtype);features=torch.cat((votes,votes.mean(-1,keepdim=True)),dim=-1);scores=self.network(features).squeeze(-1);return scores.masked_fill(votes.sum(-1)==0,-1e4)
 def choose(self,predictions):
  scores=self(predictions);ordered=predictions[:,self.priority];available=scores.gather(1,ordered);best=available.argmax(dim=1);choice=self.priority[best];return choice,predictions.gather(1,choice[:,None]).squeeze(1)

def choose_waveforms(waves,choice):
 if waves.ndim!=4 or waves.shape[2]!=2 or choice.shape!=(len(waves),) or choice.dtype!=torch.long or (choice<0).any() or (choice>=waves.shape[1]).any():raise ValueError('Batched IQ candidates and valid choice required')
 return waves[torch.arange(len(waves),device=waves.device),choice]
