"""Class-aware selector over five fixed candidate IQ waveforms."""
import torch
from torch import nn
from torch.nn import functional as F

class ClassAwareRouter(nn.Module):
 def __init__(self):
  super().__init__();self.network=nn.Sequential(nn.Linear(30,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,5));nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
  with torch.no_grad():self.network[-1].bias[4]=1.
 def forward(self,predictions):
  if predictions.ndim!=2 or predictions.shape[1]!=5 or predictions.dtype!=torch.long or torch.any((predictions<0)|(predictions>=6)):raise ValueError('Five candidate six-class decisions required')
  return self.network(F.one_hot(predictions,num_classes=6).float().flatten(1))
 def choose(self,predictions):
  choice=self(predictions).argmax(1);return choice,predictions.gather(1,choice[:,None]).squeeze(1)

def select_waveforms(waveforms,choice):
 if waveforms.ndim!=4 or waveforms.shape[1:]!=(5,2,256) or choice.shape!=(len(waveforms),) or choice.dtype!=torch.long or torch.any((choice<0)|(choice>=5)):raise ValueError('Five candidate IQ waveforms and valid per-record choices required')
 return waveforms[torch.arange(len(waveforms),device=waveforms.device),choice]

def multi_positive_loss(scores,predictions,target,weights):
 if scores.ndim!=2 or scores.shape[1]!=5 or predictions.shape!=scores.shape or target.shape!=(len(scores),) or weights.shape!=(len(scores),):raise ValueError('Matching candidate scores, labels and weights required')
 positive=predictions==target[:,None]
 if not positive.any(1).all() or not torch.isfinite(scores).all() or not torch.isfinite(weights).all() or (weights<=0).any():raise ValueError('Finite covered targets and positive weights required')
 numerator=torch.logsumexp(scores.masked_fill(~positive,-torch.inf),dim=1);denominator=torch.logsumexp(scores,dim=1);return ((denominator-numerator)*weights).mean()
