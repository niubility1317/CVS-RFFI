"""Choose among five fixed DURA-free IQ candidates using gate probability."""
import torch
from torch import nn
from torch.nn import functional as F

class GateAwareRouter(nn.Module):
 def __init__(self):
  super().__init__();self.network=nn.Sequential(nn.Linear(33,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,5));nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
  with torch.no_grad():self.network[-1].bias[1]=1.
 def forward(self,predictions,gate):
  if predictions.ndim!=2 or predictions.shape[1]!=5 or predictions.dtype!=torch.long or gate.shape!=(len(predictions),) or not torch.isfinite(gate).all() or torch.any((predictions<0)|(predictions>=6)) or torch.any((gate<0)|(gate>1)):raise ValueError('Five six-class decisions and a valid gate probability required')
  q=gate.float();features=torch.cat((F.one_hot(predictions,num_classes=6).float().flatten(1),q[:,None],q.square()[:,None],q.pow(3)[:,None]),1);return self.network(features)
 def choose(self,predictions,gate):
  choice=self(predictions,gate).argmax(1);return choice,predictions.gather(1,choice[:,None]).squeeze(1)

def select_waveforms(waveforms,choice):
 if waveforms.ndim!=4 or waveforms.shape[1:]!=(5,2,256) or choice.shape!=(len(waveforms),) or choice.dtype!=torch.long or torch.any((choice<0)|(choice>=5)):raise ValueError('Five candidate IQ waveforms and valid per-record choices required')
 return waveforms[torch.arange(len(waveforms),device=waveforms.device),choice]
