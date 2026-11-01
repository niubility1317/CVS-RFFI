"""Physics-consistency conditioned residual correction in ground IQ coordinates."""
import torch
from torch import nn
from torch.nn import functional as F

def backproject(x,z,A):
 if x.shape!=z.shape or x.ndim!=3 or x.shape[1:]!=(2,256) or A.shape!=(len(x),512,512):raise ValueError('Expected paired IQ and real action matrix')
 residual=x.reshape(len(x),512,1)-torch.bmm(A,z.reshape(len(z),512,1));scale=A.square().sum((1,2)).div(512).clamp_min(1e-6);return (torch.bmm(A.transpose(1,2),residual).squeeze(-1)/scale[:,None]).reshape_as(x)

class ResidualCorrector(nn.Module):
 def __init__(self):
  super().__init__();self.input=nn.Conv1d(4,64,7,padding=3);self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,5,padding=2*d,dilation=d),nn.SiLU(),nn.Conv1d(64,64,1)) for d in (1,2,4,8)]);self.output=nn.Conv1d(64,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,z,b):
  rms=z.square().sum(1).mean(1).sqrt().clamp_min(1e-8);v=torch.cat((z/rms[:,None,None],(b/rms[:,None,None]).clamp(-8,8)),1);v=F.silu(self.input(v))
  for block in self.blocks:v=v+block(v)
  return z+.2*rms[:,None,None]*torch.tanh(self.output(v))

def hybrid_loss(student,soft,labels):
 kl=F.kl_div(F.log_softmax(student/2,dim=1),soft,reduction='batchmean')*4;labeled=labels>=0
 if labeled.any():return .1*kl+.15*F.cross_entropy(student[labeled],labels[labeled])
 return .1*kl
