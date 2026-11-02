"""Phase-equivariant IQ flow conditioned directly on the received waveform."""
import torch
from torch import nn
from torch.nn import functional as F

class RawConditionedFlow(nn.Module):
 def __init__(self,width=48):
  super().__init__();self.entry=nn.Conv1d(5,width,7,padding=3);self.blocks=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d) for d in (1,2,4,8));self.filter_real=nn.Conv1d(1,2,7,padding=3,bias=False);self.filter_imag=nn.Conv1d(1,2,7,padding=3,bias=False);self.output=nn.Conv1d(width,12,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,state,observation,time):
  if state.ndim!=3 or state.shape[1:]!=(2,256) or observation.shape!=state.shape or time.shape!=(len(state),):raise ValueError('Matching IQ [batch,2,256] and per-sample time required')
  x=torch.complex(observation[:,0],observation[:,1]);z=torch.complex(state[:,0],state[:,1]);scale=x.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt();cross=z*x.conj()/scale.square();features=torch.stack((x.abs()/scale,z.abs()/scale,cross.real,cross.imag,time[:,None].expand_as(cross.real)),1);h=F.silu(self.entry(features))
  for block in self.blocks:h=h+.25*F.silu(block(h))
  def filtered(q):
   real=self.filter_real(q.real[:,None])-self.filter_imag(q.imag[:,None]);imag=self.filter_imag(q.real[:,None])+self.filter_real(q.imag[:,None]);return torch.complex(real,imag)
  basis=torch.cat((x[:,None],z[:,None],filtered(x),filtered(z)),1);w=self.output(h).tanh().reshape(len(x),2,6,256);coeff=torch.complex(w[:,0],w[:,1]);v=(coeff*basis).sum(1)/6;return torch.stack((v.real,v.imag),1)
 def restore(self,observation,steps=4):
  if not isinstance(steps,int) or steps<1:raise ValueError('Positive integer steps required')
  state=observation
  for i in range(steps):
   t=observation.new_full((len(observation),),i/steps);state=state+self(state,observation,t)/steps
  return state
