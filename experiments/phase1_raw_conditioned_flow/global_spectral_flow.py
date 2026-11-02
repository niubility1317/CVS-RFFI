"""Phase-equivariant conditional flow with a global adaptive spectral path."""
import torch
from torch import nn
from torch.nn import functional as F
from raw_conditioned_flow import RawConditionedFlow

class GlobalSpectralFlow(RawConditionedFlow):
 def __init__(self,width=48):
  super().__init__(width);self.spectral_entry=nn.Conv1d(5,width,7,padding=3,padding_mode='circular');self.spectral_blocks=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d,padding_mode='circular') for d in (1,2,4,8));self.spectral_output=nn.Conv1d(width,4,1);self.frequency_bias=nn.Parameter(torch.zeros(1,4,256));nn.init.zeros_(self.spectral_output.weight);nn.init.zeros_(self.spectral_output.bias)
 def forward(self,state,observation,time):
  local=super().forward(state,observation,time);x=torch.complex(observation[:,0],observation[:,1]);z=torch.complex(state[:,0],state[:,1]);X=torch.fft.fft(x,norm='ortho');Z=torch.fft.fft(z,norm='ortho');scale=X.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt();cross=Z*X.conj()/scale.square();features=torch.stack((X.abs()/scale,Z.abs()/scale,cross.real,cross.imag,time[:,None].expand_as(cross.real)),1);h=F.silu(self.spectral_entry(features))
  for block in self.spectral_blocks:h=h+.25*F.silu(block(h))
  w=(self.spectral_output(h)+self.frequency_bias).tanh();a=torch.complex(w[:,0],w[:,1]);b=torch.complex(w[:,2],w[:,3]);v=torch.fft.ifft(.5*(a*X+b*Z),norm='ortho');return local+torch.stack((v.real,v.imag),1)
