"""Phase-equivariant synthesis of original and physical-posterior IQ waves."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SpectralSynthesis(nn.Module):
 def __init__(self,width=48):
  super().__init__();self.entry=nn.Conv1d(22,width,7,padding=3,padding_mode='circular');self.blocks=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d,padding_mode='circular') for d in (1,2,4,8));self.output=nn.Conv1d(width,14,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,base,candidates,logvar):
  if base.ndim!=3 or base.shape[1:]!=(2,256) or candidates.shape!=(len(base),7,2,256) or logvar.shape!=(len(base),):raise ValueError('Base IQ, seven IQ candidates and noise scale required')
  b=torch.fft.fft(torch.complex(base[:,0],base[:,1]),norm='ortho');c=torch.fft.fft(torch.complex(candidates[:,:,0],candidates[:,:,1]),norm='ortho');scale=b.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt();cross=c*b[:,None,:].conj()/scale[:,None,:].square();lv=(logvar.clamp(-12,4)/12)[:,None,None].expand(-1,1,256);features=torch.cat((c.abs()/scale[:,None,:],cross.real,cross.imag,lv),1);h=F.silu(self.entry(features))
  for block in self.blocks:h=h+F.silu(block(h))*.25
  coeff=self.output(h).tanh().reshape(len(base),2,7,256);weight=torch.complex(coeff[:,0],coeff[:,1]);combined=b+.2/math.sqrt(7)*(weight*(c-b[:,None,:])).sum(1);out=torch.fft.ifft(combined,norm='ortho');return torch.stack((out.real,out.imag),1)
