"""Phase-equivariant nonlocal posterior-waveform synthesis."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class PosteriorNonlocalSynthesis(nn.Module):
 def __init__(self,width=48):
  super().__init__()
  self.conv_real=nn.Conv1d(8,8,9,padding=4,bias=False)
  self.conv_imag=nn.Conv1d(8,8,9,padding=4,bias=False)
  self.entry=nn.Conv1d(25,width,7,padding=3)
  self.layers=nn.ModuleList(nn.Conv1d(width,width,5,padding=2*d,dilation=d) for d in (1,2,4,8,16))
  self.coefficients=nn.Conv1d(width,16,1)
  nn.init.zeros_(self.coefficients.weight);nn.init.zeros_(self.coefficients.bias)
 def forward(self,base,candidates,logvar):
  if base.ndim!=3 or base.shape[1:]!=(2,256) or candidates.shape!=(len(base),7,2,256) or logvar.shape!=(len(base),):raise ValueError('Expected base [N,2,256], candidates [N,7,2,256], logvar [N]')
  streams=torch.cat((base[:,None],candidates),dim=1);r=streams[:,:,0];i=streams[:,:,1];scale=streams.square().mean((1,2,3),keepdim=False).clamp_min(1e-8).sqrt()[:,None,None];r=r/scale;i=i/scale
  mag=torch.sqrt((r.square()+i.square()).clamp_min(1e-12));br=r[:,0:1];bi=i[:,0:1];features=torch.cat((mag,r*br+i*bi,i*br-r*bi,(logvar.clamp(-12,4)/12)[:,None,None].expand(-1,1,256)),dim=1)
  h=F.silu(self.entry(features))
  for layer in self.layers:h=h+.2*F.silu(layer(h))
  coeff=self.coefficients(h).tanh();cr,ci=coeff[:,:8],coeff[:,8:];vr=self.conv_real(r)-self.conv_imag(i);vi=self.conv_real(i)+self.conv_imag(r);dr=(cr*vr-ci*vi).sum(1);di=(cr*vi+ci*vr).sum(1);delta=torch.stack((dr,di),1)*scale*(.2/math.sqrt(8))
  return base+delta
