"""DURA-anchored complex fusion of a frozen independent physical posterior."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase1_dura_blendgate.model import align_to_anchor


class PhysicalFusion(nn.Module):
 def __init__(self,width=32):
  super().__init__()
  self.entry=nn.Conv1d(8,width,7,padding=3)
  self.layers=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d) for d in (1,2,4,8))
  self.gate_output=nn.Conv1d(width,2,1)
  nn.init.zeros_(self.gate_output.weight);nn.init.zeros_(self.gate_output.bias)

 def forward(self,received,packed,log_variance):
  n=len(received)
  if received.shape!=(n,2,256) or packed.shape!=(n,4,256) or log_variance.shape!=(n,):
   raise ValueError('Matching received, DURA and physical posterior windows required')
  if not all(torch.isfinite(v).all() for v in (received,packed,log_variance)):
   raise FloatingPointError('Finite same-record evidence required')
  anchor=packed[:,:2];physical=align_to_anchor(packed[:,2:],anchor,preserve_edges=True)
  power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8);scale=power.sqrt()
  x=torch.complex(received[:,0],received[:,1])/scale[:,0]
  a=torch.complex(anchor[:,0],anchor[:,1])/scale[:,0]
  b=torch.complex(physical[:,0],physical[:,1])/scale[:,0]
  ab=a.conj()*b;ax=a.conj()*x
  noise=((log_variance-power[:,0,0].log()).clamp(-16,8)/8)[:,None].expand(-1,256)
  features=torch.stack((torch.log1p(x.abs().square()),torch.log1p(a.abs().square()),
      torch.log1p(b.abs().square()),ab.real.clamp(-16,16),ab.imag.clamp(-16,16),
      ax.real.clamp(-16,16),ax.imag.clamp(-16,16),noise),dim=1)
  h=F.silu(self.entry(features))
  for layer in self.layers:h=h+.2*F.silu(layer(h))
  w=.5*self.gate_output(h).tanh();gain=torch.complex(w[:,0],w[:,1])
  delta=gain*torch.complex(physical[:,0]-anchor[:,0],physical[:,1]-anchor[:,1])
  return anchor+torch.stack((delta.real,delta.imag),dim=1)
