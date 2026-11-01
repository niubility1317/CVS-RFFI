"""Source-pilot input-only periodic self-reference IQ restorer."""
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,c,d):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d),nn.SiLU(),nn.Conv1d(c,c,1))
 def forward(self,x):return x+.2*self.net(x)
class SelfReferenceNet(nn.Module):
 def __init__(self,channels=64):
  super().__init__();self.input=nn.Conv1d(5,channels,9,padding=4);self.blocks=nn.Sequential(*(Block(channels,d) for d in (1,2,4,8,16,32)));self.output=nn.Conv1d(channels,4,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 @staticmethod
 def reference(z):
  a=z[:,:64];b=z[:,80:144];dot=(a*b.conj()).sum(1);rot=dot/dot.abs().clamp_min(1e-12);corr=dot.abs()/(a.abs().square().sum(1)*b.abs().square().sum(1)).clamp_min(1e-20).sqrt();ref=z.clone();ref[:,:64]=rot[:,None]*b;ref[:,80:144]=rot.conj()[:,None]*a;return ref,corr
 def forward(self,iq):
  if iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('Expected [batch,2,256] IQ')
  rms=iq.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt();x=iq/rms;z=torch.complex(x[:,0],x[:,1]);ref,corr=self.reference(z);diff=ref-z;features=torch.stack((z.real,z.imag,diff.real,diff.imag,corr[:,None].expand_as(z.real)),1);p=self.output(self.blocks(self.input(features)));gain=torch.complex(torch.tanh(p[:,0]),torch.tanh(p[:,1]));residual=torch.complex(torch.tanh(p[:,2]),torch.tanh(p[:,3]));out=z+gain*diff+.5*residual;return torch.stack((out.real,out.imag),1)*rms
