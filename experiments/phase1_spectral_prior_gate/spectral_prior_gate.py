"""Source-pilot spectral correction to a label-free waveform prior."""
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,c,d):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d,padding_mode='circular'),nn.SiLU(),nn.Conv1d(c,c,1))
 def forward(self,x):return x+.2*self.net(x)
class SpectralPriorGate(nn.Module):
 def __init__(self,channels=48):
  super().__init__();self.input=nn.Conv1d(6,channels,5,padding=2,padding_mode='circular');self.blocks=nn.Sequential(*(Block(channels,d) for d in (1,2,4,8)));self.output=nn.Conv1d(channels,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,iq,prior):
  if iq.shape!=prior.shape or iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('matching [batch,2,256] IQ required')
  rms=iq.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt();x=iq/rms;t=prior/rms;z=torch.complex(x[:,0],x[:,1]);v=torch.complex(t[:,0],t[:,1]);X=torch.fft.fft(z,norm='ortho');T=torch.fft.fft(v,norm='ortho');D=T-X;features=torch.stack((X.real,X.imag,T.real,T.imag,D.real,D.imag),1);p=self.output(self.blocks(self.input(features)));gain=torch.complex(torch.tanh(p[:,0]),torch.tanh(p[:,1]));y=torch.fft.ifft(X+gain*D,norm='ortho');return torch.stack((y.real,y.imag),1)*rms
