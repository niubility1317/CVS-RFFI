"""Blind input-only corruption detector for direct IQ restoration."""
import torch
from torch import nn
from torch.nn import functional as F

class BlindGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(5,48,9,padding=4),nn.GELU(),nn.Conv1d(48,64,7,stride=2,padding=3),nn.GELU(),nn.Conv1d(64,96,7,stride=2,padding=3),nn.GELU(),nn.Conv1d(96,128,5,stride=2,padding=2),nn.GELU())
        self.head=nn.Sequential(nn.Linear(288,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,iq):
        if iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('Expected [batch,2,256] IQ')
        x=iq/iq.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt()
        delta=x-torch.roll(x,1,dims=-1);magnitude=x.square().sum(1,keepdim=True).clamp_min(1e-12).sqrt()
        h=self.conv(torch.cat((x,delta,magnitude),1));mean=h.mean(-1);std=h.var(-1,unbiased=False).clamp_min(1e-12).sqrt()
        z=torch.complex(x[:,0],x[:,1]);power=torch.fft.fftshift(torch.fft.fft(z).abs().square(),dim=-1);power=power/power.mean(-1,keepdim=True).clamp_min(1e-12)
        spectrum=F.adaptive_avg_pool1d(torch.log1p(power)[:,None,:],32).squeeze(1)
        return self.head(torch.cat((mean,std,spectrum),1)).squeeze(1)
