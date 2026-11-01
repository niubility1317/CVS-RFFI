"""Input-conditioned, globally bounded identity-aware IQ correction."""
import torch
from torch import nn
from torch.nn import functional as F

class IdentityStudent(nn.Module):
    def __init__(self,epsilon=.02):
        super().__init__()
        if not 0<epsilon<=.1:raise ValueError('Valid waveform correction budget required')
        self.epsilon=epsilon;self.input=nn.Conv1d(6,64,7,padding=3);self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,5,dilation=d,padding=2*d),nn.SiLU(),nn.Conv1d(64,64,1)) for d in (1,2,4,8,16)]);self.output=nn.Conv1d(64,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,received,parent,backproject):
        if received.shape!=parent.shape or parent.shape!=backproject.shape or parent.ndim!=3 or parent.shape[1:]!=(2,256):raise ValueError('Matched IQ waveforms [B,2,256] required')
        rms=parent.square().sum(1).mean(1).sqrt().clamp_min(1e-8);v=torch.cat((received/rms[:,None,None],parent/rms[:,None,None],(backproject/rms[:,None,None]).clamp(-8,8)),1);v=F.silu(self.input(v))
        for block in self.blocks:v=v+block(v)
        raw=self.epsilon*rms[:,None,None]*torch.tanh(self.output(v));size=raw.flatten(1).norm(dim=1).clamp_min(1e-12);radius=self.epsilon*parent.flatten(1).norm(dim=1);factor=torch.minimum(torch.ones_like(size),radius/size);return parent+raw*factor[:,None,None]
