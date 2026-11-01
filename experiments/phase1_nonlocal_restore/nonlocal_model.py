"""Noise-conditioned nonlocal IQ restoration around a frozen blind posterior."""
import torch
from torch import nn
from torch.nn import functional as F

class NonlocalBlock(nn.Module):
    def __init__(self,width=64,heads=4):
        super().__init__();self.n1=nn.LayerNorm(width);self.attn=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True);self.n2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))
    def forward(self,x):
        y=self.n1(x);x=x+self.attn(y,y,y,need_weights=False)[0];return x+self.ff(self.n2(x))

class NoiseNonlocalRestorer(nn.Module):
    def __init__(self,epsilon=.2):
        super().__init__()
        if not 0<epsilon<=.5:raise ValueError('Valid residual bound required')
        self.epsilon=epsilon;self.input=nn.Conv1d(7,64,7,padding=3);self.position=nn.Parameter(torch.zeros(1,256,64));self.blocks=nn.ModuleList([NonlocalBlock() for _ in range(2)]);self.output=nn.Conv1d(64,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,received,parent,backproject,log_noise):
        if received.shape!=parent.shape or parent.shape!=backproject.shape or parent.ndim!=3 or parent.shape[1:]!=(2,256) or log_noise.shape!=(len(parent),):raise ValueError('Matched IQ and per-record noise required')
        rms=parent.square().sum(1).mean(1).sqrt().clamp_min(1e-8);noise=((log_noise-2*rms.log()).clamp(-12,4)/8)[:,None,None].expand(-1,1,256);v=torch.cat((received/rms[:,None,None],parent/rms[:,None,None],(backproject/rms[:,None,None]).clamp(-8,8),noise),1);h=F.silu(self.input(v)).transpose(1,2)+self.position
        for block in self.blocks:h=block(h)
        raw=self.epsilon*rms[:,None,None]*torch.tanh(self.output(h.transpose(1,2)));size=raw.flatten(1).norm(dim=1).clamp_min(1e-12);radius=self.epsilon*parent.flatten(1).norm(dim=1);factor=torch.minimum(torch.ones_like(size),radius/size);return parent+raw*factor[:,None,None]
