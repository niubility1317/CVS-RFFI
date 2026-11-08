"""Joint time/frequency reconstruction from two frozen source-only experts."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
            nn.GroupNorm(8,width),nn.GELU(),nn.Conv1d(width,width,3,padding=1),nn.GroupNorm(8,width))
        self.act=nn.GELU()

    def forward(self,x):
        return self.act(x+self.body(x))


class CrossExpertRefiner(nn.Module):
    """Start at half mixture; learn unrestricted time/frequency IQ corrections."""
    def __init__(self,width=192):
        super().__init__()
        self.time_in=nn.Sequential(nn.Conv1d(9,width,7,padding=3),nn.GroupNorm(8,width),nn.GELU())
        self.time_blocks=nn.Sequential(*(ResidualBlock(width,d) for d in (1,2,4,8,16,32,1,2)))
        self.time_out=nn.Conv1d(width,2,3,padding=1)
        self.freq_in=nn.Sequential(nn.Conv1d(8,width,5,padding=2),nn.GroupNorm(8,width),nn.GELU())
        self.freq_blocks=nn.Sequential(*(ResidualBlock(width,d) for d in (1,2,4,8)))
        self.freq_out=nn.Conv1d(width,2,3,padding=1)
        nn.init.zeros_(self.time_out.weight);nn.init.zeros_(self.time_out.bias)
        nn.init.zeros_(self.freq_out.weight);nn.init.zeros_(self.freq_out.bias)

    def forward(self,received,anchor,innovation,proposal,variance):
        if (received.ndim!=3 or received.shape[1:]!=(2,256) or anchor.shape!=received.shape
                or innovation.shape!=received.shape or proposal.shape!=received.shape or variance.shape!=(len(received),)):
            raise ValueError('Matching 256-sample IQ and per-record variance required')
        power=anchor.square().mean((1,2),keepdim=True).clamp_min(1e-12)
        scale=power.sqrt()
        delta=proposal-anchor
        noise=torch.log1p(variance[:,None,None].clamp_min(0)/power).expand(-1,1,256)
        signals=(received/scale,anchor/scale,innovation/scale,delta/scale)
        time_input=torch.cat((*signals,noise),dim=1)
        spectrum=[torch.fft.fft(torch.complex(z[:,0],z[:,1]),norm='ortho') for z in signals]
        freq_input=torch.cat([torch.stack((z.real,z.imag),dim=1) for z in spectrum],dim=1)
        time_delta=self.time_out(self.time_blocks(self.time_in(time_input)))
        frequency_delta=self.freq_out(self.freq_blocks(self.freq_in(freq_input)))
        inverse=torch.fft.ifft(torch.complex(frequency_delta[:,0],frequency_delta[:,1]),norm='ortho')
        spectral_delta=torch.stack((inverse.real,inverse.imag),dim=1)
        return anchor+.5*delta+scale*(time_delta+spectral_delta)
