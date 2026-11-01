"""Direct input-conditioned spectral and multiscale temporal IQ restoration."""
import torch
from torch import nn
from torch.nn import functional as F

class DirectSpectralRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(2,32,9,padding=4),nn.SiLU(),nn.Conv1d(32,64,7,stride=2,padding=3),nn.SiLU(),nn.Conv1d(64,96,7,stride=2,padding=3),nn.SiLU())
        self.spectrum=nn.Linear(96,512)
        nn.init.zeros_(self.spectrum.weight);nn.init.zeros_(self.spectrum.bias)
        self.input=nn.Conv1d(4,64,7,padding=3)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,5,padding=2*d,dilation=d),nn.SiLU(),nn.Conv1d(64,64,1)) for d in (1,2,4,8,16)])
        self.output=nn.Conv1d(64,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('IQ must be [batch,2,256]')
        rms=x.square().sum(1).mean(1).sqrt().clamp_min(1e-8)
        v=x/rms[:,None,None]
        context=self.encoder(v).mean(-1)
        coeff=self.spectrum(context).reshape(-1,2,256)
        filt=torch.complex(1+2*torch.tanh(coeff[:,0]),2*torch.tanh(coeff[:,1]))
        signal=torch.complex(v[:,0],v[:,1])
        spectral=torch.fft.ifft(torch.fft.fft(signal)*filt)
        w=torch.stack((spectral.real,spectral.imag),dim=1)
        h=F.silu(self.input(torch.cat((v,w),1)))
        for block in self.blocks:h=h+block(h)
        return (w+self.output(h))*rms[:,None,None]
