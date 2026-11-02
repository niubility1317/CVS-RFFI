"""Phase-equivariant multi-resolution IQ residual recovery."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def split_haar(z):
    if z.ndim != 2 or z.shape[-1] % 2:
        raise ValueError('Even-length complex batches are required')
    return (z[:,0::2]+z[:,1::2])/math.sqrt(2), (z[:,0::2]-z[:,1::2])/math.sqrt(2)

def join_haar(low, high):
    if low.shape != high.shape:
        raise ValueError('Matching Haar coefficients are required')
    return torch.stack(((low+high)/math.sqrt(2),(low-high)/math.sqrt(2)),dim=-1).flatten(1)

class WaveletLifting(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry=nn.Conv1d(10,48,5,padding=2,padding_mode='reflect')
        self.blocks=nn.ModuleList([nn.Conv1d(48,48,5,padding=2*d,dilation=d,padding_mode='reflect') for d in (1,2,4)])
        self.gate_output=nn.Conv1d(48,8,1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def _band(self, zh, ah, zl, al, noise, level):
        hcross=zh*ah.conj();lcross=zl*al.conj()
        features=torch.stack((zh.abs(),ah.abs(),zl.abs(),al.abs(),hcross.real,hcross.imag,lcross.real,lcross.imag),dim=1)
        shape=(len(zh),1,zh.shape[-1])
        level_feature=features.new_full(shape,level/4)
        h=F.silu(self.entry(torch.cat((features,noise.expand(shape),level_feature),dim=1)))
        for block in self.blocks:h=h+0.2*F.silu(block(h))
        w=self.gate_output(h).tanh()
        return (torch.complex(w[:,0],w[:,1])*ah + torch.complex(w[:,2],w[:,3])*zh
                + torch.complex(w[:,4],w[:,5])*al + torch.complex(w[:,6],w[:,7])*zl)

    def forward(self, received, initial, log_variance):
        if (received.ndim!=3 or received.shape[1:]!=(2,256) or initial.shape!=received.shape
                or log_variance.shape!=(received.shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(t).all() for t in (received,initial,log_variance)):
            raise FloatingPointError('Finite received evidence is required')
        power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8);scale=power.sqrt()
        z=torch.complex(received[:,0],received[:,1])/scale[:,0]
        a=torch.complex(initial[:,0],initial[:,1])/scale[:,0]
        noise=((log_variance-power[:,0,0].log()).clamp(-16,8)/8)[:,None,None]
        details=[]
        for level in range(1,5):
            z,zh=split_haar(z);a,ah=split_haar(a)
            details.append(self._band(zh,ah,z,a,noise,level))
        correction=self._band(z,a,z,a,noise,4)
        for high in reversed(details):
            correction=join_haar(correction,high)
        correction=0.15*correction
        return initial+torch.stack((correction.real,correction.imag),dim=1)*scale
