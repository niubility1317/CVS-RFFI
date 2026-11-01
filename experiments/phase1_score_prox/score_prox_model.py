"""Noise-conditioned ground-signal denoiser and data-consistent proximal step."""
import torch
from torch import nn
from torch.nn import functional as F

class GroundDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(3,64,7,padding=3)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,5,dilation=d,padding=2*d),nn.SiLU(),nn.Conv1d(64,64,1)) for d in (1,2,4,8,16)])
        self.output=nn.Conv1d(64,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,noise_rms):
        if x.ndim!=3 or x.shape[1:]!=(2,256) or noise_rms.shape!=(len(x),):raise ValueError('IQ [B,2,256] and per-record noise required')
        scale=x.square().sum(1).mean(1).sqrt().clamp_min(1e-8);v=x/scale[:,None,None];s=(noise_rms/scale).clamp(0,2)
        h=F.silu(self.input(torch.cat((v,s[:,None,None].expand(-1,1,256)),1)))
        for block in self.blocks:h=h+block(h)
        return x+noise_rms[:,None,None]*self.output(h)

def data_prox(A,x,prior,strength):
    if A.ndim!=3 or A.shape[1]!=A.shape[2] or x.shape!=prior.shape or x.ndim!=2 or A.shape[:2]!=x.shape or strength<=0:raise ValueError('Square operator, matched vectors and positive strength required')
    at=A.transpose(1,2);normal=at@A;energy=normal.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-8);weight=strength*energy;eye=torch.eye(A.shape[-1],device=A.device,dtype=A.dtype);system=normal+weight[:,None,None]*eye;rhs=(at@x.unsqueeze(-1))+weight[:,None,None]*prior.unsqueeze(-1)
    return torch.cholesky_solve(rhs,torch.linalg.cholesky(system)).squeeze(-1)
