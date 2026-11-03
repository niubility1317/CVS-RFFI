"""DURA-conditioned denoising trajectories with deterministic antithetic sampling."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase1_dura_blendgate.model import align_to_anchor


class PosteriorDenoiser(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.entry = nn.Conv1d(9, width, 7, padding=3)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2*d, dilation=d) for d in (1,2,4,8))
        self.filter_real = nn.Conv1d(1,1,7,padding=3,bias=False)
        self.filter_imag = nn.Conv1d(1,1,7,padding=3,bias=False)
        self.gate_output = nn.Conv1d(width,8,1)
        nn.init.zeros_(self.gate_output.weight); nn.init.zeros_(self.gate_output.bias)
        generator=torch.Generator(device='cpu').manual_seed(2026110761)
        self.register_buffer('quadrature',torch.randn((1,2,256),generator=generator))

    def denoise(self, received, anchor, state, log_variance, sigma):
        n=len(received)
        if received.shape!=(n,2,256) or anchor.shape!=received.shape or state.shape!=received.shape or log_variance.shape!=(n,) or sigma.shape!=(n,):
            raise ValueError('Matching IQ windows, blind noise and per-record scale required')
        if not all(torch.isfinite(v).all() for v in (received,anchor,state,log_variance,sigma)) or (sigma<0).any():
            raise ValueError('Finite inputs and nonnegative scale required')
        power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8); scale=power.sqrt()
        x=torch.complex(received[:,0],received[:,1])/scale[:,0]
        a=torch.complex(anchor[:,0],anchor[:,1])/scale[:,0]
        z=torch.complex(state[:,0],state[:,1])/scale[:,0]
        zx=z*x.conj();za=z*a.conj()
        noise=((log_variance-power[:,0,0].log()).clamp(-16,8)/8)[:,None].expand(-1,256)
        features=torch.stack((torch.log1p(x.abs()),torch.log1p(a.abs()),torch.log1p(z.abs()),
            zx.real.clamp(-16,16),zx.imag.clamp(-16,16),za.real.clamp(-16,16),za.imag.clamp(-16,16),
            noise,sigma[:,None].expand(-1,256)),dim=1)
        h=F.silu(self.entry(features))
        for layer in self.layers:h=h+.2*F.silu(layer(h))
        real=self.filter_real(z.real[:,None])-self.filter_imag(z.imag[:,None])
        imag=self.filter_imag(z.real[:,None])+self.filter_real(z.imag[:,None])
        basis=torch.cat((x[:,None],a[:,None],z[:,None],torch.complex(real,imag)),dim=1)
        weight=self.gate_output(h).tanh().reshape(n,2,4,256)
        delta=.3*(torch.complex(weight[:,0],weight[:,1])*basis).sum(1)/4
        return state+torch.stack((delta.real,delta.imag),dim=1)*scale

    def forward(self, received, anchor, log_variance):
        n=len(received)
        if received.shape!=(n,2,256) or anchor.shape!=received.shape or log_variance.shape!=(n,):
            raise ValueError('Matching IQ windows and one blind variance required')
        power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8)
        z=torch.complex(received[:,0],received[:,1])
        pivot=z.gather(1,z.abs().argmax(dim=1,keepdim=True));unit=pivot/pivot.abs().clamp_min(1e-8)
        unit=torch.where(pivot.abs()>1e-8,unit,torch.ones_like(unit))
        q=torch.complex(self.quadrature[:,0],self.quadrature[:,1])*unit*power.sqrt()[:,0]
        q=torch.stack((q.real,q.imag),dim=1)
        level=(.5*(log_variance-power[:,0,0].log())).exp().clamp(.1,1.)
        outputs=[]
        for sign in (1.,-1.):
            state=anchor+sign*(.35*level)[:,None,None]*q
            for current,next_scale in ((.35,.20),(.20,.08),(.08,0.)):
                sigma=current*level
                estimate=self.denoise(received,anchor,state,log_variance,sigma)
                state=estimate+sign*(next_scale*level)[:,None,None]*q
            outputs.append(state)
        return (outputs[0]+outputs[1])/2


def denoising_loss(model,received,anchor,clean,log_variance,groups,*,sigma=None,noise=None):
    n=len(received)
    if anchor.shape!=received.shape or clean.shape!=received.shape or groups.shape!=(n,):
        raise ValueError('Paired source IQ and receiver groups required')
    if sigma is None:sigma=.08+.27*torch.rand(n,device=received.device,dtype=received.dtype)
    if noise is None:noise=torch.randn_like(received)
    if sigma.shape!=(n,) or noise.shape!=received.shape or not torch.isfinite(noise).all() or not torch.isfinite(sigma).all() or (sigma<0).any():
        raise ValueError('Finite per-record noise and nonnegative scales required')
    with torch.no_grad():
        target=align_to_anchor(clean.detach(),anchor.detach(),preserve_edges=True)
        scale=received.detach().square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt()
        state=target+sigma[:,None,None]*scale*noise
        energy=target.square().mean((1,2)).clamp_min(1e-8)
    estimate=model.denoise(received.detach(),anchor.detach(),state,log_variance.detach(),sigma.detach())
    error=(estimate-target).square().mean((1,2))/energy
    loss=torch.stack([error[groups==group].mean() for group in groups.unique()]).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Finite source denoising loss required')
    return loss
