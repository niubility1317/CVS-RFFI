"""Training-only state actions; deployed identity forward remains native CVS."""
import torch
from torch import nn
from experiments.cvs_multi_disentangle.model import (
    intermediate, identity_from_intermediate, classify_intermediate, _encoder,
    INTERMEDIATE_DIM, IDENTITY_DIM,
)
from experiments.cvs_multi_disentangle.physics import apply_linear, apply_temporal


def reference_response(identity, x):
    return _encoder(identity).response(x)


def signal_state(x):
    """Fixed recipient state shared by unified controls; no identity head."""
    z = torch.complex(x[:, 0], x[:, 1])
    spectrum = torch.fft.fft(z).abs().square()
    spectrum = spectrum.reshape(len(x), 32, 8).mean(2)
    spectrum = torch.log1p(spectrum / spectrum.mean(1, keepdim=True).clamp_min(1e-6))
    power = z.abs().square()
    normalized = power / power.mean(1, keepdim=True).clamp_min(1e-6)
    moments = torch.stack((power.mean(1).clamp_min(1e-6).log(), normalized.square().mean(1),
        normalized.pow(3).mean(1), normalized.max(1).values), 1)
    correlations = []
    for lag in (1, 20):
        corr = (z[:, lag:] * z[:, :-lag].conj()).mean(1) / power.mean(1).clamp_min(1e-6)
        correlations.extend((corr.real, corr.imag))
    return torch.cat((spectrum, 8*torch.tanh(moments/8), torch.stack(correlations,1)),1)


class UnifiedAction(nn.Module):
    """One shared nonlinear core for L/T/R; type-specific input adapters only.

    f(state,increment)-f(state,0) preserves zero identity without imposing odd
    symmetry. R has no true digital reference replacement; L/T retain exact29.
    """
    def __init__(self, hidden=96):
        super().__init__()
        self.receiver_encoder = nn.Sequential(nn.Linear(49,32),nn.SiLU(),nn.Linear(32,8))
        self.core = nn.Sequential(nn.Linear(349+40+8+3,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,349))
        nn.init.normal_(self.core[-1].weight,std=.001)
        nn.init.zeros_(self.core[-1].bias)

    def action(self,h,state,q,kind):
        types=h.new_zeros((len(h),3));types[:,kind]=1
        context=torch.cat((torch.nn.functional.layer_norm(h,(349,)),state,types),1)
        return self.core(torch.cat((context,q),1))-self.core(torch.cat((context,torch.zeros_like(q)),1))

    def digital(self,kind,h,x,p,reference_fn):
        from .actions import signal_state as matched_signal_state
        q=h.new_zeros((len(h),8));q[:,:p.shape[1]]=p
        delta=self.action(h,torch.nn.functional.layer_norm(matched_signal_state(x),(40,)),q,0 if kind=='linear' else 1)
        transformed=(apply_linear if kind=='linear' else apply_temporal)(x,p)
        return torch.cat((delta[:,:320],reference_fn(transformed)-reference_fn(x)),1)

    def receiver(self,h,v0,v1):
        q=(self.receiver_encoder(v1)-self.receiver_encoder(v0)).expand(len(h),-1)
        # Real RX distribution descriptors replace unavailable packet IQ state;
        # same fixed40 recipient information is unavailable to both R variants.
        return self.action(h,h.new_zeros((len(h),40)),q,2)


class ReceiverAdapter(nn.Module):
    def __init__(self,shared):
        super().__init__();self.shared=shared
    def forward(self,h,v0,v1):
        return self.shared.receiver(h,v0,v1)
