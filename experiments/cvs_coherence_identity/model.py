"""Regularized complex increments, with unchanged physical cue capacity."""
import torch
from torch import nn
import torch.nn.functional as F
from experiments.cvs_stability_identity.model import StabilityCVS
VARIANTS=('coherence_phase','coherence_dsq')

class CoherencePhaseStem(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.sinc_out=original.sinc_out;self.eps=original.eps
        self.proj=original.proj  # preserve the exact initial parameters;no RNG redraw
        self.cue_mode='regularized_complex_increment'

    def raw_features(self,sinc_iq):
        if sinc_iq.dim()!=3 or sinc_iq.size(1)!=2*self.sinc_out:
            raise ValueError('Expected packet filterbank IQ [B,2*sinc_out,T]')
        b,_,length=sinc_iq.shape
        y=torch.nan_to_num(sinc_iq.float(),nan=0.,posinf=0.,neginf=0.).view(b,2,self.sinc_out,length)
        i,q=y[:,0],y[:,1];amp=torch.sqrt(i*i+q*q+self.eps)
        amp_norm=torch.log1p(amp/amp.mean(dim=-1,keepdim=True).clamp_min(self.eps))
        if length>1:
            d_amp=F.pad(amp[...,1:]-amp[...,:-1],(1,0))
            cross=q[...,1:]*i[...,:-1]-i[...,1:]*q[...,:-1]
            dot=i[...,1:]*i[...,:-1]+q[...,1:]*q[...,:-1]
            denom=amp[...,1:]*amp[...,:-1]
            imaginary=F.pad(cross/denom,(1,0));real=F.pad(dot/denom,(1,0))
        else:d_amp=torch.zeros_like(amp);imaginary=torch.zeros_like(amp);real=torch.zeros_like(amp)
        return torch.cat((amp_norm.clamp(0.,8.),d_amp.clamp(-8.,8.),imaginary,real),1)

    def forward(self,sinc_iq):return self.proj(self.raw_features(sinc_iq))

class CoherenceCVS(StabilityCVS):
    def __init__(self,variant):
        super().__init__('phase_dsq' if variant=='coherence_dsq' else 'phase_delta')
        b=self.id_backbone;b.time_stability=CoherencePhaseStem(b.time_stability)
        b.time_stability_mode='coherence'

def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered complex coherence variant')
    return CoherenceCVS(variant)
