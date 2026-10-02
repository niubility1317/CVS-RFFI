"""Prototype: explicit relative complex-channel phase, no new parameters."""
import torch
from torch import nn
from experiments.cvs_energy_identity.model import EnergyCVS,energy_contract

VARIANTS=('crossphase_raw','crossphase_half')

def crossphase_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unknown cross-channel phase prototype')
    base='energy_equivariant' if variant=='crossphase_raw' else 'energy_half'
    return dict(energy_contract(base),mode=variant,prototype_only=True,
        readout='logpower_mean_std;lag1_2_complex_coherence;two_fixed_channel_anchor_complex_coherences',
        anchors='channel0 and channel floor(C/2), no packet/class/RX selection',
        temporal_position_logpower_means_replaced=True,readout_features_per_channel=10,readout_active=True,
        new_trainable_parameters=0,hardware_parameter_recovery=False)

class CrossChannelPhaseReadout(nn.Module):
    """Same 10*C dimensions; preserve inter-channel phase under a common rotation."""
    def forward(self,z):
        if z.ndim!=4 or z.shape[1]!=2 or z.shape[2]<2:raise ValueError('Expected complex[B,2,C,T], C>=2')
        r,i=z.unbind(1);power=r.square()+i.square();logpower=torch.log1p(power)
        parts=[logpower.mean(-1),torch.sqrt(logpower.var(-1,unbiased=False)+1e-6)]
        for lag in (1,2):
            ar,ai=r[...,lag:],i[...,lag:];br,bi=r[...,:-lag],i[...,:-lag]
            denominator=torch.sqrt((ar.square()+ai.square()).mean(-1)*(br.square()+bi.square()).mean(-1)+1e-6)
            parts += [(ar*br+ai*bi).mean(-1)/denominator,(ai*br-ar*bi).mean(-1)/denominator]
        for anchor in (0,z.shape[2]//2):
            br,bi=r[:,anchor:anchor+1],i[:,anchor:anchor+1]
            denominator=torch.sqrt(power.mean(-1)*(br.square()+bi.square()).mean(-1)+1e-6)
            parts += [(r*br+i*bi).mean(-1)/denominator,(i*br-r*bi).mean(-1)/denominator]
        return torch.cat(parts,1)

class CrossPhaseCVS(EnergyCVS):
    def __init__(self,variant):
        crossphase_contract(variant)
        super().__init__('energy_equivariant' if variant=='crossphase_raw' else 'energy_half')
        self.crossphase_variant=variant
        self.core.readout=CrossChannelPhaseReadout()
    def contract(self):
        d=crossphase_contract(self.crossphase_variant)
        d['readout_active']=isinstance(self.core.readout,CrossChannelPhaseReadout)
        return d

def build(variant):return CrossPhaseCVS(variant)
