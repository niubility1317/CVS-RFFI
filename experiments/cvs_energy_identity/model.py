"""Prototype: retain relative RF filter energy while respecting common phase."""
import torch
from torch import nn
from experiments.cvs_equivariant_identity.model import EquivariantCVS,equivariant_contract

VARIANTS=('energy_equivariant',)

def energy_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered energy-preserving variant')
    d=equivariant_contract('equivariant_memory')
    d.update(mode=variant,normalization_scope='one complex power mean across all channels and time per packet per block',
        per_channel_sample_dependent_rms_normalization=False,relative_filter_channel_energy_preserved_before_gate=True,
        learned_channel_scales_retained=True,new_trainable_parameters=0,relative_energy_after_gate='may change through learned radial gate;not fixed',
        interpretation='Received relative filter/polynomial energy is retained by normalization;not recovered TX PA/IQ coefficients or arbitrary RX/channel invariance')
    return d

class GlobalEnergyBlock(nn.Module):
    """Reuses only this new scratch model's modules; no historical weight input."""
    def __init__(self,block):
        super().__init__();self.conv=block.conv;self.scale=block.scale
        self.gate_a=block.gate_a;self.gate_b=block.gate_b;self.epsilon=block.epsilon

    def forward(self,z):
        z=self.conv(z);power=z.square().sum(1)
        gain=torch.rsqrt(power.mean((1,2),keepdim=True)+self.epsilon)
        z=z*gain[:,None]*self.scale[None,None,:,None]
        power=z.square().sum(1)
        gate=2*torch.sigmoid(self.gate_a[None,:,None]*torch.log1p(power)+self.gate_b[None,:,None])
        return z*gate[:,None]

class EnergyCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;energy_contract(variant)
        self.core=EquivariantCVS('equivariant_memory')
        for path in ('time','behavior'):
            blocks=getattr(self.core,path)
            for i,block in enumerate(blocks):blocks[i]=GlobalEnergyBlock(block)

    def contract(self):return energy_contract(self.variant)
    def features(self,x):return self.core.features(x)
    def classify_features(self,features):return self.core.classify_features(features)
    def forward(self,x):return self.classify_features(self.features(x))
    def diagnostics(self,x):return self.core.diagnostics(x)

def build(variant):return EnergyCVS(variant)
