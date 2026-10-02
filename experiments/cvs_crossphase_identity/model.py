"""Explicit relative complex-channel phase, no new parameters."""
import torch
from torch import nn
from experiments.cvs_energy_identity.model import EnergyCVS,energy_contract

VARIANTS=('crossphase_raw','crossphase_half')

def crossphase_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unknown cross-channel phase prototype')
    base='energy_equivariant' if variant=='crossphase_raw' else 'energy_half'
    return dict(energy_contract(base),mode=variant,prototype_only=False,
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
        d=dict(super().contract(),**crossphase_contract(self.crossphase_variant))
        d['per_channel_sample_dependent_rms_normalization']=super().contract()['per_channel_sample_dependent_rms_normalization']
        d['readout_active']=isinstance(self.core.readout,CrossChannelPhaseReadout)
        return d

    @torch.no_grad()
    def readout_diagnostics(self,x):
        records=[]
        def capture(module,inputs,output):
            z=inputs[0];power=z.square().sum(1).mean(-1);c=z.shape[2]
            anchors=(0,c//2)
            for j,a in enumerate(anchors):
                real,imag=output[:,(6+2*j)*c:(7+2*j)*c],output[:,(7+2*j)*c:(8+2*j)*c]
                absolute=torch.sqrt(real.square()+imag.square())
                records.append(dict(path='time' if len(records)<2 else 'behavior',anchor=a,channels=c,output_dimensions=output.shape[1],
                    packets=len(z),anchor_mean_power=float(power[:,a].mean()),low_anchor_packet_fraction=float((power[:,a]<1e-6).float().mean()),
                    actual_coherence_abs_mean=float(absolute.mean()),actual_coherence_abs_max=float(absolute.max())))
        hook=self.core.readout.register_forward_hook(capture)
        try:self.features(x)
        finally:hook.remove()
        return dict(scope='Actual two readout calls and fixed anchors;source lastbatch/public input only;not identified TX hardware',records=records)

    @torch.no_grad()
    def diagnostics(self,x):return dict(super().diagnostics(x),cross_channel_readout=self.readout_diagnostics(x))

def build(variant):return CrossPhaseCVS(variant)
