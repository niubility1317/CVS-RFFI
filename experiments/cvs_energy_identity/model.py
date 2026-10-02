"""Prototype: retain relative RF filter energy while respecting common phase."""
import torch
from torch import nn
from experiments.cvs_equivariant_identity.model import EquivariantCVS,equivariant_contract
from experiments.cvs_synchronized_identity.model import RepeatedFieldSynchronizer
from experiments.cvs_coordinate_identity.model import rotate

VARIANTS=('energy_equivariant','energy_half')

def energy_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered energy-preserving variant')
    d=equivariant_contract('equivariant_memory')
    d.update(mode=variant,normalization_scope='one complex power mean across all channels and time per packet per block',
        per_channel_sample_dependent_rms_normalization=False,relative_filter_channel_energy_preserved_by_normalization_only=True,
        learned_channel_scales_retained=True,new_trainable_parameters=0,relative_energy_after_gate='may change through learned radial gate;not fixed',
        learned_channel_scales_may_change_energy_ratios=True,identity_core='energy_equivariant',
        estimator_window=[80,160],estimator_period_samples=20,relative_frequency_alias_hz=1250000,
        alignment_strength=.5 if variant=='energy_half' else 0.,alignment_parameters=0,
        alignment_packet_specific=False,alignment_class_specific=False,alignment_rx_specific=False,
        waveform_operator='x[n]exp(-j alpha omega n);raw variant returns x exactly',
        whole_affine_phase_invariant=False,alignment_interpretation='fixed received frequency retention;not TX/RX oscillator proportion',
        interpretation='Received relative filter/polynomial energy is retained by normalization;not recovered TX PA/IQ coefficients or arbitrary RX/channel invariance')
    return d

class GlobalEnergyBlock(nn.Module):
    """Reuses only this new scratch model's modules; no historical weight input."""
    def __init__(self,block):
        super().__init__();self.conv=block.conv;self.scale=block.scale
        self.gate_a=block.gate_a;self.gate_b=block.gate_b;self.epsilon=block.epsilon

    def forward(self,z):
        z=self.conv(z);power=z.square().sum(1)
        gain=self.shared_gain(power)
        z=z*gain[:,None]*self.scale[None,None,:,None]
        power=z.square().sum(1)
        gate=2*torch.sigmoid(self.gate_a[None,:,None]*torch.log1p(power)+self.gate_b[None,:,None])
        return z*gate[:,None]

    def shared_gain(self,power):return torch.rsqrt(power.mean((1,2),keepdim=True)+self.epsilon)

class EnergyCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;energy_contract(variant)
        self.synchronizer=RepeatedFieldSynchronizer()
        self.core=EquivariantCVS('equivariant_memory')
        for path in ('time','behavior'):
            blocks=getattr(self.core,path)
            for i,block in enumerate(blocks):blocks[i]=GlobalEnergyBlock(block)

    def contract(self):
        d=energy_contract(self.variant)
        d['per_channel_sample_dependent_rms_normalization']=not all(isinstance(m,GlobalEnergyBlock) for path in ('time','behavior') for m in getattr(self.core,path))
        return d

    def alignment_strength(self):return next(self.parameters()).new_tensor(.5 if self.variant=='energy_half' else 0.)
    def coordinates(self,x):
        omega,valid,coherence=self.synchronizer.estimate(x);alpha=self.alignment_strength()
        partial=x if self.variant=='energy_equivariant' else rotate(x,-alpha*omega)
        return partial,omega,valid,coherence,alpha
    def features(self,x):
        if self.variant=='energy_equivariant':return self.core.features(x)
        return self.core.features(self.coordinates(x)[0])
    def classify_features(self,features):return self.core.classify_features(features)
    def forward(self,x):return self.classify_features(self.features(x))
    @torch.no_grad()
    def normalization_diagnostics(self,x):
        records=[];hooks=[]
        def capture(name,block):
            def hook(module,inputs,z):
                power=z.square().sum(1);before=power.mean(-1);gain=block.shared_gain(power)
                after=(power*gain.square()).mean(-1)
                eligible=before.sum(1)>1e-12
                if eligible.any():
                    a=before[eligible]/before[eligible].sum(1,keepdim=True)
                    b=after[eligible]/after[eligible].sum(1,keepdim=True)
                    error=float((a-b).abs().max());variation=float(a.var(1,unbiased=False).mean())
                else:error=variation=None
                records.append(dict(block=name,eligible_packets=int(eligible.sum()),relative_energy_fraction_max_error=error,relative_energy_fraction_variance_mean=variation))
            return hook
        try:
            for path in ('time','behavior'):
                for i,block in enumerate(getattr(self.core,path)):hooks.append(block.conv.register_forward_hook(capture(path+'.'+str(i),block)))
            self.features(x)
        finally:
            for hook in hooks:hook.remove()
        return dict(scope='Actual conv outputs and one shared normalization gain only;learned scales/gates excluded;not TX/RX causal separation',records=records,blocks=6)

    @torch.no_grad()
    def diagnostics(self,x):
        _,omega,valid,coherence,alpha=self.coordinates(x);hz=omega*25000000/(2*torch.pi)
        return dict(self.core.diagnostics(x),alignment_strength=float(alpha),alignment_parameters=0,
            received_relative_cfo_hz_mean=float(hz.mean()),received_relative_cfo_hz_min=float(hz.min()),received_relative_cfo_hz_max=float(hz.max()),
            nominal_residual_cfo_hz_mean=float(((1-alpha)*hz).mean()),coherence_mean=float(coherence.mean()),fallback_fraction=float((~valid).float().mean()),
            hardware_parameter_recovery=False,whole_affine_phase_invariant=False,normalization=self.normalization_diagnostics(x))

def build(variant):return EnergyCVS(variant)
