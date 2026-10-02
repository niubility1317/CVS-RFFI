"""Prototype: retain the relative carrier coordinate after packet alignment.

Alignment plus its coordinate is reversible; an invariant quotient alone is not.
Received CFO is not a uniquely identified TX oscillator parameter. This model
retains global-phase invariance, but deliberately makes no affine-phase or
arbitrary receiver/channel invariance claim for the complete identity output.
"""
import math
import torch
from torch import nn
from experiments.cvs_synchronized_identity.model import RepeatedFieldSynchronizer
from experiments.cvs_equivariant_identity.model import build as equivariant_build
from experiments.cvs_gauge_identity.model import build as gauge_build

VARIANTS=('coordinate_equivariant','coordinate_gauge')
GAIN_BOUND=.25


def coordinate_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered coordinate identity prototype')
    return dict(mode=variant,input='received_equalized1_unit_rms_IQ',sample_rate_hz=25000000,
        estimator_window=[80,160],period_samples=20,relative_frequency_alias_hz=1250000,
        all_waveform_identity_paths_after_alignment=True,relative_coordinate_retained=True,
        retained_coordinates=['cos(20omega)-1','sin(20omega)','clamp(coherence,0,1)-1'],
        gain='1+0.25*tanh(Linear3to160(coordinates))',initial_gain=1.,gain_interval=[.75,1.25],
        conditioner_parameters=640,conditioner_identity_class_specific=False,
        quality_input_gradient='detached; measured perpacket quality controls learned gain without undefined magnitude gradient at zero correlation',
        constant_phase_invariant_real_arithmetic=True,whole_affine_phase_invariant=False,
        identity_core='equivariant_memory' if variant=='coordinate_equivariant' else 'gauge_coherent',
        embedding_dimension=160,classifier_scale=30.,arbitrary_channel_rx_invariant=False,hardware_parameter_estimation=False,
        interpretation='Aligned received identity features modulated by retained relative-frequency/quality coordinate;not TX hardware recovery')


def rotate(x,omega):
    n=torch.arange(256,device=x.device,dtype=x.dtype);phase=omega[:,None]*n
    r,i=x.unbind(1);c,s=phase.cos(),phase.sin()
    return torch.stack((r*c-i*s,r*s+i*c),1)


class CoordinateCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;coordinate_contract(variant)
        self.synchronizer=RepeatedFieldSynchronizer()
        self.core=equivariant_build('equivariant_memory') if variant=='coordinate_equivariant' else gauge_build('gauge_coherent')
        self.conditioner=nn.Linear(3,160)
        nn.init.zeros_(self.conditioner.weight);nn.init.zeros_(self.conditioner.bias)
    def contract(self):return coordinate_contract(self.variant)
    def coordinates(self,x):
        omega,valid,coherence=self.synchronizer.estimate(x)
        aligned=rotate(x,-omega)
        alpha=20*omega
        # The estimator's zero-correlation magnitude is nondifferentiable.
        # Quality is observed, not fitted; detach only that scalar path.
        quality=coherence.detach().clamp(0,1)
        descriptor=torch.stack((alpha.cos()-1,alpha.sin(),quality-1),1)
        return aligned,omega,valid,descriptor
    def features(self,x):
        aligned,_,_,descriptor=self.coordinates(x)
        u=self.core.features(aligned)
        gain=1+GAIN_BOUND*torch.tanh(self.conditioner(descriptor))
        return u*gain
    def classify_features(self,features):return self.core.classify_features(features)
    def forward(self,x):return self.classify_features(self.features(x))
    @torch.no_grad()
    def diagnostics(self,x):
        _,omega,valid,descriptor=self.coordinates(x);gain=1+GAIN_BOUND*torch.tanh(self.conditioner(descriptor))
        return dict(relative_cfo_hz_mean=float((omega*25000000/(2*math.pi)).mean()),fallback_fraction=float((~valid).float().mean()),
                    gain_min=float(gain.min()),gain_max=float(gain.max()),gain_mean=float(gain.mean()),conditioner_active=True,
                    whole_affine_invariance_claim=False,hardware_parameter_recovery=False)


def build(variant):return CoordinateCVS(variant)
