"""Learn a shared received-frequency correction fraction before RF features.

No extra classifier/feature conditioner. The fraction is a source-learned
network parameter, not an estimate of the TX or RX oscillator contribution.
"""
import math
import torch
from torch import nn
from experiments.cvs_synchronized_identity.model import RepeatedFieldSynchronizer
from experiments.cvs_coordinate_identity.model import rotate
from experiments.cvs_equivariant_identity.model import build as core_build

VARIANTS=('fractional_half','fractional_learned')


def fractional_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered fractional synchronization prototype')
    return dict(mode=variant,input='received_equalized1_unit_rms_IQ',sample_rate_hz=25000000,
        estimator_window=[80,160],period_samples=20,relative_frequency_alias_hz=1250000,
        waveform_operator='x[n]*exp(-j*alpha*omega(x)*n)',
        alpha='sigmoid(classshared_scalar)' if variant=='fractional_learned' else '0.5',
        alpha_initial=.5,alpha_numeric_interval=[0.,1.],alpha_parameters=1 if variant=='fractional_learned' else 0,
        alpha_packet_specific=False,alpha_identity_class_specific=False,alpha_RX_specific=False,
        all_identity_paths_after_fractional_correction=True,preserved_sample_amplitudes=True,
        waveform_reconstruction='rotate(partial_aligned,+alpha*estimated_omega)',
        residual_frequency_transform='additional_nu -> (1-alpha)*nu when estimator valid and principal branch not crossed',
        constant_phase_invariant_real_arithmetic=True,whole_affine_phase_invariant=False,
        principal_branch_crossing='not covered by residual-frequency covariance equation',
        degenerate_behavior='estimated omega zero before atan2;no frequency correction',
        identity_core='equivariant_memory',embedding_dimension=160,classifier_scale=30.,
        feature_conditioner=False,domain_backbone=False,arbitrary_channel_rx_invariant=False,
        hardware_parameter_estimation=False,
        interpretation='Received frequency coordinate acts on the waveform before RF features;alpha is discrimination tradeoff,not identified TX/RX fraction')


class FractionalCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();fractional_contract(variant);self.variant=variant
        self.synchronizer=RepeatedFieldSynchronizer();self.core=core_build('equivariant_memory')
        if variant=='fractional_learned':self.alignment_logit=nn.Parameter(torch.zeros(()))
        else:self.register_buffer('alignment_logit',torch.zeros(()))
    def contract(self):return fractional_contract(self.variant)
    def alignment_strength(self):
        return self.alignment_logit.sigmoid() if self.variant=='fractional_learned' else self.alignment_logit.new_tensor(.5)
    def coordinates(self,x):
        omega,valid,coherence=self.synchronizer.estimate(x)
        alpha=self.alignment_strength()
        return rotate(x,-alpha*omega),omega,valid,coherence,alpha
    def features(self,x):return self.core.features(self.coordinates(x)[0])
    def classify_features(self,features):return self.core.classify_features(features)
    def forward(self,x):return self.classify_features(self.features(x))
    @torch.no_grad()
    def diagnostics(self,x):
        _,omega,valid,coherence,alpha=self.coordinates(x)
        hz=omega*25000000/(2*math.pi)
        return dict(alignment_strength=float(alpha),alignment_parameters=1 if self.variant=='fractional_learned' else 0,
            relative_cfo_hz_mean=float(hz.mean()),relative_cfo_hz_min=float(hz.min()),relative_cfo_hz_max=float(hz.max()),
            nominal_residual_cfo_hz_mean=float(((1-alpha)*hz).mean()),
            coherence_mean=float(coherence.mean()),fallback_fraction=float((~valid).float().mean()),
            hardware_parameter_recovery=False,whole_affine_invariance_claim=False)


def build(variant):return FractionalCVS(variant)
