"""Packet repetition synchronizer precedes every identity information path.

It chooses a relative-frequency coordinate, not a TX oscillator parameter.
Modulo ambiguity and non-repeating/degenerate windows are explicit boundaries.
"""
import math
import torch
from torch import nn
from experiments.cvs_equivariant_identity.model import build as equivariant_build
from experiments.cvs_gauge_identity.model import build as gauge_build

VARIANTS = ('synchronized_equivariant', 'synchronized_gauge')


def synchronized_contract(variant):
    if variant not in VARIANTS: raise ValueError('Unregistered synchronized identity variant')
    return dict(mode=variant, input='received_equalized1_unit_rms_IQ', sample_rate_hz=25000000,
        estimator_window=[80,160], period_samples=20, repeated_cycles=4,
        correlation_pairs=60, coherence_fallback_threshold=1e-6, energy_floor=1e-12,
        estimator='arg(sum(x[n]conj(x[n-20])))/20;all3 adjacent cycle pairs',
        relative_cfo_principal_interval_hz=[-625000,625000], relative_cfo_ambiguity_hz=1250000,
        synchronizer_learned_parameters=0, all_identity_paths_after_synchronization=True,
        preserved_sample_amplitudes=True, original_temporal_samples=256,
        constant_phase_invariant_real_arithmetic=True,
        affine_phase_invariant_when_correlation_valid_and_no_principal_branch_crossing=True,
        degenerate_behavior='no frequency correction;constant phase properties remain;no affine claim',
        waveforms_preserved_up_to_packet_affine_phase=True,
        identity_core='equivariant_memory' if variant=='synchronized_equivariant' else 'gauge_coherent',
        arbitrary_channel_rx_invariant=False, hardware_parameter_estimation=False,
        embedding_dimension=160, classifier_scale=30.,
        interpretation='Relative received carrier synchronization before existing identity network;not recovered TX CFO/PA/IQ hardware parameters')


class RepeatedFieldSynchronizer(nn.Module):
    def estimate(self, x):
        if x.ndim!=3 or x.shape[1:]!=(2,256): raise ValueError('Expected IQ[B,2,256]')
        r,i=x[:,0,80:160].reshape(-1,4,20),x[:,1,80:160].reshape(-1,4,20)
        cr=(r[:,1:]*r[:,:-1]+i[:,1:]*i[:,:-1]).sum((1,2))
        ci=(i[:,1:]*r[:,:-1]-r[:,1:]*i[:,:-1]).sum((1,2))
        energy1=(r[:,1:].square()+i[:,1:].square()).sum((1,2))
        energy0=(r[:,:-1].square()+i[:,:-1].square()).sum((1,2))
        energy=(energy1*energy0).clamp_min(1e-24).sqrt()
        # Magnitude is used only for validity/telemetry, not as a differentiable
        # phase divisor. Zero correlation must report zero coherence.
        magnitude=(cr.square()+ci.square()).sqrt()
        coherence=magnitude/energy
        valid=(energy>1e-12)&(magnitude>1e-6*energy)
        # atan2(0,0) has undefined input gradient. Replace both arguments before
        # atan2 in the fallback branch; this yields a zero correction/gradient.
        safe_r=torch.where(valid,cr,torch.ones_like(cr))
        safe_i=torch.where(valid,ci,torch.zeros_like(ci))
        omega=torch.atan2(safe_i,safe_r)/20
        return omega,valid,coherence

    def forward(self,x):
        omega,_,_=self.estimate(x)
        n=torch.arange(256,device=x.device,dtype=x.dtype)
        phase=-omega[:,None]*n
        c,s=phase.cos(),phase.sin();r,i=x.unbind(1)
        return torch.stack((r*c-i*s,r*s+i*c),1)

    @torch.no_grad()
    def diagnostics(self,x):
        omega,valid,coherence=self.estimate(x)
        hz=omega*25000000/(2*math.pi)
        return dict(relative_cfo_hz_mean=float(hz.mean()),relative_cfo_hz_min=float(hz.min()),
            relative_cfo_hz_max=float(hz.max()),coherence_mean=float(coherence.mean()),
            fallback_fraction=float((~valid).float().mean()),
            principal_branch_margin_hz_min=float((625000-hz.abs()).min()),
            scope='Last source batch only;relative CFO modulo1.25MHz;not TX oscillator measurement')


class SynchronizedCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;synchronized_contract(variant)
        self.synchronizer=RepeatedFieldSynchronizer()
        self.core=equivariant_build('equivariant_memory') if variant=='synchronized_equivariant' else gauge_build('gauge_coherent')
    def contract(self): return synchronized_contract(self.variant)
    def features(self,x): return self.core.features(self.synchronizer(x))
    def classify_features(self,features): return self.core.classify_features(features)
    def forward(self,x): return self.classify_features(self.features(x))
    @torch.no_grad()
    def diagnostics(self,x):
        result=self.synchronizer.diagnostics(x);aligned=self.synchronizer(x)
        if self.variant=='synchronized_equivariant':result.update(core=self.core.diagnostics(aligned))
        else:result.update(core=self.core.gauge.diagnostics(aligned))
        return result


def build(variant): return SynchronizedCVS(variant)
