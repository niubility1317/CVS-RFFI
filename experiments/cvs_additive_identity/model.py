"""Received RF coordinate injection into an aligned, phase-invariant core.

This is a testable source-derived hypothesis, not identified TX hardware. The
additive coordinate can create feature components a multiplicative gain cannot
create when the corresponding core feature is zero. No extra network width or
depth, domain path, independent classifier, or auxiliary loss is introduced.
"""
import math
import torch
from experiments.cvs_coordinate_identity.model import CoordinateCVS, coordinate_contract, rotate

VARIANTS=('additive_equivariant','additive_gauge')
DELTA_BOUND=.25
PARENTS=dict(additive_equivariant='coordinate_equivariant',additive_gauge='coordinate_gauge')


def additive_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered additive identity model')
    c=coordinate_contract(PARENTS[variant])
    for k in ('gain','initial_gain','gain_interval'):del c[k]
    c.update(mode=variant,feature_injection='u+0.25*norm(u,2)*tanh(Linear3to160(coordinates))',
        initial_delta=0.,per_component_delta_bound_relative_to_core_norm=.25,
        vector_delta_bound_relative_to_core_norm=.25*math.sqrt(160),
        exact_zero_initialized_core_equivalence=True,
        quality_input_gradient='detached; observed perpacket quality conditions learned additive feature direction',
        interpretation='Aligned received features plus shared additive direction from retained relative-frequency/quality;not TX hardware recovery')
    return c


class AdditiveCVS(CoordinateCVS):
    def __init__(self,variant):
        additive_contract(variant)
        super().__init__(PARENTS[variant]);self.variant=variant
    def contract(self):return additive_contract(self.variant)
    def inject(self,u,descriptor):
        scale=torch.linalg.vector_norm(u,dim=1,keepdim=True)
        delta=DELTA_BOUND*scale*torch.tanh(self.conditioner(descriptor))
        return u+delta,delta,scale
    def components(self,x):
        aligned,omega,valid,descriptor=self.coordinates(x)
        u=self.core.features(aligned)
        features,delta,scale=self.inject(u,descriptor)
        return features,delta,scale,omega,valid,descriptor
    def features(self,x):return self.components(x)[0]
    @torch.no_grad()
    def diagnostics(self,x):
        features,delta,scale,omega,valid,descriptor=self.components(x)
        ratio=torch.where(scale>0,delta.norm(dim=1,keepdim=True)/scale.clamp_min(torch.finfo(scale.dtype).tiny),torch.zeros_like(scale))
        hz=omega*25000000/(2*math.pi)
        return dict(relative_cfo_hz_mean=float(hz.mean()),relative_cfo_hz_min=float(hz.min()),relative_cfo_hz_max=float(hz.max()),
            coherence_mean=float((descriptor[:,2]+1).mean()),fallback_fraction=float((~valid).float().mean()),
            delta_norm_ratio_min=float(ratio.min()),delta_norm_ratio_max=float(ratio.max()),delta_norm_ratio_mean=float(ratio.mean()),
            core_norm_mean=float(scale.mean()),result_norm_mean=float(features.norm(dim=1).mean()),conditioner_active=True,
            whole_affine_invariance_claim=False,hardware_parameter_recovery=False)


def build(variant):return AdditiveCVS(variant)
