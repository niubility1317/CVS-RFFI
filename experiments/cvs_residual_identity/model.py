"""Preserve native time/frequency/PA signals; replace the expensive head cascade."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiments.cvs_identity_ce.model import IdentityOnlyCVS
from experiments.cvs_clean_design.model import MomentPool

VARIANTS = ('residual_fusion', 'residual_fusion_moment')


def prune_unused_identity_modules(model):
    """Only modules bypassed by return_aux=False; used for an equivalence control."""
    b = model.id_backbone
    b.con_proj = nn.Identity()
    b.cls_head.imp_merge = nn.Identity()
    b.cls_head.pa_head = None
    return model


class ResidualPhysicalHead(nn.Module):
    """Branch-normalized additive fusion with one learned, bounded gain/channel."""
    def __init__(self, dimension, num_classes=6, initial_gain=0.25):
        super().__init__()
        self.base_norm = nn.LayerNorm(dimension)
        self.pa_norm = nn.LayerNorm(dimension)
        self.gain = nn.Parameter(torch.full((dimension,), math.atanh(initial_gain)))
        self.weight = nn.Parameter(torch.empty(num_classes, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = 30.0

    def components(self, base, pa_local, pa_delta=None):
        identity = self.base_norm(base)
        physical = self.pa_norm(pa_local if pa_delta is None else pa_local + pa_delta)
        # No stop-gradient: CE jointly aligns both native branch projections.
        joint = identity + self.gain.tanh() * physical
        return identity, physical, joint

    def classify(self, joint):
        return self.scale * F.linear(F.normalize(joint, dim=1, eps=1e-4),
                                     F.normalize(self.weight, dim=1, eps=1e-4))

    def forward_logits(self, base, dac_local, pa_local, labels=None, dac_delta=None, pa_delta=None):
        if labels is not None:
            raise ValueError('Pure CE identity head accepts no label-dependent margin')
        return self.classify(self.components(base, pa_local, pa_delta)[2])

    def forward(self, base, dac_local, pa_local, labels=None, return_emb=False, dac_delta=None, pa_delta=None):
        if labels is not None:
            raise ValueError('Pure CE identity head accepts no label-dependent margin')
        identity, physical, joint = self.components(base, pa_local, pa_delta)
        logits = self.classify(joint)
        zero = torch.zeros_like(identity)
        scalar_zero = base.new_zeros(base.shape[0])
        if not return_emb:
            return logits, scalar_zero
        return logits, scalar_zero, scalar_zero, identity, zero, physical, physical, joint


def build(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered CVS residual variant: ' + str(variant))
    model = IdentityOnlyCVS()
    b = model.id_backbone
    if b.use_dac_path or not b.use_pa_path or b.nmfdu_gate is not None:
        raise ValueError('Residual identity design requires native time/frequency/PA and no DAC/domain gate')
    b.con_proj = nn.Identity()
    b.cls_head = ResidualPhysicalHead(b.emb_dim)
    if variant == 'residual_fusion_moment':
        for pool, projection in [('t_pool', 't_proj'), ('f_pool', 'f_proj'), ('pa_pool', 'pa_proj')]:
            proj = getattr(b, projection)
            linear = proj if isinstance(proj, nn.Linear) else next(m for m in proj.modules() if isinstance(m, nn.Linear))
            setattr(b, pool, MomentPool(linear.in_features))
    return model
