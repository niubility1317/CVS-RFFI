"""Low-rank complex cross-path relations, learned by the original TX CE only.

Feature-channel gain normalization is not a claim of physical channel removal.
All statistics are computed within one received packet; there is no fitted state.
"""
from contextlib import contextmanager

import torch
from torch import nn

from experiments.cvs_equivariant_identity.model import (
    ComplexConv, EquivariantCVS, InvariantReadout, behavior_basis,
)
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS, neural_contract

VARIANTS = ('crosspath_gram', 'crosspath_coherence')
BASE_VARIANT = 'neural_residual_shallow'
BASE_PARAMETERS = 220987
NEW_PARAMETERS = 12288
LAGS = (0, 4, 8)
POWER_FLOOR = 1e-6


def relation_statistics(a, b, channel_normalized):
    """A[:,t-lag] conj(B[:,t]); common B positions 8..63 for all lags.

    a,b: real-pair tensors [B,2,8,64] after learned complex projections.
    An uncentered cross moment, not a mean-centered covariance estimator.
    """
    if a.ndim != 4 or a.shape[1:] != (2, 8, 64) or b.shape != a.shape:
        raise ValueError('Expected paired complex projections [B,2,8,64]')
    parts = []
    for lag in LAGS:
        left, right = a[..., 8-lag:64-lag], b[..., 8:64]
        pa, pb = left.square().sum(1).mean(-1), right.square().sum(1).mean(-1)
        if not channel_normalized:
            pa, pb = pa.mean(-1, keepdim=True), pb.mean(-1, keepdim=True)
        # Floor each factor before sqrt: finite gradients for zero/quiet packets.
        left = left * torch.rsqrt(pa.clamp_min(POWER_FLOOR))[:, None, :, None]
        right = right * torch.rsqrt(pb.clamp_min(POWER_FLOOR))[:, None, :, None]
        ar, ai = left.unbind(1)
        br, bi = right.unbind(1)
        real = torch.bmm(ar, br.transpose(1, 2)) + torch.bmm(ai, bi.transpose(1, 2))
        imag = torch.bmm(ai, br.transpose(1, 2)) - torch.bmm(ar, bi.transpose(1, 2))
        parts.append(torch.stack((real, imag), 1) / (56 * 8))
    return torch.stack(parts, 1)  # [B,3,2,8,8], each lag Frobenius norm <= 1


class CrossPathRelation(nn.Module):
    def __init__(self, channel_normalized):
        super().__init__()
        self.channel_normalized = channel_normalized
        self.time_project = ComplexConv(32, 8, 1)
        self.behavior_project = ComplexConv(32, 8, 1)
        self.compress = nn.Linear(384, 16, bias=False)
        self.project = nn.Linear(16, 320, bias=False)
        nn.init.zeros_(self.project.weight)

    def statistics(self, time, behavior):
        if time.ndim != 4 or time.shape[1:] != (2, 32, 64) or behavior.shape != time.shape:
            raise ValueError('Expected paired backbone features [B,2,32,64]')
        return relation_statistics(self.time_project(time), self.behavior_project(behavior),
                                   self.channel_normalized)

    def forward(self, time, behavior):
        return self.project(self.compress(self.statistics(time, behavior).flatten(1)))


class CrossPathCore(EquivariantCVS):
    def __init__(self, original, channel_normalized):
        # Keep the scratch modules and their original state_dict paths exactly.
        nn.Module.__init__(self)
        if list(original.parameters(recurse=False)) or list(original.buffers(recurse=False)):
            raise ValueError('Unexpected root-level backbone state')
        self.variant = original.variant
        for name, module in original.named_children():
            self.add_module(name, module)
        self.relation = CrossPathRelation(channel_normalized)

    def features(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Expected IQ[B,2,256]')
        b = self.id_backbone
        sinc = b._sinc_on_iq(x).reshape(len(x), 2, 24, 256)
        time_map = self.time(sinc)
        behavior_map = self.behavior(behavior_basis(x))
        time = b.t_proj(self.readout(time_map))
        physical = b.pa_proj(self.readout(behavior_map) + self.relation(time_map, behavior_map))
        spectral, rho, dac_stats, pa_stats = b._mirror_compressed_features(x, sinc_iq=None)
        f = b.f_pool(b.f3(b.f2(b.f1(b.freq_gate(spectral))))).squeeze(-1)
        frequency = b.f_proj(f)
        if b.use_stats_path and b.use_freq_stats and b.freq_stats_proj is not None:
            frequency = frequency + b.freq_stats_proj(dac_stats)
        if b.use_stats_path and b.pa_stats_proj is not None:
            physical = physical + .25 * b.pa_stats_proj(pa_stats)
        parts = [time, frequency]
        if rho is not None:
            parts.append(rho if b.use_stats_path else torch.zeros_like(rho))
        base = b.fuse(torch.cat(parts, 1))
        return b.cls_head.components(base, physical)[2]


def readout_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered cross-path relation architecture')
    d = neural_contract(BASE_VARIANT)
    d.update(
        mode=variant, identity_core='crosspath_relation_identity', base_variant=BASE_VARIANT,
        base_trainable_parameters=BASE_PARAMETERS, new_trainable_parameters=NEW_PARAMETERS,
        learned_readout_active=True, relation_paths=['time', 'behavior'],
        relation_input_shapes=[[2, 32, 64], [2, 32, 64]],
        relation_projection_channels=8, relation_lags=list(LAGS),
        relation_lag_sign='A(t-lag)*conj(B(t)); A=time, B=behavior',
        relation_common_positions=[8, 64], relation_pairs_per_lag=56,
        relation_statistic='uncentered complex cross moment',
        relation_normalization='per_projected_channel_RMS' if variant == VARIANTS[1] else 'whole_projected_branch_RMS',
        relation_power_floor=POWER_FLOOR, relation_scale_divisor=8,
        relation_dimensions=384, relation_linear_rank=16, output_dimension=320,
        relation_exit='biasfree Linear384to16 then biasfree Linear16to320; no activation',
        readout_skip='complete original320 InvariantReadout retained independently in both paths',
        readout_rule='behavior_skip+relation; original time readout unchanged',
        readout_exit_initialization='zero final projection; random earlier projections',
        readout_scope='one received packet; common finite lag window; no cross-packet statistics/state',
        initial_function='exact own scratch neural_residual_shallow; isolated new initialization RNG',
        relation_gain_property='positive per-projected-channel gain invariance above floors' if variant == VARIANTS[1] else 'positive whole-projected-branch gain invariance above floors',
        gain_property_scope='relation statistics after projections only; not the whole model or arbitrary physical channel',
        relation_physical_alignment_claim=False, inherited_padding_retained=True,
        raw_tx_hardware_identified=False, coadaptation_prevented=False,
    )
    return d


@contextmanager
def diagnostic_mode(model, x):
    flags = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=[x.get_device()] if x.is_cuda else []):
            model.eval()
            yield
    finally:
        for module, training in flags:
            module.training = training


class CrossPathCVS(NeuralResidualCVS):
    def __init__(self, variant):
        readout_contract(variant)
        super().__init__(BASE_VARIANT)
        self.relation_variant = variant
        with torch.random.fork_rng(devices=[]):
            self.core = CrossPathCore(self.core, variant == VARIANTS[1])

    def readout_blocks(self):
        return [('crosspath.readout', self.core.relation)]

    def readout_parameters(self):
        return list(self.core.relation.parameters())

    def contract(self):
        d = readout_contract(self.relation_variant)
        r = self.core.relation
        projections = (r.time_project, r.behavior_project)
        actual = (isinstance(self.core, CrossPathCore) and isinstance(self.core.readout, InvariantReadout)
                  and isinstance(r, CrossPathRelation)
                  and r.channel_normalized == (self.relation_variant == VARIANTS[1])
                  and all(isinstance(p, ComplexConv) and (p.cin, p.cout, p.k, p.stride, p.dilation, p.causal)
                          == (32, 8, 1, 1, 1, False) for p in projections)
                  and isinstance(r.compress, nn.Linear) and isinstance(r.project, nn.Linear)
                  and (r.compress.in_features, r.compress.out_features, r.project.in_features, r.project.out_features)
                  == (384, 16, 16, 320) and r.compress.bias is None and r.project.bias is None)
        ids = {id(p) for p in self.readout_parameters()}
        d.update(learned_readout_active=actual,
                 neural_residual_active=super().contract()['neural_residual_active'],
                 new_trainable_parameters=sum(p.numel() for p in self.readout_parameters() if p.requires_grad),
                 base_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in ids))
        return d

    @torch.no_grad()
    def readout_diagnostics(self, x):
        records = []

        def capture(module, inputs, output):
            skip = self.core.readout(inputs[1])
            statistics = module.statistics(*inputs)
            a, b = module.time_project(inputs[0]), module.behavior_project(inputs[1])
            floors = []
            for lag in LAGS:
                for name, z in (('time', a[..., 8-lag:64-lag]), ('behavior', b[..., 8:64])):
                    power = z.square().sum(1).mean(-1)
                    if not module.channel_normalized:
                        power = power.mean(-1, keepdim=True)
                    floors.append(dict(path=name, lag=lag, floor_fraction=float((power < POWER_FLOOR).float().mean())))
            records.append(dict(block='crosspath.readout', packets=len(output),
                relative_output_change_mean=float((output.norm(dim=1) / skip.norm(dim=1).clamp_min(1e-12)).mean()),
                projection_norm=float(module.project.weight.norm()),
                compress_norm=float(module.compress.weight.norm()),
                relation_norm_by_lag=statistics.flatten(2).norm(dim=-1).mean(0).tolist(),
                output_dimension=320, relation_linear_rank=16, relation_pairs_per_lag=56,
                per_projected_channel_normalization=module.channel_normalized, floor_records=floors))

        hook = self.core.relation.register_forward_hook(capture)
        try:
            with diagnostic_mode(self, x):
                self.features(x)
        finally:
            hook.remove()
        return dict(active=self.contract()['learned_readout_active'], records=records,
                    scope='last source batch; measured cross-path relation; no updates or new loss')

    @torch.no_grad()
    def diagnostics(self, x):
        with diagnostic_mode(self, x):
            return dict(super().diagnostics(x), learned_readout=self.readout_diagnostics(x))


def build(variant):
    return CrossPathCVS(variant)
