"""Remove the original frequency backbone while controlling PA IQ and statistics."""
from types import MethodType
from torch import nn
import torch

from experiments.pamf_branch_factorial.graph import DisabledSinc
from experiments.cric_dura_comparison.structural import NoSequence, ZeroProjection


class EmptyPhysical(nn.Module):
    def forward(self, x):
        return x.new_zeros((x.shape[0], 160, 1))


class PassGate(nn.Module):
    def forward(self, x, _iq):
        return x


def install(model, use_time, use_path, use_pa, use_spectrum):
    if use_path and not use_time:
        raise ValueError('Path descriptor requires time backbone')
    ref = model.id_backbone.encoder
    core = ref.core.id_backbone
    if (hasattr(ref, 'response') or not core.use_time_path or not core.use_freq_path or
            not core.use_pa_path or core.use_dac_path or core.freq_feature_source != 'raw_fft' or
            core.pa_feature_source != 'raw_iq' or not core.use_circularity or not core.use_stats_path):
        raise ValueError('Unexpected no-reference source graph')
    if not use_path:
        ref.sequence_pool = NoSequence()
    if not use_time:
        def features_without_time(self, x):
            return self.core.features(x)
        ref.features = MethodType(features_without_time, ref)
        core.use_time_path = False
        core.sinc = DisabledSinc()
        for name in ('hf', 'time_fuse', 'time_down', 't1', 't2', 't3', 't_pool', 't_proj'):
            setattr(core, name, None)
    core.use_freq_path = False
    for name in ('f1', 'f2', 'f3', 'f_pool', 'f_proj', 'freq_gate', 'freq_stats_proj'):
        setattr(core, name, None)
    if not use_pa:
        # Preserve the original PA combination point and its learned layer norm,
        # but remove every input-dependent PA-IQ operator and its parameters.
        core.pa_lift = EmptyPhysical()
        core.pa_gate = PassGate()
        core.pa_b1 = core.pa_b2 = core.pa_b3 = nn.Identity()
        core.pa_pool = nn.AdaptiveAvgPool1d(1)
        core.pa_proj = nn.Identity()
    if not use_spectrum:
        core.pa_stats_proj = ZeroProjection()
    def gather_aux(module, args):
        x = args[0]
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Expected 2x256 input')
        if use_spectrum:
            _, _, _, pa_stats = module._mirror_compressed_features(x, sinc_iq=None)
        else:
            pa_stats = x.new_zeros((len(x), 3))
        module.pa_stats_proj._factorial_pa_stats = pa_stats
    def clear_aux(module, args, output):
        if hasattr(module.pa_stats_proj, '_factorial_pa_stats'):
            del module.pa_stats_proj._factorial_pa_stats
    def pa_inputs(module, args):
        return (module._factorial_pa_stats,)
    core.register_forward_pre_hook(gather_aux)
    core.register_forward_hook(clear_aux)
    core.pa_stats_proj.register_forward_pre_hook(pa_inputs)
    def fusion_inputs(module, args):
        value = args[0]
        zero = value.new_zeros((value.shape[0], 160))
        time = value[:, :160] if use_time else zero
        return (torch.cat((time, zero, value.new_zeros((value.shape[0], 1))), dim=1),)
    core.fuse.register_forward_pre_hook(fusion_inputs)
    return model
