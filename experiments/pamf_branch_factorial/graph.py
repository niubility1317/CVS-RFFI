"""Isolated branch removal; retain the matched 321-dimensional fusion interface."""
from types import MethodType

import torch
from torch import nn


# time, frequency, PA, path descriptor, retain rho when frequency is absent
SPECS = {
    "no_pa": (1, 1, 0, 1, 0),
    "strict_no_frequency": (1, 0, 1, 1, 0),
    "no_time": (0, 1, 1, 0, 0),
    "time_only": (1, 0, 0, 1, 0),
    "frequency_only": (0, 1, 0, 0, 0),
    "pa_only": (0, 0, 1, 0, 0),
    "no_frequency_with_rho": (1, 0, 1, 1, 1),
    "no_path_no_frequency": (1, 0, 1, 0, 0),
    "no_path_time_only": (1, 0, 0, 0, 0),
}


class DisabledSinc(nn.Module):
    """No trainable time front end; calling it is always a structural error."""
    def forward(self, *args, **kwargs):
        raise RuntimeError("Disabled Sinc front end was called")

    def forward_iq_pair(self, *args, **kwargs):
        raise RuntimeError("Disabled Sinc front end was called")


def install(model, arm, seed):
    if arm not in SPECS:
        raise ValueError("Unknown branch-factorial arm")
    from experiments.cric_dura_comparison.supervised_physics import install as install_old
    from experiments.cric_dura_comparison.structural import NoSequence, remove_branch

    use_time, use_freq, use_pa, use_signature, keep_rho = SPECS[arm]
    if use_signature and not use_time:
        raise ValueError("Descriptor requires the time backbone")
    model = install_old(model, "NO_REFERENCE", seed)
    ref = model.id_backbone.encoder
    core = ref.core.id_backbone
    if (core.use_time_path is not True or core.use_freq_path is not True or
            core.use_pa_path is not True or core.use_dac_path is not False or
            core.freq_feature_source != "raw_fft" or core.pa_feature_source != "raw_iq" or
            not core.use_circularity or not core.use_stats_path):
        raise ValueError("Source backbone contract differs")
    if not use_signature:
        ref.sequence_pool = NoSequence()
    if not use_time:
        # WithoutReference normally captures t2. This control has no t2 at all.
        def features_without_time(self, x):
            return self.core.features(x)
        ref.features = MethodType(features_without_time, ref)
        core.use_time_path = False
        core.sinc = DisabledSinc()
        for name in ("hf", "time_fuse", "time_down", "t1", "t2", "t3", "t_pool", "t_proj"):
            setattr(core, name, None)
    if not use_pa:
        remove_branch(ref, "F09")
        for name in ("pa_lift", "pa_gate", "pa_b1", "pa_b2", "pa_b3", "pa_pool"):
            setattr(core, name, None)
    if not use_freq:
        core.use_freq_path = False
        for name in ("f1", "f2", "f3", "f_pool", "f_proj", "freq_gate", "freq_stats_proj"):
            setattr(core, name, None)

        def gather_aux(module, args):
            x = args[0]
            if x.ndim != 3 or x.shape[1:] != (2, 256):
                raise ValueError("Expected source-compatible 2x256 input")
            if use_pa or keep_rho:
                _, rho, _, pa_stats = module._mirror_compressed_features(x, sinc_iq=None)
                module._factorial_aux = (rho, pa_stats)
            else:
                module._factorial_aux = (None, None)

        def clear_aux(module, args, output):
            if hasattr(module, "_factorial_aux"):
                del module._factorial_aux

        core.register_forward_pre_hook(gather_aux)
        core.register_forward_hook(clear_aux)
        if use_pa:
            def pa_inputs(module, args):
                return (core._factorial_aux[1],)
            core.pa_stats_proj.register_forward_pre_hook(pa_inputs)

    if not use_time or not use_freq:
        # The native constructor's fuse has the original 160 + 160 + 1 input.
        # Native forward omits disabled chunks; restore fixed zero placeholders.
        def fusion_inputs(module, args):
            value = args[0]
            b = value.shape[0]
            zero = value.new_zeros((b, 160))
            zero_rho = value.new_zeros((b, 1))
            if use_time:
                t = value[:, :160]
            else:
                t = zero
            if use_freq:
                offset = 160 if use_time else 0
                f = value[:, offset:offset + 160]
                rho = value[:, offset + 160:offset + 161]
            else:
                f = zero
                rho = core._factorial_aux[0] if keep_rho else zero_rho
            if f.shape != zero.shape or rho is None or rho.shape != zero_rho.shape:
                raise ValueError("Original time/frequency/rho fusion layout differs")
            return (torch.cat((t, f, rho), dim=1),)
        core.fuse.register_forward_pre_hook(fusion_inputs)
    model._feature_learning_report.update(branch_factorial=arm, source_U_IQ_used=False,
                                           keep_time=bool(use_time), keep_frequency=bool(use_freq),
                                           keep_pa=bool(use_pa), keep_signature=bool(use_signature),
                                           rho_without_frequency=bool(keep_rho))
    return model
