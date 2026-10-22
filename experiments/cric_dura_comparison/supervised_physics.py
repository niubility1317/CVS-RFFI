"""L-only CE architecture controls; no effect on previously released rows."""
from types import MethodType

ARMS = ('SIG_TIME', 'NO_TIME', 'NO_SIGNATURE', 'NO_PA', 'NO_REFERENCE',
        'COV_X3', 'SHUFFLE_TIME', 'WRONG_REFERENCE')


def shuffled_descriptor(self, h):
    import torch
    from experiments.cvs_feature_sequence50.features import path_signature
    u = torch.tanh(self.local_projection(h)).transpose(1, 2)
    b, t, _ = u.shape
    windows = u.reshape(b * 4, t // 4, 6)
    if windows.shape[1] != 8 or self.window_order.device != h.device:
        raise ValueError('Fixed256 IQ requires t2 length32 and device-resident order')
    return path_signature(windows[:, self.window_order], True).reshape(b, 112)


def install(model, arm, seed):
    import torch
    from torch import nn
    if arm not in ARMS:
        raise ValueError('Unregistered supervised architecture control')
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool, DescriptorView
    from .structural import no_time_descriptor, NoSequence, remove_branch
    ref = model.id_backbone.encoder
    if arm == 'SIG_TIME':
        ref.sequence_pool = CompositePool(ref.sequence_pool.channels, 'SIG_TIME', seed)
    elif arm == 'NO_TIME':
        for view in ref.sequence_pool.views:
            view.forward = MethodType(no_time_descriptor, view)
    elif arm == 'NO_SIGNATURE':
        ref.sequence_pool = NoSequence()
    elif arm in ('NO_PA', 'NO_REFERENCE'):
        remove_branch(ref, 'F09' if arm == 'NO_PA' else 'F10')
    elif arm == 'COV_X3':
        channels = ref.sequence_pool.channels
        ref.sequence_pool.views = nn.ModuleList(
            [DescriptorView(channels, 'COV', seed, i) for i in range(3)])
    elif arm == 'SHUFFLE_TIME':
        # Fixed IQ256 -> t2 length32 -> four windows of eight. Register once so
        # model.to(device) moves indices before any CUDA graph capture.
        g = torch.Generator().manual_seed(2026102207)
        order = torch.cat((torch.tensor([0]),torch.randperm(6,generator=g)+1,torch.tensor([7])))
        for view in ref.sequence_pool.views:
            view.register_buffer('window_order',order.clone(),persistent=False)
            view.forward = MethodType(shuffled_descriptor, view)
    elif arm == 'WRONG_REFERENCE':
        response = ref.response
        bank = torch.complex(response.bank_real, response.bank_imag)
        g = torch.Generator().manual_seed(2026102208)
        angles = torch.rand(bank.shape[-1], dtype=torch.float64, generator=g) * (2 * torch.pi)
        # Keep the registered CPU RNG and FP64 phase values; native training
        # may already have placed the frozen reference bank on CUDA.
        phase = torch.polar(torch.ones_like(angles), angles).to(bank.device)
        # Public, sample-independent phase scrambling; spectrum magnitudes and
        # all trainable projection dimensions remain identical to the real bank.
        wrong = torch.fft.ifft(torch.fft.fft(bank, dim=-1) * phase, dim=-1)
        response.bank_real.copy_(wrong.real)
        response.bank_imag.copy_(wrong.imag)
    model._feature_learning_report.update(supervised_architecture=arm,
        source_U_IQ_used=False, physics_claim='Structural inductive bias; no hardware-identifiability claim')
    return model


def validate_config(c):
    if c.get('supervised_architecture') not in ARMS:
        raise ValueError('Unknown L-only architecture')
    if (c.get('regime') != 'matched_sat_supervised' or c.get('ablation') != 'F15'
            or c.get('satellite_concat') is not True or c.get('U_use') is not False
            or c.get('own_EMA') is not False or c.get('native_U_gate') is not False):
        raise ValueError('Strict supervised architecture must not activate U/EMA/CRIC/RDrop')
    if c.get('G_source_roles') != 'L_s' or c.get('optimizer_updates') != 11100:
        raise ValueError('L-only ancestry and matched F update budget required')
