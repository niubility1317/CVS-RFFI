"""Two feature adapters before the original t1/t2 GroupNorms.

Original module/state names are retained.  Freezing parameters deliberately
does not wrap the student's later layers in no_grad or change train/eval mode.
"""
from numbers import Integral

import torch
from torch import nn


CORE = 'id_backbone.encoder.core.id_backbone'
LAYOUT = {'t1': (48, 72, 128), 't2': (72, 96, 64)}
RANK = 16
KERNEL = 5
SEED_OFFSETS = {'t1': 190301, 't2': 190302}
ORIGINAL_CHILDREN = ('dw', 'pw', 'norm', 'act', 'pool', 'drop')


class TemporalResidual(nn.Module):
    """B(GELU(depthwise5(A(h)))); only output B starts at zero."""

    def __init__(self, cin, cout, seed):
        super().__init__()
        # All random initialization occurs on CPU inside a restored RNG scope.
        # Moving the result to the original module's device draws no randomness.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            factory = dict(device='cpu', dtype=torch.float32)
            self.A = nn.Conv1d(cin, RANK, 1, bias=False, **factory)
            self.D = nn.Conv1d(RANK, RANK, KERNEL, padding=2,
                               groups=RANK, bias=False, **factory)
            self.B = nn.Conv1d(RANK, cout, 1, bias=False, **factory)
            nn.init.zeros_(self.B.weight)
        self.activation = nn.GELU()

    def forward(self, h):
        return self.B(self.activation(self.D(self.A(h))))


class AdaptedDSConvBlock1d(nn.Module):
    """Preserve native dw/pw/norm/act/pool/drop; add adapter before norm."""

    def __init__(self, original, layer, seed):
        super().__init__()
        if layer not in LAYOUT or type(original).__name__ != 'DSConvBlock1d':
            raise ValueError('Only actual original F t1/t2 DSConvBlocks may be adapted')
        cin, cout, length = LAYOUT[layer]
        if (tuple(original._modules) != ORIGINAL_CHILDREN
                or not isinstance(original.dw, nn.Conv1d)
                or not isinstance(original.pw, nn.Conv1d)
                or (original.dw.in_channels, original.dw.out_channels,
                    original.dw.groups, original.dw.kernel_size,
                    original.dw.stride, original.dw.padding) !=
                   (cin, cin, cin, (5,), (1,), (2,))
                or (original.pw.in_channels, original.pw.out_channels,
                    original.pw.kernel_size, original.pw.stride) !=
                   (cin, cout, (1,), (1,))
                or original.dw.bias is not None or original.pw.bias is not None
                or not isinstance(original.norm, nn.GroupNorm)
                or original.norm.num_channels != cout
                or not isinstance(original.pool, nn.MaxPool1d)
                or original.pool.kernel_size != 2 or original.pool.stride != 2
                or original.pw.weight.dtype != torch.float32):
            raise ValueError('Actual FP32 native F t1/t2 geometry changed')
        self.layer, self.input_channels, self.output_channels = layer, cin, cout
        self.input_length = length
        for name in ORIGINAL_CHILDREN:
            setattr(self, name, getattr(original, name))
        self.adapter = TemporalResidual(cin, cout, seed).to(
            device=original.pw.weight.device, dtype=original.pw.weight.dtype)
        self.training = original.training
        self.adapter.train(original.training)

    def forward(self, h):
        if (h.ndim != 3 or h.shape[0] == 0
                or tuple(h.shape[1:]) != (self.input_channels, self.input_length)):
            raise ValueError('Actual t1/t2 channel and temporal dimensions required')
        u = self.pw(self.dw(h)) + self.adapter(h)
        return self.drop(self.pool(self.act(self.norm(u))))


def _core(model):
    return model.get_submodule(CORE)


def _blocks(model):
    core = _core(model)
    blocks = {name: getattr(core, name) for name in LAYOUT}
    if any(type(block) is not AdaptedDSConvBlock1d for block in blocks.values()):
        raise ValueError('Both fixed native t1/t2 feature adapters must be installed')
    return blocks


def adapter_named_parameters(model):
    """Exact six adapter leaves, with their real full model state names."""
    result = []
    for name, block in _blocks(model).items():
        result.extend((CORE+'.'+name+'.adapter.'+key, parameter)
                      for key, parameter in block.adapter.named_parameters())
    if len(result) != 6 or sum(p.numel() for _, p in result) != 4768:
        raise ValueError('Fixed rank16/two-site adapter parameter count differs')
    return tuple(result)


def adapter_parameters(model):
    return tuple(parameter for _, parameter in adapter_named_parameters(model))


def enable(model, enabled=True):
    """Freeze original F/W and enable only adapters; retain train/eval flags."""
    if not isinstance(enabled, bool):
        raise ValueError('Adapter training enabled must be boolean')
    parameters = adapter_parameters(model)
    model.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(enabled)
    return model


def attach(model, seed=2026092701):
    """Call after strict parent F loading; reconstruct before strict new CP load."""
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError('Explicit model-seed-derived adapter initialization required')
    core = _core(model)
    head = core.cls_head
    if head.weight.shape != (6, 160) or head.scale != 30.:
        raise ValueError('Original default6x160 cosine scale30 head required')
    old_keys = set(model.state_dict())
    # Construct both before replacing either, so wrong geometry leaves the
    # student's original block registrations intact.
    replacements = {name: AdaptedDSConvBlock1d(getattr(core, name), name,
                    int(seed)+SEED_OFFSETS[name]) for name in LAYOUT}
    for name, block in replacements.items():
        setattr(core, name, block)
    new_keys = {name for name, _ in adapter_named_parameters(model)}
    if set(model.state_dict()) != old_keys | new_keys or old_keys & new_keys:
        raise ValueError('Original state keys must be retained; only six adapter keys added')
    return enable(model)


def frozen_state(model):
    """CPU snapshot of every original parameter/buffer, excluding adapters."""
    adapter_names = {name for name, _ in adapter_named_parameters(model)}
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items() if name not in adapter_names}


def assert_frozen_state(model, before):
    names = {name for name, _ in adapter_named_parameters(model)}
    actual = {name: value for name, value in model.state_dict().items() if name not in names}
    if set(actual) != set(before) or any(
            not torch.equal(value.detach().cpu(), before[name]) for name, value in actual.items()):
        raise ValueError('Original frozen F/head parameters or buffers changed')
    if any(p.requires_grad or p.grad is not None for name, p in model.named_parameters()
           if name not in names):
        raise ValueError('Only the six feature-adapter leaves may receive gradients')
    return dict(original_state_tensors=len(before), original_state_unchanged=True,
                default_W_unchanged=True, nonadapter_grads_none=True)


def adapter_state_report(model):
    blocks = _blocks(model)
    parameters = adapter_named_parameters(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = {name for name, _ in parameters}
    return dict(rank=RANK, kernel=KERNEL, bias=False, parameters=4768,
        sites={name: dict(input=[LAYOUT[name][0], LAYOUT[name][2]],
                         pre_norm_output=[LAYOUT[name][1], LAYOUT[name][2]],
                         block_output=[LAYOUT[name][1], LAYOUT[name][2]//2],
                         parameters=sum(p.numel() for p in block.adapter.parameters()))
               for name, block in blocks.items()},
        initialization='A/D native Conv1d init, only B.weight zero; no extra gate',
        init_seed_offsets=dict(SEED_OFFSETS), insertion='pw(dw(h))+adapter(h), before original GN',
        SIG_TIME_t2_tap=[96, 32], original_state_keys_retained=True,
        only_adapters_trainable=trainable == expected,
        original_head_shape=list(_core(model).cls_head.weight.shape),
        original_head_scale=float(_core(model).cls_head.scale))
