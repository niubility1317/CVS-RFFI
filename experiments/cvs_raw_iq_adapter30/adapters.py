"""Received-IQ feature supplement before the original t1/t2 GroupNorms.

All old F/state names remain intact.  Call-local pointwise-convolution hooks
use this features call's IQ, and bound methods deepcopy with the owning encoder.
"""
from numbers import Integral
from types import MethodType

import torch
from torch import nn

CORE = 'id_backbone.encoder.core.id_backbone'
ENCODER = 'id_backbone.encoder'
MODULE = ENCODER + '.raw_iq_adapter'
LAYOUT = {'t1': (48, 72, 128), 't2': (72, 96, 64)}
ORIGINAL_CHILDREN = ('dw', 'pw', 'norm', 'act', 'pool', 'drop')
WIDTH = 16
PARAMETERS = 4256
CONV_MACS = 356352
STATE_TENSORS = 8
SEED_OFFSET = 190401


class RawIQSupplement(nn.Module):
    """2x256 -> g1(16x128) -> g2(16x64), with two zero-output projections."""
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            factory = dict(device='cpu', dtype=torch.float32)
            self.conv1 = nn.Conv1d(2, WIDTH, 7, stride=2, padding=3, bias=False, **factory)
            self.gn1 = nn.GroupNorm(8, WIDTH, **factory)
            self.conv2 = nn.Conv1d(WIDTH, WIDTH, 5, stride=2, padding=2, bias=False, **factory)
            self.gn2 = nn.GroupNorm(8, WIDTH, **factory)
            self.B1 = nn.Conv1d(WIDTH, 72, 1, bias=False, **factory)
            self.B2 = nn.Conv1d(WIDTH, 96, 1, bias=False, **factory)
            nn.init.zeros_(self.B1.weight)
            nn.init.zeros_(self.B2.weight)
        self.act1, self.act2 = nn.GELU(), nn.GELU()

    def forward(self, x):
        if (x.ndim != 3 or x.shape[0] == 0 or tuple(x.shape[1:]) != (2, 256)
                or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('The current finite received IQ must be FP32[N,2,256]')
        g1 = self.act1(self.gn1(self.conv1(x)))
        g2 = self.act2(self.gn2(self.conv2(g1)))
        return self.B1(g1), self.B2(g2)


def _core(model):
    return model.get_submodule(CORE)


def _geometry(core):
    for name, (cin, cout, _) in LAYOUT.items():
        block = getattr(core, name)
        if (type(block).__name__ != 'DSConvBlock1d' or tuple(block._modules) != ORIGINAL_CHILDREN
                or not isinstance(block.dw, nn.Conv1d) or not isinstance(block.pw, nn.Conv1d)
                or (block.dw.in_channels, block.dw.out_channels, block.dw.groups,
                    block.dw.kernel_size, block.dw.stride, block.dw.padding) !=
                   (cin, cin, cin, (5,), (1,), (2,))
                or (block.pw.in_channels, block.pw.out_channels, block.pw.kernel_size,
                    block.pw.stride) != (cin, cout, (1,), (1,))
                or block.dw.bias is not None or block.pw.bias is not None
                or not isinstance(block.norm, nn.GroupNorm) or block.norm.num_channels != cout
                or not isinstance(block.pool, nn.MaxPool1d)
                or block.pool.kernel_size != 2 or block.pool.stride != 2
                or block.pw.weight.dtype != torch.float32):
            raise ValueError('Original native FP32 t1/t2 geometry required; no local adapter may be stacked')


def _module(model):
    encoder = model.get_submodule(ENCODER)
    if (not isinstance(getattr(encoder, 'raw_iq_adapter', None), RawIQSupplement)
            or not isinstance(getattr(encoder, '_raw_iq_base_features', None), MethodType)
            or encoder._raw_iq_base_features.__self__ is not encoder):
        raise ValueError('A received-IQ feature supplement bound to this encoder is required')
    return encoder.raw_iq_adapter


def _features_with_raw(self, x):
    """No persistent IQ or module-capturing hook closures survive this call."""
    if getattr(self, '_raw_iq_active', False):
        raise ValueError('Nested received-IQ feature calls cannot share temporary hooks')
    handles = []
    self._raw_iq_active = True
    try:
        deltas = dict(zip(LAYOUT, self.raw_iq_adapter(x)))
        calls = {name: 0 for name in LAYOUT}
        core = self.core.id_backbone
        for name, (_, cout, length) in LAYOUT.items():
            delta = deltas[name]
            if delta.shape != (len(x), cout, length):
                raise ValueError('Raw supplement must match the actual pre-GN temporal map')

            def inject(module, args, output, *, site=name, correction=delta):
                calls[site] += 1
                if (calls[site] != 1 or not torch.is_tensor(output)
                        or output.shape != correction.shape or output.dtype != correction.dtype
                        or output.device != correction.device):
                    raise ValueError('Exactly one same-IQ native pointwise map per site required')
                return output + correction

            handles.append(getattr(core, name).pw.register_forward_hook(inject))
        result = self._raw_iq_base_features(x)
        if (calls != {'t1': 1, 't2': 1} or not torch.is_tensor(result)
                or result.shape != (len(x), 160)):
            raise ValueError('The original F and SIG_TIME feature path must execute once')
        self._raw_iq_calls += 1
        self._raw_iq_packets += len(x)
        self._raw_iq_last_shape = list(x.shape)
        return result
    finally:
        for handle in handles:
            handle.remove()
        self._raw_iq_active = False


def adapter_named_parameters(model):
    result = tuple((MODULE + '.' + name, p) for name, p in _module(model).named_parameters())
    if len(result) != STATE_TENSORS or sum(p.numel() for _, p in result) != PARAMETERS:
        raise ValueError('Raw encoder/projections must have exactly 4256 parameters in eight leaves')
    return result


def adapter_parameters(model):
    return tuple(p for _, p in adapter_named_parameters(model))


def parameter_names(model):
    return tuple(name for name, _ in adapter_named_parameters(model))


def enable(model, enabled=True):
    if not isinstance(enabled, bool):
        raise ValueError('Feature supplement training enabled must be boolean')
    parameters = adapter_parameters(model)
    model.requires_grad_(False)
    for p in parameters:
        p.requires_grad_(enabled)
    return model


def attach(model, seed=2026092701):
    """After strict parent load, before the parent's new counted-features wrapper."""
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError('Explicit model-seed-derived raw encoder initialization required')
    core, encoder = _core(model), model.get_submodule(ENCODER)
    _geometry(core)
    if core.cls_head.weight.shape != (6, 160) or core.cls_head.scale != 30.:
        raise ValueError('Original default6x160 cosine scale30 head required')
    if hasattr(encoder, 'raw_iq_adapter') or hasattr(encoder, '_raw_iq_base_features'):
        raise ValueError('The raw supplement may be attached only once')
    if not isinstance(encoder.features, MethodType) or encoder.features.__self__ is not encoder:
        raise ValueError('Original features must be a bound method owned by this encoder')
    before = set(model.state_dict())
    supplement = RawIQSupplement(int(seed) + SEED_OFFSET).to(
        device=core.t1.pw.weight.device, dtype=core.t1.pw.weight.dtype)
    supplement.train(encoder.training)
    # Save the complete current features method, without altering an existing
    # _counted_features binding.  A subsequently installed counted wrapper can
    # save this method as its own _counted_features entry in the usual way.
    encoder.raw_iq_adapter = supplement
    encoder._raw_iq_base_features = encoder.features
    encoder._raw_iq_calls = encoder._raw_iq_packets = 0
    encoder._raw_iq_last_shape = None
    encoder._raw_iq_active = False
    encoder.features = MethodType(_features_with_raw, encoder)
    names = set(parameter_names(model))
    if set(model.state_dict()) != before | names or before & names:
        raise ValueError('Retain every original state key and add only the eight raw-feature keys')
    model._dta_adapter_names = tuple(name for name, _ in adapter_named_parameters(model))
    return enable(model)


install = attach


def frozen_state(model):
    added = set(parameter_names(model))
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name not in added}


def assert_frozen_state(model, before):
    names = set(parameter_names(model))
    actual = {name: value for name, value in model.state_dict().items() if name not in names}
    if set(actual) != set(before) or any(not torch.equal(value.detach().cpu(), before[name])
                                       for name, value in actual.items()):
        raise ValueError('Original frozen F/head parameters or buffers changed')
    if any(p.requires_grad or p.grad is not None for name, p in model.named_parameters() if name not in names):
        raise ValueError('Only the eight raw feature leaves may receive gradients')
    return dict(original_state_tensors=len(before), original_state_unchanged=True,
                default_W_unchanged=True, nonadapter_grads_none=True)


def adapter_state_report(model):
    module = _module(model)
    expected = set(parameter_names(model))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    encoder = model.get_submodule(ENCODER)
    return dict(kind='current_received_raw_IQ_feature_supplement', width=WIDTH,
        parameters=PARAMETERS, new_state_tensors=STATE_TENSORS, convolution_MACs=CONV_MACS,
        MAC_scope='four biasFalse convolutions; excludes GN/GELU/addition and original F',
        raw_input=[2, 256], g1=[16, 128], g2=[16, 64],
        sites=dict(t1=dict(pre_norm_output=[72, 128]), t2=dict(pre_norm_output=[96, 64])),
        initialization='conv1/conv2 normal Conv1d init; GN native init; only B1/B2 weights zero',
        seed_offset=SEED_OFFSET, extra_gate=False, input='this features call current received IQ',
        injection='original pw output + raw B projection, before unchanged GN/activation/pool',
        original_F_forwards_per_features_call=1, raw_encoders_per_features_call=1,
        temporary_hooks=True, hooks_removed_after_each_call=True, persistent_IQ=False,
        SIG_TIME_t2_tap=[96, 32], original_state_keys_retained=True,
        only_adapters_trainable=trainable == expected,
        original_head_shape=list(_core(model).cls_head.weight.shape),
        original_head_scale=float(_core(model).cls_head.scale),
        successful_features_calls=encoder._raw_iq_calls,
        successful_received_packets=encoder._raw_iq_packets,
        last_input_shape=encoder._raw_iq_last_shape,
        zero_output_projections=all(not bool(p.count_nonzero()) for p in (module.B1.weight, module.B2.weight)))


def check(model):
    _geometry(_core(model))
    report = adapter_state_report(model)
    if set(parameter_names(model)) != set(model._dta_adapter_names):
        raise ValueError('Actual new parameter names differ from model runtime scope')
    return report
