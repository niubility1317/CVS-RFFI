"""CPU synthetic checks of actual native blocks/head and same-call raw features."""
from copy import deepcopy
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from . import adapters as a


def _fixture_type(native_block, native_head, pool_type):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = nn.Module()
            b = self.core.id_backbone = nn.Module()
            b.stem = nn.Conv1d(2, 48, 1, bias=False)
            b.t1 = native_block(48, 72, k=5, pool=2, drop=.1)
            b.t2 = native_block(72, 96, k=5, pool=2, drop=.1)
            b.t3 = native_block(96, 96, k=3, pool=1, drop=.1)
            b.projection = nn.Linear(96, 160)
            b.physical = nn.Linear(96, 160)
            b.cls_head = native_head(160, 6)
            self.sequence_pool = pool_type(96, 'SIG_TIME_X3', 2026092701)
            self.base_calls = self.sequence_taps = 0

        def features(self, x):
            local = {}
            b = self.core.id_backbone
            handle = b.t2.register_forward_hook(lambda m, args, out: local.update(t2=out))
            try:
                self.base_calls += 1
                h = b.t3(b.t2(b.t1(F.avg_pool1d(b.stem(x), 2)))).mean(-1)
                base = b.cls_head.components(b.projection(h), b.physical(h))[2]
            finally:
                handle.remove()
            self.sequence_taps += 1
            return base + self.sequence_pool(local['t2'])

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.id_backbone = nn.Module()
            self.id_backbone.encoder = Encoder()
            for i in range(206 - len(self.state_dict())):
                self.register_buffer('synthetic_padding_' + str(i), torch.tensor(float(i)))
            if len(self.state_dict()) != 206:
                raise AssertionError('Synthetic native-layout parent must have 206 state tensors')

        def forward(self, x):
            z = self.id_backbone.encoder.features(x)
            return self.get_submodule(a.CORE).cls_head.classify(z)

    return Student


def _counted(self, x):
    self.synthetic_counted_calls += 1
    return self._counted_features(x)


def run(native_block=None, native_head=None, pool_type=None):
    if native_block is None:
        from model import DSConvBlock1d as native_block
    if native_head is None:
        from experiments.cvs_residual_identity.model import ResidualPhysicalHead as native_head
    if pool_type is None:
        from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool as pool_type
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(27431)
            return _run(native_block, native_head, pool_type)
    finally:
        torch.set_num_threads(old_threads)


def _run(native_block, native_head, pool_type):
    Student = _fixture_type(native_block, native_head, pool_type)
    negatives = 0
    def reject(call, kind=ValueError):
        nonlocal negatives
        try:
            call()
        except kind:
            negatives += 1
        else:
            raise AssertionError('Invalid raw feature path accepted')

    x = torch.randn(3, 2, 256)
    alternate = x + .125
    labels = torch.tensor([0, 1, 2])
    plain = Student().eval()
    model = deepcopy(plain)
    before = deepcopy(model.state_dict())
    block_ids = {name: id(model.get_submodule(a.CORE + '.' + name)) for name in a.LAYOUT}
    head = model.get_submodule(a.CORE).cls_head
    rng = torch.get_rng_state().clone()
    a.attach(model)
    if (not torch.equal(rng, torch.get_rng_state()) or model.get_submodule(a.CORE).cls_head is not head
            or any(id(model.get_submodule(a.CORE + '.' + n)) != ident for n, ident in block_ids.items())
            or len(model.state_dict()) != 214 or set(model.state_dict()) != set(before) | set(a.parameter_names(model))
            or any(not torch.equal(v, model.state_dict()[n]) for n, v in before.items())):
        raise AssertionError('Install must preserve RNG, original206 keys, blocks and default head')
    params = dict(a.adapter_named_parameters(model))
    module = model.id_backbone.encoder.raw_iq_adapter
    if (len(params) != 8 or sum(p.numel() for p in params.values()) != 4256
            or not bool(module.conv1.weight.count_nonzero()) or not bool(module.conv2.weight.count_nonzero())
            or bool(module.B1.weight.count_nonzero()) or bool(module.B2.weight.count_nonzero())
            or {n for n, p in model.named_parameters() if p.requires_grad} != set(params)):
        raise AssertionError('Only two projections may be zero and only4256 raw-feature parameters train')
    macs = (module.conv1.weight.numel() * 128 + module.conv2.weight.numel() * 64
            + module.B1.weight.numel() * 128 + module.B2.weight.numel() * 64)
    if macs != 356352:
        raise AssertionError('Fixed convolution MACs differ')
    with torch.no_grad():
        expected = plain(x)
        actual = model(x)
    if not torch.equal(expected, actual):
        raise AssertionError('Zero-B start must exactly preserve the complete synthetic F function')

    # Preserve an already installed _counted_features binding, not just its name.
    counted_first = Student().eval()
    ce = counted_first.id_backbone.encoder
    ce._counted_features = ce.features
    original_binding = ce._counted_features
    ce.synthetic_counted_calls = 0
    ce.features = MethodType(_counted, ce)
    a.attach(counted_first)
    counted_first(x)
    if ce._counted_features is not original_binding or ce.synthetic_counted_calls != 1:
        raise AssertionError('Existing original _counted_features binding must remain intact')

    # Exercise the actual caller order: attach, then counted wrapper, then EMA copy.
    encoder = model.id_backbone.encoder
    encoder._counted_features = encoder.features
    encoder.synthetic_counted_calls = 0
    encoder.features = MethodType(_counted, encoder)
    observed = []
    handle = module.register_forward_pre_hook(lambda m, args: observed.append(args[0]))
    base_before, tap_before = encoder.base_calls, encoder.sequence_taps
    call_before, counted_before = encoder._raw_iq_calls, encoder.synthetic_counted_calls
    try:
        model(x)
        model(alternate)
    finally:
        handle.remove()
    if (len(observed) != 2 or observed[0] is not x or observed[1] is not alternate
            or encoder.base_calls != base_before + 2 or encoder.sequence_taps != tap_before + 2
            or encoder._raw_iq_calls != call_before + 2 or encoder.synthetic_counted_calls != counted_before + 2):
        raise AssertionError('Raw/As must each use current IQ and original F/SIG_TIME exactly once')

    model.eval()
    model(x).square().mean().backward()
    if (module.B1.weight.grad is None or not bool(module.B1.weight.grad.count_nonzero())
            or module.B2.weight.grad is None or not bool(module.B2.weight.grad.count_nonzero())):
        raise AssertionError('Both zero-output B projections need feature gradients through frozen F/head')
    a.assert_frozen_state(model, before)
    opt = torch.optim.SGD(a.adapter_parameters(model), lr=.01)
    opt.step()
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(alternate), labels).backward()
    if (module.conv1.weight.grad is None or not bool(module.conv1.weight.grad.count_nonzero())
            or module.conv2.weight.grad is None or not bool(module.conv2.weight.grad.count_nonzero())):
        raise AssertionError('Raw encoder gradients must flow after nonzero B update')
    a.assert_frozen_state(model, before)
    model.zero_grad(set_to_none=True)

    ema = deepcopy(model).eval().requires_grad_(False)
    ee = ema.id_backbone.encoder
    if (ee.features.__self__ is not ee or ee._counted_features.__self__ is not ee
            or ee._raw_iq_base_features.__self__ is not ee
            or ee.raw_iq_adapter is module):
        raise AssertionError('EMA raw/count/base bound methods and parameters must belong to its own copy')
    calls_before = encoder._raw_iq_calls
    ema_calls_before = ee._raw_iq_calls
    with torch.no_grad():
        left = model(x)
        right = ema(x)
    if not torch.equal(left, right) or encoder._raw_iq_calls != calls_before + 1 or ee._raw_iq_calls != ema_calls_before + 1:
        raise AssertionError('EMA inference must use only its own original F and raw feature module')
    frozen_student = deepcopy(model.state_dict())
    with torch.no_grad():
        ee.raw_iq_adapter.B1.weight.add_(.005)
        different = ema(x)
    if torch.equal(left, different) or any(not torch.equal(v, model.state_dict()[n]) for n, v in frozen_student.items()):
        raise AssertionError('Changing EMA supplement must not modify or execute the student copy')

    # Rebuild before strict loading the new214 state; no old local-adapter keys.
    reloaded = a.attach(Student().eval())
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        if not torch.equal(model(x), reloaded(x)):
            raise AssertionError('Strict reconstructed raw-feature state must reproduce the function')

    core = model.get_submodule(a.CORE)
    hooks_before = {n: len(getattr(core, n).pw._forward_hooks) for n in a.LAYOUT}
    def fail(module, args):
        raise RuntimeError('synthetic downstream failure')
    failure = core.t2.pw.register_forward_pre_hook(fail)
    try:
        reject(lambda: model(x), RuntimeError)
    finally:
        failure.remove()
    if (encoder._raw_iq_active or any(len(getattr(core, n).pw._forward_hooks) != hooks_before[n] for n in a.LAYOUT)
            or core.t2._forward_hooks):
        raise AssertionError('Temporary raw and pre-existing SIG_TIME call-local hooks must clean up on failure')
    model(x)
    reject(lambda: a.attach(model))
    reject(lambda: model(torch.randn(3, 2, 255)))
    reject(lambda: model(torch.full_like(x, float('nan'))))
    reject(lambda: model(x.double()))
    bad = Student()
    bad.get_submodule(a.CORE).t1.pw = nn.Conv1d(48, 71, 1, bias=False)
    reject(lambda: a.attach(bad))
    if hasattr(bad.id_backbone.encoder, 'raw_iq_adapter'):
        raise AssertionError('Invalid native layout must fail before installing the supplement')
    if any(getattr(core, n).pw._forward_hooks for n in a.LAYOUT):
        raise AssertionError('No temporary hook may persist between raw/As/U/eval calls')
    report = a.check(model)
    return dict(status='PASS', protocol_negatives=negatives, parameters=4256, new_state_tensors=8,
        parent_state_tensors=206, final_state_tensors=214, convolution_MACs=macs,
        native_block=native_block.__name__, default_head=native_head.__name__, SIG_TIME_pool=pool_type.__name__,
        zero_B_complete_function_exact=True, CPU_RNG_preserved=True, original_blocks_preserved=True,
        current_raw_As_input_verified=True, one_original_F_forward_per_features=True,
        gradient_through_frozen_F_head=True, both_raw_convolutions_receive_gradient=True,
        own_EMA_deepcopy_bound_correctly=True, existing_counted_binding_preserved=True,
        temporary_hooks_cleanup_on_exception=True, strict214_reload=True,
        original206_frozen=True, report=report, synthetic_optimizer_steps=1,
        checkpoints_loaded=0, real_IQ_packets=0, remote_actions=0, real_training_updates=0)


if __name__ == '__main__':
    import json
    print(json.dumps(run(), ensure_ascii=False, allow_nan=False))
