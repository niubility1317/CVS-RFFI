"""Synthetic CPU checks using the actual native DSConvBlock1d implementation."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import (CORE, LAYOUT, AdaptedDSConvBlock1d, attach, enable,
                       adapter_named_parameters, adapter_parameters,
                       frozen_state, assert_frozen_state, adapter_state_report)


def run(native_block=None):
    if native_block is None:
        from model import DSConvBlock1d as native_block
    if native_block.__name__ != 'DSConvBlock1d':
        raise ValueError('Use the actual immutable native DSConvBlock1d class')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(27300)
            return _run(native_block)
    finally:
        torch.set_num_threads(old_threads)


def _run(native_block):
    negatives = 0
    def reject(call, kind=ValueError):
        nonlocal negatives
        try:
            call()
        except kind:
            negatives += 1
        else:
            raise AssertionError('Invalid adapter geometry/state accepted')

    for name, (cin, cout, length) in LAYOUT.items():
        original = native_block(cin, cout, k=5, pool=2, drop=.1)
        old_keys = set(original.state_dict())
        state = torch.get_rng_state().clone()
        adapted = AdaptedDSConvBlock1d(deepcopy(original), name, 27301)
        if not torch.equal(state, torch.get_rng_state()):
            raise AssertionError('Adapter construction consumed the native CPU RNG stream')
        if (set(adapted.state_dict()) - old_keys !=
                {'adapter.A.weight', 'adapter.D.weight', 'adapter.B.weight'}
                or any(not torch.equal(v, adapted.state_dict()[k])
                       for k, v in original.state_dict().items())
                or adapted.adapter.B.weight.count_nonzero()
                or not adapted.adapter.A.weight.count_nonzero()
                or not adapted.adapter.D.weight.count_nonzero()):
            raise AssertionError('Original state names or zero-B-only initialization differ')
        x = torch.randn(4, cin, length)
        for training in (False, True):
            original.train(training); adapted.train(training)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(27302); expected = original(x)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(27302); actual = adapted(x)
                if (not torch.equal(actual, expected)
                        or not torch.equal(expected_rng, torch.get_rng_state())
                        or actual.shape != (4, cout, length//2)):
                    raise AssertionError('Zero-B native train/eval/dropout path changed')
        reject(lambda: adapted(torch.zeros(4, cin+1, length)))
        reject(lambda: adapted(torch.zeros(4, cin, length+1)))
        reject(lambda: adapted(torch.zeros(cin, length)))
    reject(lambda: AdaptedDSConvBlock1d(native_block(47,72,k=5), 't1', 1))

    class FixedTestHead(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.randn(6, 160)); self.scale = 30.
        def forward(self, z):
            return self.scale * F.linear(F.normalize(z, dim=1, eps=1e-4),
                                         F.normalize(self.weight, dim=1, eps=1e-4))
    class SyntheticStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.id_backbone = nn.Module(); self.id_backbone.encoder = nn.Module()
            self.id_backbone.encoder.core = nn.Module()
            core = nn.Module(); self.id_backbone.encoder.core.id_backbone = core
            core.t1 = native_block(48, 72, k=5, pool=2, drop=.1)
            core.t2 = native_block(72, 96, k=5, pool=2, drop=.1)
            core.t3 = native_block(96, 96, k=3, pool=1, drop=.1)
            core.fixed_projection = nn.Linear(96, 160)
            core.cls_head = FixedTestHead()
        def forward(self, x):
            core = self.get_submodule(CORE)
            h = core.t3(core.t2(core.t1(x))).mean(-1)
            return core.cls_head(core.fixed_projection(h))
    student = SyntheticStudent()
    original_keys = set(student.state_dict())
    original_state = deepcopy(student.state_dict())
    old_head = student.get_submodule(CORE).cls_head
    state = torch.get_rng_state().clone()
    attach(student, seed=2026092701)
    if not torch.equal(state, torch.get_rng_state()) or student.get_submodule(CORE).cls_head is not old_head:
        raise AssertionError('Attachment consumed native RNG or replaced default W/head')
    parameters = adapter_named_parameters(student)
    if ({name for name,p in student.named_parameters() if p.requires_grad} != {n for n,_ in parameters}
            or set(student.state_dict()) != original_keys | {n for n,_ in parameters}
            or len(parameters) != 6 or sum(p.numel() for _,p in parameters) != 4768):
        raise AssertionError('Only six adapter leaves/4768 parameters may be trained')
    for name in original_keys:
        if not torch.equal(original_state[name], student.state_dict()[name]):
            raise AssertionError('Original F tensor changed during adapter attachment')
    before = frozen_state(student)
    captures = []
    handle = student.get_submodule(CORE+'.t2').register_forward_hook(
        lambda module,args,out: captures.append(tuple(out.shape)))
    optimizer = torch.optim.SGD(adapter_parameters(student), lr=.03)
    x, y = torch.randn(4, 48, 128), torch.arange(4) % 6
    student.train()
    F.cross_entropy(student(x), y).backward()
    first_B = {}
    for site in LAYOUT:
        adapter = student.get_submodule(CORE+'.'+site+'.adapter')
        for branch in ('A', 'D'):
            g = getattr(adapter,branch).weight.grad
            if g is None or bool(g.count_nonzero()):
                raise AssertionError('Zero B should initially give exactly zero A/D gradient')
        g = adapter.B.weight.grad
        if g is None or not bool(torch.isfinite(g).all()) or not bool(g.count_nonzero()):
            raise AssertionError('First step B gradient must survive frozen later layers')
        first_B[site] = float(g.norm())
    assert_frozen_state(student, before)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(student(x), y).backward()
    second_AD = {}
    for site in LAYOUT:
        adapter = student.get_submodule(CORE+'.'+site+'.adapter')
        second_AD[site] = {}
        for branch in ('A', 'D'):
            g = getattr(adapter,branch).weight.grad
            if g is None or not bool(torch.isfinite(g).all()) or not bool(g.count_nonzero()):
                raise AssertionError('A/D must receive effective gradients after B moves')
            second_AD[site][branch] = float(g.norm())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    handle.remove()
    if captures != [(4,96,32), (4,96,32)]:
        raise AssertionError('Full t2 output/signature tap changed')
    frozen = assert_frozen_state(student, before)
    saved = deepcopy(student.state_dict())
    reloaded = SyntheticStudent(); attach(reloaded, seed=2026092701)
    reloaded.load_state_dict(saved, strict=True)
    student.eval(); reloaded.eval()
    with torch.no_grad():
        if not torch.equal(student(x), reloaded(x)):
            raise AssertionError('Strict new adapter state reload changed frozen inference')
    reject(lambda: SyntheticStudent().load_state_dict(saved, strict=True), RuntimeError)
    reject(lambda: attach(student, seed=2026092701))
    reject(lambda: attach(SyntheticStudent(), seed=True))
    bad_head = SyntheticStudent(); bad_head.get_submodule(CORE).cls_head.scale = 1.
    reject(lambda: attach(bad_head))
    enable(student, False)
    if any(p.requires_grad for p in student.parameters()):
        raise AssertionError('Disabled training must leave original F/W and adapters frozen')
    enable(student)
    report = adapter_state_report(student)
    if report['sites']['t1']['parameters'] != 2000 or report['sites']['t2']['parameters'] != 2768:
        raise AssertionError('Actual two-site parameter count differs')
    return dict(status='PASS', protocol_negatives=negatives, native_block=native_block.__name__,
        native_zero_B_train_eval_same_RNG=True, CPU_RNG_preserved=True,
        first_B_grad_norm=first_B, after_B_update_AD_grad_norm=second_AD,
        frozen_later_layers_backpropagate=True, frozen=frozen, strict_reload_equal=True,
        original_default_W_retained=True, t2_signature_tap=[96,32], adapter=report,
        synthetic_optimizer_steps=2, real_IQ_packets=0, checkpoints_loaded=0,
        full_F_model_constructed=False, real_training_updates=0)


if __name__ == '__main__':
    import json
    print(json.dumps(run(), allow_nan=False))
