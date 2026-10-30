from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_open_world_model as ow


class CosineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 160))

    def classify(self, features):
        return 30 * F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.t2 = nn.Sequential(nn.Conv1d(2, 96, 1), nn.AdaptiveAvgPool1d(32))
        self.pa = nn.Linear(2, 160)
        self.base = nn.Linear(96, 160)
        self.cls_head = CosineHead()
        self.register_buffer('source_buffer', torch.tensor([7.]))

    def forward(self, x):
        h = self.t2(x)
        return dict(base=self.base(h.mean(-1)), feat_pa=self.pa(x.mean(-1)))


class Signature(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_projection = nn.Linear(96, 160)
        self.calls = 0

    def forward(self, h):
        self.calls += 1
        return .05 * self.output_projection(h.mean(-1))


class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('value', torch.randn(29))
        self.value[27] = 0

    def forward(self, x):
        return self.value[None].expand(len(x), -1)


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.id_backbone = Core()
        self.sequence_pool = Signature()
        self.response = Response()
        self.response_projection = nn.Linear(29, 160)
        self.fail = False
        self.skip_signature = False
        self.zero_base = False
        # Count-scope fixture only; this does not simulate real native weights.
        self.unused_fixture = nn.Parameter(torch.zeros(232993 - sum(p.numel() for p in self.parameters())))

    def features(self, x):
        captured = {}
        handle = self.core.id_backbone.t2.register_forward_hook(lambda m, i, o: captured.update(h=o))
        try:
            aux = self.core.id_backbone(x)
            if self.fail:
                raise RuntimeError('synthetic original-feature failure')
            sig = torch.zeros_like(aux['base']) if self.skip_signature else self.sequence_pool(captured['h'])
            base = aux['base'] + .25 * aux['feat_pa'] + sig + .1 * self.response_projection(self.response(x))
            return torch.zeros_like(base) if self.zero_base else base
        finally:
            handle.remove()

    def classify_features(self, z):
        return self.core.id_backbone.cls_head.classify(z)


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.encoder = Reader()
        self.unused_domain_parameters = nn.Parameter(torch.zeros(667729))

    def forward(self, x):
        reader = self.id_backbone.encoder
        return reader.classify_features(reader.features(x))


def fixture():
    torch.manual_seed(11)
    return Native().eval(), torch.randn(4, 2, 256)


def hook_counts(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


def test_parameter_scope_default_head_paths_and_zero_gain_exact_identity():
    model, x = fixture()
    reader = model.id_backbone.encoder
    head = reader.core.id_backbone.cls_head
    original_parameters = dict(model.named_parameters())
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    old = model(x)
    assert ow.install(model) is model
    assert model.id_backbone.encoder is reader and reader.core.id_backbone.cls_head is head
    out = ow.forward(model, x)
    assert set(out) == {'raw_fused160', 'baseline_fused160', 'logits', 'open64'}
    torch.testing.assert_close(out['logits'], old, atol=0, rtol=0)
    torch.testing.assert_close(out['raw_fused160'], out['baseline_fused160'], atol=0, rtol=0)
    torch.testing.assert_close(out['open64'].norm(dim=1), torch.ones(len(x)), atol=2e-7, rtol=0)
    for name, parameter in original_parameters.items():
        assert dict(model.named_parameters())[name] is parameter
    assert all(name.startswith('open_world.') for name in set(model.state_dict()) - set(original_parameters) - set(buffers))
    report = ow.parameter_report(model)
    assert report['native_identity_parameters'] == 232993
    assert report['open_projector_parameters'] == 37120
    assert report['identity_residual_parameters'] == 10560
    assert report['relation_parameters'] == 4161
    assert report['open_world_parameters'] == 51841
    assert report['active_parameters'] == 284834
    assert report['native_container_parameters'] == 900722
    assert report['training_container_parameters'] == 952563
    assert report['unused_native_parameters'] == 667729
    assert sum(p.numel() for p in model.open_world.parameters()) == 51841
    for name, value in buffers.items(): torch.testing.assert_close(dict(model.named_buffers())[name], value)
    assert (reader.response(x)[:, 27] == 0).all()


def test_real_taps_are_differentiable_single_forward_and_original_head_joint():
    model, x = fixture(); ow.install(model)
    reader = model.id_backbone.encoder
    calls = reader.sequence_pool.calls
    out = ow.forward(model, x)
    assert reader.sequence_pool.calls == calls + 1
    prototype = F.normalize(torch.randn(3, 64), dim=1).requires_grad_()
    ow.relation(model, out['open64'], prototype).square().mean().backward()
    for p in (reader.core.id_backbone.t2[0].weight, reader.core.id_backbone.pa.weight,
              reader.sequence_pool.output_projection.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert prototype.grad is not None and prototype.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    out = ow.forward(model, x)
    F.cross_entropy(out['logits'], torch.arange(len(x)) % 6).backward()
    assert reader.core.id_backbone.cls_head.weight.grad.abs().sum() > 0
    assert model.open_world.residual_gain.grad.abs().sum() > 0
    ow.clear(model)
    assert model._open_world_last is None


def test_same_seed_new_weights_rng_preserved_and_no_duplicate_registration():
    model, _ = fixture()
    second = deepcopy(model)
    before = torch.get_rng_state().clone()
    ow.install(model, seed=2026092701)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    ow.install(second, seed=2026092701)
    for name, tensor in model.open_world.state_dict().items():
        torch.testing.assert_close(tensor, second.open_world.state_dict()[name], atol=0, rtol=0)
    state = model.state_dict()
    assert len([k for k in state if k.endswith('residual_gain')]) == 1
    assert all('encoder.open_world' not in name for name in state)
    assert model.id_backbone.encoder._open_world_original_features.__self__ is model.id_backbone.encoder


def test_relation_slot_permutation_rows_and_forward_batch_invariance():
    model, x = fixture(); ow.install(model)
    full = ow.forward(model, x)
    rows = [ow.forward(model, x[i:i+1]) for i in range(len(x))]
    for name in full:
        torch.testing.assert_close(full[name], torch.cat([r[name] for r in rows]), atol=2e-6, rtol=1e-5)
    prototypes = F.normalize(torch.randn(5, 64), dim=1)
    score = ow.relation(model, full['open64'], prototypes)
    order = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(ow.relation(model, full['open64'], prototypes[order]), score[:, order])
    reordered = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(ow.relation(model, full['open64'][reordered], prototypes), score[reordered])
    assert score.dtype == torch.float32 and score.shape == (4, 5)


def test_residual_bound_safe_constant_std_and_no_direct_reference_input():
    model, x = fixture(); ow.install(model)
    with torch.no_grad(): model.open_world.residual_gain.fill_(10)
    result = ow.forward(model, x)
    residual = (result['raw_fused160'] - result['baseline_fused160']).norm(dim=1)
    bound = .1 * result['baseline_fused160'].norm(dim=1)
    assert (residual <= bound + 1e-6).all()
    before = result['open64'].detach().clone()
    with torch.no_grad(): model.id_backbone.encoder.response.value[0].add_(3)
    after = ow.forward(model, x)
    torch.testing.assert_close(after['open64'], before, atol=0, rtol=0)
    model.zero_grad(set_to_none=True)
    out = ow.forward(model, torch.zeros_like(x))
    ow.relation(model, out['open64'], F.normalize(torch.randn(2, 64), dim=1)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.id_backbone.encoder.zero_base = True
    out = ow.forward(model, x)
    assert not out['baseline_fused160'].any() and not out['raw_fused160'].any()


def test_hook_lifetime_clear_and_failure_recovery():
    model, x = fixture(); ow.install(model)
    initial = hook_counts(model)
    first = ow.forward(model, x)
    second = ow.forward(model, x + .1)
    assert model._open_world_last['open64'] is second['open64']
    assert model._open_world_last['open64'] is not first['open64']
    assert hook_counts(model) == initial
    model.id_backbone.encoder.fail = True
    with pytest.raises(RuntimeError, match='synthetic original-feature failure'): ow.forward(model, x)
    assert model._open_world_last is None and hook_counts(model) == initial
    model.id_backbone.encoder.fail = False
    model.id_backbone.encoder.skip_signature = True
    with pytest.raises(ValueError, match='capture'): ow.forward(model, x)
    assert model._open_world_last is None and hook_counts(model) == initial
    model.id_backbone.encoder.skip_signature = False
    assert ow.forward(model, x)['logits'].shape == (4, 6)
    ow.clear(model); assert model._open_world_last is None


def test_mode_grad_boundary_and_invalid_inputs(subtests):
    model, x = fixture(); ow.install(model)
    model.train()
    assert ow.forward(model, x)['open64'].requires_grad
    model.eval()
    with torch.no_grad(): assert not ow.forward(model, x)['open64'].requires_grad
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = ow.forward(model, x)
        assert output['open64'].dtype == output['logits'].dtype == torch.float32
        assert ow.relation(model, output['open64'], output['open64'][:2]).dtype == torch.float32
    invalid = [lambda: ow.install(model), lambda: ow.install(fixture()[0], seed=True),
               lambda: ow.forward(model, x[:, :, :255]),
               lambda: ow.forward(model, torch.full_like(x, float('nan'))),
               lambda: ow.relation(model, torch.randn(4, 63), torch.randn(3, 64)),
               lambda: ow.relation(model, torch.randn(4, 64), torch.randn(3, 65)),
               lambda: ow.relation(model, torch.randn(4, 64, dtype=torch.float64), torch.randn(3, 64)),
               lambda: ow.relation(model, torch.randn(4, 64), torch.full((3, 64), float('inf')))]
    for index, action in enumerate(invalid):
        with subtests.test(index=index):
            with pytest.raises(ValueError): action()
