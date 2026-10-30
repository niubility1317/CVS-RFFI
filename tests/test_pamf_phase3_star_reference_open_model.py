"""Synthetic native contracts for complete-contribution reference conditioning."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_reference_open_model as model_api
from experiments.pamf_phase3_open_world import star_open_world_model as old


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_norm = nn.LayerNorm(160)
        self.pa_norm = nn.LayerNorm(160)
        self.gain = nn.Parameter(torch.full((160,), math.atanh(.25)))
        self.weight = nn.Parameter(torch.randn(6, 160))

    def components(self, base, pa):
        i, p = self.base_norm(base), self.pa_norm(pa)
        return i + self.gain.tanh() * p

    def classify(self, z):
        return 30 * F.linear(F.normalize(z, dim=1), F.normalize(self.weight, dim=1))


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.t2 = nn.Sequential(nn.Conv1d(2, 96, 1), nn.AdaptiveAvgPool1d(32))
        self.base = nn.Linear(96, 160)
        self.pa = nn.Linear(4, 160)
        self.cls_head = Head()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        h = self.t2(x)
        pa = self.pa(torch.cat((x.mean(-1), x.square().mean(-1)), 1))
        return self.cls_head.components(self.base(h.mean(-1)), pa), h


class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 29)
        self.register_buffer('shift', torch.zeros(29))
        self.unmask = False
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        value = self.project(torch.cat((x.mean(-1), x.square().mean(-1)), 1)) + self.shift
        return value if self.unmask else torch.cat((value[:, :27], torch.zeros_like(value[:, 27:28]), value[:, 28:]), 1)


class Signature(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(96, 160)
        self.norm = nn.LayerNorm(160)
        self.gain = nn.Parameter(torch.full((160,), math.atanh(.25)))
        self.calls = 0

    def forward(self, h):
        self.calls += 1
        return .2 * self.gain.tanh() * self.norm(self.projection(h.mean(-1)))


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module(); self.core.id_backbone = Core()
        self.response = Response()
        self.response_projection = nn.Sequential(nn.Linear(29, 64), nn.SiLU(), nn.Linear(64, 160), nn.LayerNorm(160))
        self.response_gain = nn.Parameter(torch.full((160,), math.atanh(.15)))
        self.sequence_pool = Signature()
        self.fail = self.skip_signature = self.duplicate_response = False
        self.register_buffer('source_buffer', torch.tensor([7.]))
        self.count_padding = nn.Parameter(torch.zeros(232993 - sum(p.numel() for p in self.parameters())))

    def features(self, x):
        base, h = self.core.id_backbone(x)
        if self.fail:
            raise RuntimeError('synthetic feature failure')
        r = self.response(x)
        if self.duplicate_response: self.response(x)
        value = base + self.response_gain.tanh() * self.response_projection(r)
        sig = torch.zeros_like(value) if self.skip_signature else self.sequence_pool(h)
        return value + sig

    def classify_features(self, z):
        return self.core.id_backbone.cls_head.classify(z)


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module(); self.id_backbone.encoder = Reader()
        self.unused_domain_parameters = nn.Parameter(torch.zeros(667729))

    def forward(self, x):
        return self.id_backbone.encoder.classify_features(self.id_backbone.encoder.features(x))


def fixture():
    torch.manual_seed(11)
    return Native().eval(), torch.randn(4, 2, 256)


def hooks(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


def test_same_native_paths_zero_initial_identity_complete_contributions_and_counts():
    model, x = fixture(); reader = model.id_backbone.encoder; head = reader.core.id_backbone.cls_head
    parameters = dict(model.named_parameters()); buffers = {n: v.clone() for n, v in model.named_buffers()}
    logits = model(x); counts = (reader.core.id_backbone.calls, reader.response.calls, reader.sequence_pool.calls)
    assert model_api.install(model, use_reference=True) is model
    assert isinstance(model.open_world, old.OpenParts)
    out = model_api.forward(model, x)
    assert model.id_backbone.encoder is reader and reader.core.id_backbone.cls_head is head
    torch.testing.assert_close(out['logits'], logits, atol=0, rtol=0)
    torch.testing.assert_close(out['raw_fused160'], out['baseline_fused160'], atol=0, rtol=0)
    c = out['contributions']
    assert c.shape == (4, 4, 160) and out['reference29'].shape == (4, 29)
    torch.testing.assert_close(((c[:, 0] + c[:, 1]) + c[:, 2]) + c[:, 3], out['baseline_fused160'], atol=0, rtol=0)
    torch.testing.assert_close(out['conditioned_contributions'], c, atol=0, rtol=0)
    assert not out['branch_deltas'].any() and not out['reference29'][:, 27].any()
    assert (reader.core.id_backbone.calls, reader.response.calls, reader.sequence_pool.calls) == tuple(v+1 for v in counts)
    for n, p in parameters.items(): assert dict(model.named_parameters())[n] is p
    for n, b in buffers.items(): torch.testing.assert_close(dict(model.named_buffers())[n], b, atol=0, rtol=0)
    report = model_api.parameter_report(model)
    expected = dict(native_identity_parameters=232993, open_projector_parameters=45312,
        identity_residual_parameters=10560, relation_parameters=4161, reference_conditioner_parameters=22080,
        open_world_parameters=82113, active_parameters=315106, native_container_parameters=900722,
        training_container_parameters=982835, unused_native_parameters=667729,
        expected_open_world_parameters=82113, expected_active_parameters=315106)
    for k, v in expected.items(): assert report[k] == v
    assert sum(p.numel() for p in model.open_world.parameters() if p.requires_grad) == 82113
    assert report['trainable_open_world_parameters'] == 82113
    assert report['trainable_identity_parameters'] == 232993
    assert len([k for k in model.state_dict() if k.endswith('residual_gain')]) == 1
    assert not any('encoder.open_world' in k for k in model.state_dict())


def test_all_actual_contribution_gradients_and_conditioner_last_layer_are_live():
    model, x = fixture(); model_api.install(model, True); reader = model.id_backbone.encoder
    out = model_api.forward(model, x)
    for key in ('contributions', 'reference29'): out[key].retain_grad()
    proto = F.normalize(torch.randn(3, 64), dim=1).requires_grad_()
    model_api.relation(model, out['open64'], proto).square().mean().backward()
    assert out['contributions'].grad is not None
    assert (out['contributions'].grad.abs().sum((0, 2)) > 0).all()
    live = (reader.core.id_backbone.base.weight, reader.core.id_backbone.pa.weight,
        reader.core.id_backbone.cls_head.base_norm.weight, reader.core.id_backbone.cls_head.pa_norm.weight,
        reader.core.id_backbone.cls_head.gain, reader.response_projection[0].weight,
        reader.response_gain, reader.sequence_pool.projection.weight, reader.sequence_pool.gain,
        model.open_world.conditioner[-1].weight, model.open_world.conditioner[-1].bias)
    for p in live: assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert proto.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model_api.forward(model, x)['logits'], torch.arange(4)).backward()
    assert reader.core.id_backbone.cls_head.weight.grad.abs().sum() > 0
    assert model.open_world.residual_gain.grad.abs().sum() > 0


def test_matched_c0_c1_parameters_information_and_constant_residual_learning():
    c0, x = fixture(); c1 = deepcopy(c0); rng = torch.get_rng_state().clone()
    model_api.install(c0, False); model_api.install(c1, True)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for n, p in c0.open_world.named_parameters():
        torch.testing.assert_close(p, dict(c1.open_world.named_parameters())[n], atol=0, rtol=0)
        assert p.requires_grad and dict(c1.open_world.named_parameters())[n].requires_grad
    torch.testing.assert_close(model_api.forward(c0, x)['open64'], model_api.forward(c1, x)['open64'], atol=0, rtol=0)
    for m in (c0, c1):
        with torch.no_grad():
            m.id_backbone.encoder.response_gain.zero_()
            m.open_world.conditioner[-1].weight.normal_(0, .2)
    c1.open_world.load_state_dict(c0.open_world.state_dict())
    before0 = model_api.forward(c0, x)['open64'].detach().clone()
    before1 = model_api.forward(c1, x)['open64'].detach().clone()
    for m in (c0, c1):
        with torch.no_grad(): m.id_backbone.encoder.response.shift[0] += 3
    torch.testing.assert_close(model_api.forward(c0, x)['open64'], before0, atol=0, rtol=0)
    assert not torch.equal(model_api.forward(c1, x)['open64'], before1)
    c0.zero_grad(set_to_none=True)
    out = model_api.forward(c0, x)
    model_api.relation(c0, out['open64'], F.normalize(torch.randn(2,64),dim=1)).sum().backward()
    assert c0.open_world.conditioner[-1].bias.grad.abs().sum() > 0
    assert torch.isfinite(c0.open_world.conditioner[-1].bias.grad).all()
    assert not c0.open_world.conditioner[0].weight.grad.any()
    c1.zero_grad(set_to_none=True)
    out = model_api.forward(c1, x)
    model_api.relation(c1, out['open64'], F.normalize(torch.randn(2,64),dim=1)).sum().backward()
    assert c1.open_world.conditioner[0].weight.grad.abs().sum() > 0
    assert not c1.open_world.conditioner[0].weight.grad[:,27].any()


def test_branch_and_identity_residual_bounds_and_zero_contribution():
    model, x = fixture(); model_api.install(model, True)
    with torch.no_grad():
        model.open_world.conditioner[-1].weight.fill_(10)
        model.open_world.conditioner[-1].bias.fill_(20)
        model.open_world.residual_gain.fill_(10)
        model.id_backbone.encoder.response_gain.zero_()
    out = model_api.forward(model, x)
    assert (out['branch_deltas'].norm(dim=-1) <= .1*out['contributions'].norm(dim=-1)+2e-6).all()
    assert not out['contributions'][:, 2].any() and not out['branch_deltas'][:, 2].any()
    assert (out['raw_fused160']-out['baseline_fused160']).norm(dim=1).le(.1*out['baseline_fused160'].norm(dim=1)+2e-6).all()
    out['open64'].square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_per_record_mode_autocast_relation_permutation_and_transient_clear():
    model, x = fixture(); model_api.install(model, True); model.train()
    with torch.no_grad():
        model.open_world.conditioner[2].weight.normal_(0, .05)
        model.open_world.conditioner[2].bias.normal_(0, .05)
    out = model_api.forward(model, x); assert out['open64'].requires_grad
    model.eval()
    rows = [model_api.forward(model, x[i:i+1]) for i in range(len(x))]
    for k in out:
        torch.testing.assert_close(out[k], torch.cat([row[k] for row in rows]), atol=3e-6, rtol=2e-5)
    proto = F.normalize(torch.randn(3,64), dim=1); order = torch.tensor([2,0,1])
    scores = model_api.relation(model, out['open64'], proto)
    torch.testing.assert_close(model_api.relation(model, out['open64'], proto[order]), scores[:,order])
    with torch.no_grad(): assert not model_api.forward(model, x)['open64'].requires_grad
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        auto = model_api.forward(model, x)
        assert all(t.dtype == torch.float32 for t in auto.values())
        assert model_api.relation(model, auto['open64'], proto).dtype == torch.float32
    model_api.clear(model); assert model._open_world_last is None


def test_call_local_hooks_errors_missing_duplicate_and_cfo_mask(subtests):
    model, x = fixture(); model_api.install(model, True); reader = model.id_backbone.encoder; initial = hooks(model)
    for flag in ('fail','skip_signature','duplicate_response','unmask'):
        owner = reader.response if flag == 'unmask' else reader
        setattr(owner, flag, True)
        with subtests.test(flag=flag), pytest.raises((ValueError, RuntimeError)):
            model_api.forward(model, x)
        assert model._open_world_last is None and hooks(model) == initial
        setattr(owner, flag, False)
    for invalid in (x[:, :, :255], x.double(), torch.full_like(x, float('nan'))):
        with subtests.test(input_shape=tuple(invalid.shape), dtype=str(invalid.dtype)), pytest.raises(ValueError):
            model_api.forward(model, invalid)
        assert hooks(model) == initial and model._open_world_last is None
    assert model_api.forward(model, x)['logits'].shape == (4,6)
    with pytest.raises(ValueError): model_api.install(model, True)
    with pytest.raises(ValueError): model_api.install(fixture()[0], 1)
    with pytest.raises(ValueError): model_api.install(fixture()[0], False, seed=True)


def test_captures_use_original_return_as_base_even_when_reconstruction_rounds():
    model, x = fixture(); reader = model.id_backbone.encoder; original = reader.features
    def returned_with_roundoff(value):
        return original(value) + 1e-7
    reader.features = returned_with_roundoff
    baseline = reader.features(x); logits = reader.classify_features(baseline)
    model_api.install(model, False); out = model_api.forward(model, x)
    torch.testing.assert_close(out['baseline_fused160'], baseline, atol=0, rtol=0)
    torch.testing.assert_close(out['logits'], logits, atol=0, rtol=0)
