"""Actual-model higher-order CE subsystem and conditional response contracts."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cvsrffi.game_tracking.field import (audit_local_field, build_local_ce_field,
    fishr_proxy_higher_order, apply_response_tracking, dynamic_stopgrad_gradient,
    build_core90_field, audit_core90_field)
from cvsrffi.game_tracking.legacy.model_dual_cvsincnet import grad_reverse
from cvsrffi.game_tracking.source_audit import isolated_rng


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class ToyModel(nn.Module):
    def __init__(self, nonlinear=True):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.fuse = nn.Sequential(nn.Linear(2, 2), nn.Tanh())
        self.id_backbone.norm = nn.BatchNorm1d(2)
        self.classifier = nn.Linear(2, 2)
        self.adv_head = (nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2))
                         if nonlinear else nn.Linear(2, 2))
        self.unused = nn.Parameter(torch.tensor(2.))

    def forward(self, x, return_aux=False):
        z = self.id_backbone.fuse(self.id_backbone.norm(x))
        tx = self.classifier(z)
        return dict(tx_logits=tx, z_id=z, adv_dom_logits=self.adv_head(Reverse.apply(z))) if return_aux else tx


def snapshot(model):
    return deepcopy(model.state_dict()), {name: module.training for name, module in model.named_modules()}, {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}


def assert_unchanged(model, before, exceptions=()):
    state, training, gradients = before
    for key, value in model.state_dict().items():
        if key not in exceptions:
            assert torch.equal(value, state[key]), key
    assert training == {name: m.training for name, m in model.named_modules()}
    for name, p in model.named_parameters():
        assert p.grad is None if gradients[name] is None else torch.equal(p.grad, gradients[name])


def test_actual_local_field_signed_derivatives_and_jvp_finite_difference():
    torch.manual_seed(1)
    model = ToyModel().double().eval()
    x = torch.randn(8, 2, dtype=torch.float64)
    y, domain = torch.arange(8) % 2, torch.arange(8) // 4
    field, point, names, count = build_local_ce_field(model, x, y, domain, .4)
    point = point.requires_grad_()
    value = field(point)
    out = model(x, return_aux=True)
    selected = dict(model.named_parameters())
    tx, rx = F.cross_entropy(out['tx_logits'], y), F.cross_entropy(model.adv_head(out['z_id']), domain)
    enc = torch.autograd.grad(tx - .4 * rx, [selected[n] for n in names[:2]], retain_graph=True)
    head = torch.autograd.grad(.4 * rx, [selected[n] for n in names[2:]])
    assert torch.allclose(value, torch.cat([g.flatten() for g in enc + head]), atol=1e-12)
    direction = torch.randn_like(point)
    direction /= direction.norm()
    _, jvp = torch.autograd.functional.jvp(field, point, direction)
    fd = (field(point + 1e-5 * direction) - field(point - 1e-5 * direction)) / 2e-5
    assert torch.allclose(jvp, fd, rtol=1e-6, atol=1e-8)
    assert count['field_evaluations'] >= 4


def test_local_audit_preserves_model_buffers_gradients_flags_and_rng():
    torch.manual_seed(2)
    model = ToyModel().double().train()
    model.classifier.eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    x = torch.randn(8, 2, dtype=torch.float64)
    labels = torch.arange(8) % 2
    before = snapshot(model)
    rng, py, np_before = torch.get_rng_state(), random.getstate(), np.random.get_state()
    result = audit_local_field(model, x, labels, labels, .35, seed=12)
    assert result['valid'], result
    assert result['model_forwards'] == result['field_evaluations'] == 2
    assert result['scope'].endswith('CE_only') and 'FISHR' in result['excludes']
    assert_unchanged(model, before)
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py
    assert np.array_equal(np_before[1], np.random.get_state()[1])


def test_unsupported_actual_model_returns_precise_invalid_without_discard():
    class Unsupported(ToyModel):
        def forward(self, *args, **kwargs):
            raise NotImplementedError('custom IQ operator has no higher derivative')
    result = audit_local_field(Unsupported(), torch.zeros(4, 2), torch.zeros(4, dtype=torch.long),
                               torch.zeros(4, dtype=torch.long), .4)
    assert not result['valid'] and result['error_type'] == 'NotImplementedError'
    assert 'custom IQ operator' in result['error']


def legacy_fishr(logits, y, d):
    proxy = logits.float().softmax(1) - F.one_hot(y, logits.size(1)).to(logits)
    values = torch.stack([proxy[d == domain].var(0, unbiased=False) for domain in d.unique()])
    return (values - values.mean(0, keepdim=True).detach()).square().mean()


def test_fishr_value_and_first_derivative_match_legacy():
    torch.manual_seed(6)
    logits = torch.randn(12, 3, requires_grad=True)
    y, d = torch.arange(12) % 3, torch.arange(12) // 4
    legacy = legacy_fishr(logits, y, d)
    higher = fishr_proxy_higher_order(logits, y, d)
    assert torch.allclose(legacy, higher, atol=1e-10)
    assert torch.allclose(torch.autograd.grad(legacy, logits)[0], torch.autograd.grad(higher, logits)[0], atol=1e-9)


def test_fishr_second_derivative_matches_gradient_finite_difference():
    torch.manual_seed(9)
    point = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    y, d = torch.arange(12) % 3, torch.arange(12) // 4
    direction = torch.randn_like(point)
    def field(z):
        return torch.autograd.grad(fishr_proxy_higher_order(z, y, d), z, create_graph=True)[0]
    _, jvp = torch.autograd.functional.jvp(field, point, direction)
    fd = (field(point + 1e-5 * direction) - field(point - 1e-5 * direction)) / 2e-5
    assert torch.allclose(jvp, fd, rtol=1e-6, atol=1e-9)


def response_fixture():
    model = ToyModel(nonlinear=False).double().eval()
    with torch.no_grad():
        model.id_backbone.fuse[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        model.id_backbone.fuse[0].bias.zero_()
        model.adv_head.weight.copy_(torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64))
        model.adv_head.bias.zero_()
    x = torch.tensor([[2., -2.], [-2., 2.]], dtype=torch.float64).repeat(4, 1)
    y = torch.arange(8) % 2
    current = deepcopy(model)
    with torch.no_grad():
        current.id_backbone.fuse[0].weight.mul_(.85)
    return model, current, x, y


def test_response_uses_actual_displacement_and_only_final_layer_commit():
    reference, current, x, y = response_fixture()
    before, reference_before = snapshot(current), snapshot(reference)
    rng = torch.get_rng_state()
    result = apply_response_tracking(reference, current, x, y, monitor_x=x * .9, monitor_domain=y,
                                      damping=.1, max_iterations=10)
    assert result['accepted'], result
    assert result['monitor_ce_after'] <= result['monitor_ce_before']
    assert result['encoder_displacement_norm'] > 0.
    assert result['hvp_iterations'] > 0 and result['model_forwards'] == 2
    assert not result['optimizer_moments_updated']
    assert_unchanged(current, before, ('adv_head.weight', 'adv_head.bias'))
    assert_unchanged(reference, reference_before)
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(current.adv_head.weight, before[0]['adv_head.weight'])


def test_response_monitor_rejection_preserves_live_model_and_bad_delta_invalid():
    reference, current, x, y = response_fixture()
    before = snapshot(current)
    rejected = apply_response_tracking(reference, current, x, y, monitor_x=x, monitor_domain=1 - y)
    assert not rejected['accepted'] and rejected['reason'] == 'source_monitor_rejected'
    assert_unchanged(current, before)
    invalid = apply_response_tracking(reference, current, x, y,
        delta={'id_backbone.fuse.0.weight': torch.zeros_like(current.id_backbone.fuse[0].weight)},
        monitor_x=x, monitor_domain=y)
    assert not invalid['valid'] and 'actual mainstep' in invalid['error']
    assert_unchanged(current, before)


def test_response_spd_and_logit_linear_contract_fail_closed():
    reference, current, x, y = response_fixture()
    result = apply_response_tracking(reference, current, x, y, monitor_x=x, monitor_domain=y, damping=0)
    assert not result['valid'] and 'positive' in result['error']
    reference.adv_head = nn.Sequential(reference.adv_head, nn.Tanh())
    current.adv_head = nn.Sequential(current.adv_head, nn.Tanh())
    result = apply_response_tracking(reference, current, x, y, monitor_x=x, monitor_domain=y)
    assert not result['valid'] and 'convex conditional-head' in result['error']


@pytest.mark.parametrize('use_no_grad', [False, True])
def test_dynamic_target_response_preserves_first_field_and_correct_jacobian(use_no_grad):
    def objective(w):
        if use_no_grad:
            with torch.no_grad():
                target = w.square()
        else:
            target = w.square().detach()
        return ((w - target)**2).sum()
    point = torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True)
    expected = 2*(point-point.square())
    actual = dynamic_stopgrad_gradient(objective, point)
    torch.testing.assert_close(actual, expected)
    direction = torch.tensor([.4, -.5], dtype=torch.float64)
    _, jvp = torch.autograd.functional.jvp(lambda w: dynamic_stopgrad_gradient(objective, w), point, direction)
    torch.testing.assert_close(jvp, 2*(1-2*point)*direction)
    # A plain Hessian of the detached surrogate would incorrectly return 2v.
    assert not torch.allclose(jvp, 2*direction)
    assert torch.is_grad_enabled()


def test_dynamic_target_no_grad_random_draws_replay_same_problem():
    def objective(w):
        with torch.no_grad():
            target = w.square() + torch.rand_like(w)
        multiplier = torch.rand_like(w)
        return ((w-target).square()*multiplier).sum()
    point = torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True)
    with isolated_rng(18):
        expected = torch.autograd.grad(objective(point), point)[0]
    result = dynamic_stopgrad_gradient(objective, point, seed=18)
    torch.testing.assert_close(result, expected)


class FullObjectiveToy(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.fuse = nn.Sequential(nn.Linear(3, 4), nn.Tanh())
        self.id_backbone.norm = nn.BatchNorm1d(3)
        self.classifier = nn.Linear(4, 3)
        self.domain_branch = nn.Linear(3, 4)
        self.dom_head = nn.Linear(4, 4)
        self.adv_head = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Dropout(dropout), nn.Linear(5, 4))

    def forward(self, x, y_tx=None, grl_lambda=1., return_aux=True, domain_labels=None):
        z = self.id_backbone.fuse(self.id_backbone.norm(x))
        dom = (self.domain_branch(x)+.2*z).tanh()
        return dict(tx_logits=self.classifier(z), dom_logits=self.dom_head(dom),
                    adv_dom_logits=self.adv_head(grad_reverse(z,grl_lambda)), z_id=z, z_dom=dom)


def full_objective_fixture(*, epoch=150, dropout=0., pseudo=True, scale='legacy_weighted'):
    from cvsrffi.game_tracking.config import parse_args
    from cvsrffi.game_tracking.legacy.options import _loss_weights
    from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
    from cvsrffi.game_tracking.step_context import StepContext
    torch.manual_seed(92)
    args = parse_args(['--output_dir', 'unused-test-output'])
    args.game_head_scale = scale
    args.num_classes = 3
    args.fishr_min_domains = 4
    args.ow_feat_min_classes = 2
    args.proto_min_count = 1
    args.proxy_unknown_virtual_count = 4
    args.soft_unknown_mixup_count = 4
    args.source_episode_min_domains = 2
    # Ensure every stage-gated auxiliary is active in the complete-objective
    # parity test, even if its historical configuration had a zero coefficient.
    args.lambda_open_world_feat = .02
    args.lambda_zid_compact = .03
    args.lambda_proxy_unknown = .04
    args.lambda_soft_unknown_mixup = .05
    args.lambda_source_episode = .06
    args.lambda_proto = .07
    weights = _loss_weights(args, dict(adv_scale=1., dom_scale=1., orth_scale=1., cons_scale=1., group_ce_scale=1.))
    model = FullObjectiveToy(dropout=dropout).double().train()
    x = torch.randn(24,3,dtype=torch.float64)
    labels = torch.arange(24) % 3
    domain = (torch.arange(24)//3) % 4
    ctx = StepContext(x, x*.9+.02, labels, domain, weights, epoch, 3,
                      [str(i) for i in range(24)], 'leo_clear_weak', 24)
    if pseudo:
        ctx.strong = x[:9]*.85
        ctx.pseudo = labels[:9]
        ctx.base_mask = torch.ones(9,dtype=torch.bool)
        ctx.strong_mask = torch.tensor([True,False,True]*3)
    proto = PrototypeMemoryBank(3,4,min_count=1)
    with torch.no_grad():
        proto.update(model(x)['z_id'],labels,domain)
    return model,ctx,args,proto


@pytest.mark.parametrize('scale', ['legacy_weighted', 'separate_head_scale'])
def test_full_core90_field_matches_actual_grl_first_update_all_aux_fishr_and_dropout(scale):
    from cvsrffi.game_tracking.step_context import Core90Objective
    model, ctx, args, proto = full_objective_fixture(dropout=.2, scale=scale)
    before = snapshot(model)
    field, point, names, counters, metadata = build_core90_field(model, ctx, args, proto, seed=71)
    value = field(point.requires_grad_())
    native = deepcopy(model)
    with isolated_rng(71):
        loss = Core90Objective(native,args,deepcopy(proto))(deepcopy(ctx))
        selected = dict(native.named_parameters())
        expected = torch.autograd.grad(loss,[selected[name] for name in names])
    torch.testing.assert_close(value, torch.cat([g.flatten() for g in expected]), atol=2e-6, rtol=2e-5)
    assert {'fishr','proto','open_world_feat','zid_compact','proxy_unknown','soft_unknown_mixup','source_episode','sat_cls','unlabeled_ce','unlabeled_entropy'} <= set(metadata['loss_terms'])
    assert metadata['effective_weights']['fishr'] > 0 and counters['stop_gradient_targets'] > 0
    assert_unchanged(model,before)
    assert ctx.origin_features is None and ctx.forward_calls == 0


def test_full_core90_field_jvp_agrees_with_first_field_finite_difference():
    model, ctx, args, proto = full_objective_fixture()
    field, point, names, counters, metadata = build_core90_field(model,ctx,args,proto,seed=73)
    point.requires_grad_()
    torch.manual_seed(7)
    direction = torch.randn_like(point); direction /= direction.norm()
    _, jvp = torch.autograd.functional.jvp(field,point,direction)
    delta = 2e-3
    fd = (field(point+delta*direction)-field(point-delta*direction))/(2*delta)
    relative = float(((fd-jvp).norm()/jvp.norm()).detach())
    assert relative < .03, relative


def test_full_field_audit_fails_closed_for_unsupported_aux_operator(monkeypatch):
    import cvsrffi.game_tracking.legacy.objective as objective
    model,ctx,args,proto = full_objective_fixture()
    def unsupported(*a,**kw):
        raise NotImplementedError('active source episode third derivative unavailable')
    monkeypatch.setattr(objective,'source_episode_three_sigma_loss',unsupported)
    result = audit_core90_field(model,ctx,args,proto)
    assert not result['valid'] and result['reason'] == 'full_CORE90_higher_order_audit_failed'
    assert 'source episode' in result['error'] and not result['excluded_loss_terms']


def test_real_frozen_legacy_backbone_full_field_audit():
    from cvsrffi.game_tracking.runtime import build_model
    from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
    _,ctx,args,_ = full_objective_fixture(pseudo=False)
    args.device = 'cpu'
    args.game_synthetic = True
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        model = build_model(args,4,torch.device('cpu')).train()
        ctx.x = torch.randn(24,2,256)
        ctx.satellite = ctx.x*.95
        proto = PrototypeMemoryBank(3,4,min_count=1)
        with torch.no_grad():
            proto.update(model(ctx.x,return_aux=True)['z_id'],ctx.y,ctx.domain)
        result = audit_core90_field(model,ctx,args,proto,seed=73)
    finally:
        torch.set_num_threads(threads)
    assert result['valid'],{k:result.get(k) for k in ('reason','error','finite_difference_relative_error','jvp_norm','vjp_norm')}
    assert 'fishr' in result['active_weight_terms']
    assert result['excluded_loss_terms'] == []
    assert 4 <= result['field_evaluations'] <= 8
    assert result['finite_difference_trials'][-1]['relu_region_changes'] == 0
    assert result['finite_difference_trials'][-1]['relative_error'] <= result['finite_difference_tolerance']
    if len(result['finite_difference_trials']) > 1:
        assert result['finite_difference_trials'][0]['relu_region_changes'] > 0
