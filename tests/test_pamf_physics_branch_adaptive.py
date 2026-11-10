from contextlib import nullcontext
import pytest
import torch
from torch.nn import functional as F

from test_pamf_physics_branch_objective import TinyStudent
from experiments.pamf_physics_branch.objective import PhysicalPL, install_labeled as install_fixed
from experiments.pamf_physics_branch_adaptive import objective as adaptive_module
from experiments.pamf_physics_branch_adaptive.objective import AdaptivePL, gradient_budget, install_labeled
from experiments.pamf_physics_branch_adaptive.plan import RUN, candidate_rows
from experiments.pamf_physics_branch_adaptive.source import validate_source


def test_gradient_budget_is_detached_bounded_and_preserves_backward():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    base, relation = (2 * p).sum(), (8 * p).sum()
    scale, base_norm, relation_norm = gradient_budget(base, relation, [p])
    assert scale == pytest.approx(.25) and base_norm > 0 and relation_norm > 0 and p.grad is None
    (base + scale * relation).backward()
    assert torch.allclose(p.grad, torch.full_like(p, 4.))
    p.grad = None
    assert gradient_budget((p * 0).sum(), (p * 8).sum(), [p])[0] == .1
    assert gradient_budget((p * 2).sum(), (p * 0).sum(), [p])[0] == 1.
    with pytest.raises(FloatingPointError):
        gradient_budget((p * float('nan')).sum(), p.sum(), [p])


def test_actual_l_u_hooks_consume_one_budget_without_truth():
    torch.manual_seed(8)
    model = TinyStudent().train()
    c = {'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.},
         'source_rxs': [1, 2], 'source_days': [1],
         'physical_relation': {'beta': .5, 'l_weight': .05, 'u_weight': .05}}
    objective = AdaptivePL(c)
    holder = {'objective': objective}
    def batch(n, hidden):
        i = torch.arange(n)
        y = i.remainder(4).div(2, rounding_mode='floor')
        rx = i.remainder(2) + 1
        x = torch.randn(n, 2, 256) * .05
        x[:, 0, 0] = torch.where(y == 0, 1., -1.)
        meta = {'rx_i': rx, 'day_i': torch.ones(n, dtype=torch.long), 'branch_record_id': i,
                'tx_label_visible': torch.full((n,), not hidden, dtype=torch.bool)}
        return x, torch.full_like(y, -1) if hidden else y, torch.zeros(n, dtype=torch.long), meta
    x_u, y_u, domain_u, meta_u = batch(8, True)
    model._cd_U_iter = iter([(x_u, y_u, domain_u, meta_u)])
    model._phase1_labeled_feature_context = lambda: nullcontext()
    def original_hook(**kwargs):
        result = objective.loss(model, 'cpu')
        model._cd_pending = None
        return result, {}
    model._source_labeled_feature_hook = original_hook
    install_labeled(model, c, holder)
    x, y, domains, meta = batch(64, False)
    model._cd_pending = {'x': x}
    with model._phase1_labeled_feature_context():
        logits = model(x)
    identity = F.cross_entropy(logits, y)
    value, metrics = model._source_labeled_feature_hook(labels=y, extra=(None, meta), identity_loss=identity)
    (identity + value).backward()
    state = objective.state_dict()
    assert state['calls'] == state['adaptive_reference']['calls'] == 1
    assert .1 <= state['adaptive_reference']['minimum'] <= 1.
    assert state['physical_U']['selected_samples'] == 8 and state['U_truth_read'] is False
    assert metrics['physical_L_valid_anchors'] > 0 and objective.metrics['physical_U_valid_anchors'] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_adaptive_source_guard_rejects_wrong_contract_or_budget():
    row = candidate_rows()[0]
    c = dict(physical_row=row, run_id=RUN, row_id=row['row_id'], model_seed=row['model_seed'],
             source_rxs=row['source_rxs'], source_days=row['source_days'], num_classes=6,
             epochs=50, optimizer_updates=11100, ablation='F21', U_use=True, own_EMA=False,
             native_U_gate=False, posthoc_head_fit=False,
             paper_row=dict(structure='full', permuted=False, signature_views=3),
             hard_pl=dict(method='HPL2', threshold=.95, weight=1.), hard_pl_label_smoothing=0.,
             physical_relation=dict(beta=.5, l_weight=.05, u_weight=.05),
             adaptive_reference_budget=dict(floor=.1, eps=1e-12), algorithm_id='pamf_ptbr_adaptive_v1',
             output_root='/project/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    assert validate_source(c) == row
    with pytest.raises(ValueError):
        validate_source(dict(c, adaptive_reference_budget=dict(floor=0., eps=1e-12)))
    with pytest.raises(ValueError, match='target'):
        validate_source(dict(c, target_truth='/forbidden'))


def test_adaptive_u_empty_acceptance_is_finite_and_consumed():
    model = TinyStudent().train()
    c = {'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.},
         'source_rxs': [1, 2], 'source_days': [1],
         'physical_relation': {'beta': .5, 'l_weight': .05, 'u_weight': .05}}
    objective = AdaptivePL(c)
    meta = {'rx_i': torch.tensor([1, 2, 1, 2]), 'day_i': torch.ones(4, dtype=torch.long),
            'branch_record_id': torch.arange(4), 'tx_label_visible': torch.zeros(4, dtype=torch.bool)}
    model._cd_U_iter = iter([(torch.zeros(4, 2, 256), torch.full((4,), -1),
                              torch.zeros(4, dtype=torch.long), meta)])
    base = objective.loss(model, 'cpu')
    with pytest.raises(RuntimeError, match='consume'):
        objective.state_dict()
    ref, objective._adaptive_ref = objective._adaptive_ref, None
    scale, direct, rel = gradient_budget(base, ref, model.id_backbone.encoder.response_projection.parameters())
    objective.record_budget(scale, direct, rel)
    (base + scale * ref).backward()
    state = objective.state_dict()
    assert state['accepted'] == 0 and objective.metrics['physical_U_valid_anchors'] == 0
    assert scale == 1. and torch.isfinite(base)


def test_unit_scale_matches_original_ptbr_loss_and_parameter_gradients(monkeypatch):
    torch.manual_seed(11)
    initial = TinyStudent().state_dict()
    c = {'hard_pl': {'method': 'HPL2', 'threshold': .95, 'weight': 1.},
         'source_rxs': [1, 2], 'source_days': [1],
         'physical_relation': {'beta': .5, 'l_weight': .05, 'u_weight': .05}}
    def batch(n, hidden):
        i = torch.arange(n)
        y = i.remainder(4).div(2, rounding_mode='floor')
        x = torch.randn(n, 2, 256) * .05
        x[:, 0, 0] = torch.where(y == 0, 1., -1.)
        meta = {'rx_i': i.remainder(2) + 1, 'day_i': torch.ones(n, dtype=torch.long),
                'branch_record_id': i, 'tx_label_visible': torch.full((n,), not hidden, dtype=torch.bool)}
        return x, torch.full_like(y, -1) if hidden else y, torch.zeros(n, dtype=torch.long), meta
    labeled, unlabeled = batch(64, False), batch(8, True)
    monkeypatch.setattr(adaptive_module, 'gradient_budget', lambda *args, **kwargs: (1., 1., 1.))
    def run(adaptive):
        model = TinyStudent().train()
        model.load_state_dict(initial)
        objective = AdaptivePL(c) if adaptive else PhysicalPL(c)
        model._cd_U_iter = iter([unlabeled])
        model._phase1_labeled_feature_context = lambda: nullcontext()
        def original_hook(**kwargs):
            result = objective.loss(model, 'cpu')
            model._cd_pending = None
            return result, {}
        model._source_labeled_feature_hook = original_hook
        if adaptive:
            install_labeled(model, c, {'objective': objective})
        else:
            install_fixed(model, c)
        x, y, domains, meta = labeled
        model._cd_pending = {'x': x}
        with model._phase1_labeled_feature_context():
            logits = model(x)
        identity = F.cross_entropy(logits, y)
        correction, _ = model._source_labeled_feature_hook(labels=y, extra=(None, meta), identity_loss=identity)
        loss = identity + correction
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters()]
    old_loss, old_grads = run(False)
    new_loss, new_grads = run(True)
    assert torch.allclose(old_loss, new_loss, atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(old_grads, new_grads))
