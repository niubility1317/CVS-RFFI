"""Finite-risk algebra and paired radial-policy checks on synthetic source IQ."""
import argparse
import json
import math

import torch
from torch import nn

from .objectives import M_REQ, keep_loss, keep_per_sample, lt_loss, lt_per_sample
from .actions import (ACTION_ORDER, KINDS, action_contract, apply_proposal, fit_actions,
                      load_actions, _proposal)
from experiments.cvs_stage2_decouple import actions as old
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(512, 6)
        self.dropout = nn.Dropout(.4)

    def forward(self, x):
        return self.projection(self.dropout(x.flatten(1)))


def objective_checks():
    assert M_REQ == math.log(95.)
    clean = torch.tensor([[20., 0., 0.], [2., 0., 0.], [-1., 0., 0.]], requires_grad=True)
    y = torch.zeros(3, dtype=torch.long)
    equal = clean.detach().clone().requires_grad_()
    zero = lt_loss(clean, equal, y)
    assert zero.requires_grad and float(zero.detach()) == 0
    gc, ga = torch.autograd.grad(zero, (clean, equal), allow_unused=True)
    assert gc is None and ga is not None and not ga.any()
    augmented = torch.tensor([[5., 0., 0.], [1., 0., 0.], [-2., 0., 0.]], requires_grad=True)
    loss = lt_per_sample(clean, augmented, y)
    assert loss[0] == 0
    assert torch.allclose(loss[1:], torch.full((2,), .5 / M_REQ**2))
    gc, ga = torch.autograd.grad(loss.mean(), (clean, augmented), allow_unused=True)
    assert gc is None and ga is not None and ga[1:].abs().sum() > 0 and not ga[0].any()
    parent = torch.tensor([20., 2., -1.], requires_grad=True)
    student = augmented.detach().clone().requires_grad_()
    keep = keep_per_sample(student, parent, y)
    assert keep[0] == 0 and keep[2] == 0 and torch.allclose(keep[1], torch.tensor(.5 / M_REQ**2))
    gp, gs = torch.autograd.grad(keep_loss(student, parent, y), (parent, student), allow_unused=True)
    assert gp is None and gs is not None and not gs[2].any()
    # Even the exact same Tensor gives zero derivative with the stopped target.
    tied = torch.tensor([[M_REQ, 0., 0.]], requires_grad=True)
    zero_tied = lt_loss(tied, tied, torch.zeros(1, dtype=torch.long))
    assert float(zero_tied.detach()) == 0 and not torch.autograd.grad(zero_tied, tied)[0].any()
    return dict(exact_one_sided_square=True, log95_margin=True, clean_and_parent_reference_detached=True,
                zero_action_loss_and_gradient_zero=True, task_complete_stops_pressure=True,
                parent_wrong_no_keep=True, no_augmented_ce=True)


def action_checks(device='cpu'):
    torch.manual_seed(619)
    parent = Parent().to(device).train()
    parent.projection.bias.requires_grad_(False)
    original = ReadOnlyState(parent)
    x = torch.randn(24, 2, 256)
    with ReadOnlyState(parent):
        parent.eval()
        with torch.no_grad():
            logits = parent(x.to(device))
            y = logits.argmax(1).cpu()
            margins = old.logit_margin(logits, y.to(device)).cpu()
    cache = dict(role='L_s', x=x, y=y, state=old.signal_state(x), rx=torch.arange(24) % 2,
                 ids=['source-' + str(i) for i in range(24)], parent_margin=margins)
    payload = fit_actions(parent, cache, 623, device, passes=2, batch_size=16)
    assert all(payload['diagnostics']['parent_state_invariance'].values())
    assert all(torch.equal(p, s) and p.requires_grad == f for p, s, f, _ in original.parameters)
    assert all(m.training == f for m, f in original.modes)
    loaded = load_actions(payload, device)
    expected = action_contract()
    assert expected['conditional_fraction'] == .5 and tuple(expected['schedule']['order']) == ACTION_ORDER
    for kind in KINDS:
        batch = x.to(device)
        first = loaded.propose(batch, kind, 719, 'coverage')
        second = loaded.propose(batch, kind, 719, 'conditional')
        assert torch.equal(first['radius'], second['radius'])
        assert torch.equal(first['normalized_parameters'].abs().amax(1), first['radius'])
        assert torch.equal(second['normalized_parameters'].abs().amax(1), second['radius'])
        if kind == 'temporal':
            assert torch.equal(first['noise'], second['noise'])
            assert (second['parameters'][:, 2] >= 0).all() and not second['parameters'][:, 3].any()
        assert (second['normalized_parameters'].abs() <= 1).all()
        same_cover = ~second['conditional_mask']
        assert same_cover.any() and second['conditional_mask'].any()
        assert torch.equal(first['parameters'][same_cover], second['parameters'][same_cover])
        out, stats = loaded.transform(batch, kind, 719, 'conditional')
        assert torch.equal(out, apply_proposal(batch, second))
        assert stats['within_registered_bounds'] and stats['radius_linf_max_error'] == 0
        # Physical operator identity at zero radius, including T fractional delay.
        zero = dict(second, parameters=torch.zeros_like(second['parameters']))
        assert torch.equal(apply_proposal(batch, zero), batch)
        physical = dict(second, parameters=second['parameters'].clone().requires_grad_())
        loss = apply_proposal(batch, physical).square().mean()
        derivative, = torch.autograd.grad(loss, physical['parameters'])
        assert torch.isfinite(derivative).all()
        # No source-conditioned network can affect coverage samples or consume a
        # different random stream; coverage uses only its fixed draw plan.
        before = torch.get_rng_state().clone()
        loaded.transform(batch, kind, 731, 'coverage')
        assert torch.equal(before, torch.get_rng_state())
        assert not any(p.requires_grad for p in loaded.networks.parameters())
        branch = payload['diagnostics']['branches'][kind]
        assert branch['optimizer_steps'] == 4 and branch['sample_occurrences'] == 48
        for policy in ('coverage', 'conditional'):
            diagnostic = payload['diagnostics']['full_source_policy_diagnostics'][kind + '/' + policy]
            assert diagnostic['source_count'] == 24
            assert sum(diagnostic['normalized_radius_histogram']) == 24
    old_payload = dict(payload, format_version=1)
    try:
        load_actions(old_payload, device)
    except ValueError:
        pass
    else:
        raise AssertionError('Old-policy artifact was silently accepted')
    return dict(fit_apply_same_radial_policy=True, full_two_pass_fit=True, parent_preserved=True,
                paired_radius_exact=True, temporal_realization_paired=True,
                true_fixed_coverage_component=True, no_global_rng_consumption=True,
                original_physical_bounds_preserved=True, zero_radius_exact_iq_identity=True,
                source_risk_boundary_coverage_diagnostics=True, stale_policy_rejected=True,
                actual_iq_strength_equality_not_claimed=True, device=str(device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(dict(status='PASS', objectives=objective_checks(), actions=action_checks(args.device))), flush=True)
