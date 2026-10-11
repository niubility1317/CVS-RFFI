"""Bounded synthetic IQ checks; no checkpoint, dataset, optimizer or target I/O."""
import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from .common import identity_logits, margins, positive_huber
from .deterministic_forward import (DROPOUT_TYPES, DeterministicIdentityContractError,
                                    deterministic_identity, identity_mode_contract)


class MixStyleFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.p, self.alpha, self.strength = .3, .1, .7

    def forward(self, x):
        return x


class IdentityFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(12, 24)
        self.norm = nn.LayerNorm(24)
        self.drop = nn.Dropout(.65)
        self.alpha = nn.AlphaDropout(.3)
        self.head = nn.Linear(24, 3)
        self.branch_drop_p = 0.
        self.mixstyle_on = False
        self.mixstyle = MixStyleFixture()
        self.last_mixstyle_calls = 7
        self.last_mixstyle_changed_samples = 9

    def forward(self, x):
        self.last_mixstyle_calls = self.last_mixstyle_changed_samples = 0
        return self.head(self.alpha(self.drop(self.norm(self.encoder(x)))))


class ModelFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = IdentityFixture()
        self.dom_backbone = nn.Sequential(nn.Linear(12, 8), nn.Dropout(.7))

    def forward_identity_only(self, x):
        return {'tx_logits': self.id_backbone(x)}


def _expect_mismatch(function):
    try:
        function()
    except DeterministicIdentityContractError:
        return
    raise AssertionError('Expected deterministic identity contract mismatch')


def synthetic_checks():
    torch.manual_seed(4271)
    model = ModelFixture().train()
    model.id_backbone.alpha.eval()  # Must restore an actual mixed mode.
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    x, y = torch.randn(32, 12), torch.arange(32) % 3
    before = identity_logits(model, x)
    before2 = identity_logits(model, x)
    assert not torch.equal(before, before2)
    model.id_backbone.last_mixstyle_calls = 7
    model.id_backbone.last_mixstyle_changed_samples = 9
    counters = (7, 9)
    random_before = torch.get_rng_state().clone()
    with deterministic_identity(model) as contract:
        assert model.training and model.id_backbone.training
        assert model.dom_backbone[1].training  # Domain path was never modified.
        a = identity_logits(model, x)
        assert a.requires_grad
        with deterministic_identity(model):
            b = identity_logits(model, x)
            assert torch.equal(a, b)
        assert not model.id_backbone.drop.training
        c = identity_logits(model, x + .15)
        zero = positive_huber(margins(a, y).detach() - margins(b, y) - .5).mean()
        assert zero.requires_grad and float(zero.detach()) == 0.
        zero_grad = torch.autograd.grad(zero, model.id_backbone.encoder.weight, retain_graph=True)[0]
        assert not zero_grad.any()
        real_grad = torch.autograd.grad(F.cross_entropy(c, y), model.id_backbone.encoder.weight)[0]
        assert torch.isfinite(real_grad).all() and real_grad.abs().sum() > 0
    assert torch.equal(torch.get_rng_state(), random_before)
    assert all(m.training == f for m, f in modes)
    assert all(p.requires_grad == f for p, f in flags)
    assert (model.id_backbone.last_mixstyle_calls, model.id_backbone.last_mixstyle_changed_samples) == counters
    # Original CE mode is unchanged after the mechanism, including randomness.
    assert not torch.equal(identity_logits(model, x), identity_logits(model, x))
    model.id_backbone.last_mixstyle_calls, model.id_backbone.last_mixstyle_changed_samples = counters
    try:
        with deterministic_identity(model):
            identity_logits(model, x)
            raise ValueError('fixture failure')
    except ValueError:
        pass
    assert all(m.training == f for m, f in modes)
    assert (model.id_backbone.last_mixstyle_calls, model.id_backbone.last_mixstyle_changed_samples) == counters
    assert not model.id_backbone._forward_pre_hooks and not model.id_backbone._forward_hooks
    model.id_backbone.branch_drop_p = .1
    _expect_mismatch(lambda: identity_mode_contract(model))
    model.id_backbone.branch_drop_p = 0.
    model.id_backbone.mixstyle_on = True
    _expect_mismatch(lambda: identity_mode_contract(model))
    model.id_backbone.mixstyle_on = False
    model.id_backbone.bn = nn.BatchNorm1d(24)
    _expect_mismatch(lambda: identity_mode_contract(model))
    model.id_backbone.bn.eval()
    identity_mode_contract(model)
    del model.id_backbone.bn
    # Reject a re-enabled stochastic module before it can consume any RNG.
    with deterministic_identity(model):
        model.id_backbone.drop.train()
        _expect_mismatch(lambda: identity_logits(model, x))
    with torch.no_grad():
        def bad_grad_context():
            with deterministic_identity(model):
                pass
        _expect_mismatch(bad_grad_context)
    # Check every standard Dropout family, not only nn.Dropout.
    model.id_backbone.other_drops = nn.ModuleList([kind(.2) for kind in DROPOUT_TYPES])
    with deterministic_identity(model):
        assert all(not m.training for m in model.id_backbone.other_drops)
    assert all(m.training for m in model.id_backbone.other_drops)
    return dict(status='PASS', nested_and_exception_mode_restoration=True,
                adapter_counters_restored=True, original_ce_dropout_unchanged=True,
                zero_intervention_loss_and_gradient_zero=True, augmented_encoder_gradient_nonzero=True,
                domain_modes_and_requires_grad_unchanged=True, deterministic_scope_rng_unchanged=True,
                active_branch_drop_mixstyle_running_bn_rejected=True, all_dropout_families_disabled=True,
                no_grad_and_mode_drift_rejected=True, fixture_dropout_count=len(contract['dropout_modules']))


def native_checks(device='cpu'):
    """Actual registered architecture, random weights and 4 synthetic IQ rows."""
    from . import design as d
    from .common import parameter_groups
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    c = d.config(next(r for r in d.rows() if r['arm'] == 'C0'))
    torch.manual_seed(4371)
    args = d.make_args(c, device)
    args.num_domains, args.input_len = 5, 256
    with numerical_context(d.FULL_FP32_POLICY), installed(c) as native:
        model = native.build_baseline_model(args, torch.device(device))
        groups, _ = parameter_groups(model)
        encoder = next(g['params'] for g in groups if g['name'] == 'encoder')
        model.train()
        modes = [(m, m.training) for m in model.modules()]
        x = torch.randn(4, 2, 256, device=device)
        y = torch.arange(4, device=device) % 6
        domain_calls = []
        hook = model.dom_backbone.register_forward_hook(lambda *args: domain_calls.append(1))
        with deterministic_identity(model) as contract:
            a, b = identity_logits(model, x), identity_logits(model, x)
            assert torch.equal(a, b)
            zero = positive_huber(margins(a, y).detach() - margins(b, y) - .5).mean()
            zero_grad = torch.autograd.grad(zero, encoder, retain_graph=True, allow_unused=True)
            assert all(g is None or not g.any() for g in zero_grad)
            loss = F.cross_entropy(b, y)
            gradients = torch.autograd.grad(loss, encoder, allow_unused=True)
            norm = sum(g.detach().float().square().sum() for g in gradients if g is not None).sqrt()
            assert torch.isfinite(norm) and norm > 0
        hook.remove()
        assert not domain_calls and all(m.training == f for m, f in modes)
    return dict(status='PASS', device=device, architecture='reference_response', source_rows=4,
                checkpoint_reads=0, target_access=False, dropout_modules=len(contract['dropout_modules']),
                statistical_modules=contract['statistical_modules'], dormant_mixstyle=contract['dormant_mixstyle_modules'],
                actual_encoder_gradient_norm=float(norm), deterministic_identity_outputs_exact=True,
                zero_intervention_E_gradient_zero=True, domain_forward_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(synthetic=synthetic_checks())
    if args.native:
        result['native'] = native_checks(args.device)
    print(json.dumps(result, ensure_ascii=False), flush=True)
