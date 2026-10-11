"""Synthetic checks of source-only counterfactual semantics and immutability."""
import argparse
import json

import numpy as np
import torch
from torch import nn

from .common import margins
from .source_counterfactual_audit import (ReadOnlyState, RNGState, audit_source,
                                          iq_tensor, margin_penalty)


class TinyIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 12)
        self.norm = nn.BatchNorm1d(12)
        self.dropout = nn.Dropout(.7)
        self.head = nn.Linear(12, 3)
        self.register_buffer('scratch', torch.tensor(4.), persistent=False)
        self._filter_cache_key = ('initial',)
        self._filter_cache = torch.tensor([7.])
        self.last_mixstyle_calls = 9
        self.last_mixstyle_changed_samples = 11

    def forward_identity_only(self, x):
        self._filter_cache_key = ('forward',)
        self._filter_cache = torch.tensor([3.])
        self.last_mixstyle_calls = 0
        self.last_mixstyle_changed_samples = 0
        x = x.flatten(1)
        return {'tx_logits': 30 * self.head(self.dropout(self.norm(self.linear(x))).tanh())}


class ToyActions:
    def transform(self, x, kind, generator):
        # Exercise private stochastic action generation without fitting or IQ I/O.
        random = torch.rand(len(x), 1, 1, generator=generator).to(x)
        return x + (.05 if kind == 'linear' else -.1) * random, dict(within_registered_bounds=True, finite=True)


def run(device='cpu'):
    caller = RNGState()
    torch.manual_seed(519)
    model = TinyIdentity().to(device)
    model.train()
    model.norm.eval()  # A mixed mode must survive, not merely the root flag.
    model.linear.bias.requires_grad_(False)
    model.head.weight.grad = torch.ones_like(model.head.weight)
    count = 53
    x = torch.randn(count, 2, 4)
    with ReadOnlyState(model):
        model.eval()
        with torch.no_grad():
            z = model.forward_identity_only(x.to(device))['tx_logits']
            y = z.argmax(1)
            u = margins(z, y).cpu()
    cache = dict(x=x, y=y.cpu(), rx=torch.arange(count) % 2,
                 parent_margin=u, parent_correct=torch.ones(count, dtype=torch.bool),
                 ids=['source-' + str(i) for i in range(count)], role='L_s')
    expected = ReadOnlyState(model)
    result = audit_source(model, cache, ToyActions(), 715, device, torch.full((3,), 100.),
                          encoder_parameters=list(model.linear.parameters()), batch_size=8, gradient_batches=12)
    assert result['source_count'] == count and result['batches'] == 7
    assert result['status'] == 'SOURCE_COUNTERFACTUAL_AUDIT_COMPLETE'
    assert result['seed'] == 715 and result['count'] == count and result['model_state_restored']
    assert len(result['gradient_audits']) == 7
    assert all(result['state_invariance'].values()) and expected.rng.matches()
    assert all(torch.equal(p, saved) and p.requires_grad == flag for p, saved, flag, _ in expected.parameters)
    assert all(torch.equal(b, saved) for b, saved in expected.buffers)
    assert all(m.training == flag for m, flag in expected.modes)
    for name in ('paired_zero_action/drop_loss', 'paired_zero_action/margin_drop', 'same_parent_keep_eval_iq_1/loss'):
        assert result['metrics'][name]['all']['rms'] == 0, name
    assert result['metrics']['old_zero_action/drop_loss']['all']['mean'] > 0
    assert result['metrics']['same_parent_keep_train_iq_1/loss']['all']['mean'] > 0
    for audit in result['gradient_audits']:
        for name, loss in audit['losses'].items():
            assert math_finite(loss['raw_E_norm'])
            if name.startswith('paired_zero_action_drop'):
                assert loss['loss'] == 0 and loss['raw_E_norm'] == 0
        assert audit['losses']['clean_eval_ce']['raw_E_norm'] > 0
        assert audit['losses']['linear_eval_ce']['raw_E_norm'] > 0
        original = audit['losses']['old_zero_action_drop']['raw_E_norm']
        scaled = audit['losses']['old_zero_action_drop__unit_margin_scaled_huber']['raw_E_norm']
        assert abs(scaled - original / 30) <= max(1e-6, original * 1e-6)
    # The API keeps a differentiable zero loss, rather than hiding it in no_grad.
    probe = torch.tensor([1., 2.], device=device, requires_grad=True)
    zero = margin_penalty(probe, probe).mean()
    assert zero.requires_grad
    assert not torch.autograd.grad(zero, probe)[0].any()
    # NumPy conversion follows the buffer path even on an ABI-mismatched runtime.
    a = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    assert torch.equal(iq_tensor(a, device).cpu(), torch.arange(16).float().reshape(2, 2, 4))
    # Restoration must also happen on exceptions and include nonpersistent state.
    before = ReadOnlyState(model)
    try:
        with ReadOnlyState(model):
            model.train()
            model.scratch.add_(1.)
            model.linear.bias.requires_grad_(True)
            torch.rand(2, device=device)
            raise ValueError('intentional fixture exception')
    except ValueError:
        pass
    assert before.rng.matches() and all(torch.equal(b, s) for b, s in before.buffers)
    assert all(m.training == f for m, f in before.modes)
    caller.restore()
    assert caller.matches()
    return dict(status='PASS', device=str(device), complete_source_fixture_count=count,
                deterministic_identity_loss_and_gradient_zero=True, eval_autograd_preserved=True,
                train_zero_intervention_noise_measured=True, initial_train_keep_nonzero=True,
                state_and_rng_restored=True, exception_restores_state=True, numpy_buffer_conversion=True,
                native_forward_caches_counters_and_parameter_versions_restored=True,
                normalized_covariant_gradient_scales_one_over_30=True,
                old_zero_drop_mean=result['metrics']['old_zero_action/drop_loss']['all']['mean'],
                initial_keep_train_mean=result['metrics']['same_parent_keep_train_iq_1/loss']['all']['mean'])


def math_finite(value):
    return bool(torch.isfinite(torch.tensor(value)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.device), ensure_ascii=False), flush=True)
