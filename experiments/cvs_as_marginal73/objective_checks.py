"""Pure-logit marginal CE checks; no actual CVS forward or data access."""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


def objective_checks():
    from .objective import terms
    passed = []
    y = torch.tensor([0, 1, 2, 3])
    values = torch.arange(24, dtype=torch.float64).reshape(4, 6).sin()
    first = values.clone().requires_grad_(); second = values.clone().requires_grad_()
    delta, report = terms(first, second, y, 'PROBABILITY_MEAN_CE')
    old = .5 * (F.cross_entropy(first, y, label_smoothing=.01) + F.cross_entropy(second, y, label_smoothing=.01))
    actual = old + delta
    actual_grad = torch.autograd.grad(actual, (first, second), retain_graph=True)
    old_grad = torch.autograd.grad(old, (first, second), retain_graph=True)
    assert delta.item() == 0 and report['mean_CE'] == report['probmean_CE']
    assert torch.equal(actual, old) and all(torch.equal(a, b) for a, b in zip(actual_grad, old_grad))
    passed.append('identical_logits_exact_old_mean_CE_and_both_gradients')
    first = torch.tensor([[5., -2., 1., .3, 0., -1.], [-3., 4., 1., 0., 2., -1.], [2., 0., 4., -2., 1., 0.], [1., 2., 0., 3., -1., 0.]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[-2., 4., 0., 1., -1., .2], [3., -2., 0., 1., 2., 0.], [-2., 1., .3, 4., 0., 1.], [0., -3., 4., 2., 0., 1.]], dtype=torch.float64, requires_grad=True)
    groups = torch.tensor([0, 1, 2, 3])
    delta, report = terms(first, second, y, 'PROBABILITY_MEAN_CE', groups)
    mean_ce = .5 * (F.cross_entropy(first, y, label_smoothing=.01) + F.cross_entropy(second, y, label_smoothing=.01))
    distribution = .5 * (first.softmax(1) + second.softmax(1))
    target = torch.full_like(first, .01 / 6); target.scatter_add_(1, y[:, None], torch.full((4, 1), .99, dtype=torch.float64))
    hand = -(target * distribution.log()).sum(1).mean()
    assert torch.allclose(mean_ce + delta, hand, atol=1e-14, rtol=1e-14)
    assert not torch.allclose(hand, F.cross_entropy(.5 * (first + second), y, label_smoothing=.01))
    assert report['Jensen_gap'] >= -1e-14 and math.isclose(report['Jensen_gap'], -delta.item(), abs_tol=1e-14)
    passed.append('probability_mean_soft_label_hand_formula_not_mean_logits')
    gradients = torch.autograd.grad(mean_ce + delta, (first, second), retain_graph=True)
    hand_gradients = torch.autograd.grad(hand, (first, second), retain_graph=True)
    assert all(bool(torch.isfinite(a).all()) and torch.allclose(a, b, atol=1e-14, rtol=1e-14) for a, b in zip(gradients, hand_gradients))
    assert all(float(a.abs().sum()) > 0 for a in gradients)
    passed.append('both_branches_finite_live_gradients_match_probability_formula')
    zero, control = terms(first, second, y, 'MEAN_CE', groups)
    assert zero.requires_grad and zero.item() == 0 and control['used_CE'] == control['mean_CE']
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(zero, (first, second), retain_graph=True))
    assert len(report['group_stats']) == 4 and sum(row['count'] for row in report['group_stats']) == 4
    predictions = distribution.argmax(1); p0, p1 = first.argmax(1), second.argmax(1)
    for index, prediction in enumerate((p0, p1)):
        expected_repaired = int(((prediction != y) & (predictions == y)).sum())
        expected_new = int(((prediction == y) & (predictions != y)).sum())
        assert report['pooled_vs_mask' + str(index)]['repaired'] == expected_repaired
        assert report['pooled_vs_mask' + str(index)]['new_errors'] == expected_new
    passed.append('live_zero_control_and_fourbin_per_mask_error_accounting')
    extreme0 = torch.tensor([[1e4, -1e4, -1e4, 0., 1., -1.], [-1e4, 1e4, 1., 0., -1., 2.]], dtype=torch.float64, requires_grad=True)
    extreme1 = (-extreme0.detach()).requires_grad_(); extreme_y = torch.tensor([0, 2])
    extreme_delta, extreme_report = terms(extreme0, extreme1, extreme_y, 'PROBABILITY_MEAN_CE')
    assert math.isfinite(extreme_report['probmean_CE']) and extreme_report['Jensen_gap'] >= -1e-12
    assert all(bool(torch.isfinite(g).all()) for g in torch.autograd.grad(extreme_delta, (extreme0, extreme1)))
    passed.append('extreme_1e4_logits_stable_finite_loss_and_gradients')
    bad = [
        (first[:, :5], second[:, :5], y, 'MEAN_CE', None),
        (first, second[:3], y, 'MEAN_CE', None),
        (first, second, y.float(), 'MEAN_CE', None),
        (first, second, torch.tensor([0, 1, 2, 6]), 'MEAN_CE', None),
        (first, second, y, 'OTHER', None),
        (first, second, y, 'MEAN_CE', torch.tensor([0, 1, 2, 4])),
        (first, second, y, 'MEAN_CE', torch.ones(4)),
    ]
    nonfinite = first.detach().clone(); nonfinite[0, 0] = float('nan')
    bad.append((nonfinite, second, y, 'MEAN_CE', None))
    for args in bad:
        try: terms(*args)
        except ValueError: pass
        else: raise AssertionError('Malformed source logits/labels/groups/mode accepted')
    passed.append('bad_shapes_labels_nonfinite_groups_and_mode_rejected')
    return passed


def hook_checks():
    """Run the actual six-forward objective hook with source-free tensor stubs."""
    from .model import _labeled_hook
    from collections import Counter
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch
    import sys

    def rd(a, b, y, alpha):
        ce0, ce1 = (F.cross_entropy(z, y, label_smoothing=.01) for z in (a, b))
        p, q = a.log_softmax(1), b.log_softmax(1)
        kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1).mean()
        return .5 * (ce0 + ce1) + alpha * kl, .5 * (ce0 + ce1), kl, ce0, ce1

    rx_by_domain = [r for r in (1, 3, 4, 6, 8) for _ in range(3)]
    supcon = SimpleNamespace(schedule=lambda epoch: 0,
        clear=lambda m: setattr(m, '_supcon_pending', None),
        RX_BY_DOMAIN=rx_by_domain, LEVELS=())
    base = SimpleNamespace(terms=rd,
        capture_forward=lambda m, x, d, active: (m.forward_identity_only(x, domain_labels=d)['tx_logits'], {}))
    fake = {}
    for name, attrs in [
        ('experiments.cvs_crossview_supcon41', dict(model=base, relations=SimpleNamespace())),
        ('experiments.cvs_fullsource_concat20', dict(supcon=supcon)),
        ('experiments.cvs_noise_shape_factorial43', dict(noise=SimpleNamespace(
            apply=lambda raw, step, mode: (raw.clone(), dict(applied=0))))),
    ]:
        module = ModuleType(name); module.__dict__.update(attrs); fake[name] = module

    raw = torch.arange(128 * 2 * 256, dtype=torch.float64).reshape(128, 2, 256) / 100000
    ordinary = raw + .1; labels = torch.arange(128) % 6; domains = torch.arange(128) % 15
    states = []; snapshots = []; records = []
    for step in (1, 2):
        smask = (torch.arange(128) + step).remainder(2) != 0
        selected = torch.nonzero(smask).flatten().tolist()
        meta = dict(eq_i=torch.ones(128, dtype=torch.long), rx_i=torch.tensor(rx_by_domain)[domains],
            day_i=domains % 3 + 1, sig_i=torch.arange(128), base_index=torch.arange(128))
        ids = [[int(meta[k][i]) for k in ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')] for i in selected]
        observations = [dict(sample_id=pid, scene='toy', state_end='LOS', receiver_lock_status='locked') for pid in ids]

        class Reader:
            def __init__(self): self.origin = None
            def validate_origin(self, actual_step, epoch, *, labels, domains, physical_ids):
                assert actual_step == step and epoch == 1 and physical_ids == ids
                assert torch.equal(labels, source_y[smask]) and torch.equal(domains, source_d[smask])
                self.origin = dict(physical_ids=ids, labels=labels.tolist(), domains=domains.tolist())
                return self.origin
            def consume(self, actual_step, epoch, device):
                assert self.origin is not None and actual_step == step
                # Keep the actual P70 boundary: groups are a JSON list, not a Tensor.
                return raw[smask] + 2, labels[smask], domains[smask], observations, dict(
                    groups=[i % 4 for i in range(64)], counts=[16] * 4, p_g=[.1, .2, .3, .4],
                    used_pool_indices=list(range(64 * (step - 1), 64 * step)),
                    origin_steps=[step] * 64, origin_slots=selected)

        source_y, source_d = labels, domains
        for mode in ('MEAN_CE', 'PROBABILITY_MEAN_CE'):
            torch.manual_seed(117)
            dropout = torch.nn.Dropout(.45).train(); calls = []; logits = []
            m = SimpleNamespace(training=True, _quality_pool_bound=True, _quality_method='POOL_RANDOM',
                _feature_l_steps=step, _full_calls=step-1, _legacy_actual_L_input=ordinary.clone(),
                _supcon_pending=None, _full_L_packets=0, _supcon_calls=0, _legacy_calls=0,
                _legacy_extra_packets=0, _legacy_positive_KL_calls=0, _full_records=[],
                _full_coverage=Counter(), _supcon_applied=0, _shape_offset=4440, _shape_mode='NONE',
                _shape_NN=0, _shape_packets=0, _shape_records=[], _quality_records=[],
                _crossview_mode='ORD128', _marginal_mode=mode,
                _marginal_method='AS_MEAN_CE' if mode == 'MEAN_CE' else 'AS_PROBMEAN_CE',
                _marginal_coefficient=.25, _marginal_records=[], _full_channel=Reader())

            def forward(x, domain_labels):
                calls.append((x.detach().clone(), domain_labels.detach().clone()))
                axis = torch.arange(160, dtype=x.dtype)
                features = (x.mean((1, 2))[:, None] * (axis + .3) / 100
                    + domain_labels[:, None] * axis * .0001).requires_grad_()
                z = dropout(features)[:, :6]; logits.append(z)
                return dict(tx_logits=z)

            m.forward_identity_only = forward
            first = forward(ordinary, domains)['tx_logits']
            identity = F.cross_entropy(first, labels, label_smoothing=.01)
            with patch.dict(sys.modules, fake):
                delta, metrics = _labeled_hook(m, model=m, labeled_output=dict(tx_logits=first),
                    labels=labels, extra=(domains, meta), domains=domains, optimizer_step=step,
                    identity_loss=identity, labeled_count=128, labeled_inputs=raw)
            actual = identity + delta
            mean_as = .5 * (F.cross_entropy(logits[2][smask], labels[smask], label_smoothing=.01)
                + F.cross_entropy(logits[3][smask], labels[smask], label_smoothing=.01))
            probabilities = .5 * (logits[2][smask].softmax(1) + logits[3][smask].softmax(1))
            target = torch.full_like(probabilities, .01 / 6)
            target.scatter_add_(1, labels[smask, None], torch.full((64, 1), .99, dtype=torch.float64))
            probmean_as = -(target * probabilities.log()).sum(1).mean()
            original = (.5 * rd(logits[0], logits[1], labels, 1.)[0]
                + .5 * rd(logits[2], logits[3], labels, 1.)[0]
                + .25 * rd(logits[4], logits[5], labels, 1.)[0])
            change = .25 * (probmean_as - mean_as) if mode != 'MEAN_CE' else mean_as * 0
            expected = original + change
            assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
            actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)
            expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)
            assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-12)
                for a, b in zip(actual_grad, expected_grad))
            assert len(calls) == 6 and all(torch.equal(calls[i][0], calls[j][0])
                for i, j in ((0, 1), (2, 3), (4, 5)))
            assert all(torch.equal(d, domains) for _, d in calls)
            assert torch.equal(calls[2][0][~smask], ordinary[~smask])
            assert torch.equal(calls[2][0][smask], raw[smask] + 2)
            assert m.forward_identity_only is forward and not dropout._forward_hooks
            assert m._full_L_packets + m._shape_NN == 768
            native, q, report = m._full_records[0], m._quality_records[0], m._marginal_records[0]
            assert isinstance(q['groups'], list) and q['origin_verified'] and report['As_origin_verified']
            assert report['method'] == m._marginal_method and report['physical_ids'] == ids
            assert report['used_pool_indices'] == list(range(64 * (step-1), 64*step))
            assert sum(row['count'] for row in report['group_stats']) == 64
            assert report['KL_unchanged'] and native['extra_KL'] == report['mixed_KL']
            assert math.isclose(native['total_L'], float(actual), abs_tol=1e-12)
            assert math.isclose(native['extra_CE'], report['used_mixed_CE'], abs_tol=1e-12)
            assert math.isclose(native['extra_RD'], report['used_mixed_RD'], abs_tol=1e-12)
            assert math.isclose(native['base_L_before_shape'], report['used_L_before_shape'], abs_tol=1e-12)
            assert math.isclose(q['CE_correction_to_total'], float(change), abs_tol=1e-12)
            assert math.isclose(q['used_total_L'], float(actual), abs_tol=1e-12)
            assert metrics['quality/As_CE'] == report['used_CE']
            assert report['all_As_CE_supervised'] == 64 and report['extra_NN'] == report['new_channel_calls'] == 0
            # The old analyzer's actual mixed CE/KL algebra must describe the used loss.
            reconstructed = (.5 * (native['common_CE'] + native['common_KL'])
                + .5 * (native['extra_CE'] + native['extra_KL'])
                + .25 * m._shape_records[0]['shape_RD'])
            assert math.isclose(reconstructed, native['total_L'], abs_tol=1e-12)
            states.append(torch.get_rng_state().clone()); snapshots.append(calls); records.append(report)
        assert torch.equal(states[-1], states[-2])
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            for a, b in zip(snapshots[-1], snapshots[-2]))
    assert all(record['Jensen_gap'] >= -1e-12 for record in records)
    return ['actual_six_forward_hook_list_groups_only_As_CE_live_gradients_KL_IQ_RNG_and_both_parities']


def run(hook_only=False):
    started = time.perf_counter()
    passed = [] if hook_only else objective_checks()
    passed += hook_checks()
    return dict(status='PASS', checks=len(passed), passed=passed,
                wall_seconds=time.perf_counter() - started,
                real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0, actual_channel_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--hook-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.hook_only), allow_nan=False))
