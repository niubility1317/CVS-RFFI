"""Focused source-only checks for RX adversaries and main/auxiliary isolation."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from . import design as d
from .auxiliary import build, reverse, descriptor, domain_objective, gradient_diagnostics, capture, rx_classes


def _raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected protocol-negative check to reject input')


def _rng(device):
    return (torch.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()] if str(device).startswith('cuda') else [])


def _same_rng(a, b):
    return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(
        torch.equal(x, y) for x, y in zip(a[1], b[1]))


def _same_gradients(left, right):
    return all((a is None) == (b is None) and (a is None or torch.equal(a, b)) for a, b in zip(left, right))


def _independent_formula(auxiliary, maps, y, rx, arm):
    labels = rx_classes(rx); values = []
    for level in d.LEVELS:
        scores = auxiliary(level, descriptor(maps[level]))
        if arm == 'MARGINAL_GRL':
            # Independently average the six ordinary head CEs, rather than use
            # the implementation's flattened repeated-label reduction.
            values.append(torch.stack([F.cross_entropy(scores[:, k], labels) for k in range(6)]).mean())
        else:
            selected = torch.stack([scores[i, int(y[i])] for i in range(len(y))])
            values.append(F.cross_entropy(selected, labels))
    return torch.stack(values).mean()


def auxiliary_checks(device):
    x = torch.randn(7, 9, device=device, requires_grad=True)
    weight = torch.randn_like(x)
    for coefficient in (0., .01, .1):
        value = reverse(x, coefficient)
        if not torch.equal(value, x):
            raise AssertionError('GRL changed forward values')
        gradient = torch.autograd.grad((value * weight).sum(), x)[0]
        if not torch.equal(gradient, -coefficient * weight):
            raise AssertionError('GRL backward coefficient/sign differs')
    _raises(lambda: reverse(x, -.1))
    y = torch.arange(6, device=device).repeat_interleave(5)
    rx = torch.tensor(d.RX, device=device).repeat(6)
    if not torch.equal(rx_classes(torch.tensor(d.RX, device=device)), torch.arange(5, device=device)):
        raise AssertionError('Physical RX1/3/4/6/8 mapping differs')
    maps = dict(time_down=torch.randn(len(y), 48, 17, device=device, requires_grad=True),
                **{'t2.pool': torch.randn(len(y), 96, 9, device=device, requires_grad=True)})
    records = []
    for arm in d.ARMS:
        c = d.config(next(row for row in d.rows() if row['arm'] == arm))
        before = _rng(device); auxiliary = build(c, device); after = _rng(device)
        if not _same_rng(before, after):
            raise AssertionError('Auxiliary initialization changed identity RNG')
        if sum(p.numel() for p in auxiliary.parameters()) != 22460:
            raise AssertionError('Auxiliary parameter capacity differs')
        domain, metrics, details = domain_objective(auxiliary, maps, y, rx, c, 15)
        independent = _independent_formula(auxiliary, maps, y, rx, arm)
        if not torch.allclose(domain, independent, atol=2e-7, rtol=1e-6):
            raise AssertionError('Conditional/marginal CE independent formula differs')
        auxiliary_parameters = list(auxiliary.parameters())
        observed_aux = torch.autograd.grad(domain, auxiliary_parameters, retain_graph=True)
        expected_aux = torch.autograd.grad(independent, auxiliary_parameters, retain_graph=True)
        if any(not torch.allclose(a, b, atol=2e-7, rtol=2e-6) for a, b in zip(observed_aux, expected_aux)):
            raise AssertionError('GRL incorrectly reversed or scaled auxiliary parameter gradients')
        actual_map = torch.autograd.grad(domain, list(maps.values()), allow_unused=True, retain_graph=True)
        expected_map = torch.autograd.grad(independent, list(maps.values()), retain_graph=True)
        coefficient = metrics['GRL_coefficient']
        if arm == 'DETACHED_AUX':
            if any(value is not None for value in actual_map):
                raise AssertionError('Detached auxiliary still reaches identity maps')
        elif any(not torch.allclose(value, -coefficient * expected, atol=1e-8, rtol=3e-6)
                 for value, expected in zip(actual_map, expected_map)):
            raise AssertionError('Native descriptor gradients did not receive only negative GRL coefficient')
        warm, warm_metrics, _ = domain_objective(auxiliary, maps, y, rx, c, 5)
        warm_grads = torch.autograd.grad(warm, list(maps.values()), allow_unused=True)
        if warm_metrics['GRL_coefficient'] != 0. or any(value is not None for value in warm_grads):
            raise AssertionError('Epoch1..5 auxiliary is not detached')
        _raises(lambda: domain_objective(auxiliary, maps, y, torch.full_like(rx, 17), c, 15))
        _raises(lambda: domain_objective(auxiliary, maps, torch.full_like(y, -1), rx, c, 15))
        bad = copy.deepcopy(c); bad['adversary']['GRL_max'] = .2
        _raises(lambda: build(bad, device))
        if details['auxiliary_query_state'] or details['U_labels_read'] or not details['first_order_GRL']:
            raise AssertionError('Auxiliary data/query state semantics differ')
        records.append(dict(arm=arm, parameters=22460, actual_layers=details['actual_layers'],
                            independent_CE_formula=True, GRL_aux_gradient_ordinary=True,
                            warm_detached=True, RNG_isolated=True))
    return records


def integrated_gradient_checks(device):
    from .model import build as build_main, groups
    from .source import objective
    records = []
    x = torch.randn(128, 2, 256, device=device)
    y = torch.arange(128, device=device) % 6
    rx = torch.tensor(d.RX, device=device).repeat(26)[:128]
    for arm in d.ARMS:
        c = d.config(next(row for row in d.rows() if row['arm'] == arm))
        model = build_main(c, device).train(); auxiliary = build(c, device)
        pg, _ = groups(model, c); params = [p for group in pg for p in group['params']]
        torch.manual_seed(7001)
        total, metrics, details = objective(model, auxiliary, x, y, rx, None, None, None, None, c, 4441, 15)
        base = details['_base_tensor']; domain = details['_domain_tensor']
        audit = gradient_diagnostics(base, domain, params, model.head.weight)
        domain_cos = torch.autograd.grad(domain, [model.head.weight], allow_unused=True, retain_graph=True)[0]
        if domain_cos is not None or not audit['direct_cosW']['gradient_is_none']:
            raise AssertionError('RX domain loss directly reached ordinary cosine W')
        if arm == 'DETACHED_AUX':
            if audit['main']['domain_norm'] != 0.:
                raise AssertionError('Detached auxiliary created main gradients')
        elif audit['encoder']['domain_norm'] <= 0. or audit['main']['nonzero_domain_parameter_count'] == 0:
            raise AssertionError('Actual GRL did not reach identity parameters')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params + list(auxiliary.parameters())):
            raise AssertionError('Integrated main/auxiliary gradient is nonfinite')
        model.eval()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            out = model.forward_identity_only(x[:2]); joint = out['id_feat_joint']
            reference = 30 * F.linear(F.normalize(joint, dim=1, eps=1e-4),
                                      F.normalize(model.head.weight, dim=1, eps=1e-4))
            refit = F.linear(F.normalize(joint, dim=1), F.normalize(torch.randn(9, 160, device=device), dim=1))
        if not torch.equal(reference, out['tx_logits']) or joint.shape != (2, 160) or refit.shape != (2, 9):
            raise AssertionError('Ordinary cosine identity/refit contract changed')
        if any(not torch.equal(value, model.state_dict()[name]) for name, value in state.items()):
            raise AssertionError('Ordinary query inference changed identity state')
        if sum(p.numel() for p in params) != 177025:
            raise AssertionError('Source-only auxiliary leaked into identity parameter budget')
        shapes = [row['map_shape'][1:] for row in details['actual_layers']]
        if shapes != [[48, 128], [96, 32]]:
            raise AssertionError('Actual shallow/middle network captures differ')
        records.append(dict(arm=arm, main_parameters=177025, ordinary_cos=True,
                            refit9=True, actual_layers=details['actual_layers'], net_domain_gradient=audit))
    return records


def equivalence_checks(device):
    from .model import build as build_main, groups
    from .source import objective
    from experiments.cvs_report_mechanisms.feedback import objective as uniform, parameter_selection
    from experiments.cvs_cross_rx_feedback.checks import AcceptedBank
    x = torch.randn(128, 2, 256, device=device)
    y = torch.arange(128, device=device) % 6
    rx = torch.tensor(d.RX, device=device).repeat(26)[:128]
    ux = torch.randn_like(x); ids = [f'syntheticU{i}' for i in range(128)]
    passes = torch.full((128,), 2, dtype=torch.long)
    cases = [('DETACHED_AUX', 15, True), ('MARGINAL_GRL', 5, True),
             ('TX_COND_GRL', 5, True), ('DETACHED_AUX', 1, False)]
    records = []
    for arm, epoch, hidden_U in cases:
        c = d.config(next(row for row in d.rows() if row['arm'] == arm))
        model = build_main(c, device).train(); baseline = copy.deepcopy(model).train()
        auxiliary = build(c, device)
        pg, _ = groups(model, c); bp, _ = groups(baseline, c)
        params = [p for group in pg for p in group['params']]
        baseline_params = [p for group in bp for p in group['params']]
        opt = torch.optim.AdamW(pg, weight_decay=.0001)
        baseline_opt = torch.optim.AdamW(bp, weight_decay=.0001)
        auxiliary_opt = torch.optim.AdamW(auxiliary.parameters(), lr=.001, weight_decay=.0001)
        _, feedback_params = parameter_selection(baseline, bp)
        step = 1333 if hidden_U else 1
        torch.manual_seed(442); start_rng = _rng(device)
        total, metrics, detail = objective(model, auxiliary, x, y, rx, ux if hidden_U else None,
                                         ids if hidden_U else None, passes if hidden_U else None,
                                         AcceptedBank() if hidden_U else None, c, step, epoch)
        actual_rng = _rng(device); base = detail['_base_tensor']; domain = detail['_domain_tensor']
        actual_base_grad = torch.autograd.grad(base, params, allow_unused=True, retain_graph=True)
        actual_total_grad = torch.autograd.grad(total, params, allow_unused=True, retain_graph=True)
        if not _same_gradients(actual_base_grad, actual_total_grad):
            raise AssertionError('Detached domain changed main gradients')
        torch.set_rng_state(start_rng[0])
        if start_rng[1]:
            torch.cuda.set_rng_state_all(start_rng[1])
        previous_config = d.b.control(c['model_seed'])
        expected, _, _ = uniform(baseline, x, y, x[-25:], y[-25:], ux if hidden_U else None,
                                  ids if hidden_U else None, passes if hidden_U else None,
                                  AcceptedBank() if hidden_U else None, baseline_opt,
                                  feedback_params, previous_config, step)
        expected_rng = _rng(device)
        expected_grad = torch.autograd.grad(expected, baseline_params, allow_unused=True, retain_graph=True)
        if not torch.equal(base, expected) or not _same_gradients(actual_total_grad, expected_grad):
            raise AssertionError('Actual CE+UNIFORM_SSL main base loss/gradient changed')
        if not _same_rng(actual_rng, expected_rng):
            raise AssertionError('Auxiliary forwards changed main dropout RNG sequence')
        auxiliary_before = {name: value.clone() for name, value in auxiliary.state_dict().items()}
        total.backward(); expected.backward()
        before_clip = [p.grad.clone() if p.grad is not None else None for p in params]
        baseline_before_clip = [p.grad.clone() if p.grad is not None else None for p in baseline_params]
        if not _same_gradients(before_clip, baseline_before_clip):
            raise AssertionError('Main backward gradients changed')
        actual_norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
        expected_norm = torch.nn.utils.clip_grad_norm_(baseline_params, 10., error_if_nonfinite=True)
        auxiliary_norm = torch.nn.utils.clip_grad_norm_(list(auxiliary.parameters()), 5., error_if_nonfinite=True)
        if not torch.equal(actual_norm, expected_norm) or not _same_gradients(
                [p.grad for p in params], [p.grad for p in baseline_params]):
            raise AssertionError('Independent clip10/clip5 changed main clipped gradients')
        opt.step(); baseline_opt.step(); auxiliary_opt.step()
        if any(not torch.equal(value, baseline.state_dict()[name]) for name, value in model.state_dict().items()):
            raise AssertionError('Main AdamW one-step update changed at zero GRL')
        if not any(not torch.equal(value, auxiliary.state_dict()[name]) for name, value in auxiliary_before.items()):
            raise AssertionError('Source auxiliary did not update normally')
        records.append(dict(arm=arm, epoch=epoch, hidden_U_active=hidden_U,
                            base_loss=float(base.detach()), auxiliary_loss=float(domain.detach()),
                            total_includes_auxiliary_CE=True, main_loss_gradient_exact=True,
                            main_clip10_norm=float(actual_norm), auxiliary_clip5_norm=float(auxiliary_norm),
                            clipped_main_gradient_exact=True, AdamW_main_update_exact=True,
                            auxiliary_updated=True, RNG_exact=True))
    return records


def run(device='cpu'):
    from experiments.cvs_equivariant_identity.precision import numerical_context
    torch.set_num_threads(2); torch.manual_seed(71)
    # Exact update comparisons use deterministic convolution only in these tests.
    with numerical_context(d.FULL_FP32_POLICY), torch.backends.cudnn.flags(deterministic=True):
        auxiliary_records = auxiliary_checks(device)
        gradient_records = integrated_gradient_checks(device)
        equivalence_records = equivalence_checks(device)
    return dict(status='PASS', device=device, auxiliary=auxiliary_records,
                integrated_gradients=gradient_records, zero_GRL_equivalence=equivalence_records,
                query_auxiliary_state=False, target_access=False, U_truth_read=False,
                real_source_IQ_opened=False, synthetic_only=True, independent_optimizers=True,
                initialization_rng_isolated=True, GRL_forward_identity=True,
                GRL_first_order_backward=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.device)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], device=args.device,
                          arms=3, equivalence_cases=len(result['zero_GRL_equivalence']),
                          target_access=False, U_truth_read=False)), flush=True)
