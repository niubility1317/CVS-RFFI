"""Synthetic checks for partial matching, source grouping, and MLDG Hessians."""
import argparse
import itertools
import json
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.func import functional_call

from . import design as d
from . import matching as match
from . import updates as update
from .common import state_q, normalized_state
from experiments.cvs_source_pseudotarget_health.frozen import snapshot, assert_unchanged


def _reject(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return
    raise AssertionError('Expected negative protocol check to reject input')


def fixture():
    entries = [(tx, rx, day, packet) for tx in range(6) for rx in d.RX for day in d.DAYS for packet in range(70)]
    g = torch.Generator().manual_seed(771)
    L = dict(x=torch.randn(len(entries), 2, 256, generator=g),
             y=torch.tensor([a for a, b, c, i in entries]), rx=torch.tensor([b for a, b, c, i in entries]),
             day=torch.tensor([c for a, b, c, i in entries]), ids=[f'synthetic-TX{a}-RX{b}-day{c}-packet{i:03d}' for a, b, c, i in entries])
    folds = torch.tensor([i % 3 for a, b, c, i in entries])
    raw = state_q(L['x']); q, mean, scale = normalized_state(raw, folds != 2)
    features = {name: torch.randn(len(entries), width, generator=g) for name, width in [('time_down', 96), ('t2.pool', 192), ('joint160', 160)]}
    probability = torch.randn(len(entries), 6, generator=g).softmax(1)
    return dict(L=L, folds=folds, state_q_raw=raw, state_q=q, state_mean=mean, state_scale=scale,
                seed=2026092701, features={'L_s': features}, probabilities={'L_s': probability})


def matching_checks(folder):
    cost = torch.tensor([[.1, .7, .3, .9], [.2, .8, .4, .1], [.9, .1, .8, .2], [.6, .3, .1, .4]], dtype=torch.float64)
    plan = match.partial_plan(cost, 2); observed = float((plan * cost).sum())
    exact = min(sum(float(cost[i, j]) for i, j in zip(rows, columns)) / 4
                for rows in itertools.combinations(range(4), 2) for columns in itertools.permutations(range(4), 2))
    if abs(observed - exact) > 1e-12:
        raise AssertionError('Partial global assignment does not minimize the constrained cost')
    generator = torch.Generator().manual_seed(17)
    random = match.random_plan(4, 2, generator); greedy = match.greedy_plan(cost, 2)
    if observed > min(float((random * cost).sum()), float((greedy * cost).sum())) + 1e-12:
        raise AssertionError('Global constrained assignment is worse than a feasible control')
    for value in (plan, random, greedy):
        match.validate_plan(value, 4, 2)
    _reject(lambda: match.partial_plan(cost, 0))
    _reject(lambda: match.validate_plan(torch.zeros_like(plan), 4, 2))
    bad = plan.clone(); bad[0, 0] = 1.
    _reject(lambda: match.validate_plan(bad, 4, 2))
    _reject(lambda: match.partial_plan(torch.full((4, 4), torch.nan), 2))
    ctx = fixture(); pools = match.candidate_pools(ctx['L'], ctx['folds'], ctx['seed'])
    for section in ('fit', 'hold'):
        first = set(torch.cat([value for key, value in pools.items() if key[0] == section and key[-1] == 0]).tolist())
        second = set(torch.cat([value for key, value in pools.items() if key[0] == section and key[-1] == 1]).tolist())
        if first & second:
            raise AssertionError('Matching repeats reused physical source packets')
    out = folder / 'matching'
    rng = torch.get_rng_state().clone()
    result = match.diagnose(ctx, out, 'cpu')
    if not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('Matching or source-only q probe changed global RNG')
    raw = torch.load(out / 'source_matching_raw.pt', map_location='cpu', weights_only=False)
    if len(raw['records']) != 720 or not isinstance(result['applicability']['passed'], bool):
        raise AssertionError('Matching fixed full matrix or applicability missing')
    for record in raw['records']:
        for arm, values in record['plans'].items():
            match.validate_plan(values['P'], 10, 5)
            if len(values['anchor']) != 5 or values['validation_dimensions_square'].shape != (5, 4):
                raise AssertionError('Equal-K or excluded validation dimensions differ')
            y = ctx['L']['y']
            if not torch.equal(y[values['anchor']], y[values['positive']]) or (y[values['anchor']] == y[values['negative']]).any():
                raise AssertionError('Matching positives/negatives use wrong visible source TX roles')
        optimum = float((record['plans']['PARTIAL_OT']['P'] * record['combined_cost']).sum())
        for arm in ('RANDOM', 'GREEDY_NN'):
            if optimum > float((record['plans'][arm]['P'] * record['combined_cost']).sum()) + 1e-8:
                raise AssertionError('Full matching audit found a nonoptimal global plan')
    _reject(lambda: match.diagnose(ctx, out, 'cpu'), FileExistsError)
    return dict(exact_partial_OT_mass_and_caps=True, exhaustive_global_optimum=True,
                controls_have_equal_K_and_identical_candidates=True, repeat_PIDs_disjoint=True,
                held_dimensions_and_negative_rule_audited=True, full_synthetic_groups=720,
                applicability_is_source_only=True, output_overwrite_rejected=True), ctx


def mldg_math_check():
    torch.manual_seed(51)
    module = torch.nn.Linear(2, 3).double()
    xa = torch.tensor([[.1, .3], [.8, -.2], [-.4, .5], [.2, -.7]], dtype=torch.float64)
    xb = torch.tensor([[.5, -.4], [-.3, .7], [.9, .2]], dtype=torch.float64)
    ya = torch.tensor([0, 1, 2, 1]); yb = torch.tensor([2, 0, 1]); alpha = .2; lam = .7
    parameters = tuple(module.parameters()); original = {name: value.detach().clone() for name, value in module.state_dict().items()}
    full, _ = update.mldg_objective(module, xa, ya, xb, yb, ['weight'], alpha, lam, False)
    full_gradient = torch.autograd.grad(full, parameters)
    fo, _ = update.mldg_objective(module, xa, ya, xb, yb, ['weight'], alpha, lam, True)
    fo_gradient = torch.autograd.grad(fo, parameters)
    inner = F.cross_entropy(module(xa), ya)
    ga = torch.autograd.grad(inner, parameters, create_graph=True, retain_graph=True)
    adapted_weight = (module.weight - alpha * ga[0]).detach().requires_grad_(True)
    monitor_bias = module.bias.detach().clone().requires_grad_(True)
    outer = F.cross_entropy(F.linear(xb, adapted_weight, monitor_bias), yb)
    gb = torch.autograd.grad(outer, (adapted_weight, monitor_bias))
    hessian_product = torch.autograd.grad(ga[0], parameters, grad_outputs=gb[0].detach(), retain_graph=True)
    expected = tuple(a.detach() + lam * (b.detach() - alpha * h.detach()) for a, b, h in zip(ga, gb, hessian_product))
    expected_fo = tuple(a.detach() + lam * b.detach() for a, b in zip(ga, gb))
    if not torch.equal(full, fo) or any(not torch.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in zip(full_gradient, expected)):
        raise AssertionError('Full MLDG outer gradient omitted the block or mixed Hessian')
    if any(not torch.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in zip(fo_gradient, expected_fo)):
        raise AssertionError('First-order MLDG approximation boundary differs')
    difference = torch.cat([(a - b).flatten() for a, b in zip(full_gradient, fo_gradient)]).norm()
    if difference <= 1e-5 or hessian_product[1].norm() <= 1e-8:
        raise AssertionError('Toy check does not exercise a genuine mixed Hessian')
    flat_direction = F.normalize(torch.linspace(-.4, .7, sum(p.numel() for p in parameters), dtype=torch.float64), dim=0)
    direction = []; offset = 0
    for parameter in parameters:
        direction.append(flat_direction[offset:offset + parameter.numel()].reshape_as(parameter)); offset += parameter.numel()
    direction = tuple(direction)
    def value(sign, epsilon):
        local = {name: (parameter.detach() + sign * epsilon * vec).requires_grad_(True)
                 for (name, parameter), vec in zip(module.named_parameters(), direction)}
        a = F.cross_entropy(functional_call(module, local, (xa,), strict=True), ya)
        g = torch.autograd.grad(a, local['weight'])[0]
        virtual = dict(local, weight=local['weight'] - alpha * g)
        b = F.cross_entropy(functional_call(module, virtual, (xb,), strict=True), yb)
        return float(a.detach() + lam * b.detach())
    epsilon = 1e-5; numeric = (value(1, epsilon) - value(-1, epsilon)) / (2 * epsilon)
    analytic = float(sum((gradient * vec).sum() for gradient, vec in zip(full_gradient, direction)))
    if abs(numeric - analytic) > 2e-8:
        raise AssertionError('Full MLDG gradient does not match finite-difference objective')
    if any(not torch.equal(value, module.state_dict()[name]) for name, value in original.items()) or any(p.grad is not None for p in parameters):
        raise AssertionError('Functional MLDG toy mutated module state or accumulated gradients')
    return dict(full_Hessian_formula=True, mixed_Hessian_exercised=True, first_order_distinct=True,
                full_FO_gradient_difference=float(difference), finite_difference_error=abs(numeric - analytic),
                functional_original_model_unchanged=True)


class MiniIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.t1 = torch.nn.Module(); self.t2 = torch.nn.Module()
        self.t1.dw = torch.nn.Linear(2, 4, bias=False); self.t2.dw = torch.nn.Linear(4, 6, bias=False)
    def forward(self, x):
        return self.t2.dw(self.t1.dw(x).tanh())


def update_checks(ctx, device):
    L = ctx['L']; episodes = []
    for repeat in range(4):
        ua, tb = update.sample_episode(L, ctx['folds'], 1, repeat, ctx['seed'])
        if set(ua.flatten().tolist()) & set(tb.flatten().tolist()):
            raise AssertionError('Update and test physical packets overlap')
        episodes.append((ua, tb))
        for indices in (ua, tb):
            random = update.randomized_groups(indices, L, 113 + repeat)
            if sorted(indices.flatten().tolist()) != sorted(random.flatten().tolist()):
                raise AssertionError('Random RX groups changed source packet pool')
            for group in random:
                if not torch.equal(torch.bincount(L['y'][group], minlength=6), torch.full((6,), 3)):
                    raise AssertionError('Random RX grouping changed TX/day quota')
    for i, (a, b) in enumerate(episodes):
        for c, e in episodes[i + 1:]:
            if set(a.flatten().tolist()) & set(c.flatten().tolist()) or set(b.flatten().tolist()) & set(e.flatten().tolist()):
                raise AssertionError('Independent update repeats reuse physical samples')
    mini = MiniIdentity().to(device).eval().requires_grad_(False); before = snapshot(mini)
    small_L = dict(L, x=L['x'][:, :, :1].squeeze(-1))
    ua, tb = episodes[0]; names = update.block_names(mini)
    matrices = update.transfer_matrices(mini, small_L, ua, tb, names['t1.dw.weight'], [0., .01], device)
    if not torch.equal(matrices['G'][0.], torch.zeros(5, 5, dtype=torch.float64)):
        raise AssertionError('Zero functional virtual update changed source risk')
    assert_unchanged(mini, before)
    records = []
    for block in update.RECIPE['blocks']:
        for alpha in update.RECIPE['alphas']:
            for day in d.DAYS:
                for repeat in range(4):
                    for kind, control in [('RX', -1)] + [('RANDOM', j) for j in range(4)]:
                        matrix = torch.full((5, 5), .02 if kind == 'RX' else -.01, dtype=torch.float64)
                        matrix.diagonal().fill_(-.03)
                        records.append(dict(block=block, alpha=alpha, day=day, repeat=repeat,
                                            partition=kind, control=control, G=matrix, actual_update_CE_change=[-.04] * 5))
    positive = update.summarize(records)
    if not all(case['passed'] for case in positive):
        raise AssertionError('Prespecified source group-effect criterion failed its positive fixture')
    for record in records:
        if record['block'] == 't1.dw.weight' and record['alpha'] == .01:
            record['G'].fill_(-.01)
    negative = update.summarize(records)
    primary = next(case for case in negative if case['block'] == 't1.dw.weight' and case['alpha'] == .01)
    if primary['passed'] or not any(case['passed'] for case in negative if case['block'] == 't2.dw.weight'):
        raise AssertionError('Background cases incorrectly promote the fixed failed primary')
    # One actual native architecture instance verifies strict functional calls;
    # this creates random weights and synthetic IQ, never loads a checkpoint.
    from experiments.cvs_nocfo_foundation.model import build as native_build
    c = d.source_config(d.rows()[0]); native = native_build(c['source_loader_config'], device).eval().requires_grad_(False)
    frozen = snapshot(native); native_names = update.block_names(native)
    if any('.id_backbone.' not in name or '.dom_backbone.' in name for name in native_names.values()):
        raise AssertionError('Native virtual blocks must belong to the captured identity backbone')
    generator = torch.Generator().manual_seed(107)
    x = torch.randn(2, 2, 256, generator=generator).to(device)
    with torch.no_grad():
        normal = native(x)
    params = dict(native.named_parameters()); buffers = {k: v.clone() for k, v in native.named_buffers()}
    for name in native_names.values():
        leaf = params[name].detach().clone().requires_grad_(True); local = dict(params); local[name] = leaf
        functional = functional_call(native, (local, buffers), (x,), strict=True)
        if not torch.allclose(normal, functional, atol=1e-6, rtol=1e-5):
            raise AssertionError('Native functional_call did not preserve ordinary identity forward')
        gradient = torch.autograd.grad(functional.square().mean(), leaf)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise AssertionError('Actual native small-block functional gradient is missing')
    assert_unchanged(native, frozen)
    return dict(RX_random_same_pool_quota=True, update_test_PIDs_disjoint=True, repeat_PIDs_disjoint=True,
                functional_zero_step_exact=True, primary_failure_cannot_be_promoted_by_background=True,
                native_functional_forward_smallblock_gradients=True, original_frozen_state_unchanged=True)


def run(device='cpu'):
    torch.set_num_threads(2); torch.manual_seed(29)
    temporary_root = d.ROOT / 'local_artifacts'; temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='six_matching_updates_', dir=temporary_root) as folder:
        path = Path(folder)
        if not path.resolve().is_relative_to(temporary_root.resolve()):
            raise ValueError('Test output must remain in the project artifact root')
        matching_result, ctx = matching_checks(path)
        math_result = mldg_math_check()
        updates_result = update_checks(ctx, device)
    return dict(status='PASS', device=device, matching=matching_result, update=updates_result, MLDG_math=math_result,
                source_only=True, target_access=False, U_truth_read=False, identity_updates=0,
                actual_source_IQ_opened=False, actual_checkpoint_loaded=False, synthetic_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu'); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = run(args.device)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], device=args.device, target_access=False, identity_updates=0)), flush=True)
