"""Source-only synthetic checks for full-vector receiver geometry."""
from copy import deepcopy
from itertools import combinations, product
import importlib.util
import json
import math

import torch
from torch.nn import functional as F


def synthetic_cache():
    state, feature, ys, rxs, days = [], [], [], [], []
    for tx, day, rx, packet in product(range(6), range(2), range(3), range(10)):
        angle = 2 * math.pi * rx / 3
        a, b = .2 * math.cos(angle), .2 * math.sin(angle)
        s = torch.zeros(40, dtype=torch.float64)
        s[0], s[1], s[2], s[32] = a + .03 * tx, b, (packet - 4.5) * .01, .5
        z = torch.zeros(160, dtype=torch.float64)
        z[0], z[1], z[10 + tx] = a, b, math.sqrt(.96)
        state.append(s); feature.append(z)
        ys.append(tx); rxs.append(rx); days.append(day)
    return dict(state=torch.stack(state), parent_feature=torch.stack(feature),
                y=torch.tensor(ys), rx=torch.tensor(rxs), day=torch.tensor(days),
                ids=[str(i) for i in range(len(ys))], parent_margin=torch.ones(len(ys)) * 10,
                parent_correct=torch.ones(len(ys), dtype=torch.bool))


def check():
    assert importlib.util.find_spec('experiments.cvs_stage2_geometry.receiver') is not None, \
        'The full-vector receiver module has not been implemented'
    from .receiver import (fit_receiver, load_receiver, finite_population_loss,
                           block_potentials, vector_reliability, trusted_covariance)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cache = synthetic_cache()
        cache['state'].requires_grad_(True)
        cache['parent_feature'].requires_grad_(True)
        before = torch.random.get_rng_state().clone()
        payload = fit_receiver(cache, seed=123)
        assert torch.equal(before, torch.random.get_rng_state()), 'Fit changed caller RNG'
        assert payload['parent_feature'].shape == (360, 160)
        assert all(f['rank'] == 2 and f['c'] > .95 for f in payload['folds'])
        assert all(f['fit_ids'] == 240 and f['fit_blocks'] == 8 for f in payload['folds'])
        assert payload['optimizer_steps'] == 0 and payload['target_access'] is False

        # Closed-form 1-D coefficient inside this rank-2 example: oversampling one
        # RX changes the fit-only standardizer, but must not change RX loss weights.
        extra = torch.nonzero(cache['rx'] == 1, as_tuple=True)[0]
        expanded_ids = torch.cat((torch.arange(len(cache['y'])), extra, extra))
        expanded = {k: v.detach()[expanded_ids].clone() if torch.is_tensor(v) else deepcopy(v)
                    for k, v in cache.items()}
        expanded['ids'] = ['expanded-'+str(i) for i in range(len(expanded_ids))]
        weighted = fit_receiver(expanded, 123)['folds'][0]
        macro_state = torch.tensor([.2*math.sin(2*math.pi*r/3) for r in range(3)],
                                   dtype=torch.float32).double()
        macro_feature = torch.tensor([.2*math.sin(2*math.pi*r/3) for r in range(3)], dtype=torch.float64)
        covariance_s_z = ((macro_state-macro_state.mean())*(macro_feature-macro_feature.mean())).mean()
        variance_s = (macro_state-macro_state.mean()).square().mean()
        expected_coefficient = covariance_s_z/(variance_s+.1*weighted['standard_scale'][1].square())
        actual_coefficient = weighted['matrix'][1,1]/weighted['standard_scale'][1]
        torch.testing.assert_close(actual_coefficient, expected_coefficient, rtol=1e-10, atol=1e-12)

        # Unequal numbers of pairs cannot give large blocks extra reliability weight.
        balance_groups = []
        for position, (value, coord) in enumerate([(-.5,0),(.5,0),(-.5,1),(0.,1),(.5,1)]):
            truth = torch.zeros(160, dtype=torch.float64); truth[coord] = value
            balance_groups.append(dict(parent_feature_mean=truth,
                                       potential=truth.clone() if position < 2 else torch.zeros_like(truth)))
        balanced = vector_reliability(balance_groups, [dict(group_ids=[0,1]),dict(group_ids=[2,3,4])],
                                      torch.eye(160, dtype=torch.float64)[:,:2])
        assert abs(balanced['c']-2/3) < 1e-12

        # A flip restricted to held-out TX must not enter the fitted map or scaler.
        changed = {k: v.detach().clone() if torch.is_tensor(v) else deepcopy(v)
                   for k, v in cache.items()}
        mask = changed['y'] < 2
        changed['parent_feature'][mask, :2] *= -1
        mismatch = fit_receiver(changed, seed=123)
        for key in ('standard_mean', 'standard_scale', 'basis', 'regression', 'matrix'):
            torch.testing.assert_close(payload['folds'][0][key], mismatch['folds'][0][key],
                                       rtol=0, atol=0)
        assert mismatch['folds'][0]['c'] == 0
        changed['state'][mask, :20] += 50
        leaked = fit_receiver(changed, seed=123)
        for key in ('standard_mean', 'standard_scale', 'matrix'):
            torch.testing.assert_close(payload['folds'][0][key], leaked['folds'][0][key],
                                       rtol=0, atol=0)

        constant = {k: v.detach().clone() if torch.is_tensor(v) else deepcopy(v)
                    for k, v in cache.items()}
        constant['parent_feature'][:] = 0
        constant['parent_feature'][:, 30] = 1
        rank0 = fit_receiver(constant, seed=123)
        assert all(f['rank'] == 0 and f['c'] == 0 for f in rank0['folds'])

        bank = load_receiver(payload)
        sample = bank.sample_relations(torch.Generator().manual_seed(94))
        indices = sample['indices']
        assert indices.shape == (32,) and len(set(indices.tolist())) == 32
        assert sample['diagnostics']['valid_relations'] == 2
        student = cache['parent_feature'].detach()[indices].clone().requires_grad_(True)
        loss, info = bank.identity_loss(student, indices, sample)
        initial_expected = sum(r['c'] * .5 * r['alpha'] ** 2 *
                               float(r['prediction'].square().sum())
                               for r in sample['relations']) / 2
        torch.testing.assert_close(loss, loss.new_tensor(initial_expected), rtol=1e-12, atol=1e-14)
        assert all(r['variance_correction'] == 0 and r['control_residual_norm'] == 0
                   for r in info['relations'])
        loss.backward()
        assert student.grad is not None and student.grad.norm() > 0
        assert cache['parent_feature'].grad is None and cache['state'].grad is None
        assert all(not value.requires_grad for f in bank.folds for value in f.values()
                   if torch.is_tensor(value))
        fp32_cache = {k: v.detach().float() if torch.is_tensor(v) and v.is_floating_point()
                      else deepcopy(v) for k,v in cache.items()}
        fp32_bank = load_receiver(fit_receiver(fp32_cache, 123))
        fp32_sample = fp32_bank.sample_relations(torch.Generator().manual_seed(94))
        fp32_features = fp32_cache['parent_feature'][fp32_sample['indices']].clone().requires_grad_(True)
        fp32_loss, fp32_info = fp32_bank.identity_loss(fp32_features, fp32_sample['indices'], fp32_sample)
        fp32_loss.backward()
        assert torch.isfinite(fp32_features.grad).all()
        assert all(r['control_residual_norm'] == 0 and r['variance_correction'] == 0
                   for r in fp32_info['relations'])

        # Every arm gets the same identities even when all R weights are zero.
        disabled = deepcopy(payload)
        for f in disabled['folds']:
            f['c'] = 0.
        other_sample = load_receiver(disabled).sample_relations(torch.Generator().manual_seed(94))
        assert torch.equal(indices, other_sample['indices'])

        # Rotation changes neither full-vector reliability nor its raw error spectrum.
        gids = payload['blocks'][0]['group_ids']
        groups = [deepcopy(payload['groups'][i]) for i in gids]
        for i, g in enumerate(groups):
            g['group_id'] = i
        blocks = [dict(group_ids=list(range(len(groups))))]
        basis = payload['folds'][0]['basis'][:, :2]
        reference = vector_reliability(groups, blocks, basis)
        generator = torch.Generator().manual_seed(501)
        q, _ = torch.linalg.qr(torch.randn(160, 160, generator=generator, dtype=torch.float64))
        rotated = deepcopy(groups)
        for g in rotated:
            for key in ('parent_feature_mean', 'potential'):
                g[key] = q @ g[key]
        rotation = vector_reliability(rotated, blocks, q @ basis)
        for key in ('c', 'mse_pred', 'mse_zero', 'mse_subspace'):
            assert abs(reference[key] - rotation[key]) < 1e-12
        torch.testing.assert_close(torch.tensor(reference['error_spectrum']),
                                   torch.tensor(rotation['error_spectrum']), atol=1e-10, rtol=1e-5)

        # Block scaling and feasibility preserve both the center and every cycle.
        means = torch.zeros(3, 160, dtype=torch.float64); means[:, 5] = .995
        raw = torch.zeros_like(means); raw[:, 0] = torch.tensor([-5., 0., 5.])
        geom = block_potentials(means, raw)
        pot, target = geom['potential'], geom['target_mean']
        assert geom['kappa'] == .2 and 0 < geom['alpha'] < .25
        assert torch.linalg.vector_norm(target, dim=1).max() <= 1 + 1e-12
        torch.testing.assert_close(target.mean(0), means.mean(0), atol=1e-14, rtol=0)
        cycle = (pot[0] - pot[1]) + (pot[1] - pot[2]) + (pot[2] - pot[0])
        assert cycle.abs().max() < 1e-14
        assert max((pot[g] - pot[h]).norm() for g, h in combinations(range(3), 2)) <= 2

        # Exhaust all independent SRSWOR bags: expected loss AND gradient match population.
        theta = torch.randn(7, 160, generator=torch.Generator().manual_seed(801),
                            dtype=torch.float64, requires_grad=True)
        base = torch.randn(7, 160, generator=torch.Generator().manual_seed(802), dtype=torch.float64)
        parent = F.normalize(base, dim=1).detach()
        current = F.normalize(base + .1 * theta, dim=1)
        residual = current - parent
        prediction = torch.randn(160, generator=torch.Generator().manual_seed(803), dtype=torch.float64) * .01
        terms = []
        for a, b in product(combinations(range(3), 2), combinations(range(3, 7), 2)):
            value, _ = finite_population_loss(residual[list(a)], residual[list(b)], prediction,
                                              alpha=.25, c=.7, population_g=3, population_h=4)
            terms.append(value)
        empirical = torch.stack(terms).mean()
        expected = .7 / 2 * (residual[:3].mean(0) - residual[3:].mean(0) + .25 * prediction).square().sum()
        grad_empirical = torch.autograd.grad(empirical, theta, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected, theta)[0]
        torch.testing.assert_close(empirical, expected, rtol=1e-11, atol=1e-14)
        torch.testing.assert_close(grad_empirical, grad_expected, rtol=1e-10, atol=1e-14)

        # The signed estimate must retain a negative value and a real variance gradient.
        r = torch.zeros(2, 160, dtype=torch.float64); r[:, 0] = torch.tensor([-1., 1.])
        r.requires_grad_(True)
        negative, parts = finite_population_loss(r, torch.zeros_like(r), torch.zeros(160),
                                                alpha=.25, c=1., population_g=3, population_h=3)
        assert negative < 0 and parts['variance_correction'].requires_grad
        assert torch.autograd.grad(negative, r)[0].norm() > 0

        # A rank-1 monitor would miss e2 entirely; the full-vector loss must see it.
        escape = torch.zeros(2, 160, dtype=torch.float64); escape[:, 1] = 1
        escaped_loss, _ = finite_population_loss(escape, torch.zeros_like(escape), torch.zeros(160),
                                                alpha=.25, c=1., population_g=2, population_h=2)
        assert float(escaped_loss) == .5 and float(escape[:, 0].square().sum()) == 0

        covariance = trusted_covariance(payload)
        assert covariance.shape == (160, 160)
        torch.testing.assert_close(covariance, covariance.T)
        assert torch.linalg.eigvalsh(covariance).min() >= -1e-12
        no_alpha = deepcopy(payload)
        for block in no_alpha['blocks']:
            block['alpha'] = 0.
        torch.testing.assert_close(covariance, trusted_covariance(no_alpha), atol=0, rtol=0)
        assert torch.count_nonzero(trusted_covariance(disabled)) == 0

        # Wrong dimensional cache and mismatched online physical indices are rejected.
        bad = dict(cache, parent_feature=cache['parent_feature'][:, :2])
        try:
            fit_receiver(bad, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-joint160 cache accepted')
        try:
            bank.identity_loss(student.detach(), indices.flip(0), sample)
        except ValueError:
            pass
        else:
            raise AssertionError('Misaligned physical indices accepted')
        return dict(status='PASS', target_access=False, synthetic_ids=payload['source_count'],
                    rank2_recovery_c=[f['c'] for f in payload['folds']], rank0_valid=True,
                    heldout_leakage=False, vector_rotation_invariant=True,
                    finite_bag_pairs=len(terms), finite_loss_error=float((empirical-expected).detach().abs()),
                    finite_gradient_max_error=float((grad_empirical-grad_expected).abs().max()),
                    negative_estimate=float(negative.detach()), projection_escape_loss=float(escaped_loss.detach()),
                    block_alpha=geom['alpha'], block_kappa=geom['kappa'],
                    matched_sampler=True, parent_and_r_frozen=True,
                    equal_block_group_weighting=True, fp32_initial_control_exact=True)
    finally:
        torch.set_num_threads(old_threads)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
