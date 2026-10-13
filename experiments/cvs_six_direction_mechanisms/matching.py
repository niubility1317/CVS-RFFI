"""Fixed source-only state matching; exact-K capacity constrained partial plans."""
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import design as d

RECIPE = dict(samples_per_side=10, matched_pairs=5, rho=.5, capacity=.1,
              repeats=2, repeat_physical_ids_disjoint=True, folds={'fit': [0, 1], 'hold': [2]},
              matching_state_dimensions=list(range(8)), validation_state_dimensions=[8, 9, 10, 11],
              primary_validation_dimensions=[10, 11], reference_beta=.1, relative_margin=.2,
              arms=['RANDOM', 'GREEDY_NN', 'PARTIAL_OT'], layers=['time_down', 't2.pool', 'joint160'],
              reference='frozen CE30 joint160; detached, same reference for all plans',
              negative_rule='one fixed nearest-state different-TX candidate per anchor, same day/fold/repeat',
              state='received-envelope proxy; not identified transmitted content',
              validation='dimensions excluded from matching cost, not independent physical truth')
THRESHOLDS = dict(held_lag_validation_reduction=.10, repeated_bins_per_TX_day=2,
                  receivers_per_repeated_bin=3, held_packets_per_RX_bin=2,
                  hard_coverage_relative_to_uniform=.5, constraint_tolerance=1e-8)


def hungarian(cost):
    """Square minimum assignment, deterministic ties, no external solver needed."""
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or np.isnan(c).any():
        raise ValueError('Square non-NaN assignment cost required')
    n = len(c); u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64); way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minimum = np.full(n + 1, np.inf); used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = c[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current; way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]; j1 = j
            if not np.isfinite(delta):
                raise ValueError('Partial assignment is infeasible')
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    rows = p[1:] - 1; columns = np.arange(n)
    if not np.isfinite(c[rows, columns]).all():
        raise ValueError('Assignment used a forbidden edge')
    return rows, columns


def partial_plan(cost, k):
    """Uniform caps 1/n and exact nonzero mass k/n, without entropy regularization."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or not torch.isfinite(cost).all():
        raise ValueError('Finite square real pairing costs required')
    n = len(cost)
    if not isinstance(k, int) or not 0 < k <= n:
        raise ValueError('Fixed nonzero matched quantity required')
    padded = np.zeros((2 * n - k, 2 * n - k), dtype=np.float64)
    padded[:n, :n] = cost.detach().cpu().double().numpy()
    padded[n:, n:] = np.inf
    rows, columns = hungarian(padded)
    keep = (rows < n) & (columns < n)
    plan = torch.zeros(n, n, dtype=torch.float64)
    plan[torch.as_tensor(rows[keep]), torch.as_tensor(columns[keep])] = 1 / n
    validate_plan(plan, n, k)
    return plan


def greedy_plan(cost, k):
    n = len(cost); available = cost.detach().cpu().double().clone()
    plan = torch.zeros(n, n, dtype=torch.float64)
    if not 0 < k <= n or available.shape != (n, n) or not torch.isfinite(available).all():
        raise ValueError('Finite exact-K nearest-neighbor problem required')
    for _ in range(k):
        location = int(available.argmin()); i, j = divmod(location, n)
        plan[i, j] = 1 / n; available[i] = torch.inf; available[:, j] = torch.inf
    validate_plan(plan, n, k)
    return plan


def random_plan(n, k, generator):
    if not 0 < k <= n:
        raise ValueError('Nonzero bounded random matching quantity required')
    rows = torch.randperm(n, generator=generator)[:k]
    columns = torch.randperm(n, generator=generator)[:k]
    plan = torch.zeros(n, n, dtype=torch.float64); plan[rows, columns] = 1 / n
    validate_plan(plan, n, k)
    return plan


def validate_plan(plan, n, k):
    tolerance = THRESHOLDS['constraint_tolerance']
    if plan.shape != (n, n) or not torch.isfinite(plan).all() or (plan < 0).any():
        raise ValueError('Finite nonnegative partial plan required')
    if (plan.sum(1) > 1 / n + tolerance).any() or (plan.sum(0) > 1 / n + tolerance).any():
        raise ValueError('Partial plan exceeds sample capacity')
    if abs(float(plan.sum()) - k / n) > tolerance or int((plan > 0).sum()) != k:
        raise ValueError('Exact positive K and fixed nonzero rho required')
    if not torch.allclose(plan[plan > 0], torch.full((k,), 1 / n, dtype=plan.dtype), atol=tolerance, rtol=0):
        raise ValueError('Uniform equal-cost pair weights required')
    return True


def candidate_pools(L, folds, seed):
    result = {}; n = RECIPE['samples_per_side']; repeats = RECIPE['repeats']
    for section, allowed in RECIPE['folds'].items():
        for tx in range(6):
            for day in d.DAYS:
                for rx in d.RX:
                    mask = (L['y'] == tx) & (L['day'] == day) & (L['rx'] == rx)
                    mask &= torch.isin(folds, torch.tensor(allowed))
                    indices = mask.nonzero().flatten()
                    if len(indices) < n * repeats:
                        raise ValueError('Insufficient disjoint source cell packets for fixed repeats')
                    generator = torch.Generator().manual_seed(seed + tx * 100003 + day * 7919 + rx * 1597 + (0 if section == 'fit' else 31337))
                    shuffled = indices[torch.randperm(len(indices), generator=generator)[:n * repeats]]
                    for repeat in range(repeats):
                        result[section, tx, day, rx, repeat] = shuffled[repeat * n:(repeat + 1) * n]
    return result


def _negative_indices(q, L, pools, section, day, repeat):
    bank = torch.cat([pools[section, tx, day, rx, repeat] for tx in range(6) for rx in d.RX])
    result = {}
    for tx in range(6):
        anchors = bank[L['y'][bank] == tx]; other = bank[L['y'][bank] != tx]
        distances = torch.cdist(q[anchors, :8].double(), q[other, :8].double()).square() / 8
        chosen = other[distances.argmin(1)]
        result.update({int(a): int(b) for a, b in zip(anchors, chosen)})
    return result


def _features(ctx):
    fit = ctx['folds'] != 2; result = {}
    for layer in RECIPE['layers']:
        value = ctx['features']['L_s'][layer].double()
        mean = value[fit].mean(0); scale = value[fit].std(0, unbiased=False).clamp_min(1e-4)
        result[layer] = F.normalize((value - mean) / scale, dim=1, eps=1e-4)
    return result


def _state_repetition(ctx):
    raw = ctx['state_q_raw']; fit = ctx['folds'] != 2; hold = ctx['folds'] == 2
    thresholds = torch.quantile(raw[fit, 10].float(), torch.tensor([.25, .5, .75]))
    bins = torch.bucketize(raw[:, 10].contiguous(), thresholds)
    records = []
    for tx in range(6):
        for day in d.DAYS:
            counts = torch.zeros(4, 5, dtype=torch.long)
            for state in range(4):
                for ri, rx in enumerate(d.RX):
                    counts[state, ri] = int((hold & (ctx['L']['y'] == tx) & (ctx['L']['day'] == day) & (ctx['L']['rx'] == rx) & (bins == state)).sum())
            repeated = int(((counts >= THRESHOLDS['held_packets_per_RX_bin']).sum(1) >= THRESHOLDS['receivers_per_repeated_bin']).sum())
            records.append(dict(TX=tx, day=day, held_counts=counts.tolist(), repeated_bins=repeated))
    return bins, dict(activity='normalized-envelope lag1 energy, quartile thresholds fitted on L folds0/1',
                      thresholds=thresholds.tolist(), records=records,
                      all_TX_day_multistate=all(r['repeated_bins'] >= THRESHOLDS['repeated_bins_per_TX_day'] for r in records))


def _q_receiver_information(ctx):
    from experiments.cvs_source_pseudotarget_health.probes import evaluate_probe, rx_labels
    result = []; states = []; fit = ctx['folds'] != 2; hold = ctx['folds'] == 2
    for tx in (None, *range(6)):
        group = torch.ones_like(fit) if tx is None else ctx['L']['y'] == tx
        features = ctx['state_q'][fit & group].float()
        mean = features.mean(0); scale = features.std(0, unbiased=False).clamp_min(1e-4)
        labels = rx_labels(ctx['L']['rx'][fit & group]); standardized = (features - mean) / scale
        # CPU default generator only: do not seed any CUDA generator while
        # diagnosing an already frozen model on the selected GPU.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(ctx['seed'] + 31337)
            classifier = torch.nn.Linear(12, 5)
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=.01, weight_decay=.001)
            for _ in range(100):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(classifier(standardized), labels)
                loss.backward(); optimizer.step()
        classifier.eval().requires_grad_(False)
        value = evaluate_probe(classifier, mean, scale, ctx['state_q'][hold & group], ctx['L']['rx'][hold & group], 'cpu')
        result.append(dict(TX=tx, **value))
        states.append(dict(TX=tx, model=classifier.state_dict(), mean=mean, scale=scale,
                           fit_indices=(fit & group).nonzero().flatten(), held_indices=(hold & group).nonzero().flatten()))
    return dict(records=result, scope='RX leakage warning only; source descriptor is not transmitted-input ground truth',
                identity_updates=0, steps_per_probe=100), states


def diagnose(ctx, output: Path, device):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    L = ctx['L']; q = ctx['state_q'].detach().cpu(); folds = ctx['folds'].cpu()
    if q.shape != (len(L['ids']), 12) or not torch.isfinite(q).all() or len(set(L['ids'])) != len(L['ids']):
        raise ValueError('Finite source-state matrix and unique L IDs required')
    pools = candidate_pools(L, folds, ctx['seed']); normalized = _features(ctx)
    reference = F.normalize(ctx['features']['L_s']['joint160'].double(), dim=1, eps=1e-4)
    probability = ctx['probabilities']['L_s']; nll = -probability[torch.arange(len(L['ids'])), L['y']].clamp_min(1e-12).log()
    hard = torch.zeros(len(L['ids']), dtype=torch.bool)
    for tx in range(6):
        for day in d.DAYS:
            group = (L['y'] == tx) & (L['day'] == day)
            threshold = torch.quantile(nll[group & (folds != 2)], .75)
            hard[group] = nll[group] >= threshold
    bins, repeated = _state_repetition(ctx)
    records = []; counts = {section: {arm: dict(validation_sum=0., lag_validation_sum=0., pairs=0, available=0, selected=0, available_hard=0, selected_hard=0) for arm in RECIPE['arms']} for section in RECIPE['folds']}
    coverage = {section: {arm: torch.zeros(len(L['ids']), dtype=torch.long) for arm in RECIPE['arms']} for section in RECIPE['folds']}
    negatives = {(section, day, repeat): _negative_indices(q, L, pools, section, day, repeat)
                 for section in RECIPE['folds'] for day in d.DAYS for repeat in range(RECIPE['repeats'])}
    group_index = 0
    for section in RECIPE['folds']:
        for tx in range(6):
            for day in d.DAYS:
                for ra, rb in combinations(d.RX, 2):
                    for repeat in range(RECIPE['repeats']):
                        ia = pools[section, tx, day, ra, repeat]; ib = pools[section, tx, day, rb, repeat]
                        state_cost = torch.cdist(q[ia, :8].double(), q[ib, :8].double()).square() / 8
                        reference_cost = torch.cdist(reference[ia], reference[ib]).square()
                        cost = state_cost + RECIPE['reference_beta'] * reference_cost
                        generator = torch.Generator().manual_seed(ctx['seed'] + 104729 * group_index + 1777)
                        plans = dict(RANDOM=random_plan(10, 5, generator), GREEDY_NN=greedy_plan(cost, 5), PARTIAL_OT=partial_plan(cost, 5))
                        neg = torch.tensor([negatives[section, day, repeat][int(i)] for i in ia])
                        available = torch.cat((ia, ib))
                        result = dict(section=section, TX=tx, day=day, RX_a=ra, RX_b=rb, repeat=repeat,
                                      candidate_a=ia, candidate_b=ib, negative_for_each_anchor=neg,
                                      state_cost=state_cost, reference_cost=reference_cost, combined_cost=cost, plans={})
                        for arm, plan in plans.items():
                            rows, columns = (plan > 0).nonzero(as_tuple=True)
                            left, right, minus = ia[rows], ib[columns], neg[rows]
                            selected = torch.cat((left, right)); coverage[section][arm].index_add_(0, selected, torch.ones(len(selected), dtype=torch.long))
                            validation = (q[left, 8:] - q[right, 8:]).double().square()
                            terms = {}
                            for layer, value in normalized.items():
                                positive_distance = (value[left] - value[right]).square().sum(1)
                                negative_distance = (value[left] - value[minus]).square().sum(1)
                                hinge = (RECIPE['relative_margin'] + positive_distance - negative_distance).clamp_min(0)
                                terms[layer] = dict(positive_distance=positive_distance, negative_distance=negative_distance,
                                                    hinge=hinge, average=float(hinge.mean()), active_fraction=float((hinge > 0).float().mean()))
                            aggregate = counts[section][arm]
                            aggregate['validation_sum'] += float(validation.mean(1).sum())
                            aggregate['lag_validation_sum'] += float(validation[:, 2:].mean(1).sum())
                            aggregate['pairs'] += len(rows); aggregate['available'] += len(available); aggregate['selected'] += len(selected)
                            aggregate['available_hard'] += int(hard[available].sum()); aggregate['selected_hard'] += int(hard[selected].sum())
                            result['plans'][arm] = dict(P=plan, anchor=left, positive=right, negative=minus,
                                                        validation_dimensions_square=validation, triplets=terms,
                                                        selected_hard_count=int(hard[selected].sum()))
                        records.append(result); group_index += 1
    summaries = {}
    for section, arms in counts.items():
        summaries[section] = {}
        for arm, aggregate in arms.items():
            coverage_rate = aggregate['selected'] / aggregate['available']
            hard_rate = aggregate['selected_hard'] / max(aggregate['available_hard'], 1)
            summaries[section][arm] = dict(**aggregate,
                validation_mean=aggregate['validation_sum'] / aggregate['pairs'],
                lag_validation_mean=aggregate['lag_validation_sum'] / aggregate['pairs'],
                occurrence_coverage=coverage_rate, hard_occurrence_coverage=hard_rate,
                hard_coverage_relative_to_uniform=hard_rate / coverage_rate,
                distinct_selected_L=int((coverage[section][arm] > 0).sum()),
                selected_activity_bin_occurrences=torch.bincount(bins, weights=coverage[section][arm].float(), minlength=4).tolist())
    random = summaries['hold']['RANDOM']['lag_validation_mean']
    improvements = {arm: 1 - summaries['hold'][arm]['lag_validation_mean'] / max(random, 1e-12)
                    for arm in ('GREEDY_NN', 'PARTIAL_OT')}
    criteria = dict(nontrivial_repeated_observable_states=repeated['all_TX_day_multistate'],
                    held_lag_validation_better_than_random=all(v >= THRESHOLDS['held_lag_validation_reduction'] for v in improvements.values()),
                    hard_modes_not_systematically_excluded=all(summaries['hold'][arm]['hard_coverage_relative_to_uniform'] >= THRESHOLDS['hard_coverage_relative_to_uniform'] for arm in ('GREEDY_NN', 'PARTIAL_OT')),
                    exact_positive_mass_and_caps=True, same_K_candidates_negatives_all_arms=True,
                    repeats_use_disjoint_physical_IDs=True)
    failed = [key for key, value in criteria.items() if not value]
    q_information, q_head_states = _q_receiver_information(ctx)
    result = dict(direction='matching', recipe=RECIPE, thresholds=THRESHOLDS, groups=len(records),
                  source_only=True, target_access=False, U_truth_read=False, identity_updates=0,
                  source_state_repetition=repeated, q_RX_information=q_information,
                  summaries=summaries, held_lag_validation_reduction=improvements,
                  applicability=dict(passed=not failed, criteria=criteria,
                      reason='Observable-state partial matching warrants a fixed training falsification; no pure-content/RX claim' if not failed else 'Source prerequisites absent: ' + ', '.join(failed)),
                  audit_file=(output / 'source_matching_raw.pt').name,
                  caveat='Received-envelope states and frozen reference may contain RX/TX information; excluded dimensions are not physical content labels')
    with (output / 'source_matching_raw.pt').open('xb') as stream:
        torch.save(dict(recipe=RECIPE, thresholds=THRESHOLDS, L_ids=L['ids'], L_y=L['y'], L_rx=L['rx'], L_day=L['day'],
                        folds=folds, q=q, q_raw=ctx['state_q_raw'], state_mean=ctx['state_mean'], state_scale=ctx['state_scale'],
                        pools=pools, activity_bins=bins, hard=hard, frozen_source_NLL=nll,
                        normalized_features=normalized, normalized_reference=reference, records=records, coverage=coverage,
                        q_receiver_head_states=q_head_states), stream)
    d.write(output / 'summary.json', result)
    return result
