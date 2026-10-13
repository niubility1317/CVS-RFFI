"""Frozen source RX transfer matrices and differentiable one-step MLDG primitive."""
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.func import functional_call

from . import design as d
from experiments.cvs_source_pseudotarget_health.frozen import snapshot, assert_unchanged, preserve_rng

RECIPE = dict(blocks=['t1.dw.weight', 't2.dw.weight'], alphas=[.001, .01],
              primary_block='t1.dw.weight', primary_alpha=.01,
              block_scope='identity backbone t1/t2 depthwise kernel; domain backbone excluded',
              repeats_per_day=4, random_partitions=4, packets_per_TX_RX=3,
              days=[1, 2, 3], receivers=[1, 3, 4, 6, 8],
              update_folds=[0, 1], test_folds=[2], grouping='TX/day-balanced true RX versus matched within-TX/day permutations',
              virtual_step='one raw SGD step phi-alpha*dCE_a/dphi; no direction normalization or old critic',
              model_mode='fixed eval; identity buffers/parameters/RNG unchanged',
              metric='six-class CE computed from float64 logits for stable differences; source only',
              identity_updates=0, V_used=False, U_used=False)
THRESHOLDS = dict(harm_CE_change=1e-5, harm_fraction_excess=.10,
                  real_mean_change=1e-5, excess_mean_change=1e-5,
                  minimum_repeated_days=2, minimum_repeats=3,
                  actual_update_mean_change_max=0.)


def block_names(model):
    named = dict(model.named_parameters()); result = {}
    for suffix in RECIPE['blocks']:
        matches = [name for name in named if name.endswith('.' + suffix) or name == suffix]
        # OriginalIdentity contains a second domain backbone with the same
        # t1/t2 suffixes. The registered intervention belongs to the identity
        # encoder captured by time_down/t2.pool, never the domain branch.
        identity_matches = [name for name in matches if '.id_backbone.' in name and '.dom_backbone.' not in name]
        if identity_matches:
            matches = identity_matches
        if len(matches) != 1:
            raise ValueError('Exactly one native small parameter block required: ' + suffix)
        result[suffix] = matches[0]
    return result


def mldg_objective(module, xa, ya, xb, yb, selected_names, alpha, lam=1., first_order=False):
    """Functional raw-SGD step; full outer graph, including mixed Hessians.

    This primitive is verified by toy mathematics, not used to update the
    identity in the source diagnostic. FO explicitly detaches the inner grad.
    """
    parameters = dict(module.named_parameters()); buffers = {k: v.detach().clone() for k, v in module.named_buffers()}
    if not selected_names or any(name not in parameters or not parameters[name].requires_grad for name in selected_names):
        raise ValueError('Registered active virtual parameter block required')
    if not alpha > 0 or not lam >= 0:
        raise ValueError('Positive fixed inner step and nonnegative outer weight required')
    inner = F.cross_entropy(functional_call(module, (parameters, buffers), (xa,), strict=True), ya)
    gradients = torch.autograd.grad(inner, [parameters[name] for name in selected_names],
                                    create_graph=not first_order, retain_graph=True)
    adapted = dict(parameters)
    for name, gradient in zip(selected_names, gradients):
        adapted[name] = parameters[name] - alpha * (gradient.detach() if first_order else gradient)
    outer = F.cross_entropy(functional_call(module, (adapted, buffers), (xb,), strict=True), yb)
    return inner + lam * outer, dict(inner=inner, outer=outer, inner_gradients=gradients,
                                   second_order=not first_order, other_parameters_live_in_outer=True)


def sample_episode(L, folds, day, repeat, seed):
    n = RECIPE['packets_per_TX_RX']; update = []; test = []
    for rx in d.RX:
        ua = []; tb = []
        for tx in range(6):
            group = (L['y'] == tx) & (L['rx'] == rx) & (L['day'] == day)
            fit = (group & (folds != 2)).nonzero().flatten()
            hold = (group & (folds == 2)).nonzero().flatten()
            if min(len(fit), len(hold)) < n * RECIPE['repeats_per_day']:
                raise ValueError('Insufficient source cell for independent update/test')
            generator = torch.Generator().manual_seed(seed + day * 100003 + rx * 1597 + tx * 31337)
            start, end = repeat * n, (repeat + 1) * n
            ua.append(fit[torch.randperm(len(fit), generator=generator)[start:end]])
            tb.append(hold[torch.randperm(len(hold), generator=generator)[start:end]])
        update.append(torch.cat(ua)); test.append(torch.cat(tb))
    update = torch.stack(update); test = torch.stack(test)
    if set(update.flatten().tolist()) & set(test.flatten().tolist()):
        raise ValueError('Virtual update/test physical IDs overlap')
    return update, test


def randomized_groups(indices, L, seed):
    n = RECIPE['packets_per_TX_RX']; flattened = indices.flatten(); result = [[] for _ in d.RX]
    generator = torch.Generator().manual_seed(seed)
    for tx in range(6):
        pool = flattened[L['y'][flattened] == tx]
        if len(pool) != len(d.RX) * n:
            raise ValueError('Random controls must preserve exact TX quota')
        order = pool[torch.randperm(len(pool), generator=generator)].reshape(len(d.RX), n)
        for group in range(len(d.RX)):
            result[group].append(order[group])
    value = torch.stack([torch.cat(parts) for parts in result])
    if sorted(value.flatten().tolist()) != sorted(indices.flatten().tolist()):
        raise ValueError('Random control changed the source packet pool')
    return value


def _class_losses(logits, labels):
    if logits.shape != (len(labels), 6) or not torch.isfinite(logits).all():
        raise ValueError('Finite ordinary six-class source logits required')
    return F.cross_entropy(logits.double(), labels, reduction='none')


def transfer_matrices(model, L, update, test, name, alphas, device):
    parameters = dict(model.named_parameters()); buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    if name not in parameters or model.training or any(p.requires_grad for p in parameters.values()):
        raise ValueError('Frozen eval identity and exact source small block required')
    uindices = update.flatten(); tind = test.flatten()
    ux = L['x'][uindices].to(device); uy = L['y'][uindices].to(device)
    tx = L['x'][tind].to(device); ty = L['y'][tind].to(device)
    m = update.shape[0]; quota = update.shape[1]
    with torch.no_grad():
        base_logits = functional_call(model, (parameters, buffers), (tx,), strict=True)
        base_loss = _class_losses(base_logits, ty).reshape(m, quota).mean(1).cpu()
    matrices = {float(alpha): torch.empty(m, m, dtype=torch.float64) for alpha in alphas}
    train_changes = {float(alpha): [] for alpha in alphas}; gradients = []; displacements = {float(alpha): [] for alpha in alphas}
    for a in range(m):
        phi = parameters[name].detach().clone().requires_grad_(True)
        local = dict(parameters); local[name] = phi
        x = ux[a * quota:(a + 1) * quota]; y = uy[a * quota:(a + 1) * quota]
        logits = functional_call(model, (local, buffers), (x,), strict=True)
        inner = _class_losses(logits, y).mean()
        gradient = torch.autograd.grad(inner, phi, create_graph=False)[0].detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Source small-block gradient is nonfinite')
        gradients.append(float(gradient.norm()))
        for alpha in alphas:
            virtual = dict(parameters); virtual[name] = (phi - alpha * gradient).detach()
            with torch.no_grad():
                after = functional_call(model, (virtual, buffers), (tx,), strict=True)
                current = _class_losses(after, ty).reshape(m, quota).mean(1).cpu()
                own = functional_call(model, (virtual, buffers), (x,), strict=True)
                own_change = float(_class_losses(own, y).mean() - inner.detach())
            matrices[float(alpha)][a] = current - base_loss
            train_changes[float(alpha)].append(own_change)
            displacements[float(alpha)].append(float(alpha * gradient.norm() / phi.detach().norm().clamp_min(1e-12)))
    return dict(baseline_CE=base_loss, G=matrices, actual_update_CE_change=train_changes,
                gradient_norms=gradients, relative_block_displacements=displacements)


def _off_diagonal(matrix):
    return matrix[~torch.eye(len(matrix), dtype=torch.bool)]


def summarize(records):
    cases = []
    for block in RECIPE['blocks']:
        for alpha in RECIPE['alphas']:
            selected = [r for r in records if r['block'] == block and r['alpha'] == alpha]
            actual = [r for r in selected if r['partition'] == 'RX']
            control = [r for r in selected if r['partition'] == 'RANDOM']
            real = torch.cat([_off_diagonal(r['G']) for r in actual])
            random = torch.cat([_off_diagonal(r['G']) for r in control])
            real_harm = float((real > THRESHOLDS['harm_CE_change']).double().mean())
            random_harm = float((random > THRESHOLDS['harm_CE_change']).double().mean())
            day_details = []; repeat_details = []
            for label, axis, values, destination in [('day', 'day', d.DAYS, day_details), ('repeat', 'repeat', range(RECIPE['repeats_per_day']), repeat_details)]:
                for value in values:
                    aa = torch.cat([_off_diagonal(r['G']) for r in actual if r[axis] == value])
                    bb = torch.cat([_off_diagonal(r['G']) for r in control if r[axis] == value])
                    destination.append({label: int(value), 'real_mean': float(aa.mean()), 'random_mean': float(bb.mean()),
                                        'mean_excess': float(aa.mean() - bb.mean()),
                                        'harm_fraction_excess': float((aa > THRESHOLDS['harm_CE_change']).double().mean() - (bb > THRESHOLDS['harm_CE_change']).double().mean())})
            own = torch.tensor([change for r in actual for change in r['actual_update_CE_change']], dtype=torch.float64)
            criteria = dict(actual_update_is_helpful=float(own.mean()) < THRESHOLDS['actual_update_mean_change_max'],
                            real_RX_updates_harm_other_RX=float(real.mean()) > THRESHOLDS['real_mean_change'],
                            mean_excess_over_random=float(real.mean() - random.mean()) > THRESHOLDS['excess_mean_change'],
                            harm_rate_excess_over_random=real_harm - random_harm >= THRESHOLDS['harm_fraction_excess'],
                            repeated_across_days=sum(r['mean_excess'] > 0 and r['harm_fraction_excess'] > 0 for r in day_details) >= THRESHOLDS['minimum_repeated_days'],
                            repeated_across_batches=sum(r['mean_excess'] > 0 and r['harm_fraction_excess'] > 0 for r in repeat_details) >= THRESHOLDS['minimum_repeats'])
            cases.append(dict(block=block, alpha=alpha, real_off_diagonal_count=len(real), random_off_diagonal_count=len(random),
                              real_mean=float(real.mean()), random_mean=float(random.mean()), mean_excess=float(real.mean() - random.mean()),
                              real_harm_fraction=real_harm, random_harm_fraction=random_harm,
                              actual_update_mean_change=float(own.mean()), day=day_details, repeat=repeat_details,
                              criteria=criteria, passed=all(criteria.values())))
    return cases


def diagnose(ctx, output: Path, device):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model = ctx['model']; L = ctx['L']; names = block_names(model); before = snapshot(model)
    records = []; episodes = []
    with preserve_rng(device):
        for day in d.DAYS:
            for repeat in range(RECIPE['repeats_per_day']):
                update, test = sample_episode(L, ctx['folds'], day, repeat, ctx['seed'])
                partitions = [('RX', -1, update, test)]
                for control in range(RECIPE['random_partitions']):
                    seed = ctx['seed'] + day * 100003 + repeat * 7919 + control * 31337 + 77777
                    partitions.append(('RANDOM', control, randomized_groups(update, L, seed), randomized_groups(test, L, seed + 1597)))
                episodes.append(dict(day=day, repeat=repeat, actual_update=update, actual_test=test,
                                     partitions=[dict(kind=kind, control=control, update=ua, test=tb) for kind, control, ua, tb in partitions]))
                for block, name in names.items():
                    for kind, control, ua, tb in partitions:
                        value = transfer_matrices(model, L, ua, tb, name, RECIPE['alphas'], device)
                        for alpha in RECIPE['alphas']:
                            records.append(dict(block=block, parameter_name=name, alpha=alpha, day=day, repeat=repeat,
                                                partition=kind, control=control, update_indices=ua, test_indices=tb,
                                                baseline_CE=value['baseline_CE'], G=value['G'][alpha],
                                                actual_update_CE_change=value['actual_update_CE_change'][alpha],
                                                gradient_norms=value['gradient_norms'], relative_block_displacements=value['relative_block_displacements'][alpha]))
                        assert_unchanged(model, before)
                print(f'SOURCE_UPDATE_MATRIX day={day} repeat={repeat} blocks=2 partitions=5', flush=True)
    frozen = assert_unchanged(model, before); cases = summarize(records)
    primary = next(r for r in cases if r['block'] == RECIPE['primary_block'] and r['alpha'] == RECIPE['primary_alpha'])
    failed = [key for key, value in primary['criteria'].items() if not value]
    result = dict(direction='update', recipe=RECIPE, thresholds=THRESHOLDS, source_only=True, target_access=False,
                  U_truth_read=False, V_gradients=False, identity_updates=0, cases=cases, matrices=len(records),
                  applicability=dict(passed=primary['passed'], criteria=primary['criteria'],
                      reason='Fixed primary t1/.01 has reproducible actual-RX update harm beyond random grouping' if primary['passed'] else 'Fixed primary source criterion absent: ' + ', '.join(failed)),
                  primary_candidate=dict(block=RECIPE['primary_block'], alpha=RECIPE['primary_alpha']),
                  background_cases_cannot_promote=True, frozen=frozen,
                  audit_file='source_update_matrices_raw.pt',
                  scope='Finite one-step frozen-source diagnostic; not AdamW replay or an actual MLDG training result')
    with (output / 'source_update_matrices_raw.pt').open('xb') as stream:
        torch.save(dict(recipe=RECIPE, thresholds=THRESHOLDS, source_L_ids=L['ids'], source_L_y=L['y'], source_L_rx=L['rx'], source_L_day=L['day'],
                        folds=ctx['folds'], block_names=names, episodes=episodes, records=records, identity_parameters_buffers_unchanged=frozen), stream)
    d.write(output / 'summary.json', result)
    return result
