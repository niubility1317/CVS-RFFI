"""Source-only cross-TX state residuals and real-bag margin repair.

The receiver predictors never read IQ, identity embeddings or current losses.
Their fitted predictions and reliability are immutable during identity training.
Sampling uses only legal source groups, independently of those predictions.
"""
from collections import defaultdict
from itertools import combinations
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


FOLDS = ((0, 1), (2, 3), (4, 5))
STATE_DIM = 40
QUALITY_COLUMN = 32  # signal_state(): 32 PSD coordinates, then repeat coherence.
BAG_SIZE = 8
PLANNED_RELATIONS = 2
FIT_STEPS = 200
FIT_LR = .002
EPS_MSE = 1e-8
DIRECTION_TIE = 1e-4
SCHEMA = 'stage2_state_receiver_crossfit_v1'


def one_sided_huber(value):
    positive = value.clamp_min(0.)
    return torch.where(positive <= 1., .5 * positive.square(), positive - .5)


class StateReceiver(nn.Module):
    """40 physical scalars -> 48 -> 48 -> one signed margin residual."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(40, 48), nn.SiLU(),
                                 nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))

    def forward(self, state):
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError('Receiver input must contain only [N,40] physical state')
        return self.net(state).squeeze(-1)


def _source(cache):
    state = torch.as_tensor(cache['state']).detach().cpu().float()
    n = len(state)
    if state.shape != (n, STATE_DIM) or not n or not torch.isfinite(state).all():
        raise ValueError('Nonempty finite [N,40] L_s state cache required')
    data = dict(state=state)
    for name in ('y', 'rx', 'day'):
        original = torch.as_tensor(cache[name]).detach().cpu()
        if original.shape != (n,) or not torch.equal(original, original.long()):
            raise ValueError(name + ' must be a one-dimensional integer source role')
        data[name] = original.long()
    if set(data['y'].tolist()) != set(range(6)):
        raise ValueError('Registered receiver folds require source TX labels 0 through 5')
    data['ids'] = list(map(str, cache['ids']))
    if len(data['ids']) != n or len(set(data['ids'])) != n:
        raise ValueError('Source cache must contain globally unique physical IDs')
    margin = torch.as_tensor(cache['parent_margin']).detach().cpu().float()
    if margin.shape != (n,) or not torch.isfinite(margin).all():
        raise ValueError('Finite fixed-scale parent_margin[N] required')
    data['parent_margin'] = margin
    correct = torch.as_tensor(cache['parent_correct']).detach().cpu()
    if correct.shape != (n,):
        raise ValueError('parent_correct[N] required')
    if ((state[:, QUALITY_COLUMN] < 0) | (state[:, QUALITY_COLUMN] > 1)).any():
        raise ValueError('state[32] must be repeat coherence in [0,1]')
    return data


def _build_groups(data):
    """Merge fixed adjacent quality intervals, never TX/day, until supported.

    For each TX/day choose the finest of 4, 2, 1 equal coherence bins for
    which every occupied bin has >=2 RX with >=8 distinct physical IDs.
    RX groups with fewer than 8 IDs are excluded and counted. At one bin,
    an unsupported TX/day is explicitly absent rather than fabricated.
    """
    memberships = defaultdict(list)
    for i, key in enumerate(zip(data['y'].tolist(), data['day'].tolist())):
        memberships[key].append(i)
    groups, blocks, decisions = [], [], []
    for (tx, day), indices in sorted(memberships.items()):
        selected = None
        for bins in (4, 2, 1):
            trial = defaultdict(lambda: defaultdict(list))
            for i in indices:
                q = min(int(float(data['state'][i, QUALITY_COLUMN]) * bins), bins - 1)
                trial[q][int(data['rx'][i])].append(i)
            eligible = {q: {rx: rows for rx, rows in rxrows.items() if len(rows) >= BAG_SIZE}
                        for q, rxrows in trial.items()}
            if all(len(rxrows) >= 2 for rxrows in eligible.values()):
                selected = (bins, eligible)
                break
        if selected is None:
            decisions.append(dict(tx=tx, day=day, quality_bins=1,
                                  status='N/A_INSUFFICIENT_RX_OR_UNIQUE_IDS',
                                  distinct_ids=len(indices), used_ids=0))
            continue
        bins, eligible = selected
        used = 0
        for q, rxrows in sorted(eligible.items()):
            block_id = len(blocks)
            gids = []
            for rx, rows in sorted(rxrows.items()):
                values = data['parent_margin'][rows]
                group_id = len(groups)
                groups.append(dict(group_id=group_id, block_id=block_id, tx=tx, day=day,
                                   quality_bins=bins, quality_bin=q, rx=rx, indices=rows,
                                   count=len(rows), parent_mean=float(values.mean()),
                                   parent_variance=float(values.var(unbiased=True))))
                gids.append(group_id)
                used += len(rows)
            center = sum(groups[g]['parent_mean'] for g in gids) / len(gids)
            for g in gids:
                groups[g]['target'] = groups[g]['parent_mean'] - center
            blocks.append(dict(block_id=block_id, tx=tx, day=day,
                               quality_bins=bins, quality_bin=q, group_ids=gids))
        decisions.append(dict(tx=tx, day=day, quality_bins=bins, status='SUPPORTED',
                              distinct_ids=len(indices), used_ids=used))
    tau = torch.full((6,), float('nan'))
    for tx in range(6):
        means = [g['parent_mean'] for g in groups if g['tx'] == tx]
        if means:
            tau[tx] = torch.quantile(torch.tensor(means), .75)
    return groups, blocks, decisions, tau


def _balanced_group_loss(predictions, groups, blocks):
    block_losses = []
    for block in blocks:
        losses = []
        for gid in block['group_ids']:
            group = groups[gid]
            prediction = predictions[group['indices']].mean()
            losses.append(F.smooth_l1_loss(prediction, prediction.new_tensor(group['target']),
                                           beta=1., reduction='none'))
        block_losses.append(torch.stack(losses).mean())
    return torch.stack(block_losses).mean() if block_losses else predictions.sum() * 0.


def _reliability(groups, blocks, predictions):
    """RX-pair average per block, then equal block average for BOTH MSEs."""
    pred_mse, zero_mse, directions, relation_count = [], [], [], 0
    for block in blocks:
        errors, zeros, direction = [], [], []
        for a, b in combinations(block['group_ids'], 2):
            delta = groups[a]['target'] - groups[b]['target']
            predicted = predictions[a] - predictions[b]
            errors.append((predicted - delta) ** 2)
            zeros.append(delta ** 2)
            relation_count += 1
            if abs(delta) > DIRECTION_TIE and abs(predicted) > DIRECTION_TIE:
                direction.append(float((delta > 0) == (predicted > 0)))
        if errors:
            pred_mse.append(sum(errors) / len(errors))
            zero_mse.append(sum(zeros) / len(zeros))
            directions.extend(direction)
    mp = sum(pred_mse) / len(pred_mse) if pred_mse else None
    mz = sum(zero_mse) / len(zero_mse) if zero_mse else None
    c = min(max(1. - mp / mz, 0.), 1.) if mz is not None and mz > EPS_MSE else 0.
    return dict(c=c, mse_pred=mp, mse_zero=mz,
                direction_accuracy=sum(directions) / len(directions) if directions else None,
                direction_non_tie_pairs=len(directions), relation_pairs=relation_count,
                condition_blocks=len(pred_mse), mse_weighting='equal_block_then_equal_RX_pair',
                status='VALID' if mz is not None and mz > EPS_MSE else 'N/A_NO_VALID_VARIATION')


def fit_receiver(cache, seed, fit_steps=FIT_STEPS):
    """Fit three small R models from fixed L_s cache; never forward parent IQ.

    This deterministic fit owns its RNG, does not alter caller torch RNG, and
    records actual steps including zero when a fit fold has no legal blocks.
    """
    if not isinstance(fit_steps, int) or fit_steps < 0:
        raise ValueError('fit_steps must be a nonnegative integer')
    started = time.perf_counter()
    data = _source(cache)
    groups, blocks, decisions, tau = _build_groups(data)
    folds, group_predictions, group_fold = [], {}, {}
    state, y = data['state'], data['y']
    for fold_index, held_out in enumerate(FOLDS):
        fit_tx = tuple(tx for tx in range(6) if tx not in held_out)
        fit_mask = torch.tensor([int(tx) in fit_tx for tx in y])
        center = state[fit_mask].mean(0)
        scale = state[fit_mask].std(0, unbiased=False).clamp_min(1e-6)
        normalized = (state - center) / scale
        fit_blocks = [block for block in blocks if block['tx'] in fit_tx]
        held_blocks = [block for block in blocks if block['tx'] in held_out]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 9137 + fold_index)
            model = StateReceiver()
        optimizer = torch.optim.AdamW(model.parameters(), lr=FIT_LR, weight_decay=0.)
        first_loss = last_loss = None
        actual = 0
        fit_indices = torch.nonzero(fit_mask, as_tuple=True)[0]
        # Keep a full-index output solely for group reductions; holdout states
        # never enter training forward, preprocessing or the fitted loss.
        for _ in range(fit_steps if fit_blocks else 0):
            model.train()
            pred_fit = model(normalized[fit_indices])
            predictions = torch.zeros(len(state), dtype=pred_fit.dtype).index_copy(0, fit_indices, pred_fit)
            loss = _balanced_group_loss(predictions, groups, fit_blocks)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite source group receiver fit loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            first_loss = float(loss.detach()) if first_loss is None else first_loss
            last_loss = float(loss.detach())
            actual += 1
        model.eval().requires_grad_(False)
        with torch.no_grad():
            predicted_samples = model(normalized)
        predictions = {g['group_id']: float(predicted_samples[g['indices']].mean()) for g in groups}
        reliability = _reliability(groups, held_blocks, predictions)
        if not fit_blocks or actual == 0:
            reliability.update(c=0., status='N/A_UNFITTED')
        for group in groups:
            if group['tx'] in held_out:
                gid = group['group_id']
                group_predictions[gid] = predictions[gid]
                group_fold[gid] = fold_index
        folds.append(dict(fold=fold_index, heldout_tx=list(held_out), fit_tx=list(fit_tx),
                          standard_mean=center, standard_scale=scale,
                          standard_fit_ids=[data['ids'][i] for i in fit_indices.tolist()],
                          model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                          actual_fit_steps=actual, fit_steps_requested=fit_steps,
                          fit_blocks=len(fit_blocks), fit_ids=int(fit_mask.sum()),
                          fit_first_loss=first_loss, fit_last_loss=last_loss,
                          **reliability))
    for group in groups:
        gid = group['group_id']
        group['prediction'] = group_predictions[gid]
        group['fold'] = group_fold[gid]
    return dict(schema=SCHEMA, seed=int(seed), groups=groups, blocks=blocks, folds=folds,
                tau_y=tau, source_ids=data['ids'], source_count=len(state),
                source_y=data['y'], source_rx=data['rx'], source_day=data['day'],
                grouping_decisions=decisions, quality_rule='coherence[32]: 4 -> adjacent 2 -> 1 bins per TX/day',
                fit_steps_requested=fit_steps, fit_lr=FIT_LR,
                architecture=[40, 48, 48, 1], parent_forward_calls=0,
                fit_seconds=time.perf_counter() - started,
                diagnostics=dict(unique_physical_ids=len(data['ids']), legal_groups=len(groups),
                                 condition_blocks=len(blocks),
                                 legal_group_ids=sum(g['count'] for g in groups),
                                 tau_missing_tx=torch.nonzero(~torch.isfinite(tau), as_tuple=True)[0].tolist()))


class ReceiverBank:
    """Frozen receiver evidence with R-independent relation sampling.

    sample_relations returns 32 source indices. Relation positions refer to
    those 32 slots. Pass offset=96 to identity_loss for a 128-row main batch,
    or offset=0 when passing just the selected 32 margins. Missing relations
    use ordinary random source rows and contribute zero over denominator two.
    """
    def __init__(self, payload):
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unexpected receiver cache schema')
        self.payload = payload
        self.groups = payload['groups']
        self.blocks = payload['blocks']
        self.folds = payload['folds']
        self.tau_y = payload['tau_y'].detach().cpu().clone()
        self.by_tx = defaultdict(list)
        for block in self.blocks:
            self.by_tx[block['tx']].append(block)

    def sample_relations(self, generator):
        if generator is None or str(generator.device) != 'cpu':
            raise ValueError('Explicit private CPU generator required')
        txs = sorted(self.by_tx)
        order = torch.randperm(len(txs), generator=generator).tolist()
        chosen = [txs[i] for i in order[:PLANNED_RELATIONS]]
        indices, relations = [], []
        for slot in range(PLANNED_RELATIONS):
            if slot >= len(chosen):
                # This is ordinary main-batch fallback, never a legal R bag.
                indices.extend(torch.randint(self.payload['source_count'], (16,), generator=generator).tolist())
                relations.append(dict(slot=slot, valid=False, active=False,
                                      reason='missing_eligible_TX', c=0., independent_ids=0))
                continue
            tx = chosen[slot]
            candidates = self.by_tx[tx]
            block = candidates[int(torch.randint(len(candidates), (), generator=generator))]
            pairs = list(combinations(block['group_ids'], 2))
            gids = pairs[int(torch.randint(len(pairs), (), generator=generator))]
            positions = []
            for gid in gids:
                rows = self.groups[gid]['indices']
                take = torch.randperm(len(rows), generator=generator)[:BAG_SIZE].tolist()
                start = len(indices)
                indices.extend(rows[i] for i in take)
                positions.append(list(range(start, start + BAG_SIZE)))
            ga, gb = [self.groups[g] for g in gids]
            delta = max(.5, math.sqrt(ga['parent_variance'] / BAG_SIZE + gb['parent_variance'] / BAG_SIZE))
            gap = ga['prediction'] - gb['prediction']
            c = float(self.folds[ga['fold']]['c'])
            strong, weak = (0, 1) if gap > 0 else (1, 0)
            reliable_direction = abs(gap) > delta
            tau = float(self.tau_y[tx])
            active = c > 0 and reliable_direction and math.isfinite(tau)
            reason = ('active' if active else 'zero_reliability' if c <= 0 else
                      'ambiguous_prediction' if not reliable_direction else 'missing_tau')
            relations.append(dict(slot=slot, valid=True, active=active, reason=reason,
                                  tx=tx, day=block['day'], quality_bins=block['quality_bins'],
                                  quality_bin=block['quality_bin'], group_ids=list(gids),
                                  strong_group=gids[strong], weak_group=gids[weak],
                                  strong_positions=positions[strong], weak_positions=positions[weak],
                                  c=c, delta=delta, tau=tau, prediction_gap=gap,
                                  independent_ids=2 * BAG_SIZE, fold=ga['fold']))
        return dict(indices=torch.tensor(indices, dtype=torch.long), relations=relations,
                    planned_relations=PLANNED_RELATIONS, relative_positions=True,
                    diagnostics=dict(planned_relations=PLANNED_RELATIONS,
                                     valid_relations=sum(r['valid'] for r in relations),
                                     active_relations=sum(r['active'] for r in relations),
                                     fallback_rows=sum(not r['valid'] for r in relations) * 16,
                                     actual_independent_bag_ids=sum(r['independent_ids'] for r in relations)))

    def identity_loss(self, margins, sample, offset=96):
        if margins.ndim != 1 or sample['planned_relations'] != PLANNED_RELATIONS:
            raise ValueError('One-dimensional scale=30 margins and two planned relations required')
        total = margins.sum() * 0.
        values = []
        for relation in sample['relations']:
            record = {key: relation[key] for key in ('slot', 'valid', 'active', 'reason', 'c')}
            if not relation['valid']:
                values.append(record)
                continue
            strong_ids = torch.tensor(relation['strong_positions'], device=margins.device) + offset
            weak_ids = torch.tensor(relation['weak_positions'], device=margins.device) + offset
            if min(strong_ids.min(), weak_ids.min()) < 0 or max(strong_ids.max(), weak_ids.max()) >= len(margins):
                raise ValueError('Relation positions outside provided main-batch margins')
            strong = margins[strong_ids].mean()
            weak = margins[weak_ids].mean()
            threshold = torch.minimum(strong, strong.new_tensor(relation['tau'])).detach()
            violation = threshold - weak - relation['delta']
            unweighted = one_sided_huber(violation)
            if relation['active']:
                total = total + relation['c'] * unweighted
            record.update(strong_margin=float(strong.detach()), weak_margin=float(weak.detach()),
                          capped_target=float(threshold), violation=float(violation.detach()),
                          unweighted_loss=float(unweighted.detach()),
                          weighted_loss=float((relation['c'] * unweighted).detach()) if relation['active'] else 0.,
                          delta=relation['delta'], tau=relation['tau'])
            values.append(record)
        return total / PLANNED_RELATIONS, dict(**sample['diagnostics'], relations=values,
                                               fixed_denominator=PLANNED_RELATIONS,
                                               reliability_applications=1)


def load_receiver(payload):
    return ReceiverBank(payload)
