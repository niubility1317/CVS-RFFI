"""Frozen cross-TX rank-2 prediction with a full joint160 student objective.

No optimizer, IQ forwarding, target access, or training-loop integration lives
here. Parent features must be the final normalized cosine-classifier joint160.
The caller owns deterministic, differentiable student forwarding. Sampling is
independent of all fitted reliability, scaling, and feasibility decisions.
"""
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import math
import time

import torch

from experiments.cvs_stage2_decouple.receiver import (
    FOLDS, STATE_DIM, BAG_SIZE, PLANNED_RELATIONS, _source, _build_groups)


SCHEMA = 'stage2_geometry_receiver_full160_v1'
FEATURE_DIM = 160
MAX_RANK = 2
RIDGE = .1
MAX_ALPHA = .25
MAX_PAIR_NORM = 2.
EPS_MSE = 1e-12
UNIT_TOL = 2e-5
FEASIBILITY_ROUNDOFF = 1e-7


def _frozen(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _frozen(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_frozen(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_frozen(item) for item in value)
    return deepcopy(value)


def _check_features(features, count):
    if (not torch.is_tensor(features) or not features.is_floating_point() or
            features.shape != (count, FEATURE_DIM) or not torch.isfinite(features).all()):
        raise ValueError('Finite floating normalized joint160 features [N,160] required')
    if not torch.allclose(features.detach().norm(dim=1), features.new_ones(count),
                          rtol=0., atol=UNIT_TOL):
        raise ValueError('joint160 features must already be unit normalized')


@torch.no_grad()
def block_potentials(parent_means, raw_potential):
    """Center and scale ONE whole block, then choose ONE feasible alpha.

    All calculations are CPU float64. Alpha=0 is the conservative fallback for
    parent means just outside the unit ball by floating-point roundoff. No edge
    is clipped independently; target centers and every directed cycle persist.
    """
    means = torch.as_tensor(parent_means).detach().cpu().double()
    raw = torch.as_tensor(raw_potential).detach().cpu().double()
    if (means.ndim != 2 or means.shape[1] != FEATURE_DIM or means.shape != raw.shape or
            len(means) < 2 or not torch.isfinite(means).all() or not torch.isfinite(raw).all()):
        raise ValueError('A finite condition block with >=2 full-vector RX groups required')
    if means.norm(dim=1).max() > 1 + FEASIBILITY_ROUNDOFF:
        raise ValueError('Parent group means outside unit ball beyond numerical tolerance')
    raw = raw - raw.mean(0)
    max_pair = max(float((raw[g] - raw[h]).norm())
                   for g, h in combinations(range(len(raw)), 2))
    kappa = min(1., MAX_PAIR_NORM / max_pair) if max_pair > 0 else 1.
    potential = kappa * raw

    def feasible(alpha):
        return bool(((means - alpha * potential).square().sum(1) <= 1.).all())

    if feasible(MAX_ALPHA):
        alpha, status = MAX_ALPHA, 'FULL_ALPHA_FEASIBLE'
    elif not feasible(0.):
        alpha, status = 0., 'PARENT_ROUNDOFF_ALPHA_ZERO'
    else:
        lo, hi = 0., MAX_ALPHA
        for _ in range(64):
            middle = (lo + hi) * .5
            if feasible(middle):
                lo = middle
            else:
                hi = middle
        alpha, status = lo, 'REDUCED_ALPHA_FEASIBLE'
    target = means - alpha * potential
    return dict(raw_potential=raw, potential=potential, kappa=kappa, alpha=alpha,
                target_mean=target, raw_max_pair_norm=max_pair,
                max_pair_norm=max(float((potential[g] - potential[h]).norm())
                                  for g, h in combinations(range(len(raw)), 2)),
                max_target_norm=float(target.norm(dim=1).max()),
                center_error=float((target.mean(0) - means.mean(0)).norm()),
                feasibility_status=status, feasibility_roundoff_tolerance=FEASIBILITY_ROUNDOFF)


@torch.no_grad()
def vector_reliability(groups, blocks, basis, potential_key='potential'):
    """Equal block / equal pair FULL-vector MSE, scalar c, raw error spectrum.

    Spectrum is E[error error^T], not a centered covariance. c is computed before
    alpha; scaling the chosen basis orthogonally cannot change it. Missing and
    zero-energy data produce c=0, never an artificial positive floor.
    """
    basis = torch.as_tensor(basis).detach().cpu().double()
    if basis.ndim != 2 or basis.shape[0] != FEATURE_DIM or basis.shape[1] > MAX_RANK:
        raise ValueError('A full-vector basis with at most two columns required')
    entries = []
    for block in blocks:
        pairs = list(combinations(block['group_ids'], 2))
        if not pairs:
            continue
        block_entries = []
        for g, h in pairs:
            truth = (groups[g]['parent_feature_mean'] - groups[h]['parent_feature_mean']).double()
            prediction = (groups[g][potential_key] - groups[h][potential_key]).double()
            block_entries.append((truth, prediction))
        entries.append(block_entries)
    if not entries:
        return dict(c=0., mse_pred=None, mse_zero=None, mse_subspace=None,
                    mse_outside_subspace=None, error_spectrum=[0.] * FEATURE_DIM,
                    relation_pairs=0, condition_blocks=0, status='N/A_NO_VALID_BLOCK',
                    weighting='equal_block_then_equal_RX_pair')
    errors, truths, predictions, weights = [], [], [], []
    for block in entries:
        for truth, prediction in block:
            errors.append(prediction - truth); truths.append(truth); predictions.append(prediction)
            weights.append(1. / (len(entries) * len(block)))
    error, truth, prediction = torch.stack(errors), torch.stack(truths), torch.stack(predictions)
    weight = torch.tensor(weights, dtype=torch.float64)
    mp = float((error.square().sum(1) * weight).sum())
    mz = float((truth.square().sum(1) * weight).sum())
    ms = float(((error @ basis).square().sum(1) * weight).sum())
    covariance = error.T @ (error * weight[:, None])
    spectrum = torch.linalg.eigvalsh((covariance + covariance.T) * .5).flip(0)
    denominator = truth.norm(dim=1) * prediction.norm(dim=1)
    nonzero = denominator > EPS_MSE
    cosines = (truth * prediction).sum(1)[nonzero] / denominator[nonzero]
    direction_weight = weight[nonzero]
    direction = float((cosines * direction_weight).sum() / direction_weight.sum()) if len(cosines) else None
    return dict(c=min(max(1. - mp / mz, 0.), 1.) if mz > EPS_MSE else 0.,
                mse_pred=mp, mse_zero=mz, mse_subspace=ms, mse_outside_subspace=mp-ms,
                prediction_energy=float((prediction.square().sum(1) * weight).sum()),
                mean_direction_cosine=direction, direction_nonzero_pairs=int(nonzero.sum()),
                error_spectrum=spectrum.tolist(), relation_pairs=len(error),
                condition_blocks=len(entries), weighting='equal_block_then_equal_RX_pair',
                status='VALID' if mz > EPS_MSE else 'N/A_NO_VALID_VARIATION', zero_energy_epsilon=EPS_MSE)


@torch.no_grad()
def _fit_fold(data, parent, groups, blocks, fit_tx):
    state = data['state'].double()
    fit_mask = torch.tensor([int(y) in fit_tx for y in data['y']], dtype=torch.bool)
    fit_indices = torch.nonzero(fit_mask, as_tuple=True)[0]
    center = state[fit_mask].mean(0)
    scale = state[fit_mask].std(0, unbiased=False).clamp_min(1e-6)
    fit_blocks = [b for b in blocks if b['tx'] in fit_tx]
    ss, hh = [], []
    for block in fit_blocks:
        means_s = torch.stack([state[groups[g]['indices']].mean(0) for g in block['group_ids']])
        means_z = torch.stack([parent[groups[g]['indices']].double().mean(0) for g in block['group_ids']])
        means_s = (means_s - center) / scale
        root_weight = math.sqrt(1. / (len(fit_blocks) * len(block['group_ids'])))
        ss.append((means_s - means_s.mean(0)).T * root_weight)
        hh.append((means_z - means_z.mean(0)).T * root_weight)
    basis = torch.zeros(FEATURE_DIM, MAX_RANK, dtype=torch.float64)
    regression = torch.zeros(MAX_RANK, STATE_DIM, dtype=torch.float64)
    singular, rank, fit_mse = torch.empty(0, dtype=torch.float64), 0, None
    if ss:
        s, h = torch.cat(ss, 1), torch.cat(hh, 1)
        k = s @ s.T + RIDGE * torch.eye(STATE_DIM, dtype=torch.float64)
        eigen, vectors = torch.linalg.eigh(k)
        inv_sqrt = (vectors * eigen.rsqrt()[None, :]) @ vectors.T
        u, singular, vh = torch.linalg.svd(h @ s.T @ inv_sqrt, full_matrices=False)
        threshold = max(1e-12, float(singular[0]) * max(h.shape[0], s.shape[0]) * torch.finfo(torch.float64).eps)
        rank = min(MAX_RANK, int((singular > threshold).sum()))
        if rank:
            basis[:, :rank] = u[:, :rank]
            regression[:rank] = (singular[:rank, None] * vh[:rank]) @ inv_sqrt
        fit_mse = float((h - basis @ regression @ s).square().sum())
    return dict(standard_mean=center, standard_scale=scale,
                standard_fit_ids=[data['ids'][i] for i in fit_indices.tolist()],
                basis=basis, regression=regression, matrix=basis @ regression,
                rank=rank, singular_values=singular.tolist(), fit_ids=len(fit_indices),
                fit_blocks=len(fit_blocks), fit_mse=fit_mse, ridge=RIDGE,
                fit_method='weighted_reduced_rank_ridge_closed_form', optimizer_steps=0,
                rank_status='VALID_RANK' if rank else 'VALID_RANK_ZERO')


@torch.no_grad()
def fit_receiver(cache, seed):
    """Fit only L_s, return a frozen CPU payload; identity/target fitting absent.

    Required cache keys are those of old receiver._source, plus normalized
    parent_feature[N,160]. Three folds and ridge=.1 are fixed, not searched.
    parent_margin is retained solely for reuse of the established group builder.
    """
    started = time.perf_counter()
    data = _source(cache)
    parent = torch.as_tensor(cache['parent_feature'])
    _check_features(parent, len(data['state']))
    parent = parent.detach().cpu().clone()
    groups, blocks, decisions, _ = _build_groups(data)
    for group in groups:
        group['parent_feature_mean'] = parent[group['indices']].double().mean(0)
    folds = []
    for fold_index, heldout in enumerate(FOLDS):
        fit_tx = tuple(y for y in range(6) if y not in heldout)
        fold = _fit_fold(data, parent, groups, blocks, fit_tx)
        held_blocks = [b for b in blocks if b['tx'] in heldout]
        for block in held_blocks:
            gids = block['group_ids']
            means_s = torch.stack([data['state'][groups[g]['indices']].double().mean(0) for g in gids])
            means_s = (means_s - fold['standard_mean']) / fold['standard_scale']
            raw = (means_s - means_s.mean(0)) @ fold['matrix'].T
            means_z = torch.stack([groups[g]['parent_feature_mean'] for g in gids])
            geometry = block_potentials(means_z, raw)
            block.update({key: value for key, value in geometry.items()
                          if key not in ('raw_potential', 'potential', 'target_mean')})
            block['fold'] = fold_index
            for position, gid in enumerate(gids):
                groups[gid].update(fold=fold_index, raw_potential=geometry['raw_potential'][position],
                                   potential=geometry['potential'][position],
                                   target_feature_mean=geometry['target_mean'][position])
        reliability = vector_reliability(groups, held_blocks, fold['basis'][:, :fold['rank']])
        raw = vector_reliability(groups, held_blocks, fold['basis'][:, :fold['rank']], 'raw_potential')
        fold.update(fold=fold_index, fit_tx=list(fit_tx), heldout_tx=list(heldout),
                    raw_reliability=raw, **reliability)
        if not fold['fit_blocks'] or fold['rank'] == 0:
            fold['c'] = 0.
        folds.append(fold)
    payload = dict(schema=SCHEMA, seed=int(seed), feature_dim=FEATURE_DIM, state_dim=STATE_DIM,
                   groups=groups, blocks=blocks, folds=folds, grouping_decisions=decisions,
                   source_count=len(parent), source_ids=data['ids'], source_y=data['y'],
                   source_rx=data['rx'], source_day=data['day'], parent_feature=parent,
                   parent_feature_bytes=parent.numel()*parent.element_size(),
                   target_access=False, optimizer_steps=0, max_rank=MAX_RANK, ridge=RIDGE,
                   alpha_max=MAX_ALPHA, max_pair_norm=MAX_PAIR_NORM,
                   reliability='scalar_full160_equal_block_pair_after_kappa_before_alpha',
                   quality_rule='same old source coherence[32] 4 -> adjacent 2 -> 1 bins',
                   fit_seconds=time.perf_counter()-started,
                   diagnostics=dict(legal_groups=len(groups), condition_blocks=len(blocks),
                                    unique_physical_ids=len(parent), legal_group_ids=sum(g['count'] for g in groups)))
    payload['diagnostics']['trusted_covariance'] = trusted_covariance(payload)
    payload['diagnostics']['trusted_covariance_weighting'] = 'all_blocks_equal_then_pairs_equal; c*p*pT; alpha_excluded'
    return _frozen(payload)


@torch.no_grad()
def trusted_covariance(payload):
    """Held-out predictions only, c*p*p^T, equal block/pair, alpha excluded."""
    output = torch.zeros(FEATURE_DIM, FEATURE_DIM, dtype=torch.float64)
    blocks, groups, folds = payload['blocks'], payload['groups'], payload['folds']
    if not blocks:
        return output
    for block in blocks:
        pairs = list(combinations(block['group_ids'], 2))
        if not pairs:
            continue
        c = float(folds[block['fold']]['c'])
        for g, h in pairs:
            p = (groups[g]['potential']-groups[h]['potential']).double()
            output.add_(torch.outer(p, p), alpha=c/(len(blocks)*len(pairs)))
    return output


def finite_population_loss(q_g, q_h, prediction, *, alpha, c, population_g, population_h):
    """Full-vector SRSWOR estimate with differentiable variance trace correction.

    q are differentiable student-minus-frozen-parent residuals. This helper
    permits n>=2 for exact small-population tests; the bank enforces bags of 8.
    No detach of variance and no clamping of signed estimates is permissible.
    """
    if (q_g.ndim != 2 or q_h.ndim != 2 or q_g.shape[1] != FEATURE_DIM or
            q_h.shape[1] != FEATURE_DIM or q_g.dtype != q_h.dtype or q_g.device != q_h.device):
        raise ValueError('Two residual bags with the same [n,160] dtype/device required')
    for q, population in ((q_g, population_g), (q_h, population_h)):
        if (isinstance(population, bool) or not isinstance(population, int) or
                not 2 <= len(q) <= population or not torch.isfinite(q).all()):
            raise ValueError('Finite no-replacement bags require 2 <= n <= integer population')
    if not (math.isfinite(float(c)) and 0 <= float(c) <= 1 and
            math.isfinite(float(alpha)) and 0 <= float(alpha) <= MAX_ALPHA):
        raise ValueError('Frozen scalar reliability and feasible alpha outside registered bounds')
    p = torch.as_tensor(prediction).detach().to(device=q_g.device, dtype=q_g.dtype)
    if p.shape != (FEATURE_DIM,) or not torch.isfinite(p).all():
        raise ValueError('Frozen full-vector prediction [160] required')
    mean_g, mean_h = q_g.mean(0), q_h.mean(0)
    error = mean_g - mean_h + float(alpha) * p
    var_g = (q_g-mean_g).square().sum()/(len(q_g)-1)
    var_h = (q_h-mean_h).square().sum()/(len(q_h)-1)
    variance = (1-len(q_g)/population_g)/len(q_g)*var_g + (1-len(q_h)/population_h)/len(q_h)*var_h
    mean_term, correction = .5*float(c)*error.square().sum(), .5*float(c)*variance
    return mean_term-correction, dict(mean_term=mean_term, variance_correction=correction,
                                      variance_trace=variance, error=error,
                                      control_residual=mean_g-mean_h)


class ReceiverBank:
    """Frozen evidence plus a common, R-independent two-relation sampling stream."""
    def __init__(self, payload):
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unexpected geometry receiver payload')
        self.payload = _frozen(payload)
        self.groups, self.blocks, self.folds = (self.payload[k] for k in ('groups', 'blocks', 'folds'))
        self.parent_feature = self.payload['parent_feature']
        _check_features(self.parent_feature, self.payload['source_count'])
        self.by_tx = defaultdict(list)
        for block in self.blocks:
            self.by_tx[block['tx']].append(block)

    def sample_relations(self, generator):
        if generator is None or str(generator.device) != 'cpu':
            raise ValueError('Explicit private CPU generator required')
        txs = sorted(self.by_tx)
        chosen = [txs[i] for i in torch.randperm(len(txs), generator=generator).tolist()[:PLANNED_RELATIONS]]
        indices, relations = [], []
        for slot in range(PLANNED_RELATIONS):
            if slot >= len(chosen):
                indices.extend(torch.randint(self.payload['source_count'], (2*BAG_SIZE,), generator=generator).tolist())
                relations.append(dict(slot=slot, valid=False, active=False, reason='missing_eligible_TX'))
                continue
            candidates = self.by_tx[chosen[slot]]
            block = candidates[int(torch.randint(len(candidates), (), generator=generator))]
            pairs = list(combinations(block['group_ids'], 2))
            g, h = pairs[int(torch.randint(len(pairs), (), generator=generator))]
            positions = []
            for gid in (g, h):
                available = self.groups[gid]['indices']
                take = torch.randperm(len(available), generator=generator)[:BAG_SIZE].tolist()
                positions.append(list(range(len(indices), len(indices)+BAG_SIZE)))
                indices.extend(available[i] for i in take)
            c = float(self.folds[block['fold']]['c'])
            prediction = self.groups[g]['potential']-self.groups[h]['potential']
            relations.append(dict(slot=slot, valid=True, active=c>0, reason='active' if c>0 else 'zero_reliability',
                                  group_ids=[g,h], positions=positions, g_positions=positions[0], h_positions=positions[1],
                                  block_id=block['block_id'], fold=block['fold'], tx=block['tx'], day=block['day'],
                                  quality_bin=block['quality_bin'], c=c, alpha=block['alpha'], kappa=block['kappa'],
                                  prediction=prediction.clone(), independent_ids=2*BAG_SIZE))
        return dict(indices=torch.tensor(indices, dtype=torch.long), relations=relations,
                    planned_relations=PLANNED_RELATIONS, relative_positions=True,
                    diagnostics=dict(planned_relations=PLANNED_RELATIONS,
                                     valid_relations=sum(r['valid'] for r in relations),
                                     active_relations=sum(r['active'] for r in relations),
                                     fallback_rows=sum(not r['valid'] for r in relations)*2*BAG_SIZE))

    def identity_loss(self, features32, indices, relation):
        """Return (signed loss, scalar diagnostics); fixed denominator is two.

        `indices` are source-cache physical row indices in features32 order.
        `relation` is the whole sample_relations result, not only its list.
        The frozen bank, never caller-supplied c/p/alpha, owns target values.
        """
        _check_features(features32, 2*2*BAG_SIZE)
        ids = torch.as_tensor(indices).detach().cpu()
        if (ids.shape != (32,) or not torch.equal(ids, ids.long()) or
                not torch.equal(ids, relation['indices'].cpu()) or
                relation['planned_relations'] != PLANNED_RELATIONS or len(relation['relations']) != PLANNED_RELATIONS or
                ids.min() < 0 or ids.max() >= self.payload['source_count']):
            raise ValueError('Full 32-row source index alignment and two planned relations required')
        ids = ids.long()
        total, records = features32.sum()*0., []
        for item in relation['relations']:
            if not item['valid']:
                records.append(dict(slot=item['slot'], valid=False, active=False, reason=item['reason']))
                continue
            g, h = item['group_ids']; ga, gh = self.groups[g], self.groups[h]
            if ga['block_id'] != gh['block_id'] or ga['fold'] != gh['fold'] or g == h:
                raise ValueError('A relation must use distinct groups in one condition block and fold')
            positions = item['positions']
            for group, where in zip((ga, gh), positions):
                if (len(where) != BAG_SIZE or len(set(where)) != BAG_SIZE or
                        min(where) < 0 or max(where) >= len(ids) or
                        len(set(ids[where].tolist())) != BAG_SIZE or
                        not set(ids[where].tolist()).issubset(group['indices'])):
                    raise ValueError('Each group bag must contain eight distinct eligible physical IDs')
            if len(positions) != 2 or set(positions[0]) & set(positions[1]):
                raise ValueError('Two disjoint group positions required')
            block = self.blocks[ga['block_id']]
            c, alpha = float(self.folds[ga['fold']]['c']), float(block['alpha'])
            parent_g = self.parent_feature[ids[positions[0]]].to(features32)
            parent_h = self.parent_feature[ids[positions[1]]].to(features32)
            qg, qh = features32[positions[0]]-parent_g, features32[positions[1]]-parent_h
            prediction = ga['potential']-gh['potential']
            value, parts = finite_population_loss(qg, qh, prediction, alpha=alpha, c=c,
                                                   population_g=ga['count'], population_h=gh['count'])
            total = total+value
            records.append(dict(slot=item['slot'], valid=True, active=c>0, c=c, alpha=alpha,
                                kappa=float(block['kappa']), group_ids=[g,h],
                                loss=float(value.detach()), mean_term=float(parts['mean_term'].detach()),
                                variance_correction=float(parts['variance_correction'].detach()),
                                variance_trace=float(parts['variance_trace'].detach()),
                                error_norm=float(parts['error'].detach().norm()),
                                control_residual_norm=float(parts['control_residual'].detach().norm()),
                                prediction_norm=float(prediction.norm()), independent_ids=2*BAG_SIZE))
        return total/PLANNED_RELATIONS, dict(**relation['diagnostics'], relations=records,
                                             fixed_denominator=PLANNED_RELATIONS,
                                             full_feature_dim=FEATURE_DIM, variance_gradient_retained=True,
                                             signed_estimate_not_clamped=True, reliability_applications=1)


def load_receiver(payload):
    return ReceiverBank(payload)
