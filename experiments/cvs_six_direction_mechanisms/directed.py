"""Source-only directed lag-statistic diagnostic for the frozen 160-D identity."""
from hashlib import blake2b
import itertools
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import design as d
from experiments.cvs_source_pseudotarget_health.frozen import preserve_rng

CHANNELS = 8
LAGS = (1, 2, 4)
PROBE_STEPS = 100
PROBE_LR = 0.01
PROBE_WEIGHT_DECAY = 0.001
PROBE_SEED_OFFSET = 56789
RESIDUAL_SCALE = 0.1 * math.sqrt(160 / 84)
PAIRS = tuple(itertools.combinations(range(CHANNELS), 2))
STATIC_PAIRS = tuple((i, j) for i in range(CHANNELS) for j in range(i, CHANNELS))


def fixed_orthomap(input_dim, output_dim):
    """Return a deterministic orthogonal map with shape [output_dim,input_dim].

    The low-dimensional rows or high-dimensional columns are the first rows of
    a DCT-II basis. This avoids framework-version-dependent random QR signs.
    """
    if not isinstance(input_dim, int) or not isinstance(output_dim, int) or min(input_dim, output_dim) < 1:
        raise ValueError('Positive integer map dimensions required')
    rows, cols = (output_dim, input_dim) if output_dim <= input_dim else (input_dim, output_dim)
    n = torch.arange(cols, dtype=torch.float64) + 0.5
    k = torch.arange(rows, dtype=torch.float64)[:, None]
    basis = torch.cos((math.pi / cols) * k * n[None, :]) * math.sqrt(2 / cols)
    basis[0] = 1 / math.sqrt(cols)
    basis = basis.to(torch.float32)
    return basis if output_dim <= input_dim else basis.T.contiguous()


def fixed_projection(input_channels=96, channels=CHANNELS):
    """Fixed 96-to-8 channel projection, returned as [8,96]."""
    if input_channels < channels:
        raise ValueError('Projection output cannot exceed the input channel count')
    return fixed_orthomap(input_channels, channels)


def project_channels(sequence, projection=None):
    if sequence.ndim != 3 or sequence.shape[1] < CHANNELS or sequence.shape[-1] < 5:
        raise ValueError('Expected [N,C,T] middle sequences with C>=8 and T>=5')
    if not torch.isfinite(sequence).all():
        raise ValueError('Middle sequence must be finite')
    matrix = fixed_projection(sequence.shape[1]) if projection is None else projection
    if matrix.shape != (CHANNELS, sequence.shape[1]):
        raise ValueError('Projection shape differs from the middle feature channels')
    return torch.matmul(sequence.float().transpose(1, 2), matrix.to(sequence.device).T)


def _center(sequence):
    if sequence.ndim != 3 or sequence.shape[-1] != CHANNELS:
        raise ValueError('Projected sequence must have shape [N,T,8]')
    if sequence.shape[1] < 1 or not torch.isfinite(sequence).all():
        raise ValueError('Nonempty finite projected sequence required')
    return sequence - sequence.mean(dim=1, keepdim=True)


def lagged_antisymmetric(sequence, lag):
    """Compute the 28 upper-triangle directed values at one nonnegative lag."""
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0 or lag >= sequence.shape[1]:
        raise ValueError('Lag must be an integer within the sequence')
    e = _center(sequence)
    left = e[:, :sequence.shape[1] - lag] if lag else e
    right = e[:, lag:]
    if left.shape[1] == 0:
        raise ValueError('Lag leaves no sample pairs')
    values = [(left[:, :, i] * right[:, :, j] - left[:, :, j] * right[:, :, i]).mean(1)
              for i, j in PAIRS]
    return torch.stack(values, dim=1)


def directed_statistics(sequence):
    """Concatenate 28 antisymmetric statistics for lags 1, 2, and 4."""
    return torch.cat([lagged_antisymmetric(sequence, lag) for lag in LAGS], dim=1)


def lagged_cross_covariance(sequence, lag):
    """Return C_tau = mean(e_t e_(t+tau)^T), including C0 when lag is zero."""
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0 or lag >= sequence.shape[1]:
        raise ValueError('Lag must be an integer within the sequence')
    e = _center(sequence)
    left = e[:, :sequence.shape[1] - lag] if lag else e
    right = e[:, lag:]
    return torch.matmul(left.transpose(1, 2), right) / left.shape[1]


def static_covariance(sequence):
    """Return the symmetric C0 upper triangle, including its diagonal (36)."""
    e = _center(sequence)
    covariance = torch.matmul(e.transpose(1, 2), e) / sequence.shape[1]
    return torch.stack([covariance[:, i, j] for i, j in STATIC_PAIRS], dim=1)


def _packet_seed(seed, physical_id):
    payload = f'{seed}:{type(physical_id).__name__}:{physical_id}'.encode('utf-8')
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), 'little') % (2**63 - 1)


def shuffle_packets(sequence, physical_ids, seed=20261013):
    """Apply one stable, per-packet time permutation; never changes C0."""
    if sequence.ndim != 3 or len(sequence) != len(physical_ids) or not physical_ids:
        raise ValueError('Projected packets and physical IDs must align')
    shuffled = []
    for row, physical_id in zip(sequence, physical_ids):
        generator = torch.Generator(device='cpu').manual_seed(_packet_seed(seed, physical_id))
        permutation = torch.randperm(sequence.shape[1], generator=generator).to(sequence.device)
        shuffled.append(row[permutation])
    return torch.stack(shuffled)


class DirectedResidualBranch(nn.Module):
    """Trainable 84-to-160 residual; only its exit layer starts at zero."""
    def __init__(self, input_dim=84, output_dim=160, scale=RESIDUAL_SCALE):
        super().__init__()
        self.upstream = nn.Linear(input_dim, input_dim)
        self.exit = nn.Linear(input_dim, output_dim)
        self.scale = float(scale)
        nn.init.zeros_(self.exit.weight)
        nn.init.zeros_(self.exit.bias)

    def forward(self, joint160, auxiliary84):
        if joint160.ndim != 2 or joint160.shape[1] != 160 or auxiliary84.shape != (len(joint160), 84):
            raise ValueError('Residual branch expects [N,160] identity and [N,84] auxiliary features')
        return joint160 + self.scale * self.exit(self.upstream(auxiliary84))


def structure_checks(device='cpu'):
    generator = torch.Generator().manual_seed(65084)
    sequence = torch.randn(5, 32, CHANNELS, generator=generator, device='cpu').to(device)
    directed = directed_statistics(sequence)
    reverse = directed_statistics(sequence.flip(1))
    lag0 = lagged_antisymmetric(sequence, 0)
    covariance = lagged_cross_covariance(sequence, 2)
    reverse_covariance = lagged_cross_covariance(sequence.flip(1), 2)
    upper_difference = torch.stack([covariance[:, i, j] - covariance[:, j, i] for i, j in PAIRS], dim=1)
    lag2_directed = lagged_antisymmetric(sequence, 2)
    ids = [f'packet-{i}' for i in range(len(sequence))]
    shuffled = shuffle_packets(sequence, ids)
    c0_error = float((static_covariance(sequence) - static_covariance(shuffled)).abs().max())

    projection = fixed_projection(96).to(device)
    static_map = fixed_orthomap(36, 84).to(device)
    auxiliary_map = fixed_orthomap(84, 160).to(device)
    projection_gram_error = float((projection @ projection.T - torch.eye(8, device=device)).abs().max())
    static_gram_error = float((static_map.T @ static_map - torch.eye(36, device=device)).abs().max())
    auxiliary_gram_error = float((auxiliary_map.T @ auxiliary_map - torch.eye(84, device=device)).abs().max())

    differentiable = torch.randn(4, 24, CHANNELS, generator=generator, device='cpu').to(device).requires_grad_(True)
    statistic = directed_statistics(differentiable)
    statistic.square().mean().backward()
    finite_gradients = bool(differentiable.grad is not None and torch.isfinite(differentiable.grad).all()
                            and differentiable.grad.abs().sum() > 0)

    joint = torch.randn(8, 160, generator=generator, device='cpu').to(device)
    auxiliary = torch.randn(8, 84, generator=generator, device='cpu').to(device)
    branch = DirectedResidualBranch().to(device)
    initial = branch(joint, auxiliary)
    identity_init = torch.equal(initial, joint)
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.01)
    (initial.square().sum()).backward()
    first_exit_gradient = bool(branch.exit.weight.grad is not None and branch.exit.weight.grad.abs().sum() > 0)
    first_upstream_gradient_zero = bool(branch.upstream.weight.grad is not None
                                        and branch.upstream.weight.grad.abs().sum() == 0)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    branch(joint, auxiliary).square().sum().backward()
    second_upstream_gradient = bool(branch.upstream.weight.grad is not None
                                    and branch.upstream.weight.grad.abs().sum() > 0)

    criteria = dict(
        reversal_negates_directed=bool(torch.allclose(reverse, -directed, atol=1e-6, rtol=1e-6)),
        reversal_transposes_Ctau=bool(torch.allclose(reverse_covariance, covariance.transpose(1, 2), atol=1e-6, rtol=1e-6)),
        directed_equals_C_minus_CT=bool(torch.allclose(lag2_directed, upper_difference, atol=1e-6, rtol=1e-6)),
        lag_zero_is_exact_zero=bool(torch.count_nonzero(lag0) == 0),
        shuffle_preserves_C0=c0_error <= 1e-6,
        fixed_maps_orthogonal=max(projection_gram_error, static_gram_error, auxiliary_gram_error) <= 1e-5,
        statistic_gradients_finite=finite_gradients,
        residual_starts_as_identity=identity_init,
        first_step_updates_exit_only=first_exit_gradient and first_upstream_gradient_zero,
        second_step_restores_upstream_gradient=second_upstream_gradient,
    )
    return dict(passed=all(criteria.values()), criteria=criteria,
                reversal_maxabs=float((reverse + directed).abs().max()), lag0_maxabs=float(lag0.abs().max()),
                shuffled_C0_maxabs=c0_error,
                fixed_map_orthogonality_maxabs=max(projection_gram_error, static_gram_error, auxiliary_gram_error),
                finite_gradient_sum=float(differentiable.grad.abs().sum()),
                residual_scale=RESIDUAL_SCALE,
                exact_scope='fixed projected middle sequence only; no claim that raw-IQ reversal or full-model RX effects are invariant')


def _validate_role(cache, features, name, expected_folds=None):
    required = ('ids', 'y', 'rx', 'day')
    if any(key not in cache for key in required) or any(key not in features for key in ('joint160', 'middle_sequence')):
        raise ValueError(f'{name} is missing source labels, metadata, or cached features')
    n = len(cache['ids'])
    if (n < 1 or any(len(cache[key]) != n for key in required[1:])
            or features['joint160'].shape != (n, 160)
            or features['middle_sequence'].shape != (n, 96, 32)):
        raise ValueError(f'{name} source roles and fixed feature shapes differ')
    if len(set(cache['ids'])) != n:
        raise ValueError(f'{name} physical IDs must be unique')
    y = torch.as_tensor(cache['y']).detach().cpu().long()
    rx = torch.as_tensor(cache['rx']).detach().cpu().long()
    day = torch.as_tensor(cache['day']).detach().cpu().long()
    if (not set(y.tolist()).issubset(set(range(6))) or not set(rx.tolist()).issubset(set(d.RX))
            or not set(day.tolist()).issubset(set(d.DAYS))):
        raise ValueError(f'{name} has unregistered source metadata')
    if not torch.isfinite(features['joint160']).all() or not torch.isfinite(features['middle_sequence']).all():
        raise ValueError(f'{name} contains nonfinite frozen features')
    if expected_folds is not None and len(expected_folds) != n:
        raise ValueError('L fold assignment differs from the L_s row count')
    return dict(ids=list(cache['ids']), y=y, rx=rx, day=day,
                joint160=features['joint160'].detach().cpu().float(),
                middle_sequence=features['middle_sequence'].detach().cpu().float())


@torch.no_grad()
def _extract_role(cache, projection, static_map, device, batch_size=256):
    directed_parts, static_parts, shuffled_parts, static_c0_parts = [], [], [], []
    projected_parts, shuffled_sequence_parts = [], []
    joint_parts = []
    c0_errors = []
    for start in range(0, len(cache['ids']), batch_size):
        end = min(len(cache['ids']), start + batch_size)
        sequence = cache['middle_sequence'][start:end].to(device)
        ids = cache['ids'][start:end]
        projected = project_channels(sequence, projection.to(device))
        shuffled = shuffle_packets(projected, ids, seed=20261013)
        static = static_covariance(projected)
        shuffled_static = static_covariance(shuffled)
        c0_errors.append((static - shuffled_static).abs().amax(dim=1).cpu())
        directed_parts.append(directed_statistics(projected).cpu())
        shuffled_parts.append(directed_statistics(shuffled).cpu())
        static_c0_parts.append(static.cpu())
        projected_parts.append(projected.cpu())
        shuffled_sequence_parts.append(shuffled.cpu())
        static_parts.append((static @ static_map.to(device).T).cpu())
        joint_parts.append(cache['joint160'][start:end].cpu())
    return dict(DIRECTED=torch.cat(directed_parts), STATIC=torch.cat(static_parts),
                SHUFFLED=torch.cat(shuffled_parts), JOINT=torch.cat(joint_parts),
                STATIC_C0_raw36=torch.cat(static_c0_parts),
                projected_sequence=torch.cat(projected_parts),
                shuffled_projected_sequence=torch.cat(shuffled_sequence_parts),
                shuffled_C0_maxabs=float(torch.cat(c0_errors).max()))


def _fit_normalizer(values, fit_mask):
    selected = values[fit_mask]
    if len(selected) == 0:
        raise ValueError('Each probe must have nonempty source fit data')
    mean = selected.mean(0)
    scale = selected.std(0, unbiased=False).clamp_min(1e-4)
    return mean, scale


def _embedding_pair(aux_l, aux_v, joint_l, joint_v, fit_mask, auxiliary_map):
    joint_mean, joint_scale = _fit_normalizer(joint_l, fit_mask)
    joint_l_norm = (joint_l - joint_mean) / joint_scale
    joint_v_norm = (joint_v - joint_mean) / joint_scale
    normalizer = dict(joint_mean=joint_mean, joint_scale=joint_scale)
    if aux_l is None:
        return joint_l_norm, joint_v_norm, normalizer
    auxiliary_mean, auxiliary_scale = _fit_normalizer(aux_l, fit_mask)
    mapped_l = ((aux_l - auxiliary_mean) / auxiliary_scale) @ auxiliary_map.T
    mapped_v = ((aux_v - auxiliary_mean) / auxiliary_scale) @ auxiliary_map.T
    normalizer.update(auxiliary_mean=auxiliary_mean, auxiliary_scale=auxiliary_scale,
                      residual_scale=RESIDUAL_SCALE)
    return (joint_l_norm + RESIDUAL_SCALE * mapped_l,
            joint_v_norm + RESIDUAL_SCALE * mapped_v,
            normalizer)


def _fit_linear(features, labels, seed, device):
    x = features.to(device).float(); y = labels.to(device).long()
    if x.ndim != 2 or x.shape[1] != 160 or len(x) != len(y) or not torch.isfinite(x).all():
        raise ValueError('Finite normalized 160-D probe features and labels required')
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        classifier = nn.Linear(160, 6).to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=PROBE_LR,
                                      weight_decay=PROBE_WEIGHT_DECAY)
        losses = []
        for _ in range(PROBE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Linear source probe loss is not finite')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    classifier.eval().requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.grad = None
    return classifier, losses


@torch.no_grad()
def _evaluate(classifier, features, labels, device):
    logits = classifier(features.to(device).float()).detach().cpu()
    y = labels.detach().cpu().long()
    predictions = logits.argmax(1)
    correct = predictions == y
    return dict(logits=logits, predictions=predictions,
                metrics=dict(count=len(y), correct_count=int(correct.sum()),
                             accuracy=float(correct.float().mean()),
                             cross_entropy=float(F.cross_entropy(logits, y))))


def _net_rescue(reference, candidate, labels, mask):
    y = labels[mask]
    before = reference[mask]; after = candidate[mask]
    return int(((before != y) & (after == y)).sum() - ((before == y) & (after != y)).sum())


def diagnose(ctx, output, device):
    """Run three matched source probes and two non-promotion health readouts."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    folds = torch.as_tensor(ctx['folds']).detach().cpu().long()
    L = _validate_role(ctx['L'], ctx['features']['L_s'], 'L_s', folds)
    V = _validate_role(ctx['V'], ctx['features']['V'], 'V_s')
    if set(L['ids']) & set(V['ids']):
        raise ValueError('L_s and V physical IDs overlap')
    if not set(folds.tolist()).issubset({0, 1, 2}) or set(folds.tolist()) != {0, 1, 2}:
        raise ValueError('All three fixed source folds are required')
    projection = fixed_projection(96)
    static_map = fixed_orthomap(36, 84)
    auxiliary_map = fixed_orthomap(84, 160)
    raw_L = _extract_role(L, projection, static_map, device)
    raw_V = _extract_role(V, projection, static_map, device)
    fits = {'fit0': folds == 0, 'fit1': folds == 1, 'fit01': (folds == 0) | (folds == 1)}
    eval_fold = folds == 2
    if not eval_fold.any() or any(not (folds == i).any() for i in (0, 1)):
        raise ValueError('fit0, fit1, and held fold2 must each be nonempty')

    probes = {}; audit_probes = {}; source_seed = int(ctx['seed'])
    categories = ('JOINT', 'STATIC', 'DIRECTED', 'SHUFFLED')
    for fit_name, fit_mask in fits.items():
        fit_mask = fit_mask.detach().cpu()
        fit_seed = source_seed + PROBE_SEED_OFFSET
        feature_sets = {}
        normalizers = {}
        for name in categories:
            aux_l = None if name == 'JOINT' else raw_L[name]
            aux_v = None if name == 'JOINT' else raw_V[name]
            features_l, features_v, norm = _embedding_pair(
                aux_l, aux_v, raw_L['JOINT'], raw_V['JOINT'], fit_mask, auxiliary_map)
            feature_sets[name] = (features_l.detach(), features_v.detach())
            normalizers[name] = norm

        for name in categories:
            train_features = feature_sets[name][0][fit_mask]
            classifier, losses = _fit_linear(train_features, L['y'][fit_mask], fit_seed, device)
            fold2 = _evaluate(classifier, feature_sets[name][0][eval_fold], L['y'][eval_fold], device)
            validation = _evaluate(classifier, feature_sets[name][1], V['y'], device)
            key = f'{fit_name}:{name}'
            promotions = fit_name in ('fit0', 'fit1')
            probes[key] = dict(
                fit=fit_name, method=name, training_count=int(fit_mask.sum()),
                L_fold2=fold2['metrics'], source_V=validation['metrics'],
                fitted_steps=PROBE_STEPS, initialization_seed=fit_seed,
                promotion_eligible=promotions,
                role_policy='fold2 is held source diagnostic; V is health only; neither reads U labels or target data')
            audit_probes[key] = dict(
                fit=fit_name, method=name, training_ids=[pid for i, pid in enumerate(L['ids']) if fit_mask[i]],
                L_fold2_ids=[pid for i, pid in enumerate(L['ids']) if eval_fold[i]],
                V_ids=list(V['ids']), normalizer=normalizers[name],
                training_loss=losses, fitted_steps=PROBE_STEPS,
                classifier_state={k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
                L_fold2_logits=fold2['logits'], L_fold2_predictions=fold2['predictions'],
                source_V_logits=validation['logits'], source_V_predictions=validation['predictions'],
                source_V_promotion_eligible=False)

    rescue = {}; strata = {}
    criteria = {}
    for fit_name in ('fit0', 'fit1', 'fit01'):
        method_preds = {name: audit_probes[f'{fit_name}:{name}']['L_fold2_predictions'] for name in categories}
        labels = L['y'][eval_fold]
        comparisons = {}
        for baseline in ('JOINT', 'STATIC', 'SHUFFLED'):
            comparisons[f'DIRECTED_vs_{baseline}'] = _net_rescue(method_preds[baseline], method_preds['DIRECTED'],
                                                                  labels, torch.ones(len(labels), dtype=torch.bool))
        rescue[fit_name] = comparisons
        fold2_rx = L['rx'][eval_fold]; fold2_day = L['day'][eval_fold]
        rx_detail = {str(rx): _net_rescue(method_preds['JOINT'], method_preds['DIRECTED'], labels,
                                          fold2_rx == rx) for rx in d.RX if bool((fold2_rx == rx).any())}
        day_detail = {str(day): _net_rescue(method_preds['JOINT'], method_preds['DIRECTED'], labels,
                                            fold2_day == day) for day in d.DAYS if bool((fold2_day == day).any())}
        strata[fit_name] = dict(DIRECTED_vs_JOINT_by_RX=rx_detail, DIRECTED_vs_JOINT_by_day=day_detail,
                                RX_nonnegative_count=sum(v >= 0 for v in rx_detail.values()),
                                RX_stratum_count=len(rx_detail),
                                day_nonnegative_count=sum(v >= 0 for v in day_detail.values()),
                                day_stratum_count=len(day_detail))
        if fit_name in ('fit0', 'fit1'):
            for comparison, minimum in (('DIRECTED_vs_STATIC', 3),
                                        ('DIRECTED_vs_SHUFFLED', 3),
                                        ('DIRECTED_vs_JOINT', 1)):
                criteria[f'{fit_name}_{comparison}_net_rescue_at_least_{minimum}'] = comparisons[comparison] >= minimum
            criteria[f'{fit_name}_at_least_4_of_5_RX_nonnegative'] = (
                strata[fit_name]['RX_stratum_count'] == 5 and strata[fit_name]['RX_nonnegative_count'] >= 4)
            criteria[f'{fit_name}_at_least_2_of_3_days_nonnegative'] = (
                strata[fit_name]['day_stratum_count'] == 3 and strata[fit_name]['day_nonnegative_count'] >= 2)

    result = dict(
        direction='directed', source_only=True, identity_updates=0, U_truth_read=False,
        target_access=False, source_seed=source_seed,
        feature_recipe=dict(channel_projection='fixed DCT-II 96-to-8, no label or domain fit',
            directed='lag 1/2/4; 28 i<j antisymmetric values per lag; 84 total; mean-centred projected sequence',
            static='C0 symmetric upper triangle 36 mapped by the first-column DCT-II orthomap to 84',
            shuffled='one deterministic per-physical-packet time permutation (seed=20261013); C0 preserved',
            residual='L-fit aux mean/std then fixed DCT-II 84-to-160 orthomap; standardized joint160 plus 0.1*sqrt(160/84) times auxiliary map',
            probes='JOINT/STATIC/DIRECTED/SHUFFLED x fit0/fit1/fit01; 12 linear TX6 heads; 100 full-batch AdamW steps; lr=0.01; weight_decay=0.001; init_seed=primary+56789'),
        probes={key: {k: v for k, v in record.items() if k != 'role_policy'} for key, record in probes.items()},
        rescue=rescue, strata=strata,
        shuffled_C0_maxabs=dict(L_s=raw_L['shuffled_C0_maxabs'], V=raw_V['shuffled_C0_maxabs']),
        applicability=dict(passed=all(criteria.values()), criteria=criteria,
            reason='Both independent L fits must rescue held-fold packets against matched static, shuffled, and joint controls; rescue must span RX and day strata. This is a source mechanism gate, not an unseen-RX or target performance claim.'),
        source_roles=dict(L_s_count=len(L['ids']), V_count=len(V['ids']), fold2_count=int(eval_fold.sum()),
                          fit0_count=int(fits['fit0'].sum()), fit1_count=int(fits['fit1'].sum()),
                          fit01_count=int(fits['fit01'].sum()), V_used_for_fit=False,
                          fit01_used_for_promotion=False, U_s_read=False),
        declared_limits=['Middle-sequence statistics are not claims of raw-IQ reversal invariance.',
            'RX and channel memory may also contribute to directed lag structure; it is not inherently TX-invariant.',
            'Only fit0/fit1 fold2 rescue contributes to applicability; fit01 and V are background health.',
            'The 160-D residual branch is a trainable candidate interface; its probe and optimizer states are discarded.'],
        elapsed_seconds=time.perf_counter() - started)
    audit = dict(direction='directed', source_only=True, identity_updates=0, U_truth_read=False,
        target_access=False, projection=projection, static_map=static_map, auxiliary_map=auxiliary_map,
        roles=dict(L_s=dict(ids=L['ids'], y=L['y'], rx=L['rx'], day=L['day'], features=raw_L),
                   V=dict(ids=V['ids'], y=V['y'], rx=V['rx'], day=V['day'], features=raw_V)),
        folds=folds, probes=audit_probes, rescue=rescue, strata=strata, criteria=criteria)
    with (output / 'audit.pt').open('xb') as stream:
        torch.save(audit, stream)
    d.write(output / 'diagnostic.json', result)
    return result
