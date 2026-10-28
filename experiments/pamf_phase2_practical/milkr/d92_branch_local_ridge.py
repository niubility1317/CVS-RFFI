"""Frozen train-only Gaussian ridge on the original branch interaction map.

No I/O, source state, query fitting, candidate selection, or bandwidth search.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import fit_branch_ridge, _freeze, _plain, _readonly
from .d92_branch_support_probe import _classification, _finite, _strings


FROZEN_CONFIG = dict(
    method='D92-BranchLocalRidge-v1', schema='d92_branch_local_ridge_v1',
    views='original_received_only', arms=['branch_ridge', 'interaction_ridge', 'local_ridge'],
    selected='local_ridge', selection='fixed_no_selection',
    background='unit(concat(unit(z_id),4*unit(fft96)))',
    auxiliary='concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)',
    branches=['t_emb', 'f_emb', 'pa_local'], identity_dim=160, fft_dim=96,
    input_feature_dim=736, implicit_feature_dim=123616, norm_floor=1e-12,
    fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8,
    distance_rule='rank2_reduced_QR_nonnegative_square_sum_exact_identity_zero',
    distance_pair_chunk=256, bandwidth_rule='median_nearest_other_class_squared_distance_including_zeros',
    kernel='exp(-squared_interaction_distance/tau)', bandwidth_floor=None,
    zero_bandwidth='exact_feature_equivalence_kernel',
    trace_match='train_centered_radial_trace_to_train_centered_interaction_trace',
    target='onehot_minus_1_over_C', ridge_coefficient=1., sample_weight=1.,
    objective='0.5*sum_physical_squared_error+0.5*RKHS_norm_squared',
    solver='float64_Cholesky_two_triangular_solves_no_jitter',
    numerical_tolerance='128*eps64*max(N,C)',
    residual_denominator='(1+s0)*frobenius(alpha)+frobenius(Yc)_spectral_upper_bound',
    centering='train_support_only_reference_difference_then_mean_on_R_minus_1',
    max_folds=3, physical_folds='per_class_physical_id_sort_position_mod_min_K_3',
    oneshot_proxy_anchors='all_per_class_sorted_positions_0_to_parent_K_minus_1',
    k1='same_full_support_fit_no_independent_holdout', probe_k1='numerical_only_no_fit_no_holdout',
    query_decision_policy='per_sample_all_registered_classes', tie_break='physical_class_id_lexicographic',
    optimizer_steps=0, source_inputs=False, summary_inputs=False, query_fit=False, phase1_frozen=True)
_CONFIG = deepcopy(FROZEN_CONFIG)
_ARMS = tuple(_CONFIG['arms'])
_PAIRS = (('local_ridge', 'branch_ridge'), ('local_ridge', 'interaction_ridge'),
          ('interaction_ridge', 'branch_ridge'))
_NAMES = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')


class NumericalFailure(FloatingPointError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = deepcopy(audit)

    def audit_dict(self):
        return deepcopy(self.audit)


def _stats(value):
    v = np.asarray(value, dtype=np.float64).ravel()
    if not len(v):
        return dict(count=0, min=None, median=None, max=None, mean=None, quantiles=None)
    _finite(v)
    return dict(count=len(v), min=float(v.min()), median=float(np.median(v)),
                max=float(v.max()), mean=float(v.mean()),
                quantiles=np.quantile(v, [0., .25, .5, .75, 1.]).tolist())


def _pair_distances(bi, ai, bj, aj):
    """Batched rank-two distance without subtraction of almost equal kernels."""
    db, da = bi-bj, ai-aj
    same = np.all(db == 0, axis=1) & np.all(da == 0, axis=1)
    u = np.stack((db, bj), axis=2)
    # Only R is needed; reduced QR still computes the specified orthogonal map.
    _, triangular = np.linalg.qr(u, mode='reduced')
    v = np.stack((ai, da), axis=1)
    product = triangular @ v
    distance = np.sum(db*db, axis=1)+np.sum(da*da, axis=1)+np.sum(product*product, axis=(1, 2))
    distance[same] = 0.
    _finite(distance)
    if np.any((~same) & (distance == 0)):
        raise FloatingPointError('DISTANCE_UNDERFLOW_NONIDENTICAL_FEATURES')
    return distance


def _distances(b, a, train_b=None, train_a=None):
    symmetric = train_b is None
    tb, ta = (b, a) if symmetric else (train_b, train_a)
    out = np.zeros((len(b), len(tb)), dtype=np.float64)
    chunk = _CONFIG['distance_pair_chunk']
    for i in range(len(b)):
        for start in range(i+1 if symmetric else 0, len(tb), chunk):
            end = min(start+chunk, len(tb))
            distance = _pair_distances(np.broadcast_to(b[i], (end-start, b.shape[1])),
                np.broadcast_to(a[i], (end-start, a.shape[1])), tb[start:end], ta[start:end])
            out[i, start:end] = distance
            if symmetric:
                out[start:end, i] = distance
    return out


def _radial_minus_one(distance, tau):
    if tau == 0:
        return -(distance != 0).astype(np.float64)
    with np.errstate(over='ignore', under='ignore'):
        return np.expm1(-distance/tau)


def _radial(distance, tau):
    if tau == 0:
        return (distance == 0).astype(np.float64)
    with np.errstate(over='ignore', under='ignore'):
        return np.exp(-distance/tau)


def _geometry(b, a, labels, c):
    distance = _distances(b, a)
    n = len(b)
    s0 = float(np.sum(distance[np.triu_indices(n, 1)])/n)
    if c == 1:
        tau, delta, reason = None, np.empty(0), 'SINGLE_REGISTERED_CLASS'
    else:
        delta = np.min(np.where(labels[:, None] != labels[None, :], distance, np.inf), axis=1)
        tau = float(np.median(delta))
        reason = 'IDENTICAL_COMPLETE_FEATURES' if s0 == 0 else ('ZERO_BANDWIDTH_EQUIVALENCE_KERNEL' if tau == 0 else None)
    return distance, s0, tau, delta, reason


def _solve(b, a, labels, ids, c):
    started = time.perf_counter()
    n = len(b)
    audit = dict(arm='local_ridge', train_k=n//c, train_physical_count=n,
        training_physical_ids=list(ids), factorization_calls=0, optimizer_steps=0,
        distance_rule=_CONFIG['distance_rule'], bandwidth_rule=_CONFIG['bandwidth_rule'],
        scope='final', fold=None, trial=None, completed_stages=[])
    try:
        distance, s0, tau, delta, reason = _geometry(b, a, labels, c)
        tolerance = float(128*np.finfo(np.float64).eps*max(n, c))
        audit.update(bandwidth_tau=tau, interaction_centered_trace=s0,
                     degeneracy_reason=reason, numerical_tolerance=tolerance)
        target = np.eye(c)[labels]-1./c
        factorized = c > 1 and s0 > 0
        alpha, mean, reference = np.zeros((n, c)), np.zeros(n), np.zeros(n)
        reference_self = grand = sradial = 0.
        gamma = None
        centered = np.zeros((n, n))
        radial = np.ones((n, n))
        residual_ratio = trace_error = solve_seconds = edf_seconds = 0.
        normal_norm = edf = 0.
        if factorized:
            rm1 = _radial_minus_one(distance, tau)
            radial = _radial(distance, tau)
            sradial = float(-2*np.sum(rm1[np.triu_indices(n, 1)])/n)
            if sradial <= 0:
                raise FloatingPointError('NONPOSITIVE_RADIAL_CENTERED_TRACE')
            gamma = s0/sradial
            _finite(gamma)
            centered, reference, reference_self, mean, grand = interaction._center_kernel(rm1)
            centered *= gamma
            trace_error = abs(float(np.trace(centered))-s0)/s0
            _finite(centered, trace_error)
            if trace_error > tolerance:
                raise FloatingPointError('CENTERED_TRACE_RESIDUAL_EXCEEDED')
            matrix = centered+np.eye(n)
            begin = time.perf_counter()
            audit['factorization_calls'] = 1
            chol = np.linalg.cholesky(matrix)
            # SciPy uses triangular solves, not two further dense factorizations.
            from scipy.linalg import solve_triangular
            alpha = solve_triangular(chol.T, solve_triangular(chol, target, lower=True), lower=False)
            solve_seconds = time.perf_counter()-begin
            normal_norm = float(np.linalg.norm(matrix@alpha-target))
            denominator = (1+s0)*float(np.linalg.norm(alpha))+float(np.linalg.norm(target))
            residual_ratio = normal_norm/denominator if denominator else 0.
            _finite(alpha, residual_ratio)
            if residual_ratio > tolerance:
                raise FloatingPointError('NORMAL_EQUATION_RESIDUAL_EXCEEDED')
            begin = time.perf_counter()
            edf = float(np.trace(solve_triangular(chol.T,
                solve_triangular(chol, centered, lower=True), lower=False)))
            edf_seconds = time.perf_counter()-begin
        fitted = centered@alpha
        error = fitted-target
        loss_data, loss_ridge = float(.5*np.sum(error*error)), float(.5*np.sum(alpha*fitted))
        offdiag = radial[~np.eye(n, dtype=bool)]
        _finite(loss_data, loss_ridge, edf, s0)
        audit.update(status='CLOSED_FORM_SOLVED' if factorized else 'EXACT_ZERO_CLASSIFIER',
            solver=_CONFIG['solver'] if factorized else 'NO_FACTORIZATION_EXACT_ZERO',
            bandwidth_tau=tau, bandwidth_zero=tau == 0 if tau is not None else None,
            nearest_other_class_squared_distance=_stats(delta),
            zero_nearest_other_class_fraction=float(np.mean(delta == 0)) if len(delta) else None,
            interaction_centered_trace=s0, radial_centered_trace=sradial if factorized else None,
            trace_scale=gamma, degeneracy_reason=reason, trace_relative_error=trace_error,
            normal_equation_residual=residual_ratio, normal_equation_absolute_residual=normal_norm,
            residual_denominator=_CONFIG['residual_denominator'], numerical_tolerance=tolerance,
            condition_bound=1+s0, gram_min_eigenvalue_lower_bound=1.,
            physical_loss_mass=float(n), sample_weight=1., ridge_coefficient=1.,
            loss_data=loss_data, loss_ridge=loss_ridge, loss_total=loss_data+loss_ridge,
            gradient_norm=float(np.linalg.norm(centered@((centered+np.eye(n))@alpha-target))),
            gradient_coordinate='dual_coefficients',
            target_norm_squared=float(np.sum(target*target)),
            factorization_dim=n if factorized else 0, input_feature_dim=736, implicit_feature_dim=123616,
            output_dim=c, classification_output_dim=c, training_feature_bytes=int(b.nbytes+a.nbytes),
            gram_bytes=int(centered.nbytes) if factorized else 0, kernel_bytes=int(radial.nbytes),
            coefficient_bytes=int(alpha.nbytes), optimizer_steps=0, learning_rate=None, epoch=None,
            effective_degrees_of_freedom=edf, effective_degrees_of_freedom_seconds=edf_seconds,
            effective_degrees_of_freedom_extra_triangular_solves=2 if factorized else 0,
            training_offdiagonal_radial=_stats(offdiag) if tau is not None else None,
            training_radial_row_sum=_stats(radial.sum(axis=1)) if tau is not None else None,
            centered_training_diagonal=_stats(np.diag(centered)),
            training_accuracy=float(np.mean(np.argmax(fitted, axis=1) == labels)),
            degenerate_constant_features=s0 == 0, all_states_estimated_from_trainfold_only=True,
            solve_seconds=solve_seconds, gram_seconds=time.perf_counter()-started-solve_seconds-edf_seconds,
            fit_seconds=time.perf_counter()-started)
        return (alpha, reference, reference_self, mean, grand, tau, gamma), audit
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        audit.update(status='TECHNICAL_FAILURE', failure_reason=str(exc), fit_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc), audit) from exc


def _score_local(b, a, tb, ta, alpha, reference, reference_self, mean, grand, tau, gamma):
    scores = np.zeros((len(b), alpha.shape[1]), dtype=np.float64)
    if gamma is None:
        return scores
    for i in range(len(b)):
        distance = _distances(b[i:i+1], a[i:i+1], tb, ta)[0]
        raw = _radial_minus_one(distance, tau)
        diff = (raw-raw[0])-reference+reference_self
        scores[i] = (gamma*(diff-float(diff.mean())-mean+grand))@alpha
    _finite(scores)
    return scores


@dataclass(frozen=True)
class BranchLocalRidgeState:
    support_background: np.ndarray
    support_auxiliary: np.ndarray
    alpha: np.ndarray
    reference_kernel: np.ndarray
    reference_self: float
    center_mean: np.ndarray
    center_grand: float
    bandwidth_tau: float | None
    trace_scale: float | None
    classes: tuple
    audit: Mapping

    def __post_init__(self):
        classes = _strings(self.classes, 'classes')
        n = len(self.support_background)
        shapes = dict(support_background=(n, 256), support_auxiliary=(n, 480),
                      alpha=(n, len(classes)), reference_kernel=(n,), center_mean=(n,))
        if not n:
            raise ValueError('Empty local ridge state')
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError('Invalid local ridge state '+name)
            object.__setattr__(self, name, _readonly(value))
        _finite(self.reference_self, self.center_grand)
        if self.bandwidth_tau is not None and (not np.isfinite(self.bandwidth_tau) or self.bandwidth_tau < 0):
            raise ValueError('Invalid fixed bandwidth')
        if self.trace_scale is not None and (not np.isfinite(self.trace_scale) or self.trace_scale <= 0 or self.bandwidth_tau is None):
            raise ValueError('Invalid fixed trace scale')
        object.__setattr__(self, 'classes', classes)
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def score(self, *, z_id, fft, t_emb, f_emb, pa_local):
        b, a = interaction._blocks(z_id, fft, t_emb, f_emb, pa_local, allow_empty=True)
        return _score_local(b, a, self.support_background, self.support_auxiliary, self.alpha,
            self.reference_kernel, self.reference_self, self.center_mean, self.center_grand,
            self.bandwidth_tau, self.trace_scale)

    def predict(self, **features):
        order = np.asarray(sorted(range(len(self.classes)), key=lambda i: self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(self.score(**features)[:, order], axis=1)]]

    def audit_dict(self):
        return _plain(self.audit)


def fit_branch_local_ridge(*, z_id, fft, t_emb, f_emb, pa_local,
                           support_labels, support_ids, classes, old_classes=(), arm='local_ridge'):
    args = dict(z_id=z_id, fft=fft, t_emb=t_emb, f_emb=f_emb, pa_local=pa_local,
                support_labels=support_labels, support_ids=support_ids, classes=classes, old_classes=old_classes)
    if arm in ('branch_ridge', 'interaction_ridge'):
        try:
            return (fit_branch_ridge(**args) if arm == 'branch_ridge'
                    else interaction.fit_branch_interaction(**args, arm='interaction'))
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            n, c = len(support_ids), len(classes)
            # Original controls expose no partial audit if arithmetic fails before
            # returning. Do not invent a completed factorization from that case.
            known = isinstance(exc, np.linalg.LinAlgError)
            raise NumericalFailure(str(exc), dict(arm=arm, status='TECHNICAL_FAILURE',
                failure_reason=str(exc), scope='final', fold=None, trial=None, parent_k=n//c,
                train_k=n//c, training_physical_ids=sorted(support_ids), completed_stages=[],
                factorization_calls=int(known), factorization_count=int(known),
                factorization_count_known=known,
                factorization_count_reason='Original control reports no partial audit before return',
                optimizer_steps=0)) from exc
    if arm != 'local_ridge':
        raise ValueError('Unknown frozen local ridge arm')
    started = time.perf_counter()
    b, a, labels, ids, canonical, requested, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    try:
        parts, audit = _solve(b, a, labels, ids, len(canonical))
    except NumericalFailure as exc:
        exc.audit.update(parent_k=k, classes=list(requested))
        exc.audit['factorization_count'] = exc.audit.get('factorization_calls', 0)
        raise
    alpha, reference, reference_self, mean, grand, tau, gamma = parts
    alpha = alpha[:, [canonical.index(cls) for cls in requested]]
    arrays = dict(support_background=b, support_auxiliary=a, alpha=alpha,
                  reference_kernel=reference, center_mean=mean)
    array_bytes = {name: int(value.nbytes) for name, value in arrays.items()}
    scalar_bytes = 16+8*int(tau is not None)+8*int(gamma is not None)
    size = sum(array_bytes.values())+scalar_bytes
    audit.update(parent_k=k, all_states_estimated_from_current_support_only=True)
    top = dict(config=deepcopy(_CONFIG), classes=list(requested), old_classes=sorted(old), selected=arm,
        selection='fixed_no_selection', k=k, support_count=len(ids), final_fit=audit,
        numerical=interaction._diagonal_stats(b, a), factorization_count=audit['factorization_calls'],
        fold_count=0, folds=[], oof=None, paired=None, oneshot_proxy=None, optimizer_steps=0,
        persistent_state_bytes=size, head_bytes=size, state_array_bytes=array_bytes, state_scalar_bytes=scalar_bytes,
        state_byte_scope='five_float64_arrays_and_present_float64_scalars_excludes_registry_audit_Python_overhead',
        current_support_feature_bytes=int(b.nbytes+a.nbytes),
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG',
        query_rows_used=0, source_rows_used=0, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        source_validation=None, source_validation_reason='No source samples or source feature banks',
        phase1_frozen=True, fit_seconds=time.perf_counter()-started)
    return BranchLocalRidgeState(b, a, alpha, reference, reference_self, mean, grand, tau, gamma, requested, top)


def _paired(oof, compact=False):
    result = {}
    for left, right in _PAIRS:
        lrows, rrows = oof[left]['rows'], oof[right]['rows']
        rows = [dict(physical_id=l['physical_id'], correct_delta=int(l['correct'])-int(r['correct']))
                for l, r in zip(lrows, rrows)]
        if compact:
            counts = dict(both_correct=0, left_only_correct=0, right_only_correct=0, both_wrong=0)
            for l, r in zip(lrows, rrows):
                key = ('both_correct' if r['correct'] else 'left_only_correct') if l['correct'] else ('right_only_correct' if r['correct'] else 'both_wrong')
                counts[key] += 1
            record = dict(counts=counts, record_count=len(rows))
        else:
            record = dict(rows=rows)
        record['mean_correct_delta'] = float(np.mean([row['correct_delta'] for row in rows]))
        result[left+'_minus_'+right] = record
    return result


def _compact_proxy(oof, classes):
    lookup = {cls: i for i, cls in enumerate(classes)}
    compact = {}
    for arm, result in oof.items():
        confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        nll_sum, counts = np.zeros(len(classes)), np.zeros(len(classes), dtype=np.int64)
        for row in result['rows']:
            label, predicted = lookup[row['class_id']], lookup[row['predicted_class']]
            confusion[label, predicted] += 1
            nll_sum[label] += row['nll']
            counts[label] += 1
        compact[arm] = dict(metrics=result['metrics'], confusion=confusion.tolist(), class_order=list(classes),
            classwise_nll_sum=nll_sum.tolist(), classwise_count=counts.tolist(), record_count=len(result['rows']))
    return compact, _paired(oof, compact=True)


def _evaluate_fold(raw, labels, ids, classes, old, keep, *, scope, parent_k, fold=None, trial=None):
    train_ids = tuple(pid for i, pid in enumerate(ids) if keep[i])
    held_ids = tuple(pid for i, pid in enumerate(ids) if not keep[i])
    train, held = ({name: value[mask] for name, value in raw.items()} for mask in (keep, ~keep))
    c = len(classes)
    entry = dict(training_ids=list(train_ids), held_ids=list(held_ids), train_k=int(keep.sum())//c,
                 held_k=int((~keep).sum())//c, parent_k=parent_k, stages=[])
    scores = {}
    for arm in _ARMS:
        audit = None
        try:
            state = fit_branch_local_ridge(**train, support_labels=labels[keep], support_ids=train_ids,
                classes=classes, old_classes=old, arm=arm)
            audit = state.audit_dict()['final_fit']
            audit.update(arm=arm, scope=scope, parent_k=parent_k, fold=fold, trial=trial,
                         all_states_estimated_from_trainfold_only=True)
            begin = time.perf_counter()
            scores[arm] = state.score(**held)
            audit['score_seconds'] = time.perf_counter()-begin
            train_scores = state.score(**train)
            audit['training_accuracy'] = float(np.mean(train_scores.argmax(axis=1) == labels[keep]))
            audit['held_accuracy'] = float(np.mean(scores[arm].argmax(axis=1) == labels[~keep]))
            audit['training_minus_held_accuracy'] = audit['training_accuracy']-audit['held_accuracy']
            if arm == 'local_ridge':
                hb, ha = interaction._blocks(**held)
                distance = _distances(hb, ha, state.support_background, state.support_auxiliary)
                tau = state.bandwidth_tau
                if tau is not None:
                    radial = _radial(distance, tau)
                    correct = np.array([row[labels[keep] == label].max() for row, label in zip(radial, labels[~keep])])
                    wrong = np.array([row[labels[keep] != label].max() for row, label in zip(radial, labels[~keep])]) if c > 1 else None
                    audit['held_max_radial_similarity'] = _stats(radial.max(axis=1))
                    audit['held_correct_minus_nearest_wrong_similarity'] = _stats(correct-wrong) if wrong is not None else None
                    audit['held_no_exact_match_fraction'] = float(np.mean(~np.any(distance == 0, axis=1))) if tau == 0 else None
                    raw_grand = state.center_grand+2*float(state.reference_kernel.mean())-state.reference_self
                    norms = state.trace_scale*(-2*_radial_minus_one(distance, tau).mean(axis=1)+raw_grand) if state.trace_scale is not None else np.zeros(len(distance))
                    audit['held_centered_radial_squared_norm'] = _stats(norms)
                else:
                    audit.update(held_max_radial_similarity=None, held_correct_minus_nearest_wrong_similarity=None,
                                 held_no_exact_match_fraction=None, held_centered_radial_squared_norm=None)
            entry['stages'].append(audit)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            failure = exc.audit_dict() if isinstance(exc, NumericalFailure) else dict(audit or {})
            failure.update(arm=arm, status='TECHNICAL_FAILURE', failure_reason=str(exc), scope=scope,
                parent_k=parent_k, train_k=entry['train_k'], fold=fold, trial=trial,
                training_physical_ids=list(train_ids), held_ids=list(held_ids), completed_stages=deepcopy(entry['stages']))
            failure.setdefault('factorization_calls', 1 if arm != 'local_ridge' else 0)
            raise NumericalFailure(str(exc), failure) from exc
    return entry, scores


def probe_branch_local_ridge(*, z_id, fft, t_emb, f_emb, pa_local,
                            support_labels, support_ids, classes, old_classes=()):
    started = time.perf_counter()
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {name: np.asarray(value)[order] for name, value in zip(_NAMES, (z_id, fft, t_emb, f_emb, pa_local))}
    n, c = len(ids), len(canonical)
    folds = 0 if k == 1 else min(k, 3)
    result = dict(config=deepcopy(_CONFIG), classes=list(canonical), old_classes=sorted(old), k=k, parent_k=k,
        support_count=n, numerical=interaction._diagonal_stats(b, a), fold_count=folds,
        folds=[], oof=None, paired=None, oneshot_proxy=None, physical_fold_assignment=[],
        factorization_count=0, standard_factorization_count=0, optimizer_steps=0, persistent_state_bytes=0,
        claim_scope='SUPPORT_OOF_AND_SUPPORT_ONESHOT_PROXY_NOT_QUERY_EVALUATION',
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    completed_stages = []
    def evaluate(keep, scope, fold=None, trial=None):
        try:
            entry, values = _evaluate_fold(raw, labels, ids, canonical, old, keep,
                scope=scope, parent_k=k, fold=fold, trial=trial)
        except NumericalFailure as exc:
            exc.audit['completed_stages'] = deepcopy(completed_stages)+exc.audit['completed_stages']
            exc.audit['factorization_count'] = sum(s.get('factorization_calls', 0) for s in exc.audit['completed_stages'])+exc.audit.get('factorization_calls', 0)
            raise
        completed_stages.extend(entry['stages'])
        return entry, values
    positions = [np.flatnonzero(labels == cls) for cls in range(c)]
    assignments = np.full(n, -1, dtype=int)
    for indices in positions:
        assignments[indices] = np.arange(k) % folds
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=canonical[int(labels[i])], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    scores = {arm: np.empty((n, c)) for arm in _ARMS}
    for fold in range(folds):
        keep = assignments != fold
        entry, partial = evaluate(keep, 'support_oof', fold=fold)
        entry['fold'] = fold
        for arm in _ARMS:
            scores[arm][~keep] = partial[arm]
        result['folds'].append(entry)
    result['oof'] = {arm: _classification(value, labels, canonical, old, ids, assignments) for arm, value in scores.items()}
    result['paired'] = _paired(result['oof'])
    result['standard_factorization_count'] = sum(s['factorization_calls'] for s in completed_stages)
    proxy = dict(parent_k=k, proxy_train_k=1, trial_count=k, trials=[],
        scope='SUPPORT_ONESHOT_PROXY_FROM_PARENT_SUPPORT_NOT_FORMAL_K1',
        coverage=dict(parent_physical_count=n, training_occurrences=n, held_occurrences=n*(k-1), unique_held_physical_count=n), factorization_count=0)
    for trial in range(k):
        keep = np.zeros(n, dtype=bool)
        for indices in positions:
            keep[indices[trial]] = True
        entry, partial = evaluate(keep, 'support_oneshot_proxy', trial=trial)
        entry.update(trial=trial, proxy_train_k=1)
        held_ids = tuple(pid for i, pid in enumerate(ids) if not keep[i])
        trial_oof = {arm: _classification(value, labels[~keep], canonical, old, held_ids,
            np.full(len(held_ids), trial)) for arm, value in partial.items()}
        entry['oof'], entry['paired'] = _compact_proxy(trial_oof, canonical)
        proxy['factorization_count'] += sum(s['factorization_calls'] for s in entry['stages'])
        proxy['trials'].append(entry)
    result['oneshot_proxy'] = proxy
    result['factorization_count'] = result['standard_factorization_count']+proxy['factorization_count']
    result['fit_seconds'] = time.perf_counter()-started
    return result
