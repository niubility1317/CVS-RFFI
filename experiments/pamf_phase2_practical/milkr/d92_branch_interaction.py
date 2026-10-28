"""Fixed support-only branch interaction kernel; immutable inductive inference.

The three diagnostic arms are predetermined. They never select a runtime head.
Only current-row target support features are retained, never source features.
"""
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping
import time
import numpy as np
from .d92_branch_support_probe import _feature_blocks, _strings, _finite, _classification
from .d92_branch_ridge import _freeze, _plain, _readonly


FROZEN_CONFIG = dict(
    method='D92-BranchInteraction-v1', schema='d92_branch_interaction_v1',
    views='original_received_only', branches=['t_emb', 'f_emb', 'pa_local'],
    identity_dim=160, fft_dim=96, background_dim=256, auxiliary_dim=480,
    input_feature_dim=736, implicit_interaction_dim=122880,
    norm_floor=1e-12, fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8,
    background='unit(concat(unit(z_id),4*unit(fft96)))',
    auxiliary='concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)',
    arms=['linear', 'energy_control', 'interaction'],
    kernels=dict(linear='KB+KA', energy_control='1.5*(KB+KA)',
                 interaction='KB+KA+KB*KA'),
    selected='interaction', selection='fixed_no_selection',
    energy_match_scope='unit_nonzero_blocks_only_zero_or_floor_blocks_may_differ',
    target='onehot_minus_1_over_C', ridge_coefficient=1.,
    objective='0.5*sum_physical_squared_error+0.5*RKHS_norm_squared',
    intercept_regularized=False, sample_weight=1.,
    solver='float64_centered_kernel_Cholesky',
    centering='train_support_only_reference_difference_then_mean',
    max_folds=3, folds='per_class_physical_id_sort_position_mod_min_K_3',
    k1='same_full_support_fit_no_independent_holdout',
    probe_k1='numerical_only_no_fit_no_holdout',
    score='per_sample_kernel_all_registered_classes',
    tie_break='physical_class_id_lexicographic',
    optimizer_steps=0, source_inputs=False, summary_inputs=False,
    query_fit=False, phase1_frozen=True, retained_features='current_row_target_support_only',
)
_CONFIG = deepcopy(FROZEN_CONFIG)
_ARMS = tuple(_CONFIG['arms'])


def _blocks(z_id, fft, t_emb, f_emb, pa_local, *, allow_empty=False):
    arrays = [np.asarray(x) for x in (z_id, fft, t_emb, f_emb, pa_local)]
    if allow_empty and all(x.shape == (0, d) and x.dtype.kind in 'fiu'
                           for x, d in zip(arrays, (160, 96, 160, 160, 160))):
        return np.empty((0, 256)), np.empty((0, 480))
    _, bases, aux = _feature_blocks(z_id, fft, t_emb, f_emb, pa_local)
    return bases['zfft'], aux


def _prepare(z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes):
    background, aux = _blocks(z_id, fft, t_emb, f_emb, pa_local)
    labels = np.asarray(support_labels)
    ids = _strings(support_ids, 'support_ids')
    requested = _strings(classes, 'classes')
    old = _strings(old_classes, 'old_classes', nonempty=False)
    n, c = len(background), len(requested)
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu' or len(ids) != n
            or set(labels.tolist()) != set(range(c)) or not set(old).issubset(requested)):
        raise ValueError('Invalid support labels, physical IDs, classes, or old membership')
    counts = np.bincount(labels.astype(np.int64), minlength=c)
    if np.any(counts != counts[0]):
        raise ValueError('Equal positive physical K required')
    canonical = tuple(sorted(requested))
    remap = np.asarray([canonical.index(v) for v in requested])
    order = np.asarray(sorted(range(n), key=lambda i: ids[i]))
    return (background[order], aux[order], remap[labels.astype(np.int64)][order],
            tuple(ids[i] for i in order), canonical, requested, old, int(counts[0]))


def _combine(kb, ka, arm):
    if arm not in _ARMS:
        raise ValueError('Unknown fixed branch interaction arm')
    value = kb+ka
    if arm == 'energy_control': value = 1.5*value
    if arm == 'interaction': value = value+kb*ka
    _finite(value)
    return value


def _diagonal_stats(background, aux):
    eb, ea = np.sum(background*background, axis=1), np.sum(aux*aux, axis=1)
    diag = {arm: _combine(eb, ea, arm) for arm in _ARMS}
    stats = lambda v: dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))
    diff = diag['interaction']-diag['energy_control']
    return dict(kernel_diagonal={arm: stats(value) for arm, value in diag.items()},
                interaction_minus_energy_control_diagonal=dict(**stats(diff), max_abs=float(np.max(np.abs(diff)))),
                energy_match_scope=_CONFIG['energy_match_scope'],
                background_mean_squared_norm=float(eb.mean()), auxiliary_mean_squared_norm=float(ea.mean()))


def _center_kernel(kernel):
    # Reference differences are exactly zero when every support feature is identical.
    reference = kernel[0].copy()
    reference_self = float(kernel[0, 0])
    diff = (kernel-kernel[:, :1])-reference[None, :]+reference_self
    mean = diff.mean(axis=0)
    grand = float(mean.mean())
    centered = diff-diff.mean(axis=1, keepdims=True)-mean[None, :]+grand
    centered = .5*(centered+centered.T)
    return centered, reference, reference_self, mean, grand


def _score_rows(background, aux, train_b, train_a, alpha, reference, reference_self, mean, grand, target_mean, arm):
    scores = np.empty((len(background), alpha.shape[1]), dtype=np.float64)
    if np.all(train_b == train_b[0]) and np.all(train_a == train_a[0]):
        # Balanced labels on identical support imply an exactly zero RKHS head.
        # GEMV may otherwise introduce row-tail rounding into an exact tie.
        scores[:] = target_mean
        return scores
    for i, (b, a) in enumerate(zip(background, aux)):
        raw = _combine(train_b@b, train_a@a, arm)
        diff = (raw-raw[0])-reference+reference_self
        centered = diff-float(diff.mean())-mean+grand
        scores[i] = centered@alpha+target_mean
    _finite(scores)
    return scores


def _fit(background, aux, labels, ids, c, arm):
    started = time.perf_counter()
    n = len(labels)
    kernel = _combine(background@background.T, aux@aux.T, arm)
    centered, reference, reference_self, mean, grand = _center_kernel(kernel)
    constant_features = bool(np.all(background == background[0]) and np.all(aux == aux[0]))
    if constant_features: centered = np.zeros_like(centered)
    targets = np.eye(c)[labels]-1./c
    # Every allowed fit/fold has equal physical K for every class.
    target_mean = np.zeros(c, dtype=np.float64)
    gram = centered+np.eye(n)
    _finite(gram, targets)
    gram_seconds = time.perf_counter()-started
    begin = time.perf_counter()
    chol = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, targets))
    solve_seconds = time.perf_counter()-begin
    fitted = centered@alpha
    residual = fitted-targets
    normal = gram@alpha-targets
    loss_data = float(.5*np.sum(residual*residual))
    loss_ridge = float(.5*np.sum(alpha*(centered@alpha)))
    bound = float(1+np.trace(centered))
    _finite(alpha, residual, normal, loss_data, loss_ridge, bound)
    audit = dict(arm=arm, status='CLOSED_FORM_SOLVED', solver=_CONFIG['solver'],
        train_k=n//c, train_physical_count=n, training_physical_ids=list(ids),
        physical_loss_mass=float(n), sample_weight=1., ridge_coefficient=1.,
        input_feature_dim=736, implicit_feature_dim=123616 if arm == 'interaction' else 736,
        output_dim=c, classification_output_dim=c, factorization_dim=n, factorization_calls=1,
        condition_bound=bound, gram_min_eigenvalue_lower_bound=1.,
        loss_data=loss_data, loss_ridge=loss_ridge, loss_total=loss_data+loss_ridge,
        normal_equation_residual=float(np.linalg.norm(normal)),
        gradient_norm=float(np.linalg.norm(centered@normal)), gradient_coordinate='dual_coefficients',
        dual_objective_gradient_norm=float(np.linalg.norm(centered@normal)),
        intercept_gradient_norm=float(np.linalg.norm(residual.sum(axis=0))),
        target_norm_squared=float(np.sum(targets*targets)),
        kernel_diagonal_mean=float(np.diag(kernel).mean()),
        degenerate_constant_features=constant_features,
        gram_bytes=int(gram.nbytes), coefficient_bytes=int(alpha.nbytes),
        training_feature_bytes=int(background.nbytes+aux.nbytes),
        gram_seconds=gram_seconds, solve_seconds=solve_seconds,
        fit_seconds=time.perf_counter()-started, optimizer_steps=0, learning_rate=None, epoch=None,
        all_states_estimated_from_trainfold_only=True)
    return (alpha, reference, reference_self, mean, grand, target_mean), audit


@dataclass(frozen=True)
class BranchInteractionState:
    support_background: np.ndarray
    support_auxiliary: np.ndarray
    alpha: np.ndarray
    reference_kernel: np.ndarray
    reference_self: float
    center_mean: np.ndarray
    center_grand: float
    target_mean: np.ndarray
    classes: tuple
    arm: str
    audit: Mapping

    def __post_init__(self):
        classes = _strings(self.classes, 'classes')
        b = np.asarray(self.support_background)
        n, c = len(b), len(classes)
        expected = dict(support_background=(n, 256), support_auxiliary=(n, 480), alpha=(n, c),
                        reference_kernel=(n,), center_mean=(n,), target_mean=(c,))
        if not n or self.arm not in _ARMS:
            raise ValueError('Invalid interaction state')
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError('Invalid interaction state '+name)
            object.__setattr__(self, name, _readonly(value))
        if not np.isfinite(self.reference_self) or not np.isfinite(self.center_grand):
            raise ValueError('Invalid interaction state scalar')
        object.__setattr__(self, 'classes', classes)
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def score(self, *, z_id, fft, t_emb, f_emb, pa_local):
        b, a = _blocks(z_id, fft, t_emb, f_emb, pa_local, allow_empty=True)
        return _score_rows(b, a, self.support_background, self.support_auxiliary, self.alpha,
                          self.reference_kernel, self.reference_self, self.center_mean,
                          self.center_grand, self.target_mean, self.arm)

    def predict(self, **features):
        scores = self.score(**features)
        order = np.asarray(sorted(range(len(self.classes)), key=lambda i: self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:, order], axis=1)]]

    def audit_dict(self): return _plain(self.audit)


def fit_branch_interaction(*, z_id, fft, t_emb, f_emb, pa_local,
                           support_labels, support_ids, classes, old_classes=(), arm='interaction'):
    started = time.perf_counter()
    b, a, labels, ids, canonical, requested, old, k = _prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    parts, audit = _fit(b, a, labels, ids, len(canonical), arm)
    alpha, reference, reference_self, mean, grand, target_mean = parts
    output_order = [canonical.index(v) for v in requested]
    alpha, target_mean = alpha[:, output_order], target_mean[output_order]
    numerical_bytes = int(sum(v.nbytes for v in (b, a, alpha, reference, mean, target_mean))+16)
    audit.update(scope='final', fold=None, all_states_estimated_from_current_support_only=True)
    top = dict(config=deepcopy(_CONFIG), classes=list(requested), old_classes=sorted(old),
        k=k, support_count=len(labels), selected=arm, selection='fixed_no_selection',
        numerical=_diagonal_stats(b, a), final_fit=audit, factorization_count=1,
        fold_count=0, folds=[], oof=None, optimizer_steps=0,
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG',
        persistent_state_bytes=numerical_bytes, head_bytes=numerical_bytes,
        current_support_feature_bytes=int(b.nbytes+a.nbytes),
        state_byte_scope='six_float64_arrays_and_two_float64_scalars_excludes_registry_audit_and_Python_overhead',
        query_rows_used=0, source_rows_used=0, new_source_payload_bytes=0,
        new_ground_statistics_bytes=0, phase1_frozen=True, source_validation=None,
        source_validation_reason='No source samples or source feature banks',
        fit_seconds=time.perf_counter()-started)
    return BranchInteractionState(b, a, alpha, reference, reference_self, mean, grand,
                                  target_mean, requested, arm, top)


def probe_branch_interaction(*, z_id, fft, t_emb, f_emb, pa_local,
                             support_labels, support_ids, classes, old_classes=()):
    started = time.perf_counter()
    b, a, labels, ids, canonical, _, old, k = _prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    n, c = len(labels), len(canonical)
    folds = 0 if k == 1 else min(k, 3)
    result = dict(config=deepcopy(_CONFIG), classes=list(canonical), old_classes=sorted(old),
        k=k, support_count=n, fold_count=folds, numerical=_diagonal_stats(b, a),
        folds=[], oof=None, paired=None, physical_fold_assignment=[], optimizer_steps=0,
        factorization_count=0, persistent_state_bytes=0,
        claim_scope='SUPPORT_OOF_DIAGNOSTIC_NOT_QUERY_EVALUATION',
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    assignments = np.full(n, -1, dtype=int)
    for cls in range(c):
        positions = np.flatnonzero(labels == cls)
        assignments[positions] = np.arange(k) % folds
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=canonical[int(labels[i])],
        fold=int(assignments[i])) for i, pid in enumerate(ids)]
    scores = {arm: np.empty((n, c)) for arm in _ARMS}
    for fold in range(folds):
        keep, held = assignments != fold, assignments == fold
        train_ids = tuple(pid for i, pid in enumerate(ids) if keep[i])
        held_ids = [pid for i, pid in enumerate(ids) if held[i]]
        entry = dict(fold=fold, training_ids=list(train_ids), held_ids=held_ids,
                     train_k=int(keep.sum())//c, stages=[])
        for arm in _ARMS:
            parts, audit = _fit(b[keep], a[keep], labels[keep], train_ids, c, arm)
            alpha, reference, reference_self, mean, grand, target_mean = parts
            begin = time.perf_counter()
            scores[arm][held] = _score_rows(b[held], a[held], b[keep], a[keep], alpha,
                                          reference, reference_self, mean, grand, target_mean, arm)
            audit.update(scope='support_oof', fold=fold, score_seconds=time.perf_counter()-begin)
            entry['stages'].append(audit)
        result['folds'].append(entry)
    result['factorization_count'] = folds*len(_ARMS)
    result['oof'] = {arm: _classification(value, labels, canonical, old, ids, assignments)
                     for arm, value in scores.items()}
    result['paired'] = {}
    for left, right in (('interaction', 'linear'), ('energy_control', 'linear'),
                        ('interaction', 'energy_control')):
        lrows, rrows = result['oof'][left]['rows'], result['oof'][right]['rows']
        paired = [dict(physical_id=pid, correct_delta=int(lrows[i]['correct'])-int(rrows[i]['correct']))
                  for i, pid in enumerate(ids)]
        result['paired'][left+'_minus_'+right] = dict(rows=paired,
            mean_correct_delta=float(np.mean([row['correct_delta'] for row in paired])))
    result['fit_seconds'] = time.perf_counter()-started
    return result
