"""Fixed current-support within-class metric and independent support diagnostics."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from . import d92_branch_interaction as interaction
from .d92_branch_support_probe import _classification, _finite, _strings
from .d92_branch_ridge import _freeze, _plain, _readonly


FROZEN_CONFIG = dict(
    method='D92-BranchMetric-v1', schema='d92_branch_metric_v1', views='original_received_only',
    kernel='KB+KA+KB*KA', background='unit(concat(unit(z_id),4*unit(fft96)))',
    auxiliary='concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)',
    branches=['t_emb', 'f_emb', 'pa_local'], identity_dim=160, fft_dim=96,
    input_feature_dim=736, implicit_feature_dim=123616, norm_floor=1e-12,
    fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8,
    within_scatter='sum_physical_class_centered_outer_products', within_scatter_denominator=1,
    ridge_coefficient=1., metric='inverse(I+within_scatter)',
    score='centered_query_metric_class_mean_minus_half_class_mean_metric_norm',
    class_prior='uniform_all_registered_classes', class_norm_intercept=True,
    centering='train_support_only_reference_difference_then_mean',
    solver='float64_Cholesky_I_plus_R_Gc_R', selected='within_metric', selection='fixed_no_selection',
    arms=['interaction_ridge', 'kernel_ncm', 'within_metric'], max_folds=3,
    physical_folds='per_class_physical_id_sort_position_mod_min_K_3',
    oneshot_proxy_anchors='all_per_class_sorted_positions_0_to_parent_K_minus_1',
    oneshot_proxy_ncm_equivalence_atol=0.,
    k1='exact_kernel_ncm_no_within_variance_estimate',
    query_decision_policy='per_sample_all_registered_classes', tie_break='physical_class_id_lexicographic',
    optimizer_steps=0, source_inputs=False, summary_inputs=False, query_fit=False, phase1_frozen=True)
_CONFIG = deepcopy(FROZEN_CONFIG)
_ARMS = tuple(_CONFIG['arms'])


def _class_center(value, labels):
    result = np.array(value, dtype=np.float64, copy=True)
    for cls in np.unique(labels):
        rows = labels == cls
        result[rows] -= value[rows].mean(axis=0)
    return result


def _solve(background, aux, labels, ids, c, arm):
    if arm not in ('kernel_ncm', 'within_metric'):
        raise ValueError('Unknown fixed branch metric arm')
    started = time.perf_counter()
    n, k = len(labels), len(labels)//c
    kernel = interaction._combine(background@background.T, aux@aux.T, 'interaction')
    gc, reference, reference_self, mean, grand = interaction._center_kernel(kernel)
    constant = bool(np.all(background == background[0]) and np.all(aux == aux[0]))
    if constant: gc = np.zeros_like(gc)
    p = np.eye(c)[labels]/k
    v = p.copy()
    factorized = arm == 'within_metric' and k > 1 and not constant
    scatter_trace = 0.
    normal_norm = 0.
    objective = 0.
    gram_bytes = 0
    solve_seconds = 0.
    gram_seconds = time.perf_counter()-started
    if factorized:
        begin = time.perf_counter()
        rg = _class_center(gc, labels)
        rgr = _class_center(rg.T, labels).T
        rgr = .5*(rgr+rgr.T)
        matrix = rgr+np.eye(n)
        rhs = rg@p
        scatter_trace = float(np.trace(rgr))
        gram_bytes = int(matrix.nbytes)
        gram_seconds += time.perf_counter()-begin
        begin = time.perf_counter()
        chol = np.linalg.cholesky(matrix)
        q = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        v = p-_class_center(q, labels)
        normal = matrix@q-rhs
        normal_norm = float(np.linalg.norm(normal))
        objective = float(.5*np.sum(q*(matrix@q))-np.sum(q*rhs))
        solve_seconds = time.perf_counter()-begin
    metric_norms = np.sum(p*(gc@v), axis=0)
    bias = -.5*metric_norms
    if constant: bias = np.zeros(c)
    _finite(v, bias, metric_norms, scatter_trace, normal_norm, objective)
    audit = dict(arm=arm, status='CLOSED_FORM_SOLVED' if factorized else 'EXACT_NCM_CLOSED_FORM',
        solver=_CONFIG['solver'] if factorized else 'NO_FACTORIZATION_EXACT_PROTOTYPE',
        train_k=k, train_physical_count=n, training_physical_ids=list(ids),
        physical_loss_mass=None, loss_data=None, loss_ridge=None, loss_total=objective if factorized else None,
        loss_reason='Metric solve quadratic objective; not a supervised squared-error classification loss',
        solve_objective=objective if factorized else None,
        gradient_norm=normal_norm, gradient_coordinate='metric_solve_Q', normal_equation_residual=normal_norm,
        within_scatter_trace=scatter_trace if arm == 'within_metric' else None,
        within_scatter_denominator=1, ridge_coefficient=1.,
        class_metric_norm_min=float(metric_norms.min()), class_metric_norm_max=float(metric_norms.max()),
        class_metric_norm_mean=float(metric_norms.mean()), class_norm_intercept=True,
        condition_bound=1+scatter_trace if factorized else 1., gram_min_eigenvalue_lower_bound=1.,
        factorization_calls=int(factorized), factorization_dim=n if factorized else 0,
        input_feature_dim=736, implicit_feature_dim=123616, output_dim=c, classification_output_dim=c,
        kernel_bytes=int(kernel.nbytes), gram_bytes=gram_bytes, coefficient_bytes=int(v.nbytes),
        intercept_bytes=int(bias.nbytes), training_feature_bytes=int(background.nbytes+aux.nbytes),
        gram_seconds=gram_seconds, solve_seconds=solve_seconds, fit_seconds=time.perf_counter()-started,
        optimizer_steps=0, learning_rate=None, epoch=None, degenerate_constant_features=constant,
        all_states_estimated_from_trainfold_only=True, k1_within_variance_estimated=False if k == 1 else None)
    return (v, reference, reference_self, mean, grand, bias), audit


@dataclass(frozen=True)
class BranchMetricState:
    support_background: np.ndarray
    support_auxiliary: np.ndarray
    V: np.ndarray
    reference_kernel: np.ndarray
    reference_self: float
    center_mean: np.ndarray
    center_grand: float
    b: np.ndarray
    classes: tuple
    arm: str
    audit: Mapping

    def __post_init__(self):
        classes = _strings(self.classes, 'classes')
        background = np.asarray(self.support_background)
        if background.ndim != 2 or not len(background) or self.arm not in ('kernel_ncm', 'within_metric'):
            raise ValueError('Invalid branch metric state')
        n, c = len(background), len(classes)
        shapes = dict(support_background=(n, 256), support_auxiliary=(n, 480), V=(n, c),
                      reference_kernel=(n,), center_mean=(n,), b=(c,))
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError('Invalid metric state '+name)
            object.__setattr__(self, name, _readonly(value))
        _finite(self.reference_self, self.center_grand)
        object.__setattr__(self, 'classes', classes)
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def score(self, *, z_id, fft, t_emb, f_emb, pa_local):
        background, aux = interaction._blocks(z_id, fft, t_emb, f_emb, pa_local, allow_empty=True)
        return interaction._score_rows(background, aux, self.support_background, self.support_auxiliary,
            self.V, self.reference_kernel, self.reference_self, self.center_mean, self.center_grand, self.b, 'interaction')

    def predict(self, **features):
        scores = self.score(**features)
        order = np.asarray(sorted(range(len(self.classes)), key=lambda i: self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:, order], axis=1)]]

    def audit_dict(self): return _plain(self.audit)


def _numerical(background, aux):
    values = interaction._diagonal_stats(background, aux)
    return dict(kernel='KB+KA+KB*KA', shared_kernel_diagonal=values['kernel_diagonal']['interaction'],
        background_mean_squared_norm=values['background_mean_squared_norm'],
        auxiliary_mean_squared_norm=values['auxiliary_mean_squared_norm'])


def fit_branch_metric(*, z_id, fft, t_emb, f_emb, pa_local,
                      support_labels, support_ids, classes, old_classes=(), arm='within_metric'):
    started = time.perf_counter()
    background, aux, labels, ids, canonical, requested, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    parts, audit = _solve(background, aux, labels, ids, len(canonical), arm)
    v, reference, reference_self, mean, grand, bias = parts
    output_order = [canonical.index(value) for value in requested]
    v, bias = v[:, output_order], bias[output_order]
    size = int(sum(x.nbytes for x in (background, aux, v, reference, mean, bias))+16)
    audit.update(scope='final', fold=None, trial=None, all_states_estimated_from_current_support_only=True)
    top = dict(config=deepcopy(_CONFIG), classes=list(requested), old_classes=sorted(old),
        selected=arm, selection='fixed_no_selection', k=k, support_count=len(ids),
        numerical=_numerical(background, aux), final_fit=audit, factorization_count=audit['factorization_calls'],
        fold_count=0, folds=[], oof=None, oneshot_proxy=None,
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG',
        optimizer_steps=0, persistent_state_bytes=size, head_bytes=size,
        current_support_feature_bytes=int(background.nbytes+aux.nbytes),
        state_byte_scope='six_float64_arrays_and_two_float64_scalars_excludes_registry_audit_and_Python_overhead',
        query_rows_used=0, source_rows_used=0, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        source_validation=None, source_validation_reason='No source samples or source feature banks',
        phase1_frozen=True, fit_seconds=time.perf_counter()-started)
    return BranchMetricState(background, aux, v, reference, reference_self, mean, grand, bias, requested, arm, top)


def _paired(oof):
    result = {}
    for left, right in (('within_metric', 'interaction_ridge'), ('kernel_ncm', 'interaction_ridge'),
                        ('within_metric', 'kernel_ncm')):
        rows = [dict(physical_id=l['physical_id'], correct_delta=int(l['correct'])-int(r['correct']))
                for l, r in zip(oof[left]['rows'], oof[right]['rows'])]
        result[left+'_minus_'+right] = dict(rows=rows, mean_correct_delta=float(np.mean([r['correct_delta'] for r in rows])))
    return result


def _evaluate_fold(background, aux, labels, ids, classes, old, keep, *, scope, parent_k, fold=None, trial=None):
    held = ~keep
    train_ids = tuple(pid for i, pid in enumerate(ids) if keep[i])
    held_ids = tuple(pid for i, pid in enumerate(ids) if held[i])
    c = len(classes)
    output = dict(training_ids=list(train_ids), held_ids=list(held_ids), train_k=int(keep.sum())//c,
                  held_k=int(held.sum())//c, parent_k=parent_k, stages=[])
    scores = {}
    for arm in _ARMS:
        if arm == 'interaction_ridge':
            parts, audit = interaction._fit(background[keep], aux[keep], labels[keep], train_ids, c, 'interaction')
            audit['arm'] = arm
        else:
            parts, audit = _solve(background[keep], aux[keep], labels[keep], train_ids, c, arm)
        start = time.perf_counter()
        scores[arm] = interaction._score_rows(background[held], aux[held], background[keep], aux[keep], *parts, 'interaction')
        audit.update(scope=scope, parent_k=parent_k, fold=fold, trial=trial, score_seconds=time.perf_counter()-start)
        output['stages'].append(audit)
    return output, scores


def _compact_proxy(oof, classes):
    lookup = {cls: i for i, cls in enumerate(classes)}
    compact, paired = {}, {}
    for arm, result in oof.items():
        confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        nll_sum = np.zeros(len(classes), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for record in result['rows']:
            label, prediction = lookup[record['class_id']], lookup[record['predicted_class']]
            confusion[label, prediction] += 1
            nll_sum[label] += record['nll']
            counts[label] += 1
        compact[arm] = dict(metrics=result['metrics'], confusion=confusion.tolist(),
            class_order=list(classes), classwise_nll_sum=nll_sum.tolist(), classwise_count=counts.tolist(),
            record_count=len(result['rows']))
    for left, right in (('within_metric', 'interaction_ridge'), ('kernel_ncm', 'interaction_ridge'),
                        ('within_metric', 'kernel_ncm')):
        counts = dict(both_correct=0, left_only_correct=0, right_only_correct=0, both_wrong=0)
        for lrow, rrow in zip(oof[left]['rows'], oof[right]['rows']):
            left_ok, right_ok = lrow['correct'], rrow['correct']
            key = ('both_correct' if right_ok else 'left_only_correct') if left_ok else ('right_only_correct' if right_ok else 'both_wrong')
            counts[key] += 1
        total = sum(counts.values())
        paired[left+'_minus_'+right] = dict(counts=counts, record_count=total,
            mean_correct_delta=(counts['left_only_correct']-counts['right_only_correct'])/total)
    return compact, paired


def probe_branch_metric(*, z_id, fft, t_emb, f_emb, pa_local,
                        support_labels, support_ids, classes, old_classes=()):
    started = time.perf_counter()
    background, aux, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    n, c = len(ids), len(canonical)
    folds = 0 if k == 1 else min(k, 3)
    result = dict(config=deepcopy(_CONFIG), classes=list(canonical), old_classes=sorted(old),
        k=k, parent_k=k, support_count=n, numerical=_numerical(background, aux),
        fold_count=folds, folds=[], oof=None, paired=None, oneshot_proxy=None, physical_fold_assignment=[],
        factorization_count=0, standard_factorization_count=0, optimizer_steps=0, persistent_state_bytes=0,
        claim_scope='SUPPORT_OOF_AND_SUPPORT_ONESHOT_PROXY_NOT_QUERY_EVALUATION',
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    positions = [np.flatnonzero(labels == cls) for cls in range(c)]
    assignments = np.full(n, -1, dtype=int)
    for indices in positions: assignments[indices] = np.arange(k) % folds
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=canonical[int(labels[i])], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    scores = {arm: np.empty((n, c)) for arm in _ARMS}
    for fold in range(folds):
        keep = assignments != fold
        entry, partial = _evaluate_fold(background, aux, labels, ids, canonical, old, keep,
                                        scope='support_oof', parent_k=k, fold=fold)
        entry['fold'] = fold
        for arm in _ARMS: scores[arm][~keep] = partial[arm]
        result['folds'].append(entry)
    result['oof'] = {arm: _classification(value, labels, canonical, old, ids, assignments) for arm, value in scores.items()}
    result['paired'] = _paired(result['oof'])
    result['standard_factorization_count'] = sum(stage['factorization_calls'] for f in result['folds'] for stage in f['stages'])
    proxy = dict(parent_k=k, proxy_train_k=1, trial_count=k, trials=[],
        scope='SUPPORT_ONESHOT_PROXY_FROM_PARENT_SUPPORT_NOT_FORMAL_K1',
        coverage=dict(parent_physical_count=n, training_occurrences=n, held_occurrences=n*(k-1), unique_held_physical_count=n),
        factorization_count=0)
    for trial in range(k):
        keep = np.zeros(n, dtype=bool)
        for indices in positions: keep[indices[trial]] = True
        entry, partial = _evaluate_fold(background, aux, labels, ids, canonical, old, keep,
                                        scope='support_oneshot_proxy', parent_k=k, trial=trial)
        entry.update(trial=trial, proxy_train_k=1)
        held_ids = tuple(pid for i, pid in enumerate(ids) if not keep[i])
        trial_oof = {arm: _classification(value, labels[~keep], canonical, old, held_ids,
                         np.full(len(held_ids), trial)) for arm, value in partial.items()}
        entry['oof'], entry['paired'] = _compact_proxy(trial_oof, canonical)
        difference = float(np.max(np.abs(partial['within_metric']-partial['kernel_ncm'])))
        entry['ncm_equivalence'] = dict(max_abs_score_difference=difference,
            tolerance=_CONFIG['oneshot_proxy_ncm_equivalence_atol'],
            equivalent=difference <= _CONFIG['oneshot_proxy_ncm_equivalence_atol'],
            predicted_classes_equal=bool(np.array_equal(partial['within_metric'].argmax(axis=1), partial['kernel_ncm'].argmax(axis=1))))
        proxy['factorization_count'] += sum(stage['factorization_calls'] for stage in entry['stages'])
        proxy['trials'].append(entry)
    result['oneshot_proxy'] = proxy
    result['factorization_count'] = result['standard_factorization_count']+proxy['factorization_count']
    result['fit_seconds'] = time.perf_counter()-started
    return result
