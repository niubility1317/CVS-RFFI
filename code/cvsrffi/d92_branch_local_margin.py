"""Train-only fixed LocalRidge geometry with a certified multiclass margin head.

All optimization uses current physical support. No I/O or query fit exists here.
"""
from copy import deepcopy
import math
import time
import numpy as np
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_support_probe import _classification, _finite


FROZEN_CONFIG = dict(
    method='D92-BranchLocalMargin-v1', schema='d92_branch_local_margin_v1',
    views='original_received_only', arms=['branch_ridge', 'interaction_ridge', 'local_ridge', 'local_margin'],
    selected='local_margin', selection='fixed_no_selection', direct_baseline='local_ridge',
    background=local.FROZEN_CONFIG['background'], auxiliary=local.FROZEN_CONFIG['auxiliary'],
    branches=['t_emb', 'f_emb', 'pa_local'], identity_dim=160, fft_dim=96,
    input_feature_dim=736, implicit_feature_dim=123616, norm_floor=1e-12,
    fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8,
    distance_rule=local.FROZEN_CONFIG['distance_rule'], distance_pair_chunk=256,
    bandwidth_rule=local.FROZEN_CONFIG['bandwidth_rule'], bandwidth_floor=None,
    kernel=local.FROZEN_CONFIG['kernel'], zero_bandwidth='exact_feature_equivalence_kernel',
    trace_match=local.FROZEN_CONFIG['trace_match'],
    centering=local.FROZEN_CONFIG['centering'],
    objective='0.5*RKHS_norm_squared+0.5*sum_physical_max_rival_hinge_squared',
    margin=1., physical_loss_weight=1., sample_weight=1., ridge_coefficient=1.,
    intercept=False, class_role_weights=False, temperature=None,
    solver='float64_deterministic_cyclic_exact_dual_row_block',
    initialization='beta_zero_every_fit', row_order='physical_id_lexicographic',
    max_sweeps=1000, optimization_tolerance='sqrt(eps64)',
    certificate='relative_primal_dual_gap_and_relative_projected_KKT',
    kkt_scale='max(1,max_row_beta_sum)', rounding_tolerance='128*eps64*max(N,C)',
    full_score_recompute='after_every_complete_sweep', dense_dual_hessian=False,
    max_folds=3, physical_folds=local.FROZEN_CONFIG['physical_folds'],
    oneshot_proxy_anchors=local.FROZEN_CONFIG['oneshot_proxy_anchors'],
    k1='same_full_support_fit_no_independent_holdout', probe_k1='numerical_only_no_fit_no_holdout',
    query_decision_policy='per_sample_all_registered_classes', tie_break='physical_class_id_lexicographic',
    optimizer_step_unit='physical_row_block_solve', learning_rate=None,
    source_inputs=False, summary_inputs=False, query_fit=False, phase1_frozen=True)
_CONFIG = deepcopy(FROZEN_CONFIG)
_ARMS = tuple(_CONFIG['arms'])
_PAIRS = local._PAIRS+tuple(('local_margin', name) for name in _ARMS[:-1])
_ETA = math.sqrt(np.finfo(np.float64).eps)


class NumericalFailure(local.NumericalFailure):
    def audit_dict(self):
        # Preserve nonfinite failure state explicitly while permitting strict JSON.
        def safe(value):
            if isinstance(value, dict):
                return {key: safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(item) for item in value]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return 'NaN' if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
            return value
        return safe(deepcopy(self.audit))


TechnicalFailure = NumericalFailure


def _fsum(values):
    return math.fsum(float(v) for v in values)


def _row_solution(a, diagonal):
    """Unique nonnegative row-QP minimizer, without a full dual Hessian."""
    a = np.asarray(a, dtype=np.float64)
    _finite(a, diagonal)
    if diagonal <= 0:
        raise FloatingPointError('NONPOSITIVE_NONDEGENERATE_KERNEL_DIAGONAL')
    if not len(a) or a.max() <= 0:
        return np.zeros_like(a)
    order = np.argsort(-a, kind='stable')
    sorted_a = a[order]
    for m in range(1, len(a)+1):
        total = _fsum(sorted_a[:m])
        mass = total/(diagonal+m*(diagonal+1.))
        center = total/m
        # Scale the comparison by d so an equal-rival positive mass survives
        # even when d*mass underflows. Overflow only fixes the comparison sign.
        with np.errstate(over='ignore'):
            last = (sorted_a[m-1]-center)/diagonal+mass/m
            next_value = (sorted_a[m]-center)/diagonal+mass/m if m < len(a) else -np.inf
        if mass > 0 and last > 0 and next_value <= 0:
            value = (sorted_a[:m]-center)/diagonal+mass/m
            result = np.zeros_like(a)
            result[order[:m]] = np.maximum(value, 0.)
            _finite(result)
            return result
    raise FloatingPointError('NO_VALID_ROW_ACTIVE_SET')


def _alpha(beta, labels, rivals, c):
    n = len(labels)
    masses = np.array([_fsum(row) for row in beta])
    alpha = np.zeros((n, c), dtype=np.float64)
    alpha[np.arange(n), labels] = masses
    alpha[np.arange(n)[:, None], rivals] = -beta
    return alpha, masses


def _certificate(gram, beta, alpha, scores, labels, rivals):
    n, c = scores.shape
    masses = np.array([_fsum(row) for row in beta])
    true = scores[np.arange(n), labels]
    wrong = scores[np.arange(n)[:, None], rivals]
    slack = np.maximum(0., 1.+wrong.max(axis=1)-true)
    quadratic = float(np.sum(alpha*scores))
    loss_ridge = .5*quadratic
    loss_data = .5*float(np.sum(slack*slack))
    primal = loss_ridge+loss_data
    dual = _fsum(masses)-.5*float(np.sum(masses*masses))-loss_ridge
    gap = primal-dual
    scale = max(1., abs(primal), abs(dual))
    gradient = masses[:, None]+true[:, None]-wrong-1.
    residual = np.minimum(beta, gradient)
    kkt = float(np.max(np.abs(residual)))
    kkt_scale = max(1., float(masses.max()))
    rounding = 128*np.finfo(np.float64).eps*max(n, c)
    _finite(beta, alpha, scores, masses, quadratic, primal, dual, gap, gradient, kkt)
    if np.any(beta < 0):
        raise FloatingPointError('DUAL_NONNEGATIVITY_VIOLATION')
    if gap < -rounding*scale:
        raise FloatingPointError('NEGATIVE_DUALITY_GAP_EXCEEDS_ROUNDING')
    if quadratic < -rounding*max(1., abs(quadratic)):
        raise FloatingPointError('NEGATIVE_RKHS_NORM_EXCEEDS_ROUNDING')
    relative_gap = max(0., gap)/scale
    relative_kkt = kkt/kkt_scale
    return dict(loss_data=loss_data, loss_ridge=loss_ridge, loss_total=primal,
        primal_objective=primal, dual_objective=dual, duality_gap=gap,
        relative_duality_gap=relative_gap, duality_gap_scale=scale,
        kkt_residual=kkt, relative_kkt_residual=relative_kkt, kkt_scale=kkt_scale,
        gradient_norm=float(np.linalg.norm(gradient)), gradient_coordinate='negative_dual_beta',
        active_constraint_count=int(np.count_nonzero(beta)),
        active_physical_count=int(np.count_nonzero(masses)),
        slack_min=float(slack.min()), slack_mean=float(slack.mean()), slack_max=float(slack.max()),
        squared_slack_sum=float(np.sum(slack*slack)),
        training_accuracy=float(np.mean(scores.argmax(axis=1) == labels)),
        optimization_tolerance=_ETA, objective_rounding_tolerance=rounding,
        certified=relative_gap <= _ETA and relative_kkt <= _ETA)


def _optimize(gram, labels, c, *, context=None, log_callback=None, max_sweeps=None):
    """Internal override of sweep cap is solely for bounded failure tests."""
    context = dict(context or {})
    n = len(labels)
    cap = _CONFIG['max_sweeps'] if max_sweeps is None else max_sweeps
    rivals = np.array([[j for j in range(c) if j != y] for y in labels], dtype=np.int64)
    beta = np.zeros((n, c-1), dtype=np.float64)
    alpha = np.zeros((n, c), dtype=np.float64)
    scores = np.zeros_like(alpha)
    trace, steps = [], 0
    started = time.perf_counter()
    previous_negative_dual = 0.
    try:
        _finite(gram)
        if c < 2 or np.any(np.diag(gram) <= 0):
            raise FloatingPointError('NONPOSITIVE_NONDEGENERATE_KERNEL_DIAGONAL')
        for sweep in range(1, cap+1):
            begin = time.perf_counter()
            for i in range(n):
                diagonal = float(gram[i, i])
                without = scores[i]-diagonal*alpha[i]
                a = 1.-without[labels[i]]+without[rivals[i]]
                beta[i] = _row_solution(a, diagonal)
                updated = np.zeros(c, dtype=np.float64)
                updated[labels[i]] = _fsum(beta[i])
                updated[rivals[i]] = -beta[i]
                delta = updated-alpha[i]
                alpha[i] = updated
                scores += gram[:, i, None]*delta[None, :]
                steps += 1
            scores = gram@alpha
            certificate = _certificate(gram, beta, alpha, scores, labels, rivals)
            negative_dual = -certificate['dual_objective']
            scale = max(1., abs(previous_negative_dual), abs(negative_dual))
            if negative_dual-previous_negative_dual > certificate['objective_rounding_tolerance']*scale:
                raise FloatingPointError('DUAL_OBJECTIVE_INCREASE_EXCEEDS_ROUNDING')
            event = dict(context, event='sweep', arm='local_margin', sweep=sweep, epoch=sweep,
                optimizer_steps=steps, row_block_solves=steps, learning_rate=None,
                optimizer=_CONFIG['solver'], sweep_seconds=time.perf_counter()-begin,
                solve_seconds=time.perf_counter()-started,
                negative_dual_change=negative_dual-previous_negative_dual, **certificate)
            trace.append(event)
            if log_callback is not None:
                log_callback(deepcopy(event))
            previous_negative_dual = negative_dual
            if certificate['certified']:
                return alpha, dict(certificate, optimization_trace=trace, sweeps=sweep,
                    optimizer_steps=steps, row_block_solves=steps, solve_seconds=time.perf_counter()-started)
        raise FloatingPointError('MAX_SWEEPS_WITHOUT_CERTIFICATE')
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        failure = dict(context, arm='local_margin', status='TECHNICAL_FAILURE', failure_reason=str(exc),
            optimizer_steps=steps, row_block_solves=steps, completed_sweeps=len(trace),
            optimization_trace=trace, factorization_calls=0, solve_seconds=time.perf_counter()-started,
            solver_state=dict(beta=beta.tolist(), alpha=alpha.tolist(), F=scores.tolist(),
                              rivals=rivals.tolist(), labels=np.asarray(labels).tolist()))
        raise NumericalFailure(str(exc), failure) from exc


def _solve(b, a, labels, ids, c, *, context=None, log_callback=None):
    started = time.perf_counter()
    n, k = len(labels), len(labels)//c
    supplied_context = context or {}
    context = dict(scope='final', parent_k=k, train_k=k, fold=None, trial=None)
    context.update(supplied_context)
    audit = dict(context, arm='local_margin', train_physical_count=n,
        training_physical_ids=list(ids), factorization_calls=0, factorization_dim=0,
        completed_stages=[], optimizer_steps=0, distance_rule=_CONFIG['distance_rule'],
        bandwidth_rule=_CONFIG['bandwidth_rule'])
    try:
        distance, s0, tau, delta, reason = local._geometry(b, a, labels, c)
        numerical_tolerance = float(128*np.finfo(np.float64).eps*max(n, c))
        audit.update(bandwidth_tau=tau, interaction_centered_trace=s0, degeneracy_reason=reason,
                     numerical_tolerance=numerical_tolerance)
        reference = np.zeros(n)
        mean = np.zeros(n)
        reference_self = grand = 0.
        gamma, sradial = None, None
        gram = np.zeros((n, n))
        radial = None if tau is None else np.ones((n, n))
        trace_error = 0.
        if c > 1 and s0 > 0:
            rm1 = local._radial_minus_one(distance, tau)
            radial = local._radial(distance, tau)
            sradial = float(-2*np.sum(rm1[np.triu_indices(n, 1)])/n)
            if sradial <= 0:
                raise FloatingPointError('NONPOSITIVE_RADIAL_CENTERED_TRACE')
            gamma = s0/sradial
            _finite(gamma)
            gram, reference, reference_self, mean, grand = interaction._center_kernel(rm1)
            gram *= gamma
            trace_error = abs(float(np.trace(gram))-s0)/s0
            _finite(gram, trace_error)
            if trace_error > numerical_tolerance:
                raise FloatingPointError('CENTERED_TRACE_RESIDUAL_EXCEEDED')
        gram_seconds = time.perf_counter()-started
        if gamma is None:
            alpha = np.zeros((n, c))
            loss = n/2. if c > 1 else 0.
            optimization = dict(loss_data=loss, loss_ridge=0., loss_total=loss,
                primal_objective=loss, dual_objective=loss, duality_gap=0., relative_duality_gap=0.,
                duality_gap_scale=max(1., loss), kkt_residual=0., relative_kkt_residual=0., kkt_scale=1.,
                gradient_norm=0., gradient_coordinate='analytic_dual_certificate',
                active_constraint_count=n*(c-1), active_physical_count=n if c > 1 else 0,
                slack_min=1. if c > 1 else 0., slack_mean=1. if c > 1 else 0., slack_max=1. if c > 1 else 0.,
                squared_slack_sum=float(n) if c > 1 else 0., optimization_tolerance=_ETA,
                objective_rounding_tolerance=numerical_tolerance, certified=True,
                optimization_trace=[], sweeps=0, optimizer_steps=0, row_block_solves=0, solve_seconds=0.,
                training_accuracy=float(np.mean(labels == 0)),
                analytic_certificate='beta_ic=1/(C-1),s_i=1,W=0; stored alpha=0 is equivalent RKHS function' if c > 1 else 'No rival constraints; W=0,P=D=0')
        else:
            optimizer_context = dict(context, bandwidth_tau=tau, trace_scale=gamma,
                interaction_centered_trace=s0, radial_centered_trace=sradial,
                train_physical_count=n, class_count=c, max_sweeps=_CONFIG['max_sweeps'])
            alpha, optimization = _optimize(gram, labels, c, context=optimizer_context, log_callback=log_callback)
        audit.update(optimization)
        audit.update(status='CERTIFIED_MARGIN_SOLVED' if gamma is not None else 'EXACT_ZERO_CLASSIFIER',
            solver=_CONFIG['solver'] if gamma is not None else 'NO_OPTIMIZATION_ANALYTIC_ZERO',
            bandwidth_zero=tau == 0 if tau is not None else None,
            nearest_other_class_squared_distance=local._stats(delta),
            zero_nearest_other_class_fraction=float(np.mean(delta == 0)) if len(delta) else None,
            radial_centered_trace=sradial, trace_scale=gamma, trace_relative_error=trace_error,
            physical_loss_mass=float(n), physical_loss_weight=1., sample_weight=1., ridge_coefficient=1.,
            margin=1., intercept=False, learning_rate=None, epoch=optimization['sweeps'],
            normal_equation_residual=None, normal_equation_residual_reason='Margin QP uses duality gap and KKT, not ridge normal equations',
            effective_degrees_of_freedom=None,
            effective_degrees_of_freedom_reason='Piecewise nonlinear margin estimator has no fixed ridge hat matrix',
            input_feature_dim=736, implicit_feature_dim=123616, output_dim=c, classification_output_dim=c,
            training_feature_bytes=int(b.nbytes+a.nbytes), coefficient_bytes=int(alpha.nbytes),
            kernel_bytes=int(gram.nbytes), gram_bytes=int(gram.nbytes),
            dual_hessian_materialized=False, maximum_dual_variable_count=n*(c-1),
            centered_training_diagonal=local._stats(np.diag(gram)),
            training_offdiagonal_radial=local._stats(radial[~np.eye(n, dtype=bool)]) if radial is not None else None,
            training_radial_row_sum=local._stats(radial.sum(axis=1)) if radial is not None else None,
            degenerate_constant_features=s0 == 0, all_states_estimated_from_trainfold_only=True,
            gram_seconds=gram_seconds, fit_seconds=time.perf_counter()-started)
        return (alpha, reference, reference_self, mean, grand, tau, gamma), audit
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        failure = exc.audit_dict() if isinstance(exc, NumericalFailure) else {}
        audit.update(failure)
        audit.update(context, arm='local_margin', status='TECHNICAL_FAILURE', failure_reason=str(exc),
                     training_physical_ids=list(ids), train_physical_count=n, fit_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc), audit) from exc


def _fit_state(args, *, context=None, log_callback=None):
    started = time.perf_counter()
    b, a, labels, ids, canonical, requested, old, k = interaction._prepare(
        *(args[name] for name in local._NAMES), args['support_labels'], args['support_ids'], args['classes'], args['old_classes'])
    try:
        parts, audit = _solve(b, a, labels, ids, len(canonical), context=context, log_callback=log_callback)
    except NumericalFailure as exc:
        exc.audit.update(classes=list(requested), factorization_count=0)
        raise
    alpha, reference, reference_self, mean, grand, tau, gamma = parts
    alpha = alpha[:, [canonical.index(cls) for cls in requested]]
    arrays = dict(support_background=b, support_auxiliary=a, alpha=alpha, reference_kernel=reference, center_mean=mean)
    sizes = {key: int(value.nbytes) for key, value in arrays.items()}
    scalar_bytes = 16+8*int(tau is not None)+8*int(gamma is not None)
    size = sum(sizes.values())+scalar_bytes
    audit['all_states_estimated_from_current_support_only'] = True
    top = dict(config=deepcopy(_CONFIG), classes=list(requested), old_classes=sorted(old), selected='local_margin',
        selection='fixed_no_selection', k=k, support_count=len(ids), final_fit=audit,
        numerical=interaction._diagonal_stats(b, a), factorization_count=0,
        fold_count=0, folds=[], oof=None, paired=None, oneshot_proxy=None,
        optimizer_steps=audit['optimizer_steps'], sweep_count=audit['sweeps'],
        persistent_state_bytes=size, head_bytes=size, state_array_bytes=sizes, state_scalar_bytes=scalar_bytes,
        state_byte_scope='five_float64_arrays_and_present_float64_scalars_excludes_registry_audit_and_optimizer_work_arrays',
        current_support_feature_bytes=int(b.nbytes+a.nbytes),
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG',
        query_rows_used=0, source_rows_used=0, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        source_validation=None, source_validation_reason='No source samples or source feature banks',
        phase1_frozen=True, fit_seconds=time.perf_counter()-started)
    return local.BranchLocalRidgeState(b, a, alpha, reference, reference_self, mean, grand, tau, gamma, requested, top)


def fit_branch_local_margin(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids,
                            classes, old_classes=(), arm='local_margin', log_callback=None):
    args = dict(z_id=z_id, fft=fft, t_emb=t_emb, f_emb=f_emb, pa_local=pa_local,
                support_labels=support_labels, support_ids=support_ids, classes=classes, old_classes=old_classes)
    if arm in _ARMS[:-1]:
        return local.fit_branch_local_ridge(**args, arm=arm)
    if arm != 'local_margin':
        raise ValueError('Unknown frozen local margin arm')
    return _fit_state(args, log_callback=log_callback)


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
        record['mean_correct_delta'] = float(np.mean([r['correct_delta'] for r in rows]))
        result[left+'_minus_'+right] = record
    return result


def _compact(oof, classes):
    # The frozen helper compacts any supplied arm; its fixed old pairs remain valid.
    compact, _ = local._compact_proxy(oof, classes)
    return compact, _paired(oof, compact=True)


def _evaluate_fold(raw, labels, ids, classes, old, keep, *, scope, parent_k, fold=None, trial=None, log_callback=None):
    try:
        entry, scores = local._evaluate_fold(raw, labels, ids, classes, old, keep,
            scope=scope, parent_k=parent_k, fold=fold, trial=trial)
    except local.NumericalFailure as exc:
        raise NumericalFailure(str(exc), exc.audit_dict()) from exc
    context = dict(scope=scope, parent_k=parent_k, train_k=entry['train_k'], fold=fold, trial=trial)
    train = {name: value[keep] for name, value in raw.items()}
    held = {name: value[~keep] for name, value in raw.items()}
    audit = None
    try:
        args = dict(**train, support_labels=labels[keep], support_ids=entry['training_ids'], classes=classes, old_classes=old)
        state = _fit_state(args, context=context, log_callback=log_callback)
        audit = state.audit_dict()['final_fit']
        begin = time.perf_counter()
        scores['local_margin'] = state.score(**held)
        audit['score_seconds'] = time.perf_counter()-begin
        audit['held_accuracy'] = float(np.mean(scores['local_margin'].argmax(axis=1) == labels[~keep]))
        audit['training_minus_held_accuracy'] = audit['training_accuracy']-audit['held_accuracy']
        true = scores['local_margin'][np.arange((~keep).sum()), labels[~keep]]
        if len(classes) > 1:
            wrong = scores['local_margin'].copy()
            wrong[np.arange(len(wrong)), labels[~keep]] = -np.inf
            margin = true-wrong.max(axis=1)
            audit['held_true_minus_max_wrong_score'] = local._stats(margin)
        else:
            audit['held_true_minus_max_wrong_score'] = None
        hb, ha = interaction._blocks(**held)
        if state.bandwidth_tau is not None:
            distance = local._distances(hb, ha, state.support_background, state.support_auxiliary)
            radial = local._radial(distance, state.bandwidth_tau)
            correct = np.array([row[labels[keep] == y].max() for row, y in zip(radial, labels[~keep])])
            wrong = np.array([row[labels[keep] != y].max() for row, y in zip(radial, labels[~keep])])
            audit['held_max_radial_similarity'] = local._stats(radial.max(axis=1))
            audit['held_correct_minus_nearest_wrong_similarity'] = local._stats(correct-wrong)
            audit['held_no_exact_match_fraction'] = float(np.mean(~np.any(distance == 0, axis=1))) if state.bandwidth_tau == 0 else None
        else:
            audit.update(held_max_radial_similarity=None, held_correct_minus_nearest_wrong_similarity=None, held_no_exact_match_fraction=None)
        entry['stages'].append(audit)
        return entry, scores
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        failure = exc.audit_dict() if isinstance(exc, NumericalFailure) else dict(audit or {})
        failure.update(context, arm='local_margin', status='TECHNICAL_FAILURE', failure_reason=str(exc),
            training_physical_ids=entry['training_ids'], held_ids=entry['held_ids'], completed_stages=deepcopy(entry['stages']))
        raise NumericalFailure(str(exc), failure) from exc


def probe_branch_local_margin(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids,
                              classes, old_classes=(), log_callback=None):
    started = time.perf_counter()
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {name: np.asarray(value)[order] for name, value in zip(local._NAMES, (z_id, fft, t_emb, f_emb, pa_local))}
    n, c = len(ids), len(canonical)
    folds = 0 if k == 1 else min(k, 3)
    result = dict(config=deepcopy(_CONFIG), classes=list(canonical), old_classes=sorted(old), k=k, parent_k=k,
        support_count=n, numerical=interaction._diagonal_stats(b, a), fold_count=folds,
        folds=[], oof=None, paired=None, oneshot_proxy=None, physical_fold_assignment=[],
        factorization_count=0, standard_factorization_count=0, optimizer_steps=0, sweep_count=0, persistent_state_bytes=0,
        claim_scope='SUPPORT_OOF_AND_SUPPORT_ONESHOT_PROXY_NOT_QUERY_EVALUATION',
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    completed = []
    def evaluate(keep, scope, fold=None, trial=None):
        try:
            entry, values = _evaluate_fold(raw, labels, ids, canonical, old, keep,
                scope=scope, parent_k=k, fold=fold, trial=trial, log_callback=log_callback)
        except NumericalFailure as exc:
            exc.audit['completed_stages'] = deepcopy(completed)+exc.audit.get('completed_stages', [])
            exc.audit['factorization_count'] = sum(s.get('factorization_calls', 0) for s in exc.audit['completed_stages'])+exc.audit.get('factorization_calls', 0)
            exc.audit['completed_optimizer_steps'] = sum(s.get('optimizer_steps', 0) for s in exc.audit['completed_stages'])+exc.audit.get('optimizer_steps', 0)
            raise
        completed.extend(entry['stages'])
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
    result['standard_factorization_count'] = sum(s['factorization_calls'] for s in completed)
    proxy = dict(parent_k=k, proxy_train_k=1, trial_count=k, trials=[],
        scope='SUPPORT_ONESHOT_PROXY_FROM_PARENT_SUPPORT_NOT_FORMAL_K1',
        coverage=dict(parent_physical_count=n, training_occurrences=n, held_occurrences=n*(k-1), unique_held_physical_count=n),
        factorization_count=0)
    for trial in range(k):
        keep = np.zeros(n, dtype=bool)
        for indices in positions:
            keep[indices[trial]] = True
        entry, partial = evaluate(keep, 'support_oneshot_proxy', trial=trial)
        entry.update(trial=trial, proxy_train_k=1)
        held_ids = tuple(pid for i, pid in enumerate(ids) if not keep[i])
        trial_oof = {arm: _classification(value, labels[~keep], canonical, old, held_ids,
            np.full(len(held_ids), trial)) for arm, value in partial.items()}
        entry['oof'], entry['paired'] = _compact(trial_oof, canonical)
        proxy['factorization_count'] += sum(s['factorization_calls'] for s in entry['stages'])
        proxy['trials'].append(entry)
    result['oneshot_proxy'] = proxy
    result['factorization_count'] = result['standard_factorization_count']+proxy['factorization_count']
    result['optimizer_steps'] = sum(s['optimizer_steps'] for s in completed)
    result['sweep_count'] = sum(s.get('sweeps', 0) for s in completed)
    result['fit_seconds'] = time.perf_counter()-started
    return result
