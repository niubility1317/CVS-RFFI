"""Fixed support-only branch probes. No query API, I/O, or deployable state.

Ridge is one against a SUM of physical squared errors. Six predetermined arms
include a repeated-background control; none is selected by this diagnostic.
"""
from copy import deepcopy
import time
import numpy as np


FROZEN_CONFIG = dict(
    method='D92-BranchSupportProbe-v1', schema='d92_branch_support_probe_v1',
    views='original_received_only', branches=['t_emb', 'f_emb', 'pa_local'],
    disabled_branches=['dac_local'], identity_dim=160, fft_dim=96,
    aux_dim=480, norm_floor=1e-12,
    background_z='unit(z_id)', background_zfft='unit(concat(unit(z_id),4*unit(fft96)))',
    auxiliary='concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)',
    arms=['z', 'z_duplicate', 'z_aux', 'zfft', 'zfft_duplicate', 'zfft_aux'],
    concatenate_renormalize=False, duplicate='concat(B,B)',
    target='onehot_minus_1_over_C',
    objective='0.5*sum_physical_squared_error+0.5*||W||F^2',
    ridge_coefficient=1.0, intercept_regularized=False, sample_weight=1.0,
    solver='float64_Cholesky_primal_if_D_le_N_else_dual',
    max_folds=3, folds='per_class_physical_id_sort_position_mod_min_K_3',
    k1='numerical_only_no_fit_no_holdout', selection='none_diagnostic_only',
    reconstruction='same_ridge_B_to_A_trainfold_centered_joint_RHS',
    numerical_rank='max(N,D)*eps64*max_singular_value',
    optimizer_steps=0, source_inputs=False, summary_inputs=False, query_inputs=False,
    claim_scope='SUPPORT_OOF_DIAGNOSTIC_NOT_QUERY_EVALUATION',
)
_CONFIG = deepcopy(FROZEN_CONFIG)
_EPS = 1e-12


def _finite(*values):
    if any(not np.isfinite(v).all() for v in values):
        raise FloatingPointError('Nonfinite branch support probe computation')


def _numeric(value, dim, name):
    x = np.asarray(value)
    if x.ndim != 2 or x.shape[1] != dim or not len(x) or x.dtype.kind not in 'fiu':
        raise ValueError(name + ' must be a nonempty real numeric matrix with width ' + str(dim))
    if not np.isfinite(x).all():
        raise ValueError(name + ' must be finite')
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(name + ' cannot be represented in float64')
    return x


def _strings(value, name, nonempty=True):
    if isinstance(value, (str, bytes)):
        raise ValueError(name + ' must be a sequence')
    try:
        v = tuple(value)
    except TypeError as exc:
        raise ValueError(name + ' must be a sequence') from exc
    if ((nonempty and not v) or any(not isinstance(a, str) or not a for a in v)
            or len(set(v)) != len(v)):
        raise ValueError(name + ' must contain unique nonempty strings')
    return v


def _unit(x):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        norms = np.sqrt(np.sum(x*x, axis=1, keepdims=True))
        value = x / np.maximum(norms, _EPS)
    _finite(value)
    return value


def _feature_blocks(z_id, fft, t_emb, f_emb, pa_local):
    raw = {name: _numeric(value, dim, name) for name, value, dim in (
        ('z_id', z_id, 160), ('fft', fft, 96), ('t_emb', t_emb, 160),
        ('f_emb', f_emb, 160), ('pa_local', pa_local, 160))}
    if len({len(v) for v in raw.values()}) != 1:
        raise ValueError('All feature arrays must have the same physical count')
    u = {name: _unit(value) for name, value in raw.items()}
    aux = np.concatenate([u[name] for name in _CONFIG['branches']], axis=1) / np.sqrt(3.)
    bases = dict(z=u['z_id'], zfft=_unit(np.concatenate((u['z_id'], 4*u['fft']), axis=1)))
    return raw, bases, aux


def _center(x):
    # Reference differences give exactly zero for identical floating-point rows.
    differences = x-x[0]
    offset = differences.mean(axis=0)
    return x[0]+offset, differences-offset


def _spectrum(x):
    _, xc = _center(x)
    singular = np.linalg.svd(xc, compute_uv=False)
    top = float(singular[0]) if len(singular) else 0.
    energy = float(np.sum(xc*xc))
    threshold = float(max(xc.shape)*np.finfo(np.float64).eps*top)
    _finite(singular, energy)
    return dict(dimension=int(x.shape[1]), centered_energy=energy,
                stable_rank=float(energy/(top*top)) if top else 0.,
                numerical_rank=int(np.sum(singular > threshold)),
                rank_threshold=threshold, largest_singular_value=top,
                mean_squared_norm=float(np.mean(np.sum(x*x, axis=1))))


def _numerical(raw, bases, aux):
    started = time.perf_counter()
    norms = {}
    for name, x in raw.items():
        with np.errstate(over='raise', invalid='raise'):
            norm = np.sqrt(np.sum(x*x, axis=1))
        _finite(norm)
        norms[name] = dict(min_norm=float(norm.min()), max_norm=float(norm.max()),
                           mean_norm=float(norm.mean()), exact_zero_count=int(np.sum(norm == 0)),
                           below_norm_floor_count=int(np.sum(norm < _EPS)))
    spectra = {'aux': _spectrum(aux)}
    energies = {}
    for name, b in bases.items():
        spectra[name] = _spectrum(b)
        spectra[name+'_aux'] = _spectrum(np.concatenate((b, aux), axis=1))
        be = float(np.mean(np.sum(b*b, axis=1)))
        ae = float(np.mean(np.sum(aux*aux, axis=1)))
        energies[name] = dict(base=be, duplicate=2*be, auxiliary=be+ae,
                              auxiliary_minus_duplicate=ae-be)
    return dict(raw_norms=norms, spectra=spectra, arm_mean_squared_norm=energies,
                spectrum_decomposition_count=5, seconds=time.perf_counter()-started,
                interpretation='arithmetic_structure_not_predictive_or_causal_information')


def _ridge_fit(x, targets, ridge=1.):
    """Internal primitive; ridge override is for equivalence tests, not public fit."""
    started = time.perf_counter()
    n, d = x.shape
    center, xc = _center(x)
    target_center, tc = _center(targets)
    center_seconds = time.perf_counter()-started
    begin = time.perf_counter()
    primal = d <= n
    gram = (xc.T@xc if primal else xc@xc.T) + ridge*np.eye(min(n, d))
    rhs = xc.T@tc if primal else tc
    _finite(gram, rhs)
    gram_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    chol = np.linalg.cholesky(gram)
    solution = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    w = solution if primal else xc.T@solution
    b = target_center-center@w
    _finite(w, b)
    solve_seconds = time.perf_counter()-begin
    audit = dict(solver='primal' if primal else 'dual', factorization_dim=min(n, d),
                 factorization_calls=1, condition_bound=float(1+np.sum(xc*xc)/ridge),
                 gram_min_eigenvalue_lower_bound=float(ridge),
                 center_seconds=center_seconds, gram_seconds=gram_seconds, solve_seconds=solve_seconds,
                 gram_bytes=int(gram.nbytes), cross_moment_bytes=int(rhs.nbytes),
                 training_feature_bytes=int(x.nbytes), coefficient_bytes=int(w.nbytes),
                 intercept_bytes=int(b.nbytes), temporary_state_bytes=int(w.nbytes+b.nbytes))
    return w, b, audit


def _objective(x, target, w, b):
    residual = x@w+b-target
    grad_w = x.T@residual+w
    grad_b = residual.sum(axis=0)
    _, xc = _center(x)
    _, tc = _center(target)
    normal = xc.T@(xc@w-tc)+w
    data = float(.5*np.sum(residual*residual))
    penalty = float(.5*np.sum(w*w))
    grad = float(np.sqrt(np.sum(grad_w*grad_w)+np.sum(grad_b*grad_b)))
    _finite(data, penalty, grad, normal)
    return dict(loss_data=data, loss_ridge=penalty, loss_total=data+penalty,
                target_norm_squared=float(np.sum(target*target)), gradient_norm=grad,
                normal_equation_residual=float(np.linalg.norm(normal)),
                intercept_gradient_norm=float(np.linalg.norm(grad_b)))


def _fit_stage(x, y, aux, *, background, arm, train_k, classification_dim):
    started = time.perf_counter()
    reconstruct = arm == background
    targets = np.concatenate((y, aux), axis=1) if reconstruct else y
    w, b, audit = _ridge_fit(x, targets)
    begin = time.perf_counter()
    audit.update(_objective(x, y, w[:, :classification_dim], b[:classification_dim]))
    recon = _objective(x, aux, w[:, classification_dim:], b[classification_dim:]) if reconstruct else None
    for key in ('loss_data', 'loss_ridge', 'loss_total', 'target_norm_squared', 'gradient_norm',
                'normal_equation_residual', 'intercept_gradient_norm'):
        audit['reconstruction_'+key] = recon[key] if recon else None
    audit.update(arm=arm, background=background, design_dim=int(x.shape[1]),
                 output_dim=int(targets.shape[1]), classification_output_dim=classification_dim,
                 train_physical_count=len(x), train_k=train_k, physical_loss_mass=float(len(x)),
                 sample_weight=1., ridge_coefficient=1., optimizer_steps=0, learning_rate=None,
                 epoch=None, optimization_reason='CLOSED_FORM_NO_ITERATIVE_OPTIMIZER',
                 all_states_estimated_from_trainfold_only=True,
                 objective_seconds=time.perf_counter()-begin, fit_seconds=time.perf_counter()-started,
                 status='CLOSED_FORM_SOLVED')
    return w, b, audit


def _classification(scores, labels, classes, old, ids, folds):
    maximum = scores.max(axis=1)
    nll = maximum+np.log(np.exp(scores-maximum[:, None]).sum(axis=1))-scores[np.arange(len(scores)), labels]
    predictions = np.argmax(scores, axis=1)  # canonical physical class order
    correct = predictions == labels
    _finite(nll)
    classwise = [dict(class_id=cls, count=int(np.sum(labels == i)),
                      accuracy=float(correct[labels == i].mean()), nll=float(nll[labels == i].mean()))
                 for i, cls in enumerate(classes)]
    old_acc = [v['accuracy'] for v in classwise if v['class_id'] in old]
    new_acc = [v['accuracy'] for v in classwise if v['class_id'] not in old]
    o = float(np.mean(old_acc)) if old_acc else None
    new = float(np.mean(new_acc)) if new_acc else None
    h = (2*o*new/(o+new) if o+new else 0.) if o is not None and new is not None else None
    rows = [dict(physical_id=pid, class_id=classes[int(labels[i])], fold=int(folds[i]),
                 predicted_class=classes[int(predictions[i])], correct=bool(correct[i]), nll=float(nll[i]))
            for i, pid in enumerate(ids)]
    return dict(rows=rows, metrics=dict(accuracy=float(correct.mean()),
                macro_accuracy=float(np.mean([v['accuracy'] for v in classwise])), old_accuracy=o,
                new_accuracy=new, h=h, macro_nll=float(nll.mean()), classwise=classwise,
                nll_interpretation='fixed_softmax_of_uncalibrated_ridge_scores'))


def probe_branch_support(*, z_id, fft, t_emb, f_emb, pa_local,
                         support_labels, support_ids, classes, old_classes=()):
    """Diagnose one row; no data from another row or query can enter this API."""
    started = time.perf_counter()
    raw, bases, aux = _feature_blocks(z_id, fft, t_emb, f_emb, pa_local)
    ids = _strings(support_ids, 'support_ids')
    requested = _strings(classes, 'classes')
    old = _strings(old_classes, 'old_classes', nonempty=False)
    labels = np.asarray(support_labels)
    n, c = len(aux), len(requested)
    if (len(ids) != n or labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or set(labels.tolist()) != set(range(c)) or not set(old).issubset(requested)):
        raise ValueError('Invalid support labels, identifiers, classes, or old membership')
    canonical = tuple(sorted(requested))
    mapping = np.asarray([canonical.index(v) for v in requested])
    labels = mapping[labels.astype(np.int64)]
    counts = np.bincount(labels, minlength=c)
    if np.any(counts != counts[0]):
        raise ValueError('Equal positive physical K required for every class')
    k = int(counts[0])
    order = np.asarray(sorted(range(n), key=lambda i: ids[i]))
    ids = tuple(ids[i] for i in order)
    labels = labels[order]
    bases = {name: value[order] for name, value in bases.items()}
    aux = aux[order]
    raw = {name: value[order] for name, value in raw.items()}
    result = dict(config=deepcopy(_CONFIG), classes=list(canonical), old_classes=sorted(old),
                  k=k, support_count=n, fold_count=0 if k == 1 else min(k, 3),
                  numerical=_numerical(raw, bases, aux), folds=[], oof=None, reconstruction=None,
                  paired=None, physical_fold_assignment=[], optimizer_steps=0, factorization_count=0,
                  persistent_state_bytes=0, claim_scope=_CONFIG['claim_scope'],
                  heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    assignments = np.full(n, -1, dtype=int)
    for cls in range(c):
        positions = np.flatnonzero(labels == cls)
        assignments[positions] = np.arange(k) % result['fold_count']
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=canonical[int(labels[i])],
                                                  fold=int(assignments[i])) for i, pid in enumerate(ids)]
    targets = np.eye(c)[labels]-1./c
    scores = {arm: np.empty((n, c)) for arm in _CONFIG['arms']}
    reconstruction = {name: [] for name in bases}
    for fold in range(result['fold_count']):
        keep = assignments != fold
        held = ~keep
        training_ids = [pid for i, pid in enumerate(ids) if keep[i]]
        held_ids = [pid for i, pid in enumerate(ids) if held[i]]
        train_k = int(np.sum(keep)//c)
        entry = dict(fold=fold, training_ids=training_ids, held_ids=held_ids,
                     train_k=train_k, stages=[])
        for name, base in bases.items():
            for suffix in ('', '_duplicate', '_aux'):
                arm = name+suffix
                x = base if not suffix else np.concatenate((base, base if suffix == '_duplicate' else aux), axis=1)
                w, b, audit = _fit_stage(x[keep], targets[keep], aux[keep], background=name,
                                         arm=arm, train_k=train_k, classification_dim=c)
                begin = time.perf_counter()
                scores[arm][held] = x[held]@w[:, :c]+b[:c]
                audit['score_seconds'] = time.perf_counter()-begin
                audit['reconstruction_score_seconds'] = None
                if not suffix:
                    begin = time.perf_counter()
                    predicted = x[held]@w[:, c:]+b[c:]
                    errors = np.sum((predicted-aux[held])**2, axis=1)
                    reference = np.sum((aux[keep].mean(axis=0)-aux[held])**2, axis=1)
                    _finite(errors, reference)
                    audit['reconstruction_score_seconds'] = time.perf_counter()-begin
                    reconstruction[name].extend(dict(physical_id=pid, fold=fold,
                        squared_error=float(errors[j]), mean_baseline_squared_error=float(reference[j]))
                        for j, pid in enumerate(held_ids))
                entry['stages'].append(audit)
        result['folds'].append(entry)
    result['factorization_count'] = sum(s['factorization_calls'] for f in result['folds'] for s in f['stages'])
    result['oof'] = {arm: _classification(value, labels, canonical, old, ids, assignments)
                     for arm, value in scores.items()}
    result['reconstruction'] = {}
    result['paired'] = {}
    for name, rows in reconstruction.items():
        rows.sort(key=lambda row: row['physical_id'])
        sse = float(sum(row['squared_error'] for row in rows))
        sst = float(sum(row['mean_baseline_squared_error'] for row in rows))
        result['reconstruction'][name] = dict(rows=rows, squared_error_sum=sse,
            mean_baseline_squared_error_sum=sst, r2_linear=1.-sse/sst if sst else None,
            r2_unavailable_reason=None if sst else 'ZERO_TRAINMEAN_REFERENCE_ERROR')
        result['paired'][name] = {}
        for key, left, right in (('aux_minus_base', name+'_aux', name),
                                 ('duplicate_minus_base', name+'_duplicate', name),
                                 ('aux_minus_duplicate', name+'_aux', name+'_duplicate')):
            lrows, rrows = result['oof'][left]['rows'], result['oof'][right]['rows']
            paired = [dict(physical_id=pid, correct_delta=int(lrows[i]['correct'])-int(rrows[i]['correct']))
                      for i, pid in enumerate(ids)]
            result['paired'][name][key] = dict(rows=paired,
                mean_correct_delta=float(np.mean([row['correct_delta'] for row in paired])))
    result['fit_seconds'] = time.perf_counter()-started
    return result
