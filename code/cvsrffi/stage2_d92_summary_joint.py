"""D92-SGJoint-v1: fixed summary normalization and physical-support-only CV.

No file I/O, source examples, teacher, query fitting, or cross-row state.
The caller verifies the existing summary's checkpoint/provenance binding.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .d92_ground_summary import GroundSummary, NUMERIC_MEMBERS
from . import phase1_center_lowrank_prototype_bundle as codec


FROZEN_CONFIG = dict(
    method='D92-SGJoint-v1', alphas=[0.0, 0.5, 1.0, 4.0], betas=[0, 1],
    shrinkages=[0.1, 1.0], max_folds=3,
    folds='per_class_physical_id_sort_then_position_mod_min_K_3',
    k1=dict(alpha=1.0, beta=1, shrinkage=1.0, cosine_scale=10.0, temperature=1.0),
    temperature_bounds=[0.1, 10.0], temperature_xatol=1e-8, temperature_maxiter=200,
    selection='min_max_old_new_oof_nll_then_macro_nll_then_stronger_shrinkage_smaller_alpha_beta',
    selection_round_decimals=10, class_prior='equal_1_over_registered_class_count',
    summary_formula='q=1/(1+rho); class_equal_weighted_domain_covariance; A=(I+160G/traceG)^(-1/2)',
    covariance='class_equal_unbiased_support_residual_covariance_then_spherical_shrinkage',
    training_score_scale='class_centered_training_score_rms_floor_1e-12',
    precision='float64', feature_schema='identity160+same_received_IQ_FFT96',
    norm_floor=1e-12, covariance_trace_floor=1e-24, new_source_payload=False,
)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _copy(value):
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy(item) for item in value]
    return value


_CONFIG = _freeze(FROZEN_CONFIG)


def _readonly(value):
    x = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)


def _strings(values, name, *, minimum=1):
    if isinstance(values, (str, bytes)):
        raise ValueError(name + ' must be an explicit sequence of physical identifiers')
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(name + ' must be a sequence') from exc
    if (len(result) < minimum or any(not isinstance(v, str) or not v for v in result)
            or len(set(result)) != len(result)):
        raise ValueError(name + ' requires unique nonempty string identifiers')
    return result


def _unit(x, *, allow_zero=False):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (not allow_zero and np.any(norms <= 1e-12)):
        raise ValueError('Degenerate or nonfinite feature norm')
    if allow_zero:
        return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)
    return x / norms


def _features(features):
    x = np.asarray(features)
    if (x.ndim != 2 or x.shape[1] not in (256, 288)
            or x.dtype.kind not in 'fiu' or not np.isfinite(x).all()):
        raise ValueError('Expected finite numeric identity160+FFT96 features [N,256/288]')
    return np.asarray(x[:, :256], dtype=np.float64)


def _row_affine(x, coefficient, intercept):
    # Reduction shape never depends on the number of query rows.
    if not len(x):
        return np.empty((0, len(coefficient)), dtype=np.float64)
    result = np.stack([np.sum(coefficient * row[None, :], axis=1) + intercept for row in x])
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite affine scores')
    return result


@dataclass(frozen=True)
class FrozenSummaryOperator:
    matrix: np.ndarray
    class_registry: tuple
    feature_schema: str
    audit: Mapping

    def __post_init__(self):
        matrix = np.asarray(self.matrix)
        if (matrix.shape != (160, 160) or not np.isfinite(matrix).all()
                or self.feature_schema != codec.FEATURE_SCHEMA
                or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)):
            raise ValueError('Invalid fixed summary normalization operator')
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() <= 0 or eigenvalues.max() > 1 + 1e-10:
            raise ValueError('Summary normalization must be positive definite and nonexpansive')
        object.__setattr__(self, 'matrix', _readonly(matrix))
        object.__setattr__(self, 'class_registry', _strings(self.class_registry, 'summary classes'))
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def audit_dict(self):
        return _copy(self.audit)


def build_summary_operator(summary):
    """Build a class-independent fixed normalization, never a sample bank."""
    started = time.perf_counter()
    if not isinstance(summary, GroundSummary) or summary.feature_schema != codec.FEATURE_SCHEMA:
        raise ValueError('Expected an already-loaded bound v2 GroundSummary')
    component = summary.component
    if (not isinstance(component.manifest, MappingProxyType)
            or component.manifest.get('schema') != codec.SCHEMA
            or component.manifest.get('feature_dim') != 160
            or component.manifest.get('formal_phase2_eligible') is not True
            or any(getattr(component, name).flags.writeable for name in NUMERIC_MEMBERS)):
        raise ValueError('Summary schema/dimension/immutability mismatch')
    payload = dict(schema=np.asarray(codec.SCHEMA), feature_schema=np.asarray(codec.FEATURE_SCHEMA),
                   residual_rank=np.asarray(3, dtype=np.int16),
                   domain_registry=np.asarray(component.domain_registry),
                   residual_domain_registry=np.asarray(component.residual_domain_registry),
                   class_registry=np.asarray(component.class_registry),
                   center_domain_handle=np.asarray(component.center_domain_handle),
                   **{name: getattr(component, name) for name in NUMERIC_MEMBERS})
    codec._validate_payload(payload)
    classes = _strings(summary.class_registry, 'summary classes')
    domains = sorted(_strings(summary.domain_registry, 'summary domains'))
    class_order = np.asarray(sorted(range(len(classes)), key=lambda i: classes[i]))
    means = np.zeros((len(classes), 160), dtype=np.float64)
    mass = np.zeros(len(classes), dtype=np.float64)

    def domain_arrays(domain):
        centers = np.asarray(summary.reconstruct_domain(domain), dtype=np.float64)[class_order]
        radius = np.asarray(summary.radius_for_domain(domain), dtype=np.float64)[class_order]
        if (centers.shape != means.shape or radius.shape != mass.shape
                or not np.isfinite(centers).all() or not np.isfinite(radius).all()
                or np.any(radius < 0) or np.any(radius > 2)):
            raise ValueError('Invalid aggregate domain center/radius shape or value')
        return _unit(centers), 1.0 / (1.0 + radius)

    for domain in domains:
        centers, q = domain_arrays(domain)
        means += q[:, None] * centers
        mass += q
    means /= mass[:, None]
    geometry = np.zeros((160, 160), dtype=np.float64)
    for domain in domains:
        centers, q = domain_arrays(domain)
        residual = centers - means
        for index in range(len(classes)):
            geometry += (q[index] / mass[index] / len(classes)) * np.outer(residual[index], residual[index])
    geometry = (geometry + geometry.T) * .5
    trace = float(np.trace(geometry))
    if trace <= _CONFIG['covariance_trace_floor']:
        matrix = np.eye(160)
    else:
        values, vectors = np.linalg.eigh(geometry)
        if values.min() < -1e-10 * max(trace, 1e-30):
            raise FloatingPointError('Aggregate geometry is not positive semidefinite')
        factors = 1.0 / np.sqrt(1.0 + 160 * np.maximum(values, 0) / trace)
        matrix = (vectors * factors) @ vectors.T
        matrix = (matrix + matrix.T) * .5
    audit = dict(schema=codec.SCHEMA, feature_schema=codec.FEATURE_SCHEMA,
                 class_count=len(classes), domain_count=len(domains), geometry_trace=trace,
                 geometry_degenerate=bool(trace <= _CONFIG['covariance_trace_floor']),
                 matrix_bytes=int(matrix.nbytes), source_rows_read=0, virtual_samples_generated=0,
                 dense_bank_persisted=False, build_seconds=time.perf_counter() - started,
                 payload=_copy(summary.payload_audit), formula=_CONFIG['summary_formula'])
    return FrozenSummaryOperator(matrix, tuple(sorted(classes)), codec.FEATURE_SCHEMA, audit)


def _transform(raw, alpha, beta, operator):
    identity = _unit(raw[:, :160])
    if beta:
        identity = _unit(_row_affine(identity, operator.matrix, np.zeros(160)))
    if alpha == 0:
        return identity
    fft = _unit(raw[:, 160:256], allow_zero=True)
    return np.concatenate((identity, alpha * fft), axis=1) / np.sqrt(1 + alpha * alpha)


def _statistics(x, labels, c):
    n = len(x) // c
    means = np.stack([x[labels == index].mean(axis=0) for index in range(c)])
    residual = x - means[labels]
    covariance = residual.T @ residual / (c * (n - 1)) if n > 1 else np.zeros((x.shape[1], x.shape[1]))
    covariance = (covariance + covariance.T) * .5
    variance = float(np.trace(covariance)) / x.shape[1]
    if not np.isfinite(variance):
        raise FloatingPointError('Nonfinite support covariance')
    if variance * x.shape[1] <= _CONFIG['covariance_trace_floor']:
        covariance.fill(0)
        variance = 1.0 / x.shape[1]
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-10 * max(float(values.max()), 1e-30):
        raise FloatingPointError('Support covariance is not positive semidefinite')
    return means, np.maximum(values, 0), vectors, variance


def _fit_from_statistics(x, means, values, vectors, variance, shrinkage):
    spectrum = (1 - shrinkage) * values + shrinkage * variance
    coefficient = ((means @ vectors) / spectrum) @ vectors.T
    intercept = -.5 * np.sum(means * coefficient, axis=1)
    coefficient -= coefficient.mean(axis=0, keepdims=True)
    intercept -= intercept.mean()
    scores = _row_affine(x, coefficient, intercept)
    scores -= scores.mean(axis=1, keepdims=True)
    rms = max(float(np.sqrt(np.mean(scores * scores))), 1e-12)
    if not np.isfinite(rms) or not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise FloatingPointError('Nonfinite fitted support geometry')
    return coefficient / rms, intercept / rms, dict(training_scale=rms,
        covariance_scale=variance, covariance_eigenvalue_min=float(spectrum.min()),
        covariance_eigenvalue_max=float(spectrum.max()))


def _risk(scores, labels, old_mask, temperature=1.0):
    values = temperature * scores
    losses = logsumexp(values, axis=1) - values[np.arange(len(labels)), labels]
    macro = float(losses.mean())  # every held class has the same physical count
    old_rows = old_mask[labels]
    old = float(losses[old_rows].mean())
    new = float(losses[~old_rows].mean()) if np.any(~old_rows) else None
    objective = max(old, new) if new is not None else old
    if not np.isfinite(losses).all():
        raise FloatingPointError('Nonfinite held-support NLL')
    return dict(objective=objective, macro_nll=macro, old_nll=old, new_nll=new)


def _temperature(scores, labels, old_mask):
    result = minimize_scalar(lambda value: _risk(scores, labels, old_mask, value)['objective'],
                             method='bounded', bounds=(.1, 10.0),
                             options=dict(xatol=1e-8, maxiter=200))
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError('Held-support scalar temperature optimization did not converge')
    choices = [(.1, _risk(scores, labels, old_mask, .1)),
               (1.0, _risk(scores, labels, old_mask)),
               (10.0, _risk(scores, labels, old_mask, 10.0)),
               (float(result.x), _risk(scores, labels, old_mask, float(result.x)))]
    value, risk = min(choices, key=lambda item: (item[1]['objective'], item[1]['macro_nll'], abs(np.log(item[0])), item[0]))
    return value, risk, int(result.nfev)


@dataclass(frozen=True)
class SummaryJointState:
    coefficient: np.ndarray
    intercept: np.ndarray
    classes: tuple
    alpha: float
    beta: int
    operator: FrozenSummaryOperator
    cosine_mode: bool
    audit: Mapping

    def __post_init__(self):
        object.__setattr__(self, 'coefficient', _readonly(self.coefficient))
        object.__setattr__(self, 'intercept', _readonly(self.intercept))
        object.__setattr__(self, 'classes', tuple(self.classes))
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def audit_dict(self):
        return _copy(self.audit)

    def score(self, features):
        x = _transform(_features(features), self.alpha, self.beta, self.operator)
        if self.cosine_mode:
            x = _unit(x)
        return _row_affine(x, self.coefficient, self.intercept)

    def predict(self, features):
        scores = self.score(features)
        order = np.asarray(sorted(range(len(self.classes)), key=lambda index: self.classes[index]))
        return order[np.argmax(scores[:, order], axis=1)]


def fit_summary_joint(support_features, support_labels, support_ids, classes, old_classes, summary_operator):
    """Fit only this row's physical support, returning an immutable predictor."""
    started = time.perf_counter()
    raw = _features(support_features)
    registry = _strings(classes, 'classes', minimum=2)
    identifiers = _strings(support_ids, 'support_ids')
    old_classes = _strings(old_classes, 'old_classes')
    labels = np.asarray(support_labels)
    c = len(registry)
    if (not isinstance(summary_operator, FrozenSummaryOperator)
            or summary_operator.feature_schema != codec.FEATURE_SCHEMA
            or summary_operator.matrix.flags.writeable
            or set(old_classes) != set(summary_operator.class_registry)
            or not set(old_classes).issubset(registry)
            or len(identifiers) != len(raw) or labels.shape != (len(raw),)
            or labels.dtype.kind not in 'iu' or set(labels.tolist()) != set(range(c))):
        raise ValueError('Support registry/labels/IDs/summary membership mismatch')
    counts = np.bincount(labels.astype(np.int64), minlength=c)
    if counts.min() < 1 or np.any(counts != counts[0]):
        raise ValueError('Equal positive physical K is required')
    k = int(counts[0])
    # A canonical physical-class ordering removes floating reduction dependence
    # on input label indices. Restore the caller's column order only at the end.
    canonical = tuple(sorted(registry))
    remap = np.asarray([canonical.index(name) for name in registry])
    labels = remap[labels.astype(np.int64)]
    order = np.asarray(sorted(range(len(raw)), key=lambda index: (int(labels[index]), identifiers[index])))
    raw, labels = raw[order], labels[order]
    identifiers = tuple(identifiers[index] for index in order)
    _unit(raw[:, :160])
    old_mask = np.asarray([name in old_classes for name in canonical])
    folds, candidate_trace = [], []
    decomposition_count = 0
    if k == 1:
        alpha, beta, shrinkage, temperature = 1.0, 1, 1.0, 1.0
        x = _unit(_transform(raw, alpha, beta, summary_operator))
        coefficient = 10 * np.stack([x[labels == index].mean(axis=0) for index in range(c)])
        intercept = np.zeros(c)
        final_geometry = dict(training_scale=None, covariance_scale=None,
                              reason='K1 fixed unit-sphere cosine; no covariance or validation estimated')
        selected_risk = None
    else:
        f = min(k, 3)
        for index in range(f):
            held = np.asarray([i for i in range(len(raw)) if (i % k) % f == index])
            keep = np.ones(len(raw), dtype=bool); keep[held] = False
            folds.append((keep, held))
        for alpha in _CONFIG['alphas']:
            for beta in _CONFIG['betas']:
                candidate_start = time.perf_counter()
                x = _transform(raw, alpha, beta, summary_operator)
                predictions = {value: np.empty((len(raw), c)) for value in _CONFIG['shrinkages']}
                logs = {value: [] for value in _CONFIG['shrinkages']}
                for fold_index, (keep, held) in enumerate(folds):
                    fold_start = time.perf_counter()
                    statistics = _statistics(x[keep], labels[keep], c)
                    decomposition_count += 1
                    for shrinkage in _CONFIG['shrinkages']:
                        coef, bias, geometry = _fit_from_statistics(x[keep], *statistics, shrinkage)
                        predictions[shrinkage][held] = _row_affine(x[held], coef, bias)
                        logs[shrinkage].append(dict(fold=fold_index, train_per_class=int(keep.sum() // c),
                            held_per_class=int(len(held) // c), train_rows=int(keep.sum()), held_rows=len(held),
                            **geometry, held_before_temperature=_risk(predictions[shrinkage][held], labels[held], old_mask),
                            fit_elapsed_seconds=time.perf_counter() - fold_start))
                for shrinkage in _CONFIG['shrinkages']:
                    temperature, risk, evaluations = _temperature(predictions[shrinkage], labels, old_mask)
                    for log, (_, held) in zip(logs[shrinkage], folds):
                        log['held_after_temperature'] = _risk(predictions[shrinkage][held], labels[held], old_mask, temperature)
                    candidate_trace.append(dict(alpha=alpha, beta=beta, shrinkage=shrinkage,
                        temperature=temperature, temperature_evaluations=evaluations, **risk,
                        folds=logs[shrinkage], elapsed_seconds=time.perf_counter() - candidate_start))
        best = min(candidate_trace, key=lambda row: (round(row['objective'], 10), round(row['macro_nll'], 10),
                                                    -row['shrinkage'], row['alpha'], row['beta']))
        alpha, beta, shrinkage, temperature = (best[key] for key in ('alpha', 'beta', 'shrinkage', 'temperature'))
        selected_risk = {key: best[key] for key in ('objective', 'macro_nll', 'old_nll', 'new_nll')}
        x = _transform(raw, alpha, beta, summary_operator)
        coefficient, intercept, final_geometry = _fit_from_statistics(x, *_statistics(x, labels, c), shrinkage)
        decomposition_count += 1
        coefficient *= temperature; intercept *= temperature
    restore = np.asarray([canonical.index(name) for name in registry])
    coefficient, intercept = coefficient[restore], intercept[restore]
    head_bytes = int(coefficient.nbytes + intercept.nbytes)
    matrix_bytes = int(summary_operator.matrix.nbytes)
    audit = dict(method='D92-SGJoint-v1', algorithm=_copy(_CONFIG), k=k,
        class_count=c, old_classes=list(sorted(old_classes)), support_rows=len(raw),
        feature_dim=coefficient.shape[1], selected=dict(alpha=alpha, beta=beta, shrinkage=shrinkage, temperature=temperature),
        selection='fixed_K1' if k == 1 else 'row_physical_support_cross_validation', selected_oof_risk=selected_risk,
        candidate_trace=candidate_trace, fold_count=len(folds),
        folds=[dict(fold=i, held_ids=[identifiers[index] for index in held],
                    train_rows=int(keep.sum()), held_rows=len(held)) for i, (keep, held) in enumerate(folds)],
        final_geometry=final_geometry, eigendecompositions=decomposition_count,
        zero_fft_support_rows=int(np.sum(np.linalg.norm(raw[:, 160:256], axis=1) <= 1e-12)),
        fit_seconds=time.perf_counter() - started, optimizer_steps=0, optimizer_status='CLOSED_FORM_WITH_SUPPORT_CV',
        loss=None, loss_reason='closed-form shrinkage LDA; held-support NLL recorded for each candidate/fold',
        query_rows_used_for_fit=0, source_runtime_access=False, teacher_used=False, encoder_updated=False,
        virtual_samples_generated=0, new_ground_payload_bytes=0,
        head_bytes=head_bytes, summary_operator_bytes=matrix_bytes,
        persistent_state_bytes=head_bytes + matrix_bytes,
        support_matrix_bytes=int(raw.nbytes), covariance_matrix_bytes=int(coefficient.shape[1] ** 2 * 8),
        workspace_note='Logical array bytes, not measured allocator/BLAS peak; trace JSON and shared model excluded',
        summary=summary_operator.audit_dict())
    return SummaryJointState(coefficient, intercept, registry, alpha, beta, summary_operator, k == 1, audit)
