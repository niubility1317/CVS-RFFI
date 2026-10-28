"""Single fixed branch-augmented ridge head; immutable per-query inference."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
import time
import numpy as np
from .d92_branch_support_probe import _feature_blocks, _ridge_fit, _objective, _strings, _finite


FROZEN_CONFIG = dict(
    method='D92-BranchRidge-v1', schema='d92_branch_ridge_v1', views='original_received_only',
    branches=['t_emb', 'f_emb', 'pa_local'], identity_dim=160, fft_dim=96, feature_dim=736,
    norm_floor=1e-12, background='unit(concat(unit(z_id),4*unit(fft96)))',
    auxiliary='concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)',
    features='concat(background,auxiliary)', concatenate_renormalize=False,
    fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8,
    target='onehot_minus_1_over_C', objective='0.5*sum_physical_squared_error+0.5*||W||F^2',
    ridge_coefficient=1., intercept_regularized=False, sample_weight=1.,
    solver='float64_Cholesky_primal_if_D_le_N_else_dual', selection='fixed_no_selection',
    selected='zfft_aux', runtime_cv=False, k1='same_full_support_fit_no_independent_holdout',
    score='per_sample_linear_all_registered_classes', tie_break='physical_class_id_lexicographic',
    optimizer_steps=0, source_inputs=False, summary_inputs=False, query_fit=False, phase1_frozen=True,
)
_CONFIG = deepcopy(FROZEN_CONFIG)


def _freeze(value):
    if isinstance(value, Mapping): return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if isinstance(value, Mapping): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple): return [_plain(v) for v in value]
    return value


def _readonly(value):
    x = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)


def _features(*, z_id, fft, t_emb, f_emb, pa_local):
    values = (z_id, fft, t_emb, f_emb, pa_local)
    dims = (160, 96, 160, 160, 160)
    arrays = [np.asarray(v) for v in values]
    if all(v.ndim == 2 and v.shape == (0, d) and v.dtype.kind in 'fiu' for v, d in zip(arrays, dims)):
        return np.empty((0, 736), dtype=np.float64)
    _, bases, aux = _feature_blocks(z_id, fft, t_emb, f_emb, pa_local)
    return np.concatenate((bases['zfft'], aux), axis=1)


@dataclass(frozen=True)
class BranchRidgeState:
    W: np.ndarray
    b: np.ndarray
    classes: tuple
    audit: Mapping

    def __post_init__(self):
        classes = _strings(self.classes, 'classes')
        w, b = np.asarray(self.W), np.asarray(self.b)
        if (w.shape != (736, len(classes)) or b.shape != (len(classes),)
                or any(v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in (w, b))):
            raise ValueError('Invalid branch ridge state')
        object.__setattr__(self, 'classes', classes)
        object.__setattr__(self, 'W', _readonly(w))
        object.__setattr__(self, 'b', _readonly(b))
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def score(self, *, z_id, fft, t_emb, f_emb, pa_local):
        x = _features(z_id=z_id, fft=fft, t_emb=t_emb, f_emb=f_emb, pa_local=pa_local)
        scores = np.empty((len(x), len(self.classes)), dtype=np.float64)
        for i, row in enumerate(x): scores[i] = row@self.W+self.b
        _finite(scores)
        return scores

    def predict(self, **features):
        scores = self.score(**features)
        order = np.asarray(sorted(range(len(self.classes)), key=lambda i: self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:, order], axis=1)]]

    def audit_dict(self): return _plain(self.audit)


def fit_branch_ridge(*, z_id, fft, t_emb, f_emb, pa_local,
                     support_labels, support_ids, classes, old_classes=()):
    started = time.perf_counter()
    x = _features(z_id=z_id, fft=fft, t_emb=t_emb, f_emb=f_emb, pa_local=pa_local)
    labels = np.asarray(support_labels)
    ids = _strings(support_ids, 'support_ids')
    requested = _strings(classes, 'classes')
    old = _strings(old_classes, 'old_classes', nonempty=False)
    c, n = len(requested), len(x)
    if (not n or len(ids) != n or labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or set(labels.tolist()) != set(range(c)) or not set(old).issubset(requested)):
        raise ValueError('Invalid support labels, physical IDs, classes, or old membership')
    canonical = tuple(sorted(requested))
    remap = np.asarray([canonical.index(v) for v in requested])
    labels = remap[labels.astype(np.int64)]
    counts = np.bincount(labels, minlength=c)
    if np.any(counts != counts[0]): raise ValueError('Equal positive physical K required')
    k = int(counts[0])
    order = np.asarray(sorted(range(n), key=lambda i: ids[i]))
    x, labels = x[order], labels[order]
    ids = tuple(ids[i] for i in order)
    feature_seconds = time.perf_counter()-started
    targets = np.eye(c)[labels]-1./c
    begin = time.perf_counter()
    w, b, audit = _ridge_fit(x, targets)
    constant_features = bool(np.all(x == x[0]))
    if constant_features:
        # Balanced onehot-minus-1/C has exact mathematical mean zero. Reference
        # centering guarantees W=0 here; avoid class-specific rounding in b.
        w = np.zeros_like(w)
        b = np.zeros_like(b)
    solved_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    audit.update(_objective(x, targets, w, b))
    objective_seconds = time.perf_counter()-begin
    audit.update(scope='final', fold=None, status='CLOSED_FORM_SOLVED',
        train_k=k, train_physical_count=n, training_physical_ids=list(ids),
        physical_loss_mass=float(n), sample_weight=1., ridge_coefficient=1.,
        design_dim=736, output_dim=c, feature_seconds=feature_seconds,
        degenerate_constant_features=constant_features,
        objective_seconds=objective_seconds, fit_seconds=solved_seconds+objective_seconds,
        optimizer_steps=0, learning_rate=None, epoch=None,
        all_states_estimated_from_current_support_only=True,
        mean_feature_squared_norm=float(np.mean(np.sum(x*x, axis=1))),
        feature_energy_condition_bound=float(1+2*n))
    # Reorder only final output columns; fit arithmetic used physical canonical order.
    output_order = [canonical.index(v) for v in requested]
    w, b = w[:, output_order], b[output_order]
    numerical_bytes = int(w.nbytes+b.nbytes)
    top = dict(config=deepcopy(_CONFIG), classes=list(requested), old_classes=sorted(old),
        k=k, support_count=n, fold_count=0, folds=[], oof=None,
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG',
        selection='fixed_no_selection', selected='zfft_aux', optimizer_steps=0,
        factorization_count=1, head_bytes=numerical_bytes, persistent_state_bytes=numerical_bytes,
        state_byte_scope='W_and_b_float64_nbytes_only_excludes_registry_audit_and_Python_object_overhead',
        query_rows_used=0, source_rows_used=0, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        phase1_frozen=True, source_validation=None, source_validation_reason='No source samples or feature banks',
        final_fit=audit, fit_seconds=time.perf_counter()-started)
    return BranchRidgeState(w, b, requested, top)
