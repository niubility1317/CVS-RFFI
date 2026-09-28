"""Fixed received-IQ views and a physical-support-only nonlinear ridge head.

No file I/O, model update, ground input, query fitting, or cross-row state.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import time

import numpy as np
from scipy.special import logsumexp


FROZEN_CONFIG = dict(
    method='D92-MVKME-v1', input_shape=[2, 256],
    crop_length=192, crop_starts=[0, 32, 64], crop_padding=[32, 32],
    phases_quarter_turns=[0, 1, 2, 3], view_order='window_then_phase',
    identity_dim=160, fft_dim=96, feature_block_weight='equal_1_over_sqrt2',
    fft='historical_spectral_logmag_sketch_on_each_padded_window',
    fft_norm_floor=1e-8,
    fourier_generator='numpy.Generator(PCG64(0)).standard_normal',
    fourier_seed=0, fourier_frequencies=128, fourier_bandwidth=1.0,
    fourier_map='concat_cos_sin_div_sqrt128',
    blocks=['original', 'mean_full_4_phases', 'mean_local_12_views'],
    block_renormalization=False, etas=[0.0, 0.5, 1.0], gammas=[0.01, 0.1, 1.0],
    k1=dict(eta=0.5, gamma=0.1), max_folds=3,
    folds='per_class_physical_id_sort_then_position_mod_min_K_3',
    target='onehot_minus_1_over_C', regularizer='K_times_gamma', score_scale=10.0,
    selection='min_max_old_new_oof_nll_then_macro_nll_then_larger_gamma_smaller_eta',
    selection_round_decimals=10, precision='float64', norm_floor=1e-12,
    source_inputs=False, summary_inputs=False, query_fit=False,
)


def _freeze(x):
    if isinstance(x, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    return x


def _plain(x):
    if isinstance(x, Mapping):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [_plain(v) for v in x]
    return x


_CONFIG = _freeze(FROZEN_CONFIG)


def _readonly(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)


_OMEGA = _readonly(np.random.Generator(np.random.PCG64(0)).standard_normal((128, 256)))


def _numeric(value, shape, name, *, nonempty=False):
    x = np.asarray(value)
    if (x.ndim != len(shape) + 1 or x.shape[1:] != shape or x.dtype.kind not in 'fiu'
            or not np.isfinite(x).all() or (nonempty and len(x) == 0)):
        raise ValueError(name + ' has invalid shape, type, or finite values')
    result = np.asarray(x, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name + ' cannot be represented as float64')
    return result


def _strings(values, name, *, nonempty=True):
    if isinstance(values, (str, bytes)):
        raise ValueError(name + ' must be an explicit sequence')
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(name + ' must be an explicit sequence') from exc
    if ((nonempty and not values) or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(name + ' must contain unique nonempty physical identifiers')
    return values


def make_received_views(received_iq):
    """Return [N,4 windows,4 phases,2,256]; no new physical observations."""
    iq = _numeric(received_iq, (2, 256), 'received IQ')
    windows = np.zeros((len(iq), 4, 2, 256), dtype=np.float64)
    windows[:, 0] = iq
    for j, start in enumerate((0, 32, 64), 1):
        windows[:, j, :, 32:224] = iq[:, :, start:start+192]
    views = np.empty((len(iq), 4, 4, 2, 256), dtype=np.float64)
    views[:, :, 0] = windows
    views[:, :, 1, 0] = -windows[:, :, 1]
    views[:, :, 1, 1] = windows[:, :, 0]
    views[:, :, 2] = -windows
    views[:, :, 3, 0] = windows[:, :, 1]
    views[:, :, 3, 1] = -windows[:, :, 0]
    return _readonly(views)


def fourier_map(features):
    """Fixed pointwise map with unit norm, including for a zero input."""
    x = _numeric(features, (256,), 'Fourier input')
    result = np.empty((len(x), 256), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise'):
        for i, row in enumerate(x):
            angles = np.sum(_OMEGA * row[None, :], axis=1)
            result[i] = np.concatenate((np.cos(angles), np.sin(angles))) / np.sqrt(128.0)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite Fourier map')
    return _readonly(result)


def received_fft96(received_iq):
    """Historical FFT96 formula, local to avoid shadowing the native package.

    This duplicates spectral_logmag_sketch's fixed dim=96 numerical recipe;
    tests compare against that existing implementation on synthetic IQ.
    """
    iq = _numeric(received_iq, (2, 256), 'received FFT IQ')
    with np.errstate(over='raise', invalid='raise'):
        raw = iq.astype(np.float32)
    result = np.empty((len(raw), 96), dtype=np.float32)
    target = np.linspace(0.0, 1.0, 96, dtype=np.float64)
    source = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    for i, row in enumerate(raw):
        value = row[0].astype(np.float64) + 1j*row[1].astype(np.float64)
        value -= np.mean(value)
        rms = float(np.sqrt(np.mean(np.abs(value)**2)))
        if rms > 1e-8:
            value /= rms
        logmag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(value*np.hanning(256)))))
        sketch = np.interp(target, source, logmag).astype(np.float32)
        sketch -= np.mean(sketch, dtype=np.float64).astype(np.float32)
        sketch /= max(float(np.linalg.norm(sketch)), 1e-8)
        result[i] = sketch
    return _readonly(result)


def build_fourier_blocks(*, received_iq, identity_views):
    """Combine frozen identity [N,4,4,160] and same-received FFT into [N,3,256]."""
    iq = _numeric(received_iq, (2, 256), 'received IQ')
    z = _numeric(identity_views, (4, 4, 160), 'frozen identity views')
    if len(z) != len(iq):
        raise ValueError('Identity and IQ physical sample counts differ')
    views = make_received_views(iq)
    blocks = np.empty((len(iq), 3, 256), dtype=np.float64)
    for i in range(len(iq)):
        fft = received_fft96(views[i, :, 0])
        with np.errstate(over='raise', invalid='raise'):
            norms = np.linalg.norm(z[i], axis=-1, keepdims=True)
            zn = np.divide(z[i], norms, out=np.zeros_like(z[i]), where=norms > 1e-12)
            u = np.concatenate((zn, np.repeat(fft[:, None, :], 4, axis=1)), axis=-1) / np.sqrt(2.0)
        mapped = fourier_map(u.reshape(16, 256)).reshape(4, 4, 256)
        blocks[i, 0] = mapped[0, 0]
        blocks[i, 1] = mapped[0].mean(axis=0)
        blocks[i, 2] = mapped[1:].reshape(12, 256).mean(axis=0)
    return _readonly(blocks)


def _phi(blocks, eta):
    weights = np.sqrt([1.0-eta, eta/2.0, eta/2.0])
    return (blocks * weights[None, :, None]).reshape(len(blocks), 768)


def _scores(phi, coefficient):
    result = np.empty((len(phi), coefficient.shape[1]), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise'):
        for i, row in enumerate(phi):
            result[i] = 10.0 * np.sum(row[:, None] * coefficient, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite ridge score')
    return result


def _decompose(phi):
    with np.errstate(over='raise', invalid='raise'):
        gram = phi @ phi.T
    if not np.isfinite(gram).all():
        raise FloatingPointError('Nonfinite support Gram matrix')
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    # A Gram matrix is PSD; only roundoff-scale negative eigenvalues are legal.
    tolerance = 64 * np.finfo(float).eps * max(1, len(phi)) * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -tolerance:
        raise FloatingPointError('Support Gram matrix is not numerically PSD')
    return np.maximum(eigenvalues, 0.0), eigenvectors


def _ridge(phi, targets, k, gamma, decomposition=None):
    ev, vectors = _decompose(phi) if decomposition is None else decomposition
    coefficient = phi.T @ (vectors @ ((vectors.T @ targets) / (ev[:, None] + k*gamma)))
    if not np.isfinite(coefficient).all():
        raise FloatingPointError('Nonfinite ridge coefficient')
    return coefficient


def _losses(scores, labels, classes, old_classes):
    nll = logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels]
    per_class = [float(nll[labels == c].mean()) for c in range(len(classes))]
    old = [v for c, v in zip(classes, per_class) if c in old_classes]
    new = [v for c, v in zip(classes, per_class) if c not in old_classes]
    groups = [float(np.mean(g)) for g in (old, new) if g]
    result = dict(objective=max(groups), macro_nll=float(np.mean(per_class)),
                  old_nll=float(np.mean(old)) if old else None,
                  new_nll=float(np.mean(new)) if new else None)
    if any(v is not None and not np.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite support validation loss')
    return result


def _training_metrics(phi, targets, coefficient, gamma):
    residual = phi @ coefficient - targets
    data_loss = float(np.sum(residual**2) / len(phi))
    regularization_loss = float(gamma / targets.shape[1] * np.sum(coefficient**2))
    result = dict(train_squared_error=data_loss, regularization_loss=regularization_loss,
                  train_objective=data_loss+regularization_loss)
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite ridge training objective')
    return result


@dataclass(frozen=True)
class MVKMEState:
    coefficient: np.ndarray
    classes: tuple
    eta: float
    audit: Mapping

    def __post_init__(self):
        classes = _strings(self.classes, 'classes')
        w = np.asarray(self.coefficient)
        if (w.shape != (768, len(classes)) or w.dtype.kind not in 'fiu'
                or not np.isfinite(w).all() or self.eta not in (0.0, 0.5, 1.0)):
            raise ValueError('Invalid immutable MV-KME state')
        object.__setattr__(self, 'classes', classes)
        object.__setattr__(self, 'coefficient', _readonly(w))
        object.__setattr__(self, 'audit', _freeze(self.audit))

    def score(self, blocks):
        values = _numeric(blocks, (3, 256), 'query blocks')
        return _scores(_phi(values, self.eta), self.coefficient)

    def predict(self, blocks):
        scores = self.score(blocks)
        order = np.asarray(sorted(range(len(self.classes)), key=lambda i: self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:, order], axis=1)]]

    def audit_dict(self):
        return _plain(self.audit)


def fit_mv_kme(*, support_blocks, support_labels, support_ids, classes, old_classes):
    """Fit only this row's physical support; K1 never performs holdout selection."""
    started = time.perf_counter()
    blocks = _numeric(support_blocks, (3, 256), 'support blocks', nonempty=True)
    labels = np.asarray(support_labels)
    classes = _strings(classes, 'classes')
    ids = _strings(support_ids, 'support IDs')
    old = _strings(old_classes, 'old classes', nonempty=False)
    if (len(ids) != len(blocks) or labels.shape != (len(blocks),)
            or labels.dtype.kind not in 'iu' or not set(old).issubset(classes)
            or set(labels.tolist()) != set(range(len(classes)))):
        raise ValueError('Support labels, physical IDs, or old-class membership mismatch')
    counts = np.bincount(labels.astype(np.int64), minlength=len(classes))
    if np.any(counts != counts[0]):
        raise ValueError('Equal positive physical K required for every class')
    k, c = int(counts[0]), len(classes)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: (classes[int(labels[i])], ids[i])))
    blocks, labels = blocks[order], labels[order].astype(np.int64)
    ids = tuple(ids[i] for i in order)
    targets = np.eye(c)[labels] - 1.0/c
    steps, candidates = [], []
    fold_count = 0 if k == 1 else min(k, 3)
    fold = np.full(len(blocks), -1, dtype=np.int64)
    if fold_count:
        for label in range(c):
            positions = np.flatnonzero(labels == label)
            fold[positions] = np.arange(k) % fold_count
        for eta in (0.0, 0.5, 1.0):
            phi = _phi(blocks, eta)
            oof = {gamma: np.empty((len(blocks), c)) for gamma in (0.01, 0.1, 1.0)}
            for f in range(fold_count):
                keep, hold = fold != f, fold == f
                train_k = int(np.count_nonzero(keep) // c)
                decomp_start = time.perf_counter()
                decomposition = _decompose(phi[keep])
                decomp_seconds = time.perf_counter() - decomp_start
                for gamma in (0.01, 0.1, 1.0):
                    step_start = time.perf_counter()
                    w = _ridge(phi[keep], targets[keep], train_k, gamma, decomposition)
                    score = _scores(phi[hold], w)
                    oof[gamma][hold] = score
                    steps.append(dict(event='PHYSICAL_SUPPORT_FOLD', eta=eta, gamma=gamma,
                        fold=f, train_k=train_k, heldout_k=int(np.count_nonzero(hold)//c),
                        train_physical_count=int(np.count_nonzero(keep)),
                        heldout_physical_count=int(np.count_nonzero(hold)),
                        gram_dimension=int(np.count_nonzero(keep)), regularizer=train_k*gamma,
                        score_scale=10.0, temperature=1.0, training_scale='fixed; no data-estimated normalization',
                        decomposition_shared_across_gammas=True, decomposition_seconds=decomp_seconds,
                        solve_score_seconds=time.perf_counter()-step_start,
                        elapsed_seconds=time.perf_counter()-started,
                        **_training_metrics(phi[keep], targets[keep], w, gamma),
                        **_losses(score, labels[hold], classes, old)))
            for gamma, score in oof.items():
                candidates.append(dict(eta=eta, gamma=gamma, **_losses(score, labels, classes, old)))
        chosen = min(candidates, key=lambda r: (round(r['objective'], 10), round(r['macro_nll'], 10),
                                                -r['gamma'], r['eta']))
        eta, gamma = chosen['eta'], chosen['gamma']
        selection = 'physical_support_oof'
    else:
        eta, gamma, chosen = 0.5, 0.1, None
        selection = 'k1_preregistered_no_holdout'
    full_start = time.perf_counter()
    phi = _phi(blocks, eta)
    w = _ridge(phi, targets, k, gamma)
    steps.append(dict(event='FINAL_SUPPORT_FIT', eta=eta, gamma=gamma,
        train_k=k, train_physical_count=len(blocks), gram_dimension=len(blocks), regularizer=k*gamma,
        score_scale=10.0, temperature=1.0, training_scale='fixed; no data-estimated normalization',
        fit_seconds=time.perf_counter()-full_start, elapsed_seconds=time.perf_counter()-started,
        **_training_metrics(phi, targets, w, gamma)))
    audit = dict(method='D92-MVKME-v1', config=_plain(_CONFIG), k=k, classes=c,
        selected=dict(eta=eta, gamma=gamma), selection=selection,
        selected_objective=None if chosen is None else chosen['objective'],
        selected_macro_nll=None if chosen is None else chosen['macro_nll'],
        selected_old_nll=None if chosen is None else chosen['old_nll'],
        selected_new_nll=None if chosen is None else chosen['new_nll'],
        candidate_count=len(candidates), fold_count=fold_count, candidates=candidates, steps=steps,
        physical_fold_assignment=[dict(physical_id=id_, fold=int(f)) for id_, f in zip(ids, fold)],
        fit_seconds=time.perf_counter()-started, optimizer_status='closed_form_complete',
        optimizer_steps=None, learning_rate=None, gradient_norm=None,
        unavailable_reason='Closed-form ridge; no optimizer iterations, epochs, learning rate, or gradient updates',
        head_bytes=int(w.nbytes), fourier_matrix_bytes=int(_OMEGA.nbytes),
        persistent_state_bytes=int(w.nbytes+_OMEGA.nbytes),
        persistent_state_bytes_scope='float64 head plus shared mathematical Fourier matrix; identifiers/config counted separately',
        support_feature_cache_bytes=int(blocks.nbytes), gram_bytes=int(len(blocks)**2*8),
        query_rows_used_for_fit=0, source_rows_used_for_fit=0, new_source_payload_bytes=0,
        ground_summary_used=False, cross_row_adapted_state_reuse=False)
    return MVKMEState(w, classes, eta, audit)
