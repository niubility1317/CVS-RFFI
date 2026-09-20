"""Space-ground channel used for STAR-RFFI data generation.

The STAR-RFFI reproduction keeps clean and space-ground samples paired at the
raw IQ level.  This module only transforms raw time-domain complex IQ signals;
it must not be applied to the already concatenated 2048-point CNN feature.
"""

from __future__ import annotations

import numpy as np


def _as_rng(rng=None, seed=None):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _rms(x: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    eps = 1e-12
    return np.sqrt(np.mean(np.abs(x) ** 2, axis=axis, keepdims=keepdims) + eps)


def apply_space_ground_channel(
    x_complex,
    rng=None,
    seed=None,
    fr=2.45e9,
    h=500e3,
    gtgr_db=159.0,
    loo_mu=0.0,
    loo_delta2=9.0,
    loo_sigma2=0.0049,
    snr_db=None,
    normalize="rms",
    fading_mode="sample",
):
    """Apply a Loo/shadowed-Rician space-ground channel to raw IQ samples.

    Parameters
    ----------
    x_complex:
        Complex IQ with shape ``(L,)`` for one signal or ``(N, L)`` for a batch.
    rng, seed:
        Randomness is controlled by ``np.random.default_rng``.  If ``rng`` is
        supplied it is used first; otherwise ``seed`` creates a fresh generator.
    fading_mode:
        ``"sample"`` draws one Loo fading value per time sample.  ``"block"``
        draws one fading value per input sequence.
    normalize:
        With ``"rms"`` the channel output is rescaled to the input RMS.  The
        physical link gain includes free-space path loss, but that absolute
        attenuation is extremely small for neural-network training.  RMS
        normalization keeps fading, phase, and relative amplitude perturbations
        while avoiding numerically tiny features.

    Returns
    -------
    np.ndarray
        Complex64 array with the same shape as ``x_complex``.
    """

    eps = 1e-12
    rng = _as_rng(rng, seed)
    x = np.asarray(x_complex, dtype=np.complex64)
    original_shape = x.shape

    if x.ndim == 1:
        work = x[None, :]
        squeeze = True
    elif x.ndim == 2:
        work = x
        squeeze = False
    else:
        raise ValueError(f"x_complex must have shape (L,) or (N, L), got {x.shape}")

    if fading_mode not in {"sample", "block"}:
        raise ValueError("fading_mode must be 'sample' or 'block'")
    if normalize not in {None, "none", "rms"}:
        raise ValueError("normalize must be None, 'none', or 'rms'")

    n, length = work.shape
    fade_shape = (n, length) if fading_mode == "sample" else (n, 1)
    loo_delta = float(np.sqrt(loo_delta2))

    # Loo fading: lognormal LoS amplitude plus complex Gaussian diffuse scatter.
    z_los = rng.lognormal(mean=loo_mu, sigma=loo_delta, size=fade_shape)
    diffuse_scale = float(np.sqrt(loo_sigma2 / 2.0))
    diffuse = diffuse_scale * (
        rng.normal(size=fade_shape) + 1j * rng.normal(size=fade_shape)
    )
    g = z_los + diffuse

    gtgr_linear = 10.0 ** (gtgr_db / 10.0)
    c = 3e8
    path_factor = c / (4.0 * np.pi * h * fr)
    amplitude_gain = np.sqrt(gtgr_linear) * path_factor
    y = work * (amplitude_gain * g).astype(np.complex64)

    if snr_db is not None:
        signal_power = np.mean(np.abs(y) ** 2, axis=1, keepdims=True)
        noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
        noise_sigma = np.sqrt(np.maximum(noise_power, eps) / 2.0)
        noise = noise_sigma * (
            rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)
        )
        y = y + noise

    if normalize == "rms":
        in_rms = _rms(work, axis=1, keepdims=True)
        out_rms = _rms(y, axis=1, keepdims=True)
        y = y / np.maximum(out_rms, eps) * np.maximum(in_rms, eps)

    if not np.isfinite(y).all():
        raise FloatingPointError("space-ground channel produced nan or inf")

    y = y.astype(np.complex64, copy=False)
    if squeeze:
        y = y[0]
    if y.shape != original_shape:
        raise RuntimeError(f"channel shape changed from {original_shape} to {y.shape}")
    return y


def _describe_complex(name: str, value: np.ndarray) -> None:
    stacked = np.stack([value.real, value.imag], axis=0)
    print(
        f"{name}: shape={value.shape}, finite={np.isfinite(value).all()}, "
        f"mean={stacked.mean():.6g}, std={stacked.std():.6g}, "
        f"min={stacked.min():.6g}, max={stacked.max():.6g}"
    )


if __name__ == "__main__":
    test_rng = np.random.default_rng(299)
    x_test = (
        test_rng.normal(size=(4, 256)) + 1j * test_rng.normal(size=(4, 256))
    ).astype(np.complex64)
    y_test = apply_space_ground_channel(x_test, rng=test_rng)
    _describe_complex("input", x_test)
    _describe_complex("output", y_test)
