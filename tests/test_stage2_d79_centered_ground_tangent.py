from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "code" / "cvsrffi" / "stage2_d79_centered_ground_tangent.py"
SPEC = importlib.util.spec_from_file_location("d79_core_test", PATH)
assert SPEC is not None and SPEC.loader is not None
d79 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d79
SPEC.loader.exec_module(d79)


def _lda_fit(rows, labels, class_count, k_shot):
    del k_shot
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([np.mean(x[y == c], axis=0) for c in range(class_count)])
    return means, -0.5 * np.sum(means * means, axis=1), {}


def _support(seed=31):
    rng = np.random.default_rng(seed)
    classes, k_shot, dimension = 3, 4, 8
    centers = np.asarray(
        [[1.3, 0.2, 0.0, 0, 0, 0, 0, 0], [0.4, 1.1, 0.2, 0, 0, 0, 0, 0], [-0.4, 0.1, 1.0, 0, 0, 0, 0, 0]],
        dtype=np.float64,
    )
    rows, labels = [], []
    for c in range(classes):
        for _ in range(k_shot):
            rows.append(centers[c] + rng.normal(scale=0.35, size=dimension))
            labels.append(c)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, 2)))
    return np.asarray(rows), np.asarray(labels), centers * 0.4, basis


def test_centered_affine_compile_is_exact_at_support_center():
    rows, labels, base, basis = _support()
    delta_w, delta_b, audit = d79.fit_centered_ground_tangent_margin(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    center = np.mean(rows, axis=0)
    np.testing.assert_allclose(delta_w @ center + delta_b, 0.0, atol=2e-7)
    direct = rows @ delta_w.T + delta_b[None, :]
    centered = (rows - center[None, :]) @ delta_w.T
    np.testing.assert_allclose(direct, centered, atol=3e-7)
    assert audit["support_centering_enabled"] is True
    assert audit["centered_affine_compile"] is True
    assert audit["residual_logit_at_support_center_max_abs"] <= 1e-12


def test_centering_is_translation_invariant_and_deterministic():
    rows, labels, base, basis = _support(43)
    first_w, first_b, first_audit = d79.fit_centered_ground_tangent_margin(
        rows, labels, 3, 4, base_coefficient=base, tangent_basis=basis, lda_fit=_lda_fit
    )
    shift = np.asarray([0.9, -0.4, 0.6, 0.1, -0.2, 0.3, 0.0, 0.2])
    second_w, second_b, second_audit = d79.fit_centered_ground_tangent_margin(
        rows + shift,
        labels,
        3,
        4,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    np.testing.assert_allclose(first_w, second_w, atol=5e-7)
    np.testing.assert_allclose(second_b, first_b - first_w @ shift, atol=8e-7)
    assert first_audit["objective_delta"] == second_audit["objective_delta"]


def test_k1_keeps_zero_weight_and_bias_residuals():
    rows, labels, base, basis = _support()
    chosen = np.asarray([0, 4, 8])
    delta_w, delta_b, audit = d79.fit_centered_ground_tangent_margin(
        rows[chosen],
        labels[chosen],
        3,
        1,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    np.testing.assert_array_equal(delta_w, np.zeros_like(delta_w))
    np.testing.assert_array_equal(delta_b, np.zeros_like(delta_b))
    assert audit["status"] == "k1_exact_d62_fallback"

