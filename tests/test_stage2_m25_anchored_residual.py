from __future__ import annotations

import importlib.util

import numpy as np

from cvsrffi.stage2_ablation_quantization import F0
from cvsrffi.stage2_m24_compiler import compile_m24_head
from cvsrffi.stage2_m24_features import IF_DIM, physical_if256

import cvsrffi.stage2_m25_anchored_residual as m25


def test_m25_anchored_residual_module_is_available() -> None:
    assert importlib.util.find_spec("cvsrffi.stage2_m25_anchored_residual") is not None


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _support(k_shot: int = 5) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(7282101 + k_shot)
    classes = ("old-a", "old-b", "new-c", "new-d")
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, name in enumerate(classes):
        for shot in range(k_shot):
            identity = 0.02 * rng.normal(size=160)
            fft = 0.02 * rng.normal(size=96)
            identity[(index * 17 + shot) % 160] += 1.0
            identity[(index * 31 + 7) % 160] += 1.5
            fft[(index * 13 + shot) % 96] += 0.8
            fft[(index * 19 + 5) % 96] += 1.2
            rows.append(np.concatenate([identity, fft, np.ones(10)]))
            labels.append(name)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), classes


def _base_state(blocks: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]):
    features = physical_if256(blocks)
    centres = np.stack([features[labels == name].mean(axis=0) for name in classes])
    centres = _unit(centres)
    state, _resource, _quant = compile_m24_head(
        6.0 * centres,
        np.linspace(0.03, -0.03, len(classes), dtype=np.float32),
        classes=classes,
        domain_digest="test-domain",
        config_hash="test-base",
        support_features=features,
        transient_workspace_bytes=0,
        block_sizes=(160, 96),
        input_log_diag=np.zeros(IF_DIM, dtype=np.float32),
        compile_arm=F0,
    )
    return state


def test_k1_and_k2_are_exact_base_fallbacks() -> None:
    for k_shot in (1, 2):
        blocks, labels, classes = _support(k_shot)
        base = _base_state(blocks, labels, classes)
        state, audit = m25.fit_m25_anchored_residual(
            arm=m25.B3,
            base_state=base,
            support_blocks=blocks,
            support_labels=labels,
            classes=classes,
            k_shot=k_shot,
            old_class_count=2,
            domain_digest="test-domain",
        )
        expected = base.score(physical_if256(blocks))
        assert np.array_equal(state.score(blocks), expected)
        assert audit["selected_strength"] == 0.0
        assert audit["fallback_reason"] == "K_LT_5_EXACT_G0"


def test_margin_gate_preserves_high_margin_and_bounds_low_margin_delta() -> None:
    base = np.asarray([[2.0, 0.0, -1.0], [0.05, 0.0, -0.1]], dtype=np.float64)
    residual = np.asarray([[-1.0, 1.0, 0.0], [2.0, -1.0, 0.5]], dtype=np.float64)
    adjusted, audit = m25.apply_bounded_residual(
        base,
        residual,
        strength=0.08,
        margin_gate=0.10,
    )
    assert np.array_equal(adjusted[0], base[0])
    assert np.max(np.abs(adjusted[1] - base[1])) <= 0.08 + 1.0e-12
    assert abs(float(np.mean(adjusted[1] - base[1]))) <= 1.0e-12
    assert audit["gated_query_count"] == 1


def test_support_selector_accepts_bilateral_improvement_and_rejects_old_harm() -> None:
    targets = np.asarray([0, 1, 2, 3], dtype=np.int64)
    base = np.asarray(
        [[0.10, 0.09, 0.00, 0.00], [0.09, 0.10, 0.00, 0.00],
         [0.00, 0.00, 0.10, 0.09], [0.00, 0.00, 0.09, 0.10]],
        dtype=np.float64,
    )
    helpful = np.full_like(base, -1.0 / 3.0)
    helpful[np.arange(4), targets] = 1.0
    selected, audit = m25.select_residual_strength(
        base,
        helpful,
        targets,
        old_class_count=2,
        k_shot=5,
    )
    assert selected > 0.0
    assert audit["selected_old_ce_delta"] <= 0.0
    assert audit["selected_new_ce_delta"] <= 0.0

    harmful = helpful.copy()
    harmful[0] *= -1.0
    rejected, rejected_audit = m25.select_residual_strength(
        base,
        harmful,
        targets,
        old_class_count=2,
        k_shot=5,
    )
    assert rejected == 0.0
    assert rejected_audit["fallback_to_zero"] is True


def test_shrinkage_radius_is_query_dependent_and_class_symmetric() -> None:
    rows = np.zeros((10, IF_DIM), dtype=np.float64)
    rows[:5, 0] = 1.0
    rows[:5, 2] = np.linspace(-0.30, 0.30, 5)
    rows[5:, 1] = 1.0
    rows[5:, 3] = np.linspace(-0.03, 0.03, 5)
    rows = _unit(rows)
    targets = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
    model = m25.build_local_evidence_model(rows, targets, 2, arm=m25.B2, k_shot=5)
    assert model.radius_squared[0] > model.radius_squared[1]
    near_wide = _unit(np.asarray([[1.0, 0.0, 0.18] + [0.0] * (IF_DIM - 3)]))
    far_wide = _unit(np.asarray([[1.0, 0.0, 0.70] + [0.0] * (IF_DIM - 3)]))
    assert model.score(near_wide)[0, 0] > model.score(far_wide)[0, 0]

    permutation = np.asarray([1, 0])
    remapped = permutation[targets]
    permuted = m25.build_local_evidence_model(rows, remapped, 2, arm=m25.B2, k_shot=5)
    np.testing.assert_allclose(permuted.score(near_wide)[:, permutation], model.score(near_wide))


def test_dual_prototype_requires_balanced_stable_split() -> None:
    stable = np.asarray(
        [[1.0, 0.05, 0.0], [1.0, -0.05, 0.0], [1.0, 0.0, 0.05],
         [0.05, 1.0, 0.0], [-0.05, 1.0, 0.0], [0.0, 1.0, 0.05]],
        dtype=np.float64,
    )
    stable = _unit(stable)
    prototypes, weights, audit = m25.fit_stable_prototypes(stable)
    assert len(prototypes) == 2
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert audit["split_accepted"] is True

    outlier = np.asarray(
        [[1.0, 0.02, 0.0], [1.0, -0.02, 0.0], [1.0, 0.0, 0.02],
         [1.0, 0.0, -0.02], [1.0, 0.01, 0.01], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    outlier = _unit(outlier)
    prototypes, weights, audit = m25.fit_stable_prototypes(outlier)
    assert len(prototypes) == 1
    np.testing.assert_allclose(weights, [1.0])
    assert audit["split_accepted"] is False

