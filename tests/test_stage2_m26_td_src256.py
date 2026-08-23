from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_ablation_quantization import F0
from cvsrffi.stage2_m24_compiler import compile_m24_head
from cvsrffi.stage2_m24_features import IF_DIM, physical_if256
from cvsrffi.stage2_m26_spectral_anchor import build_phase1_spectral_anchor
from cvsrffi.stage2_m26_td_src256 import (
    T1,
    T2,
    T3,
    T4,
    T5,
    apply_m26_bounded_residual,
    estimate_target_domain_state,
    fit_m26_td_src256,
    m26_arm_config_hash,
)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _fixture(k_shot: int = 5):
    rng = np.random.default_rng(82610 + k_shot)
    old = tuple(f"old-{index}" for index in range(6))
    new = ("new-0", "new-1")
    classes = old + new
    source_rows: list[np.ndarray] = []
    source_labels: list[str] = []
    support_rows: list[np.ndarray] = []
    support_labels: list[str] = []
    identity_shift = np.zeros(160)
    identity_shift[150:156] = 0.12
    fft_shift = np.linspace(-0.05, 0.05, 96)
    for index, name in enumerate(old):
        source_id = np.zeros(160)
        source_fft = np.zeros(96)
        source_id[index * 13] = 1.0
        source_fft[index * 11] = 1.0
        for _ in range(8):
            source_rows.append(
                np.concatenate(
                    [source_id + 0.01 * rng.normal(size=160), source_fft + 0.01 * rng.normal(size=96)]
                )
            )
            source_labels.append(name)
        for _ in range(k_shot):
            support_rows.append(
                np.concatenate(
                    [source_id + identity_shift + 0.02 * rng.normal(size=160), source_fft + fft_shift + 0.02 * rng.normal(size=96)]
                )
            )
            support_labels.append(name)
    for offset, name in enumerate(new):
        for _ in range(k_shot):
            identity = 0.02 * rng.normal(size=160)
            fft = 0.02 * rng.normal(size=96)
            identity[90 + offset * 7] += 1.0
            fft[70 + offset * 7] += 1.0
            support_rows.append(np.concatenate([identity, fft]))
            support_labels.append(name)
    anchor, _audit = build_phase1_spectral_anchor(
        np.asarray(source_rows, dtype=np.float32),
        np.asarray(source_labels),
        class_registry=old,
        checkpoint_sha256="c" * 64,
    )
    blocks = np.asarray(support_rows, dtype=np.float32)
    labels = np.asarray(support_labels)
    features = physical_if256(blocks)
    centres = _unit(np.stack([features[labels == name].mean(axis=0) for name in classes]))
    base_state, _resource, _quant = compile_m24_head(
        7.0 * centres,
        np.zeros(len(classes), dtype=np.float32),
        classes=classes,
        domain_digest="synthetic-domain",
        config_hash="synthetic-base",
        support_features=features,
        transient_workspace_bytes=0,
        block_sizes=(160, 96),
        input_log_diag=np.zeros(IF_DIM, dtype=np.float32),
        compile_arm=F0,
    )
    return blocks, labels, classes, old, anchor, base_state


def test_domain_state_uses_old_support_only_and_detects_shared_shift() -> None:
    blocks, labels, _classes, old, anchor, _base = _fixture(5)
    old_mask = np.isin(labels, old)
    state = estimate_target_domain_state(
        blocks[old_mask], labels[old_mask], old, anchor
    )
    assert np.linalg.norm(state.shared_identity_shift) > 0.01
    assert np.linalg.norm(state.shared_envelope_shift) > 0.001
    assert np.linalg.norm(state.shared_geometry_shift) > 0.001
    assert 0.0 <= state.identity_reliability <= 1.0
    assert 0.0 <= state.envelope_reliability <= 1.0
    assert 0.0 <= state.geometry_reliability <= 1.0

    permuted_new = blocks.copy()
    permuted_new[~old_mask] *= -3.0
    repeated = estimate_target_domain_state(
        permuted_new[old_mask], labels[old_mask], old, anchor
    )
    np.testing.assert_array_equal(repeated.shared_identity_shift, state.shared_identity_shift)
    np.testing.assert_array_equal(repeated.shared_envelope_shift, state.shared_envelope_shift)
    np.testing.assert_array_equal(repeated.shared_geometry_shift, state.shared_geometry_shift)


def test_k1_and_zero_selection_are_exact_b0_fallbacks() -> None:
    blocks, labels, classes, old, anchor, base = _fixture(1)
    for arm in (T1, T2, T3, T4, T5):
        state, audit = fit_m26_td_src256(
            arm=arm,
            base_state=base,
            support_blocks=blocks,
            support_labels=labels,
            classes=classes,
            k_shot=1,
            old_class_count=len(old),
            source_anchor=anchor,
            domain_digest="synthetic-domain",
        )
        expected = base.score(physical_if256(blocks))
        assert np.array_equal(state.score(blocks), expected)
        assert audit["selected_strength"] == 0.0
        assert audit["fallback_reason"] == "K1_EXACT_B0"


def test_bounded_residual_preserves_high_margin_and_caps_each_logit() -> None:
    base = np.asarray([[2.0, 0.0, -1.0], [0.05, 0.0, -0.1]])
    residual = np.asarray([[-2.0, 2.0, 0.0], [4.0, -2.0, 1.0]])
    adjusted, audit = apply_m26_bounded_residual(
        base, residual, strength=0.04, margin_gate=0.10
    )
    np.testing.assert_array_equal(adjusted[0], base[0])
    assert np.max(np.abs(adjusted[1] - base[1])) <= 0.04 + 1.0e-12
    assert audit["gated_query_count"] == 1


def test_m26_candidate_identity_binds_anchor_content() -> None:
    first = m26_arm_config_hash(T5, "a" * 64)
    second = m26_arm_config_hash(T5, "b" * 64)
    assert first != second
    with pytest.raises(ValueError, match="anchor component"):
        m26_arm_config_hash(T5, "not-a-digest")


def test_t1_to_t5_expose_separate_support_only_mechanisms() -> None:
    blocks, labels, classes, old, anchor, base = _fixture(5)
    audits = {}
    states = {}
    for arm in (T1, T2, T3, T4, T5):
        states[arm], audits[arm] = fit_m26_td_src256(
            arm=arm,
            base_state=base,
            support_blocks=blocks,
            support_labels=labels,
            classes=classes,
            k_shot=5,
            old_class_count=len(old),
            source_anchor=anchor,
            domain_digest="synthetic-domain",
        )
        assert audits[arm]["query_rows_used"] == 0
        assert audits[arm]["support_only"] is True
        assert audits[arm]["feature_dim"] == 256
    assert audits[T1]["active_blocks"] == ["identity160"]
    assert audits[T2]["active_blocks"] == ["fft_envelope32", "fft_ripple64"]
    assert audits[T3]["active_blocks"] == [
        "identity160",
        "fft_envelope32",
        "fft_ripple64",
    ]
    assert audits[T4]["active_blocks"] == ["fft_magnitude_geometry96"]
    assert audits[T5]["active_blocks"] == [
        "identity160",
        "fft_magnitude_geometry96",
    ]
    assert states[T3].domain_state.digest == states[T1].domain_state.digest
    assert states[T5].domain_state.digest == states[T1].domain_state.digest
