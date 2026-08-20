from __future__ import annotations

from dataclasses import fields

import numpy as np

from cvsrffi.stage2_ablation_quantization import (
    F0,
    F3,
    compile_affine_state,
    decode_affine_state,
)
from cvsrffi.stage2_m24_compiler import (
    M24InferenceState,
    compile_m24_head,
    margin_normalized_quantization_audit,
)
from cvsrffi.stage2_m24_rf_residual import safe_rf_residual
from cvsrffi.stage2_m24_safe_residual import (
    D1,
    D1_REFIT,
    D3,
    D4,
    M24_ARMS,
    compile_m24_d1_from_f1_head,
    fit_m24_safe_residual,
)


def test_rf_residual_never_exceeds_alpha_point_one_and_k2_is_off() -> None:
    base = np.array([[1.0, 0.0], [0.0, 1.0]])
    rf = np.array([[4.0], [-4.0]])
    support = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    support_rf = np.array([[1.0], [0.9], [-1.0], [-0.9]])
    off, off_bias, off_audit = safe_rf_residual(
        base, np.zeros(2), rf, np.zeros(2), support, support_rf, labels, k_shot=2
    )
    np.testing.assert_array_equal(off, np.zeros_like(rf))
    np.testing.assert_array_equal(off_bias, np.zeros(2))
    assert off_audit["mode"] == "forced_off_k_le_2"
    gated, gated_bias, audit = safe_rf_residual(
        base, np.zeros(2), rf, np.zeros(2), support, support_rf, labels, k_shot=10
    )
    assert np.max(np.abs(gated / rf), initial=0.0) <= 0.1 + 1.0e-12
    assert gated_bias.shape == (2,)
    assert audit["alpha_max"] <= 0.1


def test_rf_gate_stays_off_without_two_support_side_helps() -> None:
    base = np.eye(2)
    support = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    support_rf = np.ones((4, 1))
    gated, bias, audit = safe_rf_residual(
        base,
        np.zeros(2),
        np.ones((2, 1)),
        np.zeros(2),
        support,
        support_rf,
        np.array([0, 0, 1, 1]),
        k_shot=5,
    )
    np.testing.assert_array_equal(gated, np.zeros((2, 1)))
    np.testing.assert_array_equal(bias, np.zeros(2))
    assert audit["global_help"] < 2


def test_margin_normalized_quantization_audit_reports_tail_risk() -> None:
    reference = np.array([[1.0, 0.9], [0.8, 0.1], [0.5, 0.49], [2.0, 0.0]])
    compiled = np.array([[0.99, 0.91], [0.79, 0.11], [0.48, 0.51], [1.99, 0.01]])
    audit = margin_normalized_quantization_audit(reference, compiled)
    assert audit["r_p50"] <= audit["r_p95"] <= audit["r_p99"] <= audit["r_max"]
    assert audit["fraction_r_gt_0_1"] > 0.0
    assert audit["fraction_r_gt_0_5"] > 0.0


def test_compiled_inference_state_contains_no_registration_workspace() -> None:
    coefficient = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    state, resource, _audit = compile_m24_head(
        coefficient,
        np.zeros(2, dtype=np.float32),
        classes=("a", "b"),
        domain_digest="d" * 64,
        config_hash="c" * 64,
        support_features=coefficient,
        transient_workspace_bytes=4096,
    )
    assert isinstance(state, M24InferenceState)
    assert {field.name for field in fields(state)} == {
        "schema",
        "classes",
        "compiled_affine_state",
        "input_log_diag_fp32",
        "domain_digest",
        "config_hash",
        "audit",
    }
    assert resource["persistent_update_state_bytes"] == 0
    assert resource["transient_registration_workspace_peak_bytes"] == 4096
    assert resource["compiled_inference_state_bytes"] >= state.compiled_affine_state.state_bytes


def test_compiled_head_applies_frozen_f1_metric_before_scoring() -> None:
    coefficient = np.array([[1.0, -0.4], [-0.2, 0.8]], dtype=np.float32)
    bias = np.array([0.15, -0.05], dtype=np.float32)
    support = np.array([[1.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    log_diag = np.log(np.array([2.0, 0.5], dtype=np.float32))
    state, resource, _audit = compile_m24_head(
        coefficient,
        bias,
        classes=("a", "b"),
        domain_digest="d" * 64,
        config_hash="c" * 64,
        support_features=support,
        transient_workspace_bytes=0,
        input_log_diag=log_diag,
    )
    prepared = support * np.exp(log_diag)[None, :]
    prepared /= np.linalg.norm(prepared, axis=1, keepdims=True)
    expected = prepared @ coefficient.T + bias[None, :]
    assert np.array_equal(np.argmax(state.score(support), axis=1), np.argmax(expected, axis=1))
    assert resource["compiled_inference_state_bytes"] >= state.compiled_affine_state.state_bytes + log_diag.nbytes


def test_d1_is_exact_f1_and_d3_d4_enable_only_their_declared_quality_path() -> None:
    rng = np.random.default_rng(2410)
    classes = ("a", "b")
    labels = np.repeat(classes, 5)
    blocks = rng.normal(size=(10, 266))
    coefficient = rng.normal(size=(2, 256))
    bias = rng.normal(size=2)
    quality = np.linspace(0.1, 1.0, 10)
    d1, d1_audit, _ = fit_m24_safe_residual(
        arm=D1,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        support_quality=quality,
        k_shot=5,
        old_class_count=1,
        f1_coefficient=coefficient,
        f1_bias=bias,
        domain_digest="d" * 64,
    )
    features = __import__("cvsrffi.stage2_m24_safe_residual", fromlist=["prepare_query_features"]).prepare_query_features(blocks, feature_dim=256)
    expected = features @ coefficient.T + bias[None, :]
    assert np.array_equal(np.argmax(d1.score(features), axis=1), np.argmax(expected, axis=1))
    assert d1_audit["modules"]["quality_center_enabled"] is False

    for arm, center_enabled, covariance_enabled in ((D3, True, False), (D4, False, True)):
        _state, audit, _workspace = fit_m24_safe_residual(
            arm=arm,
            support_blocks=blocks,
            support_labels=labels,
            classes=classes,
            support_quality=quality,
            k_shot=5,
            old_class_count=1,
            f1_coefficient=coefficient,
            f1_bias=bias,
            domain_digest="d" * 64,
        )
        assert audit["modules"]["quality_center_enabled"] is center_enabled
        assert audit["modules"]["quality_covariance_enabled"] is covariance_enabled


def test_d1_compile_is_a_pure_288_to_256_head_compiler() -> None:
    rng = np.random.default_rng(2411)
    blocks = rng.normal(size=(10, 266))
    coefficient = rng.normal(size=(2, 288))
    bias = rng.normal(size=2)
    log_diag = rng.normal(scale=0.05, size=288)
    state, audit, workspace = compile_m24_d1_from_f1_head(
        f1_coefficient=coefficient,
        f1_bias=bias,
        f1_log_diag=log_diag,
        classes=("a", "b"),
        domain_digest="d" * 64,
        support_blocks=blocks,
    )
    features = __import__(
        "cvsrffi.stage2_m24_safe_residual", fromlist=["prepare_query_features"]
    ).prepare_query_features(blocks, feature_dim=256)
    prepared = features * np.exp(log_diag[:256])[None, :]
    prepared /= np.linalg.norm(prepared, axis=1, keepdims=True)
    expected = prepared @ coefficient[:, :256].T + bias[None, :]
    assert np.array_equal(np.argmax(state.score(features), axis=1), np.argmax(expected, axis=1))
    assert workspace.nbytes == 0
    assert audit["method_identity"] == "P2_A1_IF256_COMPILE_PARITY"
    assert audit["fresh_support_refit"] is False
    assert audit["resource"]["transient_registration_workspace_peak_bytes"] == 0


def test_d1_compile_preserves_fp32_source_order_below_fp16_bias_resolution() -> None:
    blocks = np.ones((2, 256), dtype=np.float32)
    coefficient = np.zeros((2, 288), dtype=np.float32)
    bias = np.array([1.0, 1.0001], dtype=np.float32)
    state, audit, _workspace = compile_m24_d1_from_f1_head(
        f1_coefficient=coefficient,
        f1_bias=bias,
        f1_log_diag=np.zeros(288, dtype=np.float32),
        f1_compiled_affine_state=None,
        classes=("a", "b"),
        domain_digest="d" * 64,
        support_blocks=blocks,
    )
    features = __import__(
        "cvsrffi.stage2_m24_safe_residual", fromlist=["prepare_query_features"]
    ).prepare_query_features(blocks, feature_dim=256)
    assert state.compiled_affine_state.arm_id == F0
    assert state.predict(features).tolist() == ["b", "b"]
    assert audit["source_storage_precision"] == "P2-FP32"


def test_d1_compile_crops_existing_f3_state_without_requantizing_prefix() -> None:
    rng = np.random.default_rng(2412)
    source_coefficient = rng.normal(size=(3, 288)).astype(np.float32)
    source_bias = rng.normal(size=3).astype(np.float32)
    source = compile_affine_state(
        source_coefficient,
        source_bias,
        arm_id=F3,
        block_sizes=(160, 96, 32),
    )
    decoded_coefficient, decoded_bias = decode_affine_state(source)
    blocks = rng.normal(size=(5, 256)).astype(np.float32)
    state, audit, _workspace = compile_m24_d1_from_f1_head(
        f1_coefficient=decoded_coefficient,
        f1_bias=decoded_bias,
        f1_log_diag=np.zeros(288, dtype=np.float32),
        f1_compiled_affine_state=source,
        classes=("a", "b", "c"),
        domain_digest="d" * 64,
        support_blocks=blocks,
    )
    compact = state.compiled_affine_state
    assert compact.arm_id == F3
    assert compact.block_offsets == (0, 160, 256)
    for compact_layer, source_layer in zip(
        compact.coefficient_layers, source.coefficient_layers
    ):
        np.testing.assert_array_equal(compact_layer, source_layer[:, :256])
    for compact_scale, source_scale in zip(compact.scale_layers, source.scale_layers):
        np.testing.assert_array_equal(compact_scale, source_scale[:, :2])
    np.testing.assert_array_equal(compact.bias, source.bias)
    assert audit["source_storage_precision"] == "P2-F3"


def test_m24_arm_catalog_is_complete_d0_through_d10() -> None:
    assert len(M24_ARMS) == 12
    assert M24_ARMS[0].startswith("M24-D0-")
    assert M24_ARMS[1] == "M24-D1-COMPILE-PARITY"
    assert D1_REFIT == "M24-D1-REFIT"
    assert M24_ARMS[-1].startswith("M24-D10-")
