from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    ALLOWED_NPZ_MEMBERS,
    CenterLowRankPrototypeComponent,
    FEATURE_DIM,
    NPZ_NAME,
    PENDING_OUTER_JOINT_SEAL,
    RESIDUAL_RANK,
    SCHEMA,
    build_center_lowrank_component,
    compress_v1_dense_component,
    load_center_lowrank_component,
    radius_generation_proof_sha256,
    save_center_lowrank_component,
    validate_center_lowrank_component,
    v1_payload_sha256,
)


CHECKPOINT_SHA = "a" * 64
BINDING_SHA = "b" * 64
V1_SHA = "c" * 64
CODE_SHA = "d" * 64
CONFIG_SHA = "e" * 64
STREAM_SHA = "f" * 64
CLASSES = tuple(f"tx-{index}" for index in range(6))


def _quantize_dense(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(value), axis=-1)
    scale32 = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(value / scale32[..., None]), -127, 127).astype(np.int8)
    return q, scale32.astype(np.float16)


def _v1_payload(*, inactive_tail: int = 2) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(1707)
    active_d, c, p = 14, 6, FEATURE_DIM
    core = rng.normal(size=(c, p))
    core /= np.linalg.norm(core, axis=1, keepdims=True)
    basis = np.empty((c, RESIDUAL_RANK, p), dtype=np.float64)
    for class_index in range(c):
        raw = rng.normal(size=(p, RESIDUAL_RANK))
        q_basis, _ = np.linalg.qr(raw)
        basis[class_index] = q_basis.T
    coeff = rng.normal(scale=0.025, size=(active_d, c, RESIDUAL_RANK))
    coeff[0] = 0.0
    dense = core[None, :, :] + np.einsum("dcr,crp->dcp", coeff, basis)
    dense /= np.linalg.norm(dense, axis=-1, keepdims=True)
    total_d = active_d + inactive_tail
    vectors = np.zeros((total_d, c, p), dtype=np.float32)
    vectors[:active_d] = dense.astype(np.float32)
    q, scale = _quantize_dense(vectors)
    mask = np.zeros((total_d, c), dtype=np.uint8)
    mask[:active_d] = 1
    q[active_d:] = 0
    scale[active_d:] = np.float16(1.0)
    payload = {
        "domain_class_q": q,
        "domain_class_scale": scale,
        "domain_class_mask": mask,
        "domain_registry": np.arange(100, 100 + total_d, dtype=np.int16),
        "class_registry": np.asarray(CLASSES, dtype=np.str_),
        "feature_schema": np.asarray(
            "ADV3B02:z_id:unit_l2:160:v1", dtype=np.str_
        ),
    }
    radius = np.zeros((total_d, c), dtype=np.float32)
    radius[:active_d] = rng.uniform(0.01, 0.12, size=(active_d, c))
    return payload, radius


def _build(
    *,
    radius: np.ndarray | None = None,
    formal: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    v1, default_radius = _v1_payload()
    selected_radius = default_radius if radius is None else radius
    proof = radius_generation_proof_sha256(
        v1,
        selected_radius,
        phase1_stream_sha256=STREAM_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=BINDING_SHA,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
    )
    return build_center_lowrank_component(
        v1,
        radius_p90_cosine_distance=selected_radius,
        phase1_stream_sha256=STREAM_SHA,
        radius_generation_proof_sha256_value=proof,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=BINDING_SHA,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
        provenance_status=PENDING_OUTER_JOINT_SEAL,
        formal_phase2_eligible=formal,
    )


def _save_pending(root: Path) -> tuple[dict[str, np.ndarray], dict, dict[str, str]]:
    payload, manifest = _build()
    result = save_center_lowrank_component(root, payload, manifest)
    return payload, manifest, result


def test_fixed_rank3_compression_is_deterministic_and_component_only() -> None:
    v1, radius = _v1_payload()
    first, audit_first = compress_v1_dense_component(
        v1, radius_p90_cosine_distance=radius, formal_phase2_eligible=True
    )
    second, audit_second = compress_v1_dense_component(
        v1, radius_p90_cosine_distance=radius * 1.5, formal_phase2_eligible=True
    )
    assert int(first["residual_rank"]) == RESIDUAL_RANK == 3
    assert str(first["schema"]) == SCHEMA
    assert str(first["center_domain_handle"]) == str(second["center_domain_handle"])
    for key in (
        "core_q",
        "core_scale",
        "residual_basis_q",
        "residual_basis_scale",
        "residual_coeff_q",
        "residual_coeff_scale",
    ):
        np.testing.assert_array_equal(first[key], second[key])
    assert audit_first["center_domain_handle"] == audit_second["center_domain_handle"]
    assert audit_first["svd_sign_canonicalization"].startswith("largest_abs")
    for key in ("core_q", "residual_basis_q", "residual_coeff_q"):
        assert first[key].dtype == np.int8
        assert not np.any(first[key] == -128)
    basis = first["residual_basis_q"].astype(np.float32) * first[
        "residual_basis_scale"
    ][..., None].astype(np.float32)
    for row in basis.reshape(-1, FEATURE_DIM):
        assert row[int(np.argmax(np.abs(row)))] >= 0.0


def test_rank3_14x6_numeric_payload_and_reconstruction_macs_are_exact() -> None:
    payload, manifest = _build()
    resource = manifest["resource_audit"]
    assert payload["core_q"].shape == (6, 160)
    assert payload["residual_basis_q"].shape == (6, 3, 160)
    assert payload["residual_coeff_q"].shape == (13, 6, 3)
    assert payload["residual_coeff_scale"].shape == (13, 6)
    assert payload["radius_q"].shape == (14, 6)
    assert resource["direction_numeric_payload_bytes"] == 4278
    assert resource["radius_numeric_payload_bytes"] == 96
    assert resource["compressed_numeric_payload_bytes"] == 4374
    assert resource["single_class_prototype_reconstruction_macs"] == 480
    assert resource["single_domain_all_class_reconstruction_macs"] == 2880
    assert resource["all_residual_domain_enrollment_reconstruction_macs"] == 37440
    assert resource["center_only_reconstruction_macs"] == 0
    assert resource["persistent_dense_float_bank_bytes"] == 0


def test_roundtrip_load_exposes_only_center_or_one_domain_and_one_radius_row(
    tmp_path: Path,
) -> None:
    _, _, result = _save_pending(tmp_path)
    validated = validate_center_lowrank_component(
        tmp_path,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=BINDING_SHA,
        expected_pre_sign_content_root_sha256=result[
            "pre_sign_content_root_sha256"
        ],
    )
    component = load_center_lowrank_component(
        tmp_path,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=BINDING_SHA,
        expected_pre_sign_content_root_sha256=result[
            "pre_sign_content_root_sha256"
        ],
        allow_pending_outer_joint_seal_development=True,
    )
    assert isinstance(component, CenterLowRankPrototypeComponent)
    assert validated["formal_phase2_eligible"] is False
    assert validated["component_state"] == PENDING_OUTER_JOINT_SEAL
    assert validated["outer_bundle_signature_required"] is True
    center = component.dequantized_center()
    np.testing.assert_array_equal(
        center, component.reconstruct_domain(component.center_domain_handle)
    )
    assert center.shape == (6, FEATURE_DIM) and not center.flags.writeable
    other = component.residual_domain_registry[0]
    assert component.reconstruct_domain(other).shape == (6, FEATURE_DIM)
    assert component.radius_for_domain(other).shape == (6,)
    assert not hasattr(component, "dequantized_dense_bank")
    assert not hasattr(component, "reconstruct_all_domains")
    assert not any(
        value.dtype == np.float32 and value.ndim >= 3
        for value in vars(component).values()
        if isinstance(value, np.ndarray)
    )
    with pytest.raises(ValueError, match="unknown pre-registered"):
        component.reconstruct_domain("target-selected-domain")


def test_direction_and_radius_quantization_error_are_bounded(tmp_path: Path) -> None:
    v1, radius = _v1_payload()
    proof = radius_generation_proof_sha256(
        v1,
        radius,
        phase1_stream_sha256=STREAM_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=BINDING_SHA,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
    )
    payload, manifest = build_center_lowrank_component(
        v1,
        radius_p90_cosine_distance=radius,
        phase1_stream_sha256=STREAM_SHA,
        radius_generation_proof_sha256_value=proof,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=BINDING_SHA,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
        provenance_status=PENDING_OUTER_JOINT_SEAL,
        formal_phase2_eligible=False,
    )
    result = save_center_lowrank_component(tmp_path, payload, manifest)
    component = load_center_lowrank_component(
        tmp_path,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=BINDING_SHA,
        expected_pre_sign_content_root_sha256=result[
            "pre_sign_content_root_sha256"
        ],
        allow_pending_outer_joint_seal_development=True,
    )
    retained = np.flatnonzero(np.all(v1["domain_class_mask"] == 1, axis=1))
    reference = v1["domain_class_q"][retained].astype(np.float32) * v1[
        "domain_class_scale"
    ][retained, :, None].astype(np.float32)
    reconstructed = np.stack(
        [component.reconstruct_domain(handle) for handle in component.domain_registry]
    )
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
    reconstructed /= np.linalg.norm(reconstructed, axis=-1, keepdims=True)
    cosine = np.sum(reference * reconstructed, axis=-1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    assert float(np.max(angle)) < 2.5
    assert manifest["resource_audit"]["max_reconstruction_angle_deg"] < 2.5

    radius_reference = radius[retained]
    radius_hat = np.stack(
        [component.radius_for_domain(handle) for handle in component.domain_registry]
    )
    half_step = float(np.max(component.radius_scale.astype(np.float32))) / 2.0
    assert float(np.max(np.abs(radius_reference - radius_hat))) <= half_step + 2.0e-5


def test_pending_component_requires_radius_and_formal_load_fails_closed(
    tmp_path: Path,
) -> None:
    v1, radius = _v1_payload()
    with pytest.raises(ValueError, match="requires offline aggregated P90 radius"):
        compress_v1_dense_component(
            v1, radius_p90_cosine_distance=None, formal_phase2_eligible=True
        )
    proof = radius_generation_proof_sha256(
        v1,
        radius,
        phase1_stream_sha256=STREAM_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=BINDING_SHA,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
    )
    common = {
        "phase1_stream_sha256": STREAM_SHA,
        "radius_generation_proof_sha256_value": proof,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "class_handle_binding_sha256": BINDING_SHA,
        "generation_code_sha256": CODE_SHA,
        "generation_config_sha256": CONFIG_SHA,
        "provenance_status": PENDING_OUTER_JOINT_SEAL,
    }
    with pytest.raises(ValueError, match="requires aggregated P90 radius"):
        build_center_lowrank_component(
            v1,
            radius_p90_cosine_distance=None,
            formal_phase2_eligible=False,
            **common,
        )
    with pytest.raises(ValueError, match="cannot be formally Phase2 eligible"):
        build_center_lowrank_component(
            v1,
            radius_p90_cosine_distance=radius,
            formal_phase2_eligible=True,
            **common,
        )
    payload, manifest = _build(radius=radius, formal=False)
    assert manifest["formal_phase2_eligible"] is False
    assert manifest["component_state"] == PENDING_OUTER_JOINT_SEAL
    assert manifest["outer_bundle_signature_required"] is True
    assert "detached_signature_sha256" not in manifest
    result = save_center_lowrank_component(tmp_path, payload, manifest)
    kwargs = {
        "expected_checkpoint_sha256": CHECKPOINT_SHA,
        "expected_class_handle_binding_sha256": BINDING_SHA,
        "expected_pre_sign_content_root_sha256": result[
            "pre_sign_content_root_sha256"
        ],
    }
    with pytest.raises(ValueError, match="pending outer joint seal"):
        load_center_lowrank_component(tmp_path, **kwargs)
    development = load_center_lowrank_component(
        tmp_path, **kwargs, allow_pending_outer_joint_seal_development=True
    )
    assert development.radius_for_domain(development.center_domain_handle).shape == (6,)


def test_arbitrary_radius_without_matching_generation_proof_is_rejected() -> None:
    v1, radius = _v1_payload()
    with pytest.raises(ValueError, match="radius generation proof"):
        build_center_lowrank_component(
            v1,
            radius_p90_cosine_distance=radius,
            phase1_stream_sha256=STREAM_SHA,
            radius_generation_proof_sha256_value="0" * 64,
            checkpoint_sha256=CHECKPOINT_SHA,
            class_handle_binding_sha256=BINDING_SHA,
            generation_code_sha256=CODE_SHA,
            generation_config_sha256=CONFIG_SHA,
            provenance_status=PENDING_OUTER_JOINT_SEAL,
            formal_phase2_eligible=False,
        )


def test_radius_must_be_explicit_aggregate_matrix_not_prototype_offset() -> None:
    v1, radius = _v1_payload()
    with pytest.raises(ValueError, match="match original v1"):
        compress_v1_dense_component(
            v1,
            radius_p90_cosine_distance=radius[:14],
            formal_phase2_eligible=True,
        )
    bad = radius.copy()
    bad[0, 0] = -0.1
    with pytest.raises(ValueError, match="finite non-negative"):
        compress_v1_dense_component(
            v1, radius_p90_cosine_distance=bad, formal_phase2_eligible=True
        )


def test_payload_shape_range_scale_and_partial_coverage_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    payload, manifest = _build()
    extra = dict(payload)
    extra["dense_bank"] = np.zeros((14, 6, FEATURE_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="strict allowlist"):
        save_center_lowrank_component(tmp_path / "extra", extra, manifest)

    forbidden = {key: np.array(value, copy=True) for key, value in payload.items()}
    forbidden["core_q"][0, 0] = -128
    with pytest.raises(ValueError, match="forbidden -128"):
        save_center_lowrank_component(tmp_path / "q", forbidden, manifest)

    zero_scale = {key: np.array(value, copy=True) for key, value in payload.items()}
    zero_scale["residual_basis_scale"][0, 0] = np.float16(0.0)
    with pytest.raises(ValueError, match="finite positive"):
        save_center_lowrank_component(tmp_path / "scale", zero_scale, manifest)

    v1, radius = _v1_payload()
    v1["domain_class_mask"][1, 0] = 0
    v1["domain_class_q"][1, 0] = 0
    with pytest.raises(ValueError, match="complete class coverage"):
        compress_v1_dense_component(
            v1, radius_p90_cosine_distance=radius, formal_phase2_eligible=True
        )


def test_serialized_hash_member_and_binding_tamper_fail_closed(tmp_path: Path) -> None:
    _, _, result = _save_pending(tmp_path)
    npz_path = tmp_path / NPZ_NAME
    npz_path.write_bytes(npz_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="NPZ SHA256 mismatch"):
        validate_center_lowrank_component(tmp_path)

    clean = tmp_path / "clean"
    _, _, result = _save_pending(clean)
    with pytest.raises(ValueError, match="checkpoint_sha256 binding mismatch"):
        validate_center_lowrank_component(
            clean, expected_checkpoint_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="pre_sign_content_root_sha256 binding mismatch"):
        load_center_lowrank_component(
            clean,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            expected_class_handle_binding_sha256=BINDING_SHA,
            expected_pre_sign_content_root_sha256="1" * 64,
        )
    (clean / "unexpected.sidecar").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="directory member allowlist"):
        validate_center_lowrank_component(clean)
    assert len(result["pre_sign_content_root_sha256"]) == 64


def test_manifest_npz_allowlist_and_protocol_fields_are_strict(tmp_path: Path) -> None:
    _, _, result = _save_pending(tmp_path)
    manifest = validate_center_lowrank_component(tmp_path)
    with np.load(tmp_path / NPZ_NAME, allow_pickle=False) as arrays:
        assert set(arrays.files) == ALLOWED_NPZ_MEMBERS
        assert not any(
            token in key.lower()
            for key in arrays.files
            for token in ("sample", "member", "count", "path", "dense")
        )
    assert manifest["phase2_phase1_prototype_residual_rank"] == 3
    assert manifest["phase2_phase1_prototype_dense_bank_persistent"] is False
    assert manifest["phase2_phase1_prototype_dequantized_export"] is False
    assert manifest["phase2_phase1_prototype_update_access"] is False
    assert manifest["radius_definition"].startswith("p90_cosine_distance")
    assert len(result["manifest_sha256"]) == 64


def test_zero_residual_uses_safe_scale_and_public_api_has_no_data_access_surface() -> None:
    v1, radius = _v1_payload(inactive_tail=0)
    v1["domain_class_q"][1:] = v1["domain_class_q"][0]
    v1["domain_class_scale"][1:] = v1["domain_class_scale"][0]
    payload, _ = compress_v1_dense_component(
        v1, radius_p90_cosine_distance=radius, formal_phase2_eligible=True
    )
    assert np.all(payload["residual_coeff_q"] == 0)
    assert np.all(payload["residual_coeff_scale"] == np.float16(1.0))

    public = (
        compress_v1_dense_component,
        build_center_lowrank_component,
        save_center_lowrank_component,
        validate_center_lowrank_component,
        load_center_lowrank_component,
        CenterLowRankPrototypeComponent.dequantized_center,
        CenterLowRankPrototypeComponent.reconstruct_domain,
        CenterLowRankPrototypeComponent.radius_for_domain,
    )
    forbidden = ("query", "source_path", "dataset", "sample", "member", "count")
    for function in public:
        parameters = inspect.signature(function).parameters
        assert not any(
            token in name.lower() for name in parameters for token in forbidden
        )
    assert "rank" not in inspect.signature(compress_v1_dense_component).parameters
