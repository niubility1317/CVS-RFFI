from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from cvsrffi import stage2_grb_jp4_cfm_d92 as cfm
from cvsrffi.phase1_grb_jp4_cfm_bundle import (
    COMPONENT_PROFILE,
    GRBJP4CFMPhase1Component,
    METHOD_ID,
    METHOD_LOCK_SCHEMA,
    PROTOCOL_SCHEMA,
    SCHEMA as PHASE1_SCHEMA,
    _resource_audit as phase1_resource_audit,
    canonical_array_sha256,
    class_handle_binding_sha256,
)
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


def _sha(char: str) -> str:
    return char * 64


def _quantized_basis(rows: int, width: int, offset: int = 0):
    codes = np.zeros((rows, width), dtype=np.int8)
    for row in range(rows):
        codes[row, offset + row] = np.int8(127)
    scales = np.full((rows,), np.float16(1.0 / 127.0), dtype="<f2")
    return codes, scales


def _fixture_ground():
    classes = tuple(f"c{index}" for index in range(6))
    prototype_codes = np.zeros((6, 3, 160), dtype=np.int8)
    prototype_scales = np.zeros((6, 3), dtype="<f2")
    prototype_mask = np.zeros((6, 3), dtype=np.bool_)
    prototype_weights = np.zeros((6, 3), dtype="<f2")
    prototype_radii = np.zeros((6, 3), dtype="<f2")
    for index in range(6):
        prototype_codes[index, 0, 20 + index] = np.int8(127)
        prototype_scales[index, 0] = np.float16(1.0 / 127.0)
        prototype_mask[index, 0] = True
        prototype_weights[index, 0] = np.float16(1.0)
        prototype_radii[index, 0] = np.float16(0.125)
    left_codes, left_scales = _quantized_basis(4, 160, 0)
    right_codes, right_scales = _quantized_basis(4, 320, 0)
    energy = np.full((4,), np.float16(0.5), dtype="<f2")
    weight = np.zeros((160, 320), dtype=np.float32)
    weight[np.arange(160), np.arange(160)] = np.float32(1.0)
    joint_sha = cfm._array_sha(weight)
    factor_numeric = int(
        left_codes.nbytes
        + left_scales.nbytes
        + right_codes.nbytes
        + right_scales.nbytes
        + energy.nbytes
        + 6
    )
    resource = {
        "ground_wire_bytes": 8192,
        "jp4_update_factor_numeric_bytes": factor_numeric,
        "jp4_update_factor_receipt_bytes": 256,
        "jp4_update_factor_wire_bytes": factor_numeric + 256,
        "phase1_margin_wire_bytes": 4,
        "component_metadata_wire_bytes": 512,
        "total_component_bytes": 8192 + factor_numeric + 256 + 4 + 512,
        "jp4_update_factor_wire_limit_bytes": 4096,
        "arm_state_limit_bytes": 256 * 1024,
        "persistent_dense_float_bank_bytes": 0,
        "ground_direction_rank": 4,
    }
    ground = cfm.GroundCFMInput(
        prototype_codes,
        prototype_scales,
        prototype_mask,
        prototype_weights,
        prototype_radii,
        left_codes,
        left_scales,
        right_codes,
        right_scales,
        energy,
        np.asarray(np.float16(0.25), dtype="<f2"),
        np.asarray(np.float16(1.0), dtype="<f2"),
        classes,
        _sha("a"),
        joint_sha,
        _sha("b"),
        _sha("c"),
        resource,
        cfm._sha_json(resource),
    )
    return ground, weight


def _lock(k: int) -> cfm.CFMMethodLock:
    return cfm.CFMMethodLock(
        qknn_neighbor_count=k,
        student_nu=4.0,
        kernel_effective_dim=8.0,
        kernel_volume_gamma=1.0,
        kernel_scale=0.5,
        qknn_lock_digest=_sha("d"),
        phase1_method_lock_sha256=_sha("b"),
        delta_q=0.25,
        tau_q=1.0,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
    )


def _phase1_lock_mapping(
    classes: tuple[str, ...], qknn_lock_sha256_by_k: dict[str, str]
):
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "method_id": METHOD_ID,
        "candidate_id": METHOD_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
        "checkpoint_sha256": _sha("a"),
        "class_handle_binding_sha256": class_handle_binding_sha256(classes),
        "qknn_lock_sha256_by_k": dict(qknn_lock_sha256_by_k),
        "rank": 4,
        "old_class_count": 6,
        "allowed_k": [1, 5, 10],
        "ground_old_multiprototype_enabled": True,
        "ground_old_multiprototype_max_per_class": 3,
        "ground_old_multiprototype_min_physical_samples": 2,
        "ground_old_multiprototype_old_classes_only": True,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
        "ground_component_phase2_mutable": False,
        "delta_tau_source": (
            "phase1_receiver_lodo_correct_held_pseudoquery_only"
        ),
        "active_set_steps": 2,
        "ridge_fraction": 0.01,
        "theta_box_abs": 1.0,
        "trust_divisor_squared": 160,
        "g_denominator": 4,
        "target25_release_authorized": False,
        "query_fit_access": False,
        "query_rows_used_for_fit": 0,
        "delta_q": 0.25,
        "tau_q": 1.0,
    }


def _qknn_lock(k: int):
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=4.0,
        kernel_effective_dim=8,
        kernel_volume_gamma=1.0,
        shared_h0=0.5,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256=_sha("e"),
        quantization_margin_audit_sha256=_sha("f"),
    )


def _support(k: int, *, mutate_new: bool = False):
    labels = []
    tokens = []
    rows = []
    jacobians = []
    rng = np.random.default_rng(60720260724 + k)
    sample_index = 0
    for class_index in range(6):
        for within in range(k):
            labels.append(f"c{class_index}")
            tokens.append(f"physical-{sample_index:04d}")
            row = np.zeros((160,), dtype=np.float32)
            row[20 + class_index] = np.float32(1.0)
            row[:4] = np.asarray(
                [0.08, -0.06, 0.04, -0.02], dtype=np.float32
            ) + np.float32(0.003 * within)
            jac = np.zeros((4, 160), dtype=np.float32)
            jac[np.arange(4), np.arange(4)] = np.asarray(
                [
                    0.12 + 0.002 * within,
                    0.10 + 0.003 * class_index,
                    0.08 + 0.001 * sample_index,
                    0.06 + 0.002 * within,
                ],
                dtype=np.float32,
            )
            jac += rng.normal(0.0, 2.0e-4, jac.shape).astype(np.float32)
            if mutate_new and class_index == 5:
                row = rng.normal(0.0, 1.0, (160,)).astype(np.float32)
                jac = rng.normal(0.0, 0.2, (4, 160)).astype(np.float32)
            rows.append(row)
            jacobians.append(jac)
            sample_index += 1
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(jacobians, dtype=np.float32),
        tuple(labels),
        tuple(tokens),
    )


def _fit(k: int, *, mutate_new: bool = False):
    ground, weight = _fixture_ground()
    rows, jacobian, labels, tokens = _support(k, mutate_new=mutate_new)
    state = cfm.fit_cfm_from_precomputed(
        base_support_zid=rows,
        support_jacobian=jacobian,
        support_labels=labels,
        support_physical_tokens=tokens,
        registered_old_classes=("c0", "c1", "c2", "c3", "c4"),
        registered_new_classes=("c5",),
        ground=ground,
        lock=_lock(k),
        checkpoint_weight=weight,
        checkpoint_sha256=_sha("a"),
    )
    return state, ground, weight


def test_second_active_set_accumulates_increment_instead_of_overwrite(
    monkeypatch,
):
    ground, weight = _fixture_ground()
    rows, jacobian, labels, tokens = _support(5)
    original = cfm._solve_increment

    def controlled(H, b, *, round_index):
        _increment, receipt = original(H, b, round_index=round_index)
        value = np.asarray(
            [0.10, -0.08, 0.06, -0.04]
            if round_index == 0
            else [0.03, 0.02, -0.01, 0.04],
            dtype=np.float64,
        )
        receipt["increment_sha256"] = cfm._array_sha(value)
        return value, receipt

    monkeypatch.setattr(cfm, "_solve_increment", controlled)
    state = cfm.fit_cfm_from_precomputed(
        base_support_zid=rows,
        support_jacobian=jacobian,
        support_labels=labels,
        support_physical_tokens=tokens,
        registered_old_classes=("c0", "c1", "c2", "c3", "c4"),
        registered_new_classes=("c5",),
        ground=ground,
        lock=_lock(5),
        checkpoint_weight=weight,
        checkpoint_sha256=_sha("a"),
    )
    rounds = state.fit_receipt["rounds"]
    assert rounds[1]["theta_base_sha256"] == rounds[0]["theta_full_sha256"]
    assert rounds[1]["increment_accumulated_not_overwritten"] is True
    g = 5.0 / 9.0
    theta1 = cfm._project_theta(
        g * np.asarray([0.10, -0.08, 0.06, -0.04]),
        ground=ground,
        base_weight=weight,
    )
    expected = cfm._project_theta(
        theta1 + g * np.asarray([0.03, 0.02, -0.01, 0.04]),
        ground=ground,
        base_weight=weight,
    )
    overwrite = cfm._project_theta(
        g * np.asarray([0.03, 0.02, -0.01, 0.04]),
        ground=ground,
        base_weight=weight,
    )
    assert np.linalg.norm(state.theta() - expected) < 0.01
    assert np.linalg.norm(state.theta() - expected) < np.linalg.norm(
        state.theta() - overwrite
    )


def test_fold_closure_rejects_any_held_physical_sample_contamination():
    good = cfm.FoldClosure(
        "held",
        ("a", "b"),
        ("a", "b"),
        ("a", "b"),
        ("a", "b"),
        ("a", "b"),
    )
    assert cfm.validate_fold_closure(good)["held_fully_excluded"] is True
    bad = cfm.FoldClosure(
        "held",
        ("a", "b"),
        ("a", "held"),
        ("a", "b"),
        ("a", "b"),
        ("a", "b"),
    )
    with pytest.raises(cfm.GRBJP4CFMError, match="contaminated"):
        cfm.validate_fold_closure(bad)
    rows, jacobian, labels, tokens = _support(5)
    snapshot = cfm.SupportSnapshot(rows, jacobian, labels, tokens)
    fold, receipt = cfm.strict_physical_loo_fold(
        snapshot, held_physical_token=tokens[7]
    )
    assert tokens[7] not in fold.physical_tokens
    assert len(fold.z_id) == len(rows) - 1
    assert receipt["shared_fold_for_qknn_d92_calibration_and_normal"] is True


@pytest.mark.parametrize("k_shot", [5, 10])
def test_kgt1_fold_refit_matches_explicit_physical_delete_and_not_subtraction(
    k_shot,
):
    ground, weight = _fixture_ground()
    rows, jacobian, labels, tokens = _support(k_shot)
    rows = np.array(rows, copy=True)
    jacobian = np.array(jacobian, copy=True)
    held_index = 0
    neighbor_query_index = 1
    rows[held_index] = rows[neighbor_query_index]
    full_backend = cfm.AffineSupportBackend(
        rows, jacobian, labels, tokens
    )
    theta0 = np.zeros((4,), dtype=np.float64)
    full_snapshot = full_backend.snapshot(theta0)
    same_class_candidates = [
        index
        for index, label in enumerate(labels)
        if label == labels[neighbor_query_index]
        and index != neighbor_query_index
    ]
    nearest = min(
        same_class_candidates,
        key=lambda index: (
            float(
                np.sum(
                    (
                        full_snapshot.z_id[neighbor_query_index].astype(
                            np.float64
                        )
                        - full_snapshot.z_id[index].astype(np.float64)
                    )
                    ** 2
                )
            ),
            tokens[index].encode("utf-8"),
        ),
    )
    assert nearest == held_index

    old_classes = ("c0", "c1", "c2", "c3", "c4")
    new_classes = ("c5",)
    registered = (*old_classes, *new_classes)
    filtered_backend = cfm._ExcludedPhysicalBackend(
        full_backend, tokens[held_index]
    )
    strict_theta, strict_receipt = (
        cfm._fit_kgt1_reduced_support_two_rounds(
            backend=filtered_backend,
            excluded_physical_token=tokens[held_index],
            ground=ground,
            lock=_lock(k_shot),
            registered_classes=registered,
            old_classes=old_classes,
            new_classes=new_classes,
            base_weight=weight,
            k_shot=k_shot,
            ground_equation_enabled=True,
        )
    )
    keep = np.asarray(
        [index for index in range(len(tokens)) if index != held_index],
        dtype=np.intp,
    )
    explicit_backend = cfm.AffineSupportBackend(
        np.ascontiguousarray(rows[keep]),
        np.ascontiguousarray(jacobian[keep]),
        tuple(labels[index] for index in keep),
        tuple(tokens[index] for index in keep),
    )
    reference_theta, reference_receipt = (
        cfm._fit_kgt1_reduced_support_two_rounds(
            backend=explicit_backend,
            excluded_physical_token=tokens[held_index],
            ground=ground,
            lock=_lock(k_shot),
            registered_classes=registered,
            old_classes=old_classes,
            new_classes=new_classes,
            base_weight=weight,
            k_shot=k_shot,
            ground_equation_enabled=True,
        )
    )
    assert strict_theta.tobytes() == reference_theta.tobytes()
    assert np.array_equal(strict_theta, reference_theta)
    assert strict_receipt == reference_receipt
    expected_fold_root = cfm._sha_json(
        sorted(tokens[index] for index in keep)
    )
    assert (
        strict_receipt["fold_support_token_root_sha256"]
        == expected_fold_root
        == strict_receipt["d92_fold_token_root_sha256"]
    )
    assert strict_receipt[
        "outer_held_absent_from_every_bank_ground_oof"
    ] is True
    assert strict_receipt[
        "subtractive_normal_equation_approximation_used"
    ] is False
    assert all(
        round_receipt[
            "outer_held_absent_from_support_ground_and_oof"
        ]
        and round_receipt["normal_equation_rebuilt_from_fold_only"]
        and round_receipt[
            "class_task_weights_recomputed_after_outer_holdout"
        ]
        for round_receipt in strict_receipt["rounds"]
    )

    G_full, bg_full, ground_parts, _ground_tokens = cfm._ground_rows(
        full_snapshot, ground, active_old_classes=old_classes
    )
    tuples_full = [
        cfm._qknn_oof_tuple(
            full_snapshot,
            index,
            lock=_lock(k_shot),
            registered_classes=registered,
        )
        for index in range(len(full_snapshot.z_id))
    ]
    C_full, bc_full, cfm_parts, _A, _r = cfm._cfm_system(
        tuples_full,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    factor = float(k_shot - 1) / float(k_shot)
    legacy_H = (
        G_full
        + factor * C_full
        - ground_parts[held_index][0]
        - factor * cfm_parts[held_index][0]
    )
    legacy_b = (
        bg_full
        + factor * bc_full
        - ground_parts[held_index][1]
        - factor * cfm_parts[held_index][1]
    )
    legacy_increment, _legacy_receipt = cfm._solve_increment(
        legacy_H, legacy_b, round_index=0
    )
    legacy_theta1 = cfm._project_theta(
        (float(k_shot) / float(k_shot + 4)) * legacy_increment,
        ground=ground,
        base_weight=weight,
    )
    explicit_snapshot0 = explicit_backend.snapshot(theta0)
    G_ref, bg_ref, _ground_parts_ref, _ground_tokens_ref = (
        cfm._ground_rows(
            explicit_snapshot0,
            ground,
            active_old_classes=old_classes,
        )
    )
    tuples_ref = [
        cfm._qknn_oof_tuple(
            explicit_snapshot0,
            index,
            lock=_lock(k_shot),
            registered_classes=registered,
        )
        for index in range(len(explicit_snapshot0.z_id))
    ]
    C_ref, bc_ref, _parts_ref, _A_ref, _r_ref = cfm._cfm_system(
        tuples_ref,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    strict_increment1, _strict_solve = cfm._solve_increment(
        G_ref + factor * C_ref,
        bg_ref + factor * bc_ref,
        round_index=0,
    )
    strict_theta1 = cfm._project_theta(
        (float(k_shot) / float(k_shot + 4)) * strict_increment1,
        ground=ground,
        base_weight=weight,
    )
    assert not np.allclose(
        legacy_theta1, strict_theta1, rtol=0.0, atol=1.0e-12
    )


def test_k1_pseudo_new_is_registration_only_and_does_not_change_theta():
    state_a, _ground, _weight = _fit(1, mutate_new=False)
    state_b, _ground, _weight = _fit(1, mutate_new=True)
    assert state_a.theta_codes.tobytes() == state_b.theta_codes.tobytes()
    assert state_a.theta_scale.tobytes() == state_b.theta_scale.tobytes()
    assert (
        state_a.joint_weight_semantic_sha256
        == state_b.joint_weight_semantic_sha256
    )
    assert state_a.fit_receipt["fold_receipt"]["new_support_rows_used"] == 0
    assert state_a.fit_receipt["loco_receipt"]["new_support_rows_used"] == 0
    assert state_a.fit_receipt["registry_receipt"] == {
        "old_count": 5,
        "new_count": 1,
        "old_registry_sha256": cfm._sha_json(
            ["c0", "c1", "c2", "c3", "c4"]
        ),
        "new_registry_sha256": cfm._sha_json(["c5"]),
        "explicit_old_new_partition": True,
        "held_pseudo_new_ground_rows_used": 0,
    }


def test_quantization_uses_rne_fp16_downward_scale_and_preserves_trust():
    ground, weight = _fixture_ground()
    theta = np.asarray([8.1, -7.3, 6.7, -5.9], dtype=np.float64)
    codes, scale, receipt = cfm._quantize_theta(
        theta, ground=ground, base_weight=weight
    )
    projected = cfm._project_theta(
        theta, ground=ground, base_weight=weight
    )
    s0 = float(np.max(np.abs(projected))) / 127.0
    unit_delta = cfm._delta_weight(codes.astype(np.float64), ground)
    r_weight = float(np.linalg.norm(weight.astype(np.float64))) / np.sqrt(160)
    s_trust = r_weight / max(
        float(np.linalg.norm(unit_delta.astype(np.float64))), 2.0**-24
    )
    assert float(scale) <= min(s0, s_trust)
    assert np.array_equal(
        codes,
        np.clip(np.rint(projected / s0), -127, 127).astype(np.int8),
    )
    assert receipt["trust_verified"] is True
    assert receipt["quantized_delta_fro"] <= receipt["trust_radius_fro"]
    assert receipt["rank_used_for_fallback_or_selection"] is False


def test_fit_state_serialization_replay_binding_and_tamper_rejection():
    state, ground, _weight = _fit(1)
    wire = cfm.serialize_cfm_fit_state(state)
    restored = cfm.deserialize_cfm_fit_state(
        wire,
        expected_ground_digest=ground.digest,
        expected_lock_digest=_lock(1).digest,
        expected_checkpoint_sha256=_sha("a"),
        expected_joint_weight_sha256_before=ground.joint_weight_sha256,
    )
    assert restored.receipt_sha256 == state.receipt_sha256
    assert restored.theta_codes.tobytes() == state.theta_codes.tobytes()
    tampered = bytearray(wire)
    tampered[len(tampered) // 2] ^= 1
    with pytest.raises(cfm.GRBJP4CFMError, match="digest"):
        cfm.deserialize_cfm_fit_state(
            bytes(tampered),
            expected_ground_digest=ground.digest,
            expected_lock_digest=_lock(1).digest,
            expected_checkpoint_sha256=_sha("a"),
            expected_joint_weight_sha256_before=ground.joint_weight_sha256,
        )
    resources = state.fit_receipt["resource_receipt"]
    assert resources["update_factor_wire_bytes"] == ground.phase1_resource_receipt[
        "jp4_update_factor_wire_bytes"
    ]
    assert resources["ground_wire_bytes"] == ground.phase1_resource_receipt[
        "ground_wire_bytes"
    ]
    assert resources["total_component_bytes"] == ground.phase1_resource_receipt[
        "total_component_bytes"
    ]
    assert (
        resources["arm_state_base_component_bytes"]
        == resources["total_component_bytes"]
    )


def test_same_quantized_theta_merges_into_two_real_joint_proj_instances():
    state, ground, weight = _fit(1)
    first = torch.nn.Linear(320, 160, bias=False)
    second = torch.nn.Linear(320, 160, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(weight))
        second.weight.copy_(torch.from_numpy(weight))
    receipt_a = cfm.merge_into_joint_proj(
        first, state=state, ground=ground, lock=_lock(1)
    )
    receipt_b = cfm.merge_into_joint_proj(
        second, state=state, ground=ground, lock=_lock(1)
    )
    assert torch.equal(first.weight, second.weight)
    assert (
        receipt_a["theta_code_sha256"]
        == receipt_b["theta_code_sha256"]
        == cfm._array_sha(state.theta_codes)
    )
    assert receipt_a["same_theta_bytes_reusable_by_M_DA_and_M_DA92"] is True


def test_query_api_has_no_fit_inputs_and_state_is_read_only_contract():
    parameters = inspect.signature(cfm.predict_frozen_queries).parameters
    assert "query_labels" not in parameters
    assert "query_truth" not in parameters
    assert "registered_old_classes" not in parameters
    assert "registered_new_classes" not in parameters
    fit_parameters = inspect.signature(cfm.fit_cfm_from_support_iq).parameters
    assert "query_iq" not in fit_parameters
    assert "query_labels" not in fit_parameters


class _MethodNames:
    @staticmethod
    def _method_names():
        return ("grb_jp4_forward",)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, weight: np.ndarray):
        super().__init__()
        self.cls_head = torch.nn.Module()
        linear = torch.nn.Linear(320, 160, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(weight))
        self.cls_head.joint_proj = torch.nn.Sequential(
            linear, torch.nn.ReLU()
        )


class _TinyTapModel(torch.nn.Module):
    def __init__(self, weight: np.ndarray):
        super().__init__()
        self.id_backbone = _TinyBackbone(weight)
        self._c = _MethodNames()

    def grb_jp4_forward(self, iq: torch.Tensor):
        hidden = torch.cat((iq[:, 0], iq[:, 1]), dim=1)
        pre = self.id_backbone.cls_head.joint_proj[0](hidden)
        z_id = torch.relu(pre)
        z_dom = torch.zeros(
            (len(iq), 1), dtype=iq.dtype, device=iq.device
        )
        return z_id, z_dom, hidden, pre


def test_tap_archive_backend_matches_support_iq_and_recomputes_relu_masks():
    ground, weight = _fixture_ground()
    rows, _jacobian, labels, tokens = _support(1)
    hidden = np.zeros((len(rows), 320), dtype=np.float32)
    hidden[:, :160] = rows
    iq = torch.from_numpy(hidden.reshape(len(rows), 2, 160))
    model = _TinyTapModel(weight).eval()
    with torch.no_grad():
        z_id, _zdom, hidden_t, pre_t = model.grb_jp4_forward(iq)
    common = {
        "support_labels": labels,
        "support_physical_tokens": tokens,
        "registered_old_classes": ("c0", "c1", "c2", "c3", "c4"),
        "registered_new_classes": ("c5",),
        "ground": ground,
        "lock": _lock(1),
        "checkpoint_sha256": _sha("a"),
    }
    from_iq = cfm.fit_cfm_from_support_iq(
        model=model, support_iq=iq, **common
    )
    from_taps = cfm.fit_cfm_from_taps(
        base_z_id=z_id.numpy(),
        hidden=hidden_t.numpy(),
        pre_relu=pre_t.numpy(),
        checkpoint_weight=weight,
        **common,
    )
    assert from_taps.theta_codes.tobytes() == from_iq.theta_codes.tobytes()
    assert from_taps.theta_scale.tobytes() == from_iq.theta_scale.tobytes()
    assert from_taps.fit_receipt == from_iq.fit_receipt
    tampered = z_id.numpy().copy()
    tampered[0, 0] += np.float32(1.0)
    with pytest.raises(cfm.GRBJP4CFMError, match="byte-exact"):
        cfm.fit_cfm_from_taps(
            base_z_id=tampered,
            hidden=hidden_t.numpy(),
            pre_relu=pre_t.numpy(),
            checkpoint_weight=weight,
            **common,
        )


def test_query_api_repeated_calls_cannot_update_weight_or_fit_state():
    state, ground, weight = _fit(1)
    model = _TinyTapModel(weight).eval()
    cfm.merge_into_joint_proj(
        model, state=state, ground=ground, lock=_lock(1)
    )
    query_hidden = np.zeros((3, 320), dtype=np.float32)
    query_hidden[0, 20] = 1.0
    query_hidden[1, 21] = 1.0
    query_hidden[2, 20:22] = 0.5
    query_iq = torch.from_numpy(query_hidden.reshape(3, 2, 160))
    state_before = cfm.serialize_cfm_fit_state(state)
    weight_before = cfm._array_sha(
        model.id_backbone.cls_head.joint_proj[0]
        .weight.detach().cpu().numpy()
    )

    def score(z_id):
        return np.ascontiguousarray(z_id[:, 20:22], dtype=np.float32)

    first, receipt_a = cfm.predict_frozen_queries(
        model=model,
        query_iq=query_iq,
        state=state,
        score_function=score,
    )
    second, receipt_b = cfm.predict_frozen_queries(
        model=model,
        query_iq=query_iq,
        state=state,
        score_function=score,
    )
    assert np.array_equal(first, second)
    assert receipt_a["state_updated"] is False
    assert receipt_b["state_updated"] is False
    assert cfm.serialize_cfm_fit_state(state) == state_before
    assert (
        cfm._array_sha(
            model.id_backbone.cls_head.joint_proj[0]
            .weight.detach().cpu().numpy()
        )
        == weight_before
    )


def test_ground_off_is_an_explicit_tap_only_nonformal_falsifier():
    ground, weight = _fixture_ground()
    ground = replace(
        ground,
        delta_q_fp16=np.asarray(np.float16(10.0), dtype="<f2"),
    )
    lock = replace(_lock(5), delta_q=10.0)
    rows, _jacobian, labels, tokens = _support(5)
    hidden = np.zeros((len(rows), 320), dtype=np.float32)
    hidden[:, :160] = rows
    pre = hidden[:, :160].copy()
    z_id = np.maximum(pre, np.float32(0.0))
    state = cfm.fit_cfm_ground_off_falsifier_from_taps(
        base_z_id=z_id,
        hidden=hidden,
        pre_relu=pre,
        support_labels=labels,
        support_physical_tokens=tokens,
        registered_old_classes=("c0", "c1", "c2", "c3", "c4"),
        registered_new_classes=("c5",),
        ground=ground,
        lock=lock,
        checkpoint_weight=weight,
        checkpoint_sha256=_sha("a"),
    )
    assert state.fit_receipt["claim_scope"] == cfm.HELD_FALSIFIER_SCOPE
    assert state.fit_receipt["ground_equation_enabled"] is False
    assert state.fit_receipt["status"] == "ground_off_cfm_solved"
    assert all(
        item["all_oof_tuple_count"] == len(rows)
        for item in state.fit_receipt["rounds"]
    )
    assert (
        "ground_equation_enabled"
        not in inspect.signature(cfm.fit_cfm_from_support_iq).parameters
    )


@pytest.mark.parametrize("k_shot", [5, 10])
def test_ground_off_exact_zero_information_returns_identity(k_shot):
    ground, weight = _fixture_ground()
    rows, _jacobian, labels, tokens = _support(k_shot)
    hidden = np.zeros((len(rows), 320), dtype=np.float32)
    hidden[:, :160] = rows
    pre = hidden[:, :160].copy()
    z_id = np.maximum(pre, np.float32(0.0))
    state = cfm.fit_cfm_ground_off_falsifier_from_taps(
        base_z_id=z_id,
        hidden=hidden,
        pre_relu=pre,
        support_labels=labels,
        support_physical_tokens=tokens,
        registered_old_classes=("c0", "c1", "c2", "c3", "c4"),
        registered_new_classes=("c5",),
        ground=ground,
        lock=_lock(k_shot),
        checkpoint_weight=weight,
        checkpoint_sha256=_sha("a"),
    )
    assert (
        state.fit_receipt["status"]
        == "ground_off_zero_information_identity"
    )
    assert np.array_equal(state.theta_codes, np.zeros(4, dtype=np.int8))
    assert float(state.theta_scale) == 0.0
    assert all(
        item["status"] == "ground_off_zero_information_identity"
        and item["exact_zero_H"] is True
        and item["exact_zero_b"] is True
        for item in state.fit_receipt["rounds"]
    )
    fold_summary = state.fit_receipt["fold_receipt"]
    assert fold_summary["fold_count"] == len(tokens)
    assert fold_summary["all_folds_delete_then_two_round_refit"] is True
    assert fold_summary["d92_statistics_materialized_in_solver"] is False
    token_by_sha = {cfm._sha_json(token): token for token in tokens}
    for fold in fold_summary["folds"]:
        held = token_by_sha[fold["held_token_sha256"]]
        expected_root = cfm._sha_json(
            sorted(token for token in tokens if token != held)
        )
        assert fold[
            "two_active_set_rounds_rebuilt_from_fold_only"
        ] is True
        assert fold[
            "outer_held_absent_from_every_bank_ground_oof"
        ] is True
        assert fold[
            "subtractive_normal_equation_approximation_used"
        ] is False
        assert (
            fold["fold_support_token_root_sha256"]
            == expected_root
            == fold["held_evaluation_bank_token_root_sha256"]
            == fold["d92_fold_token_root_sha256"]
        )


def test_zero_information_exception_never_masks_nonzero_degenerate_system():
    zero_H = np.zeros((4, 4), dtype=np.float64)
    nonzero_b = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(cfm.GRBJP4CFMError, match="trace is degenerate"):
        cfm._solve_increment_with_ground_off_zero_information(
            zero_H,
            nonzero_b,
            round_index=0,
            ground_equation_enabled=False,
        )
    trace_zero_nonzero_H = np.diag([1.0, -1.0, 0.0, 0.0])
    with pytest.raises(cfm.GRBJP4CFMError, match="trace is degenerate"):
        cfm._solve_increment_with_ground_off_zero_information(
            trace_zero_nonzero_H,
            np.zeros(4, dtype=np.float64),
            round_index=0,
            ground_equation_enabled=False,
        )


def test_formal_solver_cannot_reach_ground_off_zero_information_exception():
    with pytest.raises(cfm.GRBJP4CFMError, match="trace is degenerate"):
        cfm._solve_increment_with_ground_off_zero_information(
            np.zeros((4, 4), dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            round_index=0,
            ground_equation_enabled=True,
        )
    assert (
        "ground_equation_enabled"
        not in inspect.signature(cfm.fit_cfm_from_support_iq).parameters
    )
    assert (
        "ground_equation_enabled"
        not in inspect.signature(cfm.fit_cfm_from_taps).parameters
    )


def test_fit_state_in_memory_arrays_are_read_only():
    state, _ground, _weight = _fit(1)
    with pytest.raises(ValueError):
        state.theta_codes[0] = np.int8(1)
    with pytest.raises(ValueError):
        state.theta_scale[...] = np.float16(1.0)


def test_phase1_cross_k_method_lock_selects_exact_active_qknn_digest():
    classes = tuple(f"c{index}" for index in range(6))
    qknn = {k: _qknn_lock(k) for k in (1, 5, 10)}
    mapping = _phase1_lock_mapping(
        classes, {str(k): qknn[k].lock_digest for k in (1, 5, 10)}
    )
    for k in (1, 5, 10):
        typed = cfm.CFMMethodLock.from_mapping(
            mapping, qknn_lock=qknn[k]
        )
        assert typed.qknn_neighbor_count == k
        assert typed.qknn_lock_digest == qknn[k].lock_digest
        assert typed.phase1_method_lock_sha256 == cfm._sha_json(mapping)
    bad = dict(mapping)
    bad["qknn_lock_sha256_by_k"] = dict(
        mapping["qknn_lock_sha256_by_k"]
    )
    bad["qknn_lock_sha256_by_k"]["5"] = qknn[10].lock_digest
    with pytest.raises(cfm.GRBJP4CFMError, match="constants drift"):
        cfm.CFMMethodLock.from_mapping(bad, qknn_lock=qknn[5])


def test_ground_weight_radius_contract_is_consumed_and_digest_bound():
    ground, _weight = _fixture_ground()
    assert ground.ground_multiprototype_numeric_bytes == (
        6 * 3 * 160 + 7 * 6 * 3
    )
    before = ground.digest
    weights = np.array(ground.prototype_weights, copy=True)
    weights[0, 0] = np.float16(0.5)
    with pytest.raises(cfm.GRBJP4CFMError, match="equal 1/M"):
        cfm.GroundCFMInput(
            ground.prototype_codes,
            ground.prototype_scales,
            ground.prototype_mask,
            weights,
            ground.prototype_radii,
            ground.left_codes,
            ground.left_scales,
            ground.right_codes,
            ground.right_scales,
            ground.direction_energy,
            ground.delta_q_fp16,
            ground.tau_q_fp16,
            ground.old_class_order,
            ground.checkpoint_sha256,
            ground.joint_weight_sha256,
            ground.phase1_method_lock_sha256,
            ground.component_digest,
            ground.phase1_resource_receipt,
            ground.phase1_resource_receipt_sha256,
        )
    radii = np.array(ground.prototype_radii, copy=True)
    radii[0, 0] = np.float16(0.25)
    changed = cfm.GroundCFMInput(
        ground.prototype_codes,
        ground.prototype_scales,
        ground.prototype_mask,
        ground.prototype_weights,
        radii,
        ground.left_codes,
        ground.left_scales,
        ground.right_codes,
        ground.right_scales,
        ground.direction_energy,
        ground.delta_q_fp16,
        ground.tau_q_fp16,
        ground.old_class_order,
        ground.checkpoint_sha256,
        ground.joint_weight_sha256,
        ground.phase1_method_lock_sha256,
        ground.component_digest,
        ground.phase1_resource_receipt,
        ground.phase1_resource_receipt_sha256,
    )
    assert changed.digest != before
    bad_resource = dict(ground.phase1_resource_receipt)
    bad_resource["ground_wire_bytes"] += 1
    with pytest.raises(
        cfm.GRBJP4CFMError, match="resource receipt drift"
    ):
        cfm.GroundCFMInput(
            ground.prototype_codes,
            ground.prototype_scales,
            ground.prototype_mask,
            ground.prototype_weights,
            ground.prototype_radii,
            ground.left_codes,
            ground.left_scales,
            ground.right_codes,
            ground.right_scales,
            ground.direction_energy,
            ground.delta_q_fp16,
            ground.tau_q_fp16,
            ground.old_class_order,
            ground.checkpoint_sha256,
            ground.joint_weight_sha256,
            ground.phase1_method_lock_sha256,
            ground.component_digest,
            bad_resource,
            cfm._sha_json(bad_resource),
        )


def test_exact_phase1_component_and_lock_adapt_to_typed_stage2_inputs():
    fixture, weight = _fixture_ground()
    classes = fixture.old_class_order
    qknn = {k: _qknn_lock(k) for k in (1, 5, 10)}
    method = _phase1_lock_mapping(
        classes, {str(k): qknn[k].lock_digest for k in (1, 5, 10)}
    )
    counts = np.where(fixture.prototype_mask, 2, 0).astype(np.int16)
    prototype_receipts = np.full((6, 3), b"", dtype="S64")
    prototype_receipts[fixture.prototype_mask] = _sha("9").encode("ascii")
    source_sha = np.full((6, 3), b"", dtype="S64")
    source_sha[fixture.prototype_mask] = _sha("7").encode("ascii")
    quant_error = np.zeros((6, 3), dtype="<f2")
    quant_error[fixture.prototype_mask] = np.float16(1.0e-3)
    quant_cert = np.full((6, 3), b"", dtype="S64")
    quant_cert[fixture.prototype_mask] = _sha("6").encode("ascii")
    arrays = {
        "p_g_q": np.asarray(fixture.prototype_codes),
        "p_g_scale": np.asarray(fixture.prototype_scales),
        "p_g_mask": np.asarray(fixture.prototype_mask),
        "p_g_weight": np.asarray(fixture.prototype_weights),
        "p_g_radius": np.asarray(fixture.prototype_radii),
        "p_g_physical_counts": counts,
        "p_g_receipt_sha256": prototype_receipts,
        "p_g_source_prototype_sha256": source_sha,
        "p_g_quantization_max_abs_error": quant_error,
        "p_g_quantization_certificate_sha256": quant_cert,
        "l_g_q": np.asarray(fixture.left_codes),
        "l_g_scale": np.asarray(fixture.left_scales),
        "r_q": np.asarray(fixture.right_codes),
        "r_scale": np.asarray(fixture.right_scales),
        "direction_energy_a": np.asarray(fixture.direction_energy),
        "delta_q": np.asarray(np.float16(0.25), dtype="<f2"),
        "tau_q": np.asarray(np.float16(1.0), dtype="<f2"),
        "class_registry": np.asarray(classes),
        "feature_schema": np.asarray(
            "ADV3B02:z_id:unit_l2:160:v1"
        ),
        "protocol_schema": np.asarray(PROTOCOL_SCHEMA),
    }
    manifest = {
        "schema": PHASE1_SCHEMA,
        "component_profile": COMPONENT_PROFILE,
        "method_lock_schema": METHOD_LOCK_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
        "method_lock": method,
        "method_lock_sha256": cfm._sha_json(method),
        "class_handle_binding_sha256": class_handle_binding_sha256(classes),
        "checkpoint_sha256": _sha("a"),
        "pre_sign_content_root_sha256": _sha("8"),
        "array_sha256": {
            name: canonical_array_sha256(value)
            for name, value in arrays.items()
        },
        "resource_audit": phase1_resource_audit(arrays),
        "ground_old_multiprototype_enabled": True,
        "phase2_phase1_component_immutable": True,
        "phase2_phase1_component_update_access": False,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
    }
    component = GRBJP4CFMPhase1Component(
        p_g_q=arrays["p_g_q"],
        p_g_scale=arrays["p_g_scale"],
        p_g_mask=arrays["p_g_mask"],
        p_g_weight=arrays["p_g_weight"],
        p_g_radius=arrays["p_g_radius"],
        p_g_physical_counts=counts,
        p_g_receipt_sha256=prototype_receipts,
        p_g_source_prototype_sha256=source_sha,
        p_g_quantization_max_abs_error=quant_error,
        p_g_quantization_certificate_sha256=quant_cert,
        l_g_q=arrays["l_g_q"],
        l_g_scale=arrays["l_g_scale"],
        r_q=arrays["r_q"],
        r_scale=arrays["r_scale"],
        direction_energy_a=arrays["direction_energy_a"],
        delta_q=0.25,
        tau_q=1.0,
        class_registry=classes,
        method_lock=method,
        manifest=manifest,
    )
    typed_ground = cfm.GroundCFMInput.from_phase1_component(
        component, checkpoint_weight=weight
    )
    typed_lock = cfm.CFMMethodLock.from_mapping(
        method, qknn_lock=qknn[5]
    )
    assert typed_ground.phase1_method_lock_sha256 == typed_lock.phase1_method_lock_sha256
    assert np.array_equal(
        typed_ground.barycenters(), fixture.barycenters()
    )
    tampered_radius = np.array(arrays["p_g_radius"], copy=True)
    tampered_radius[0, 0] = np.float16(0.5)
    tampered = GRBJP4CFMPhase1Component(
        **{
            **component.__dict__,
            "p_g_radius": tampered_radius,
        }
    )
    with pytest.raises(cfm.GRBJP4CFMError, match="lifecycle drift"):
        cfm.GroundCFMInput.from_phase1_component(
            tampered, checkpoint_weight=weight
        )
    tampered_manifest = dict(manifest)
    tampered_resource = dict(manifest["resource_audit"])
    tampered_resource["ground_wire_bytes"] += 1
    tampered_manifest["resource_audit"] = tampered_resource
    tampered_component = GRBJP4CFMPhase1Component(
        **{
            **component.__dict__,
            "manifest": tampered_manifest,
        }
    )
    with pytest.raises(cfm.GRBJP4CFMError, match="lifecycle drift"):
        cfm.GroundCFMInput.from_phase1_component(
            tampered_component, checkpoint_weight=weight
        )
