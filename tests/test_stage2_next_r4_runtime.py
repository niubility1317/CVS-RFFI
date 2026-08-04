from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from cvsrffi import stage2_next_r4_fa_rdce3 as fa
from cvsrffi import stage2_next_r4_artifact as artifact
from cvsrffi import stage2_next_r4_matrix as matrix
from cvsrffi import stage2_next_r4_runtime as runtime
from cvsrffi import stage2_zid_student_t_qknn as qknn


CLASSES = tuple(f"tx-{index}" for index in range(6))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _root(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _unit(primary: int, variant: int = 0) -> np.ndarray:
    value = np.zeros(runtime.Z_DIM, dtype=np.float32)
    value[20 + primary] = np.float32(0.94)
    # A deterministic class-symmetric coordinate keeps the held/unregistered
    # query from creating an accidental exact REG0 qKNN tie in this fixture.
    value[0] = np.float32(0.30 + 0.008 * primary + 0.005 * variant)
    if variant:
        value[80 + (primary + variant) % 40] = np.float32(0.01 * variant)
    value /= np.float32(np.linalg.norm(value.astype(np.float64)))
    return value


def _lock(active_k: int) -> qknn.Phase1ZIDStudentTLock:
    return qknn.Phase1ZIDStudentTLock(
        active_k=active_k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _rows() -> tuple[matrix.NextR4ProxyRow, matrix.NextR4ProxyRow, matrix.NextR4ProxyRow]:
    plan = matrix.build_next_r4_proxy24_plan(CLASSES)
    parsed = tuple(matrix.outer_key_from_mapping(item) for item in plan["rows"])
    k1 = next(item for item in parsed if item.held_receiver == "1-1" and item.held_class == "tx-0" and item.active_k == 1)
    k5 = next(item for item in parsed if item.held_receiver == "1-1" and item.held_class == "tx-0" and item.active_k == 5)
    return k1, k5, k1


def _physical_binding(k1: matrix.NextR4ProxyRow, k5: matrix.NextR4ProxyRow):
    support5 = {class_id: tuple(f"support-{class_id}-{index}" for index in range(5)) for class_id in CLASSES}
    support1 = {class_id: values[:1] for class_id, values in support5.items()}
    query = {class_id: tuple(f"query-{class_id}-{index}" for index in range(2)) for class_id in CLASSES}
    observation = {class_id: tuple(f"observation-{class_id}-{index}" for index in range(2)) for class_id in CLASSES}
    view_ids = {"K1": query, "K5": query, **{state: query for state in matrix.STATE_IDS}}
    view_observation = {"K1": observation, "K5": observation, **{state: observation for state in matrix.STATE_IDS}}
    binding = matrix.bind_next_r4_physical_ids(
        row_k1=k1,
        row_k5=k5,
        phase1_fit_ids=("phase1-a", "phase1-b"),
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
        query_observation_ids_by_class=observation,
        query_ids_by_view=view_ids,
        query_observation_ids_by_view=view_observation,
    )
    return binding, support1, support5, query, observation


def _asset(old_classes: tuple[str, ...]) -> fa.FARDCE3Phase1Asset:
    basis = np.zeros((fa.RANK, fa.Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    basis[2, 2] = 1.0
    return fa.build_fa_rdce3_phase1_asset(
        old_classes=old_classes,
        aggregate_samples_per_class=tuple(3 for _ in old_classes),
        class_centers_3d=np.zeros((len(old_classes), fa.RANK), dtype=np.float32),
        fisher_precision_3d=np.asarray([0.5, 0.6, 0.7], dtype=np.float32),
        residual_variance_3d=np.asarray([0.8, 0.9, 1.0], dtype=np.float32),
        fisher_radius=np.asarray([0.75], dtype=np.float32),
        rdce_kappa_3d=np.asarray([0.25, 0.10, 0.05], dtype=np.float32),
        basis_3x160=basis,
        checkpoint_sha256=_sha("checkpoint"),
        phase1_bundle_sha256=_sha("bundle"),
        phase1_aggregate_receipt_sha256=_sha("aggregate"),
        method_lock_sha256=_sha("method-lock"),
    )


def _case(active_k: int):
    k1, k5, _ = _rows()
    row = k1 if active_k == 1 else k5
    binding, support1, support5, _, _ = _physical_binding(k1, k5)
    physical = support1 if active_k == 1 else support5
    support: dict[str, np.ndarray] = {}
    for index, class_id in enumerate(CLASSES):
        support[class_id] = np.stack([_unit(index, variant + 1) for variant in range(active_k)]).astype(np.float32)
    query = np.concatenate(
        [np.stack([_unit(index, 0), _unit(index, 6)]) for index in range(len(CLASSES))], axis=0
    ).astype(np.float32)
    query_ids = tuple(binding["query_physical_ids"])
    observation_ids = tuple(binding["query_observation_ids"])
    reg0 = runtime.NextR4RegistrationInput(
        registration_id="REG0",
        registered_classes=row.retained_classes,
        support_r0_by_class={item: support[item] for item in row.retained_classes},
        support_physical_ids_by_class={item: physical[item] for item in row.retained_classes},
        query_r0_zid160=query,
        query_physical_ids=query_ids,
        query_observation_ids=observation_ids,
    )
    reg1 = runtime.NextR4RegistrationInput(
        registration_id="REG1",
        registered_classes=row.all_registered_classes,
        support_r0_by_class=support,
        support_physical_ids_by_class=physical,
        query_r0_zid160=query.copy(),
        query_physical_ids=query_ids,
        query_observation_ids=observation_ids,
    )
    asset = _asset(row.retained_classes)
    fa_binding = fa.FARDCE3RuntimeBinding(
        checkpoint_sha256=asset.checkpoint_sha256,
        capsule_id=_sha("capsule"),
        split_id=_sha("split"),
        row_id=row.row_id,
        seed=7,
        active_k=active_k,
        old_classes=row.retained_classes,
        support_physical_root_sha256=_root(reg0.support_physical_ids),
        support_authority_sha256=_sha("support-authority"),
    )
    return row, binding, asset, fa_binding, reg0, reg1, _lock(active_k), support, query


def _run(active_k: int):
    row, binding, asset, fa_binding, reg0, reg1, lock, support, query = _case(active_k)
    return runtime.execute_next_r4_logical_row(
        row=row,
        binding_receipt=binding,
        fa_asset=asset,
        fa_binding=fa_binding,
        reg0=reg0,
        reg1=reg1,
        qknn_lock=lock,
    ), (row, binding, asset, fa_binding, reg0, reg1, lock, support, query)


def test_k1_four_state_truth_free_closure_uses_direct_qknn_and_aliases_h(monkeypatch: pytest.MonkeyPatch) -> None:
    _, values = _run(1)
    row, binding, asset, fa_binding, reg0, reg1, lock, support, query = values
    r0_before = query.tobytes()
    support_before = {key: value.tobytes() for key, value in support.items()}

    def _public_normalize_forbidden(*args, **kwargs):
        raise AssertionError("runtime must not invoke public qKNN normalization")

    monkeypatch.setattr(qknn, "normalize_zid_rows", _public_normalize_forbidden)
    result = runtime.execute_next_r4_logical_row(
        row=row, binding_receipt=binding, fa_asset=asset, fa_binding=fa_binding,
        reg0=reg0, reg1=reg1, qknn_lock=lock,
    )
    assert set(result) == {
        "row_id", "held_receiver", "held_class", "active_k", "binding_receipt",
        "fa_state_reuse_receipt", "registrations", "resource_receipt", "query_isolation_receipt",
    }
    assert result["fa_state_reuse_receipt"]["source_sha256"] == result["fa_state_reuse_receipt"]["target_sha256"]
    assert result["fa_state_reuse_receipt"]["same_state_sha256"] is True
    assert query.tobytes() == r0_before
    assert {key: value.tobytes() for key, value in support.items()} == support_before
    for registration, states in (("REG0", ("DA0_REG0", "DA1_REG0")), ("REG1", ("DA0_REG1", "DA1_REG1"))):
        payload = result["registrations"][registration]
        assert set(payload["states"]) == set(states)
        for state_id in states:
            state = payload["states"][state_id]
            assert state["state_id"] == state_id
            assert state["state_name_zh"] == matrix.STATE_NAMES_ZH[state_id]
            assert state["query_physical_ids"] == tuple(reg0.query_physical_ids)
            assert state["query_observation_ids"] == tuple(reg0.query_observation_ids)
            assert state["arms"]["H"]["predictions"] == state["arms"]["Q"]["predictions"]
            assert state["arms"]["H"]["receipt"]["exact_qknn_alias"] is True
    assert result["resource_receipt"]["metric_availability_by_state"]["DA0_REG0"]["H_old_new"] == "N/A"
    isolation = result["query_isolation_receipt"]
    assert all(isolation[field] == 0 for field in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count", "global_reassignment_calls"))
    assert isolation["post_representation_l2_normalization_applied"] is False


def test_k5_cer_and_direct_cosine_path_are_scale_invariant() -> None:
    result, values = _run(5)
    row, _, _, _, reg0, _, lock, _, _ = values
    for state_id in matrix.STATE_IDS:
        registration = "REG0" if state_id.endswith("REG0") else "REG1"
        h_receipt = result["registrations"][registration]["states"][state_id]["arms"]["H"]["receipt"]
        assert h_receipt["head_status"] in {"FUNCTIONAL", "NO_HEAD_FUNCTION"}
        if h_receipt["head_status"] == "FUNCTIONAL":
            assert h_receipt["exact_qknn_alias"] is False
            assert h_receipt["unique_prediction"] is True
    support = reg0.support_r0_zid160
    bank = runtime._build_direct_qknn_bank(
        support=support, labels=reg0.support_labels, classes=reg0.registered_classes, lock=lock
    )
    metric = qknn.identity_shared_psd_metric(config=lock)
    logits = runtime._score_direct_qknn(bank=bank, query=reg0.query_r0_zid160, metric=metric)
    scaled_bank = runtime._build_direct_qknn_bank(
        support=(support * np.float32(3.0)).astype(np.float32), labels=reg0.support_labels,
        classes=reg0.registered_classes, lock=lock,
    )
    scaled_logits = runtime._score_direct_qknn(
        bank=scaled_bank, query=(reg0.query_r0_zid160 * np.float32(7.0)).astype(np.float32), metric=metric
    )
    np.testing.assert_allclose(logits, scaled_logits, rtol=0.0, atol=3.0e-5)
    q_receipt = result["resource_receipt"]["states"]["DA1_REG1"]["qknn"]
    assert q_receipt["explicit_precision_cosine_denominator"] is True
    assert q_receipt["k5_class_scale_explicit_cosine_denominator"] is True
    assert q_receipt["post_representation_l2_normalization_applied"] is False


def test_state_reuse_is_object_level_and_r1_has_no_post_transform() -> None:
    row, binding, asset, fa_binding, reg0, reg1, lock, _, _ = _case(1)
    state = fa.fit_fa_rdce3_reg0(asset, dict(reg0.support_r0_by_class), binding=fa_binding)
    reused = fa.reuse_fa_rdce3_state_for_reg1(state, registered_classes=reg1.registered_classes)
    assert reused is state
    direct_r1 = fa.transform_fa_rdce3_r1(state, reg0.query_r0_zid160)
    result = runtime.execute_next_r4_logical_row(
        row=row, binding_receipt=binding, fa_asset=asset, fa_binding=fa_binding,
        reg0=reg0, reg1=reg1, qknn_lock=lock,
    )
    resource = result["resource_receipt"]
    assert resource["fa_rdce3_reg1_reuse_core_receipt"]["same_state_object"] is True
    assert resource["fa_rdce3_reg1_reuse_core_receipt"]["bitwise_state_reuse"] is True
    for state_id in ("DA1_REG0", "DA1_REG1"):
        state_resource = resource["states"][state_id]
        assert state_resource["post_representation_relu_applied"] is False
        assert state_resource["post_representation_l2_normalization_applied"] is False
        assert state_resource["post_representation_translation_applied"] is False
    assert np.allclose(np.linalg.norm(direct_r1, axis=1), 1.0, rtol=0.0, atol=2.0e-6)


def test_invalid_unit_id_lock_and_tie_paths_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    row, binding, asset, fa_binding, reg0, reg1, lock, support, query = _case(1)
    bad_rows = support.copy()
    bad_rows[row.retained_classes[0]] = bad_rows[row.retained_classes[0]].copy()
    bad_rows[row.retained_classes[0]][0, 0] *= np.float32(0.2)
    with pytest.raises(runtime.NextR4RuntimeError, match="unit R0"):
        runtime.NextR4RegistrationInput(
            registration_id="REG0", registered_classes=row.retained_classes,
            support_r0_by_class={key: bad_rows[key] for key in row.retained_classes},
            support_physical_ids_by_class={key: reg0.support_physical_ids_by_class[key] for key in row.retained_classes},
            query_r0_zid160=query, query_physical_ids=reg0.query_physical_ids,
            query_observation_ids=reg0.query_observation_ids,
        )
    bad_binding = dict(binding)
    bad_binding["k1_row_id"] = "r4-drift"
    with pytest.raises(runtime.NextR4RuntimeError, match="binding"):
        runtime.execute_next_r4_logical_row(
            row=row, binding_receipt=bad_binding, fa_asset=asset, fa_binding=fa_binding,
            reg0=reg0, reg1=reg1, qknn_lock=lock,
        )
    with pytest.raises(runtime.NextR4RuntimeError, match="K-class"):
        runtime.execute_next_r4_logical_row(
            row=row, binding_receipt=binding, fa_asset=asset, fa_binding=fa_binding,
            reg0=reg0, reg1=reg1, qknn_lock=_lock(5),
        )
    original = runtime._score_direct_qknn
    def _tied(*, bank, query, metric):
        return np.zeros((len(query), len(bank.classes)), dtype=np.float32)
    monkeypatch.setattr(runtime, "_score_direct_qknn", _tied)
    with pytest.raises(runtime.NextR4RuntimeError, match="TIE_UNRESOLVED"):
        runtime.execute_next_r4_logical_row(
            row=row, binding_receipt=binding, fa_asset=asset, fa_binding=fa_binding,
            reg0=reg0, reg1=reg1, qknn_lock=lock,
        )
    monkeypatch.setattr(runtime, "_score_direct_qknn", original)


def test_public_runtime_signature_has_no_truth_role_or_quota_input() -> None:
    parameters = set(inspect.signature(runtime.execute_next_r4_logical_row).parameters)
    forbidden = {
        "query_truth", "query_labels", "query_role", "query_roles", "class_quota",
        "true_batch_class_count", "global_reassignment", "scorer_output", "optimizer",
    }
    assert forbidden.isdisjoint(parameters)


def test_runtime_rejects_old_class_grouped_binding() -> None:
    row, binding, asset, fa_binding, reg0, reg1, lock, _, _ = _case(1)
    old = dict(binding)
    old["schema"] = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.row_binding.v1"
    old["query_ids_by_class"] = {"leaked-class": list(reg0.query_physical_ids)}
    old["query_observation_ids_by_class"] = {
        "leaked-class": list(reg0.query_observation_ids)
    }
    old["query_count_by_class"] = {"leaked-class": len(reg0.query_physical_ids)}
    old.pop("binding_sha256")
    old["binding_sha256"] = matrix.canonical_sha256(old)
    with pytest.raises(runtime.NextR4RuntimeError, match="binding"):
        runtime.execute_next_r4_logical_row(
            row=row, binding_receipt=old, fa_asset=asset, fa_binding=fa_binding,
            reg0=reg0, reg1=reg1, qknn_lock=lock,
        )


def test_runtime_mapping_matches_the_strict_artifact_row_contract() -> None:
    result, _ = _run(1)
    plan = matrix.build_next_r4_proxy24_plan(CLASSES)
    artifact_row = artifact._validate_row(result, plan["rows"][0])
    assert artifact_row["row_id"] == result["row_id"]
    assert set(artifact_row["registrations"]) == {"REG0", "REG1"}
