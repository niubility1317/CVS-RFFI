from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_bundle import build_rxid_metabias4_bundle
from cvsrffi.stage2_d105_cbrc import (
    ACTIVE,
    FALLBACK_LEGAL_DATA,
    D105CBRCError,
    audit_d105_cbrc_resources,
    compute_d105_bundle_receipt_root,
    compute_d105_bundle_validator_receipt,
    compute_d105_support_binding_root,
    fit_d105_cbrc_state,
    make_d105_cbrc_bundle_handle,
    serialize_d105_cbrc_state,
    solve_d105_cbrc_support,
    transform_d105_canonical,
    transform_d105_cbrc,
    validate_d105_physical_split,
)


OLD = ("old-b", "old-a")
NEW = ("new-b", "new-a")
ALL = OLD + NEW
HASHES = tuple(f"{index:x}" * 64 for index in range(1, 9))
VALIDATED_BUNDLE_ID = "9" * 64


def _bundle(*, zero_t: bool = False, nonisometric_b: bool = False):
    rng = np.random.default_rng(105731)
    u = np.zeros((32, 160), dtype=np.float32)
    u[:, :32] = np.eye(32, dtype=np.float32)
    b = rng.normal(0.0, 0.03, (160, 4)).astype(np.float32)
    if nonisometric_b:
        b[:, 0] = np.float32(0.01)
        b[1, 0] = np.float32(5.0)
    g = np.zeros((6, 32), dtype=np.float32)
    for index in range(6):
        g[index, index] = 1.0
        g[index, (index + 7) % 32] = 0.25
    t = np.zeros((6, 4), dtype=np.float32)
    if not zero_t:
        t[:] = np.asarray(
            [
                [0.18, 0.06, -0.05, 0.09],
                [0.15, 0.04, -0.03, 0.08],
                [0.12, 0.05, -0.02, 0.06],
                [0.16, 0.07, -0.04, 0.10],
                [0.10, 0.03, -0.01, 0.05],
                [0.14, 0.06, -0.02, 0.07],
            ],
            dtype=np.float32,
        )
    return build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        np.full((6, 4), 4.0, dtype=np.float32),
        np.full(6, 1.7, dtype=np.float32),
        cell_min_physical_count=np.full(6, 2, dtype=np.int16),
        cell_class_count=np.full(6, 3, dtype=np.int16),
        checkpoint_sha256=HASHES[0],
        runtime_sha256=HASHES[1],
        method_lock_sha256=HASHES[2],
        training_receipt_sha256=HASHES[3],
        nested_receipt_sha256=HASHES[4],
        tx_probe_receipt_sha256=HASHES[5],
        aggregation_receipt_sha256=HASHES[6],
        quantization_receipt_sha256=HASHES[7],
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.24,
    )


def _support(classes: tuple[str, ...], k: int, *, reverse: bool = False):
    rng = np.random.default_rng(60700 + 10 * k + len(classes))
    pre, zdom, labels, physical_ids = [], [], [], []
    for class_index, label in enumerate(classes):
        for shot in range(k):
            row = rng.normal(0.08, 0.025, 160).astype(np.float32)
            row[20 + class_index] += np.float32(0.9)
            domain = rng.normal(0.0, 0.02, 160).astype(np.float32)
            domain[class_index] += np.float32(1.0)
            pre.append(row)
            zdom.append(domain)
            labels.append(label)
            physical_ids.append(f"physical-{label}-{shot}")
    order = list(range(len(labels)))
    if reverse:
        order.reverse()
    return (
        np.asarray([pre[index] for index in order], dtype=np.float32),
        np.asarray([zdom[index] for index in order], dtype=np.float32),
        tuple(labels[index] for index in order),
        tuple(physical_ids[index] for index in order),
    )


def _handle(
    asset,
    *,
    validated_bundle_id: str = VALIDATED_BUNDLE_ID,
    validator_receipt: str | None = None,
):
    receipt_root = compute_d105_bundle_receipt_root(asset)
    if validator_receipt is None:
        validator_receipt = compute_d105_bundle_validator_receipt(
            validated_bundle_id_sha256=validated_bundle_id,
            expected_content_root_sha256=asset.content_root_sha256,
            checkpoint_sha256=asset.checkpoint_sha256,
            runtime_sha256=asset.runtime_sha256,
            method_lock_sha256=asset.method_lock_sha256,
            receipt_root_sha256=receipt_root,
        )
    return make_d105_cbrc_bundle_handle(
        asset,
        validated_bundle_id_sha256=validated_bundle_id,
        validator_receipt_sha256=validator_receipt,
        expected_content_root_sha256=asset.content_root_sha256,
    )


def _fit(
    k: int,
    *,
    stage: str = "S_C",
    bundle=None,
    reverse: bool = False,
    registered_classes: tuple[str, ...] | None = None,
    old_classes: tuple[str, ...] | None = None,
    new_classes: tuple[str, ...] | None = None,
):
    if stage == "S_B":
        classes = OLD
        registered = OLD if registered_classes is None else registered_classes
        old = OLD if old_classes is None else old_classes
        new = () if new_classes is None else new_classes
    else:
        classes = ALL
        registered = ALL if registered_classes is None else registered_classes
        old = OLD if old_classes is None else old_classes
        new = NEW if new_classes is None else new_classes
    asset = _bundle() if bundle is None else bundle
    pre, zdom, labels, physical_ids = _support(classes, k, reverse=reverse)
    support_root = compute_d105_support_binding_root(
        pre,
        zdom,
        labels,
        physical_ids,
        registered,
        old,
        new,
        active_k=k,
        stage=stage,
    )
    state = fit_d105_cbrc_state(
        asset,
        _handle(asset),
        pre,
        zdom,
        labels,
        physical_ids,
        registered,
        old,
        new,
        active_k=k,
        stage=stage,
        support_receipt_sha256=support_root,
    )
    return state, pre, zdom, labels, physical_ids


def _fit_support(
    asset,
    pre: np.ndarray,
    zdom: np.ndarray,
    labels: tuple[str, ...],
    physical_ids: tuple[str, ...],
    registered: tuple[str, ...],
    old: tuple[str, ...],
    new: tuple[str, ...],
    *,
    k: int,
    stage: str = "S_C",
):
    support_root = compute_d105_support_binding_root(
        pre,
        zdom,
        labels,
        physical_ids,
        registered,
        old,
        new,
        active_k=k,
        stage=stage,
    )
    return fit_d105_cbrc_state(
        asset,
        _handle(asset),
        pre,
        zdom,
        labels,
        physical_ids,
        registered,
        old,
        new,
        active_k=k,
        stage=stage,
        support_receipt_sha256=support_root,
    )


def _distance(query: np.ndarray, support: np.ndarray) -> float:
    return float(2.0 * (1.0 - np.dot(query, support)))


def test_k1_uses_unified_prior_and_query_is_read_only() -> None:
    state, pre, _, _, _ = _fit(1)
    assert state.status == ACTIVE
    assert state.fit_audit["k1_unified_lambda0"] is True
    assert state.fit_audit["k1_precision_mode"] == "unified_lambda0"
    assert state.fit_audit["task_weight_sums"] == {"old": 0.5, "new": 0.5}
    assert state.fit_audit["loo_excluded_class_count"] == len(ALL)
    before = serialize_d105_cbrc_state(state)
    int8_before = state.bundle.bank_t_codes_qint8.tobytes(order="C")
    assert not np.any(state.bundle.bank_t_codes_qint8 == np.int8(-128))
    together = transform_d105_cbrc(state, pre[:4])
    chunked = np.concatenate(
        (transform_d105_cbrc(state, pre[:1]), transform_d105_cbrc(state, pre[1:4]))
    )
    repeated = transform_d105_cbrc(state, pre[:4])
    after = serialize_d105_cbrc_state(state)
    np.testing.assert_array_equal(together, chunked)
    np.testing.assert_array_equal(together, repeated)
    assert before == after
    assert int8_before == state.bundle.bank_t_codes_qint8.tobytes(order="C")
    resources = audit_d105_cbrc_resources(state)
    assert resources["query_state_updates"] == 0
    assert resources["fp32_persistent_sidecar_bytes"] == 0
    assert resources["fp16_persistent_learned_sidecar_bytes"] == 0
    assert resources["coefficient_fp16_bytes"] == 8
    assert resources["passes_state_gate"] is True
    assert resources["passes_query_da_mac_gate"] is True
    expected_b_mac = (len(ALL) + 1) * 160 * 4
    assert resources["b_projection_algorithmic_mac"] == expected_b_mac
    assert (
        resources["support_fit_algorithmic_mac"]
        == state.fit_audit["support_encoding_algorithmic_mac"]
        + state.fit_audit["support_statistic_algorithmic_mac"]
        + state.fit_audit["irls_main_algorithmic_mac"]
        + state.fit_audit["irls_loo_algorithmic_mac"]
        + expected_b_mac
    )
    assert "main/current LOO shifts" in state.fit_audit["temporary_fit_accounting"]
    assert "LOO shift" in state.fit_audit["temporary_fit_accounting"]
    assert state.fit_audit["data_information_rank_basis"].startswith(
        "centered_class_coefficient_targets"
    )
    assert state.fit_audit["data_information_rank"] <= min(len(ALL) - 1, 4)
    resources["fit_task_weight_sums_per_iteration"][0]["old"] = -1.0
    assert state.fit_audit["irls_task_weight_sums_per_iteration"][0]["old"] == 0.5
    with pytest.raises(TypeError):
        state.fit_audit["irls_task_weight_sums_per_iteration"][0]["old"] = -1.0


def test_stage_c_task_balance_and_synchronized_role_permutation() -> None:
    first, _, _, _, _ = _fit(5)
    second, _, _, _, _ = _fit(
        5,
        reverse=True,
        registered_classes=tuple(reversed(ALL)),
        old_classes=tuple(reversed(OLD)),
        new_classes=tuple(reversed(NEW)),
    )
    np.testing.assert_array_equal(first.coefficient_fp16, second.coefficient_fp16)
    assert first.fit_audit["task_mass_old"] == 0.5
    assert first.fit_audit["task_mass_new"] == 0.5
    assert first.fit_audit["task_weight_sums"] == {"old": 0.5, "new": 0.5}
    assert first.fit_audit["loo_self_exclusion"] is True
    assert first.fit_audit["ground_old_multiprototype_enabled"] is False


def test_stage_b_weights_and_legal_zero_coefficient_remain_prediction_complete() -> None:
    zero_asset = _bundle(zero_t=True)
    state, pre, _, _, _ = _fit(1, stage="S_B", bundle=zero_asset)
    assert state.status == FALLBACK_LEGAL_DATA
    assert state.fit_audit["legal_low_information_fallback"] is True
    assert state.fit_audit["task_mass_old"] == 1.0
    assert state.fit_audit["task_mass_new"] == 0.0
    assert state.fit_audit["task_weight_sums"] == {"old": 1.0, "new": 0.0}
    expected = transform_d105_canonical(
        zero_asset, state.bundle_handle, np.zeros(4, dtype=np.float16), pre[:2]
    )
    np.testing.assert_array_equal(transform_d105_cbrc(state, pre[:2]), expected)
    assert len(serialize_d105_cbrc_state(state)) > 0


def test_physical_split_and_typed_bundle_handle_tampering_are_rejected() -> None:
    asset = _bundle()
    handle = _handle(asset)
    pre, zdom, labels, physical_ids = _support(ALL, 1)
    with pytest.raises(D105CBRCError, match="overlap"):
        validate_d105_physical_split(physical_ids, (physical_ids[0], "query-other"))
    duplicate = (physical_ids[0],) + physical_ids[1:-1] + (physical_ids[0],)
    with pytest.raises(D105CBRCError, match="unique"):
        solve_d105_cbrc_support(
            asset,
            handle,
            zdom,
            labels,
            duplicate,
            ALL,
            OLD,
            NEW,
            active_k=1,
            stage="S_C",
        )
    with pytest.raises(D105CBRCError, match="validator receipt"):
        replace(handle, expected_content_root_sha256="f" * 64)
    with pytest.raises(D105CBRCError, match="target_rows"):
        replace(handle, target_rows=1)


def test_external_validator_seal_rejects_replacement_and_runtime_resealing() -> None:
    asset = _bundle()
    sealed = _handle(asset)
    replacement = _bundle(zero_t=True)
    pre, zdom, labels, physical_ids = _support(ALL, 1)
    with pytest.raises(D105CBRCError, match="identity drift"):
        solve_d105_cbrc_support(
            replacement,
            sealed,
            zdom,
            labels,
            physical_ids,
            ALL,
            OLD,
            NEW,
            active_k=1,
            stage="S_C",
        )
    with pytest.raises(D105CBRCError, match="expected content root"):
        make_d105_cbrc_bundle_handle(
            replacement,
            validated_bundle_id_sha256=VALIDATED_BUNDLE_ID,
            validator_receipt_sha256=sealed.validator_receipt_sha256,
            expected_content_root_sha256=asset.content_root_sha256,
        )
    with pytest.raises(D105CBRCError, match="validator receipt"):
        make_d105_cbrc_bundle_handle(
            replacement,
            validated_bundle_id_sha256=sealed.validated_bundle_id_sha256,
            validator_receipt_sha256=sealed.validator_receipt_sha256,
            expected_content_root_sha256=replacement.content_root_sha256,
        )
    rebuilt = _handle(
        asset,
        validated_bundle_id="b" * 64,
    )
    assert rebuilt != sealed
    assert rebuilt.expected_content_root_sha256 == sealed.expected_content_root_sha256
    with pytest.raises(TypeError):
        make_d105_cbrc_bundle_handle(asset)


def test_query_revalidates_complete_bundle_payload_and_state_coefficient() -> None:
    payload_state, pre, _, _, _ = _fit(1)
    codes = payload_state.bundle.bank_t_codes_qint8
    codes.setflags(write=True)
    codes[0, 0] = np.int8(0 if codes[0, 0] != np.int8(0) else 1)
    codes.setflags(write=False)
    with pytest.raises(D105CBRCError, match="payload validation"):
        transform_d105_cbrc(payload_state, pre[:1])

    coefficient_state, pre, _, _, _ = _fit(1)
    coefficient = coefficient_state.coefficient_fp16
    coefficient.setflags(write=True)
    coefficient[0] = np.float16(coefficient[0] + np.float16(0.125))
    coefficient.setflags(write=False)
    with pytest.raises(D105CBRCError, match="state receipt"):
        transform_d105_cbrc(coefficient_state, pre[:1])
    with pytest.raises(D105CBRCError, match="state receipt"):
        serialize_d105_cbrc_state(coefficient_state)
    with pytest.raises(D105CBRCError, match="state receipt"):
        audit_d105_cbrc_resources(coefficient_state)


@pytest.mark.parametrize("changed_surface", ("pre_relu", "zdom", "physical_ids"))
def test_support_receipt_binds_both_views_and_physical_ids(
    changed_surface: str,
) -> None:
    asset = _bundle()
    pre, zdom, labels, physical_ids = _support(ALL, 1)
    support_root = compute_d105_support_binding_root(
        pre,
        zdom,
        labels,
        physical_ids,
        ALL,
        OLD,
        NEW,
        active_k=1,
        stage="S_C",
    )
    changed_pre = pre.copy()
    changed_zdom = zdom.copy()
    changed_ids = physical_ids
    if changed_surface == "pre_relu":
        changed_pre[0, 0] += np.float32(0.25)
    elif changed_surface == "zdom":
        changed_zdom[0, 0] += np.float32(0.25)
    else:
        changed_ids = ("replacement-physical-id",) + physical_ids[1:]
    with pytest.raises(D105CBRCError, match="does not bind"):
        fit_d105_cbrc_state(
            asset,
            _handle(asset),
            changed_pre,
            changed_zdom,
            labels,
            changed_ids,
            ALL,
            OLD,
            NEW,
            active_k=1,
            stage="S_C",
            support_receipt_sha256=support_root,
        )


@pytest.mark.parametrize("new_class_count", (5, 10, 20))
def test_new_task_size_keeps_half_mass_in_every_irls_round(
    new_class_count: int,
) -> None:
    old = ("old-00", "old-01")
    new = tuple(f"new-{index:02d}" for index in range(new_class_count))
    registered = old + new
    pre, zdom, labels, physical_ids = _support(registered, 1)
    state = _fit_support(
        _bundle(),
        pre,
        zdom,
        labels,
        physical_ids,
        registered,
        old,
        new,
        k=1,
    )
    history = state.fit_audit["irls_task_weight_sums_per_iteration"]
    assert len(history) == 4
    for iteration in history:
        assert iteration["old"] == pytest.approx(0.5, abs=1.0e-12)
        assert iteration["new"] == pytest.approx(0.5, abs=1.0e-12)


def test_real_label_role_bijection_is_invariant_and_mismatch_fails_closed() -> None:
    asset = _bundle()
    pre, zdom, labels, physical_ids = _support(ALL, 5)
    original = _fit_support(
        asset, pre, zdom, labels, physical_ids, ALL, OLD, NEW, k=5
    )
    rename = {label: f"renamed-{label}" for label in ALL}
    renamed_labels = tuple(rename[label] for label in labels)
    renamed_all = tuple(rename[label] for label in ALL)
    renamed_old = tuple(rename[label] for label in OLD)
    renamed_new = tuple(rename[label] for label in NEW)
    renamed = _fit_support(
        asset,
        pre,
        zdom,
        renamed_labels,
        physical_ids,
        renamed_all,
        renamed_old,
        renamed_new,
        k=5,
    )
    np.testing.assert_array_equal(original.coefficient_fp16, renamed.coefficient_fp16)
    with pytest.raises(D105CBRCError, match="old/new lifecycle classes overlap"):
        compute_d105_support_binding_root(
            pre,
            zdom,
            renamed_labels,
            physical_ids,
            renamed_all,
            renamed_old,
            (renamed_old[0],) + renamed_new[1:],
            active_k=5,
            stage="S_C",
        )


def test_common_transform_has_only_the_frozen_nonlinear_nonisometric_path() -> None:
    asset = _bundle(nonisometric_b=True)
    coefficient = np.asarray([0.1, 0.0, 0.0, 0.0], dtype=np.float16)
    values = np.zeros((3, 160), dtype=np.float32)
    values[0, :3] = np.asarray([0.30, -0.10, 0.10], dtype=np.float32)
    values[1, :3] = np.asarray([0.05, 0.30, 0.10], dtype=np.float32)
    values[2, :3] = np.asarray([0.22, -0.05, 0.10], dtype=np.float32)
    linear_shift = np.asarray([0.4, -0.1, 0.2], dtype=np.float64)
    np.testing.assert_array_equal(
        (values[2, :3].astype(np.float64) + linear_shift)
        - (values[0, :3].astype(np.float64) + linear_shift),
        values[2, :3].astype(np.float64) - values[0, :3].astype(np.float64),
    )
    orthogonal = np.diag(np.asarray([-1.0, 1.0, 1.0], dtype=np.float64))
    orthogonal_delta = values[2, :3].astype(np.float64) - values[0, :3].astype(
        np.float64
    )
    np.testing.assert_allclose(
        np.linalg.norm(orthogonal @ orthogonal_delta),
        np.linalg.norm(orthogonal_delta),
        atol=1.0e-15,
        rtol=0.0,
    )
    handle = _handle(asset)
    base = transform_d105_canonical(
        asset, handle, np.zeros(4, dtype=np.float16), values
    )
    adapted = transform_d105_canonical(asset, handle, coefficient, values)
    decoded_bias = asset.decode_b().astype(np.float64) @ coefficient.astype(np.float64)
    assert np.any((values < 0.0) != ((values.astype(np.float64) + decoded_bias) < 0.0))
    base_margin = _distance(base[2], base[1]) - _distance(base[2], base[0])
    adapted_margin = _distance(adapted[2], adapted[1]) - _distance(adapted[2], adapted[0])
    assert adapted_margin != pytest.approx(base_margin, abs=1.0e-7)


def test_actual_fit_state_query_chain_exercises_nonisometric_relu_path() -> None:
    asset = _bundle(nonisometric_b=True)
    state, _, _, _, _ = _fit(5, bundle=asset)
    assert state.status == ACTIVE
    bias = asset.decode_b().astype(np.float64) @ state.coefficient_fp16.astype(
        np.float64
    )
    crossing_index = int(np.argmax(np.abs(bias)))
    assert abs(float(bias[crossing_index])) > 1.0e-8
    query = np.zeros((2, 160), dtype=np.float32)
    query[:, (crossing_index + 1) % 160] = np.asarray([0.8, 0.4], dtype=np.float32)
    query[0, crossing_index] = np.float32(-0.5 * bias[crossing_index])
    query[1, crossing_index] = np.float32(
        max(0.2, abs(float(bias[crossing_index])) + 0.2)
    )
    before_positive = query[0, crossing_index] > 0.0
    after_positive = query[0, crossing_index] + bias[crossing_index] > 0.0
    assert before_positive != after_positive
    baseline = transform_d105_canonical(
        asset,
        state.bundle_handle,
        np.zeros(4, dtype=np.float16),
        query,
    )
    adapted = transform_d105_cbrc(state, query)
    assert not np.allclose(baseline, adapted, atol=1.0e-7, rtol=0.0)
    assert _distance(baseline[0], baseline[1]) != pytest.approx(
        _distance(adapted[0], adapted[1]), abs=1.0e-7
    )


def test_public_fit_and_query_surfaces_exclude_forbidden_query_roles() -> None:
    forbidden = {
        "query_truth",
        "query_labels",
        "receiver",
        "tx",
        "class_quota",
        "global_assignment",
    }
    assert not set(inspect.signature(fit_d105_cbrc_state).parameters).intersection(forbidden)
    assert not set(inspect.signature(transform_d105_cbrc).parameters).intersection(forbidden)
