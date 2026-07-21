from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from cvsrffi.stage2_zid_student_t_qknn import (
    Z_DIM,
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)
from cvsrffi.stage2_zid_support_nuisance_metric import (
    MAX_C_ID_RANK,
    Phase1ZIDSupportNuisanceLock,
    ZIDSupportNuisanceMetricError,
    fit_zid_support_nuisance_metric,
)


CLASSES = ("cls_a", "cls_b", "cls_c")
SUPPORT_RECEIPT = "4" * 64


def _qknn_lock(k_shot: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
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


def _nuisance_lock(
    k_shot: int,
    *,
    qknn_config: Phase1ZIDStudentTLock | None = None,
    **updates,
) -> Phase1ZIDSupportNuisanceLock:
    config = _qknn_lock(k_shot) if qknn_config is None else qknn_config
    values = dict(
        active_k=k_shot,
        max_rank=2,
        attenuation=0.35,
        between_guard_weight=1.0,
        minimum_nuisance_fraction=0.75,
        minimum_within_energy=1.0e-7,
        qknn_config_lock_digest=config.lock_digest,
        qknn_identity_metric_receipt_sha256=identity_shared_psd_metric(
            config=config
        ).metric_receipt_sha256,
        phase1_nested_lodo_receipt_sha256="3" * 64,
    )
    values.update(updates)
    return Phase1ZIDSupportNuisanceLock(**values)


def _support(k_shot: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    nuisance = np.linspace(-0.24, 0.24, k_shot, dtype=np.float32)
    for class_index, class_name in enumerate(CLASSES):
        for local_index in range(k_shot):
            row = np.zeros(Z_DIM, dtype=np.float32)
            row[class_index] = 1.0
            row[12] = nuisance[local_index]
            row[20 + class_index] = 0.005 * float(local_index - (k_shot - 1) / 2.0)
            rows.append(row)
            labels.append(class_name)
    return np.stack(rows), np.asarray(labels, dtype=str)


def _fit(k_shot: int = 5):
    rows, labels = _support(k_shot)
    return fit_zid_support_nuisance_metric(
        rows,
        labels,
        CLASSES,
        qknn_config=_qknn_lock(k_shot),
        nuisance_lock=_nuisance_lock(k_shot),
        support_receipt_sha256=SUPPORT_RECEIPT,
    )


def _decoded_basis(fit) -> np.ndarray:
    return fit.metric.basis_codes_qint8.astype(np.float32) * fit.metric.basis_scales_fp16.astype(
        np.float32
    )[:, None]


def test_k1_is_exact_patch_a_identity_and_uses_no_target_update() -> None:
    fit = _fit(k_shot=1)
    expected = identity_shared_psd_metric(config=_qknn_lock(1))

    assert fit.metric.exact_identity
    assert fit.metric.metric_receipt_sha256 == expected.metric_receipt_sha256
    assert fit.audit.effective_rank == 0
    assert fit.audit.fallback_reason == "k1_unidentifiable_identity"
    assert fit.audit.query_rows_used_for_fit == 0
    assert fit.audit.target_optimizer_steps == 0
    assert fit.audit.classifier_formula_unchanged


def test_k5_finds_class_shared_nuisance_and_keeps_rank_at_most_two() -> None:
    fit = _fit(k_shot=5)
    basis = _decoded_basis(fit)

    assert 1 <= fit.metric.effective_rank <= MAX_C_ID_RANK
    assert fit.metric.source == "c_id_support_nuisance_v1"
    assert fit.metric.class_shared
    assert fit.metric.provenance.fit_scope == "target_support_only"
    assert fit.metric.provenance.query_rows_used_for_fit == 0
    assert fit.metric.minimum_eigenvalue > 0.0
    assert np.max(np.abs(basis[:, 12])) > 0.98
    assert all(value >= 0.75 for value in fit.audit.selected_nuisance_fraction)


def test_support_row_registry_and_label_renaming_are_geometry_invariant() -> None:
    rows, labels = _support(k_shot=5)
    baseline = _fit(k_shot=5)
    order = np.asarray([7, 0, 14, 3, 10, 1, 12, 5, 8, 2, 11, 4, 13, 6, 9])

    reordered = fit_zid_support_nuisance_metric(
        rows[order],
        labels[order],
        ("cls_c", "cls_a", "cls_b"),
        qknn_config=_qknn_lock(5),
        nuisance_lock=_nuisance_lock(5),
        support_receipt_sha256=SUPPORT_RECEIPT,
    )
    renamed = {"cls_a": "z", "cls_b": "x", "cls_c": "y"}
    renamed_labels = np.asarray([renamed[value] for value in labels], dtype=str)
    relabeled = fit_zid_support_nuisance_metric(
        rows,
        renamed_labels,
        ("x", "y", "z"),
        qknn_config=_qknn_lock(5),
        nuisance_lock=_nuisance_lock(5),
        support_receipt_sha256=SUPPORT_RECEIPT,
    )

    assert reordered.metric.metric_receipt_sha256 == baseline.metric.metric_receipt_sha256
    assert relabeled.metric.metric_receipt_sha256 == baseline.metric.metric_receipt_sha256
    assert reordered.audit.selected_nuisance_fraction == baseline.audit.selected_nuisance_fraction
    assert relabeled.audit.selected_nuisance_fraction == baseline.audit.selected_nuisance_fraction


def test_c_id_changes_only_patch_a_metric_not_bank_or_classifier_formula() -> None:
    support, labels = _support(k_shot=5)
    config = _qknn_lock(5)
    bank = build_typed_zid_support_bank(support, labels, CLASSES, config=config)
    fit = fit_zid_support_nuisance_metric(
        support,
        labels,
        CLASSES,
        qknn_config=config,
        nuisance_lock=_nuisance_lock(5),
        support_receipt_sha256=SUPPORT_RECEIPT,
    )
    bank_receipt_before = bank.bank_receipt_sha256
    query = support[[0, 5, 10]].copy()
    query[:, 12] += np.asarray([0.18, -0.16, 0.12], dtype=np.float32)

    baseline = score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=config)
    )
    adapted = score_zid_student_t_logits(bank, query, metric=fit.metric)

    assert bank.bank_receipt_sha256 == bank_receipt_before
    assert fit.metric.config_lock_digest == bank.config_lock_digest
    assert fit.audit.classifier_formula_unchanged
    assert not np.allclose(adapted, baseline, rtol=0.0, atol=1.0e-7)


def test_no_guarded_direction_falls_back_to_exact_identity() -> None:
    rows = []
    labels = []
    for class_index, class_name in enumerate(CLASSES):
        row = np.zeros(Z_DIM, dtype=np.float32)
        row[class_index] = 1.0
        rows.extend([row.copy() for _ in range(5)])
        labels.extend([class_name] * 5)
    config = _qknn_lock(5)
    fit = fit_zid_support_nuisance_metric(
        np.stack(rows),
        labels,
        CLASSES,
        qknn_config=config,
        nuisance_lock=_nuisance_lock(5),
        support_receipt_sha256=SUPPORT_RECEIPT,
    )

    assert fit.metric.metric_receipt_sha256 == identity_shared_psd_metric(
        config=config
    ).metric_receipt_sha256
    assert fit.audit.fallback_reason == "no_guarded_nuisance_direction"


def test_fit_surface_has_no_query_role_receiver_scenario_or_ground_input() -> None:
    names = set(inspect.signature(fit_zid_support_nuisance_metric).parameters)
    forbidden = {"query", "query_zid", "role", "receiver", "scenario", "ground", "source"}
    assert not names.intersection(forbidden)


def test_unbalanced_or_unknown_support_and_lock_drift_fail_closed() -> None:
    rows, labels = _support(k_shot=5)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="exactly active_k"):
        fit_zid_support_nuisance_metric(
            rows[:-1],
            labels[:-1],
            CLASSES,
            qknn_config=_qknn_lock(5),
            nuisance_lock=_nuisance_lock(5),
            support_receipt_sha256=SUPPORT_RECEIPT,
        )
    altered = labels.copy()
    altered[0] = "unknown"
    with pytest.raises(ZIDSupportNuisanceMetricError, match="registered class set"):
        fit_zid_support_nuisance_metric(
            rows,
            altered,
            CLASSES,
            qknn_config=_qknn_lock(5),
            nuisance_lock=_nuisance_lock(5),
            support_receipt_sha256=SUPPORT_RECEIPT,
        )
    with pytest.raises(ZIDSupportNuisanceMetricError, match="active K drift"):
        fit_zid_support_nuisance_metric(
            rows,
            labels,
            CLASSES,
            qknn_config=_qknn_lock(5),
            nuisance_lock=_nuisance_lock(10),
            support_receipt_sha256=SUPPORT_RECEIPT,
        )
    drifted_config = replace(_qknn_lock(5), temperature=0.9)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="bound to this Patch A config"):
        fit_zid_support_nuisance_metric(
            rows,
            labels,
            CLASSES,
            qknn_config=drifted_config,
            nuisance_lock=_nuisance_lock(5),
            support_receipt_sha256=SUPPORT_RECEIPT,
        )
    with pytest.raises(ZIDSupportNuisanceMetricError, match="identity metric"):
        fit_zid_support_nuisance_metric(
            rows,
            labels,
            CLASSES,
            qknn_config=_qknn_lock(5),
            nuisance_lock=replace(
                _nuisance_lock(5),
                qknn_identity_metric_receipt_sha256="5" * 64,
            ),
            support_receipt_sha256=SUPPORT_RECEIPT,
        )


def test_phase1_lock_rejects_rank_or_attenuation_expansion() -> None:
    with pytest.raises(ZIDSupportNuisanceMetricError, match="active K/rank"):
        replace(_nuisance_lock(5), max_rank=3)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="attenuation"):
        replace(_nuisance_lock(5), attenuation=0.9)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="exact float"):
        replace(_nuisance_lock(5), attenuation="0.35")
    with pytest.raises(ZIDSupportNuisanceMetricError, match="exact float"):
        replace(_nuisance_lock(5), between_guard_weight=True)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="exact string SHA256"):
        replace(
            _nuisance_lock(5),
            phase1_nested_lodo_receipt_sha256=int("3" * 64),
        )


def test_support_and_audit_sha_fields_reject_non_strings() -> None:
    rows, labels = _support(k_shot=5)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="exact string SHA256"):
        fit_zid_support_nuisance_metric(
            rows,
            labels,
            CLASSES,
            qknn_config=_qknn_lock(5),
            nuisance_lock=_nuisance_lock(5),
            support_receipt_sha256=int("4" * 64),
        )
    fit = _fit(k_shot=5)
    with pytest.raises(ZIDSupportNuisanceMetricError, match="exact string SHA256"):
        replace(fit.audit, support_receipt_sha256=int("4" * 64))
