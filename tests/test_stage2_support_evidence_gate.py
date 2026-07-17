from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from cvsrffi.stage2_support_evidence_gate import (
    RIDGE_LAMBDA_GRID,
    SupportEvidenceGateConfig,
    SupportEvidenceGateError,
    apply_support_evidence_gate,
    extract_e5_per_row,
    fit_support_evidence_gate,
    predict_with_support_evidence_gate,
)


def _support_scores(
    k_shot: int,
    *,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int]:
    rng = np.random.default_rng(seed)
    old_classes = ("old_a", "old_b", "old_c")
    new_classes = ("new_d", "new_e", "new_f")
    classes = old_classes + new_classes
    rows: list[np.ndarray] = []
    labels: list[str] = []
    ranks: list[int] = []
    for class_index, class_name in enumerate(classes):
        for rank in range(k_shot):
            row = np.full(len(classes), 1.0, dtype=np.float32)
            if class_index < len(old_classes):
                row[: len(old_classes)] = np.float32(2.0)
                row[len(old_classes) :] = np.float32(2.0)
                row[class_index] = np.float32(4.5)
                row[len(old_classes) + class_index] = np.float32(4.35)
            else:
                row[: len(old_classes)] = np.float32(2.8)
                row[len(old_classes) :] = np.float32(2.0)
                row[class_index] = np.float32(4.4)
                row[class_index - len(old_classes)] = np.float32(4.5)
            row += np.float32(0.01) * rng.normal(size=len(classes)).astype(np.float32)
            rows.append(row)
            labels.append(class_name)
            ranks.append(rank)
    return (
        np.stack(rows).astype(np.float32),
        np.asarray(labels),
        np.asarray(ranks, dtype=np.int64),
        classes,
        len(old_classes),
    )


def test_e5_exact_formula_is_row_local() -> None:
    scores = np.asarray(
        (
            (4.0, 3.0, 1.0, 5.0, 2.0, 0.0),
            (1.0, 2.0, 6.0, 4.0, 3.0, 2.0),
        ),
        dtype=np.float32,
    )
    evidence = extract_e5_per_row(scores, 3)
    expected_first = np.asarray(
        (
            1.0,
            3.0,
            1.0,
            np.mean((5.0, 2.0, 0.0)) - np.mean((4.0, 3.0, 1.0)),
            (5.0 - np.mean((5.0, 2.0, 0.0)))
            - (4.0 - np.mean((4.0, 3.0, 1.0))),
        ),
        dtype=np.float32,
    )

    assert evidence.shape == (2, 5)
    assert evidence.dtype == np.float32
    assert evidence[0] == pytest.approx(expected_first)
    assert np.array_equal(
        evidence[1], extract_e5_per_row(scores[1:2], 3)[0]
    )


def test_k5_fit_uses_preregistered_oof_and_preserves_score_prefixes() -> None:
    support_scores, labels, ranks, classes, old_count = _support_scores(5, seed=2)
    state = fit_support_evidence_gate(
        support_scores, labels, ranks, classes, old_count
    )
    audit = json.loads(state.audit_json)
    query_like_scores = np.asarray(
        (
            (4.0, 2.0, 1.0, 3.1, 2.2, 0.5),
            (2.5, 2.0, 1.5, 4.0, 3.0, 2.0),
        ),
        dtype=np.float32,
    )
    adjusted = apply_support_evidence_gate(state, query_like_scores)

    assert state.enabled is True
    assert state.selected_lambda in RIDGE_LAMBDA_GRID
    assert state.coefficients.shape == (6,)
    assert state.coefficients.dtype == np.float32
    assert not state.coefficients.flags.writeable
    assert state.feature_mean.shape == (5,)
    assert state.feature_std.shape == (5,)
    assert np.all(state.feature_std >= 1.0e-4)
    assert not state.feature_mean.flags.writeable
    assert not state.feature_std.flags.writeable
    assert audit["ridge_lambda_grid"] == list(RIDGE_LAMBDA_GRID)
    assert len(audit["lambda_evidence"]) == 3
    assert all(
        len(candidate["fold_evidence"]) == 5
        for candidate in audit["lambda_evidence"]
    )
    assert audit["selected_oof_gate_safety_pass"] is True
    selected_gated = audit["selected_gated_identity_metrics"]
    raw_identity = audit["raw_identity_metrics"]
    assert selected_gated["old_overall_accuracy"] >= raw_identity[
        "old_overall_accuracy"
    ]
    assert selected_gated["new_overall_accuracy"] >= raw_identity[
        "new_overall_accuracy"
    ]
    assert selected_gated["new_class_floor_accuracy"] >= raw_identity[
        "new_class_floor_accuracy"
    ]
    for class_name, raw_accuracy in raw_identity["per_old_class_accuracy"].items():
        assert selected_gated["per_old_class_accuracy"][class_name] >= raw_accuracy
    diagnostics = audit["full_fit_ridge_diagnostics"]
    assert diagnostics["condition_number"] <= 1.0e6
    assert diagnostics["coefficient_l2_norm"] <= 8.0
    assert diagnostics["minimum_feature_std"] >= 1.0e-4
    assert adjusted.dtype == np.float32
    assert not adjusted.flags.writeable
    assert adjusted[:, :old_count].tobytes() == query_like_scores[:, :old_count].tobytes()
    before_new_differences = (
        query_like_scores[:, old_count:, None]
        - query_like_scores[:, None, old_count:]
    )
    after_new_differences = (
        adjusted[:, old_count:, None] - adjusted[:, None, old_count:]
    )
    assert after_new_differences == pytest.approx(before_new_differences)
    correction = adjusted[:, old_count] - query_like_scores[:, old_count]
    assert np.max(np.abs(correction)) <= state.config.delta + 1.0e-6
    predicted = predict_with_support_evidence_gate(state, query_like_scores)
    assert predicted.tolist() == np.asarray(classes)[np.argmax(adjusted, axis=1)].tolist()


def test_k1_is_exact_disabled_d27_passthrough_without_oof() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(1, seed=3)
    state = fit_support_evidence_gate(scores, labels, ranks, classes, old_count)
    audit = json.loads(state.audit_json)
    adjusted = apply_support_evidence_gate(state, scores)
    resource = state.resource_audit()

    assert state.enabled is False
    assert state.selected_lambda is None
    assert state.coefficients.shape == (0,)
    assert audit["selection_policy"] == "k1_disabled_exact_d27_passthrough"
    assert adjusted.tobytes() == scores.tobytes()
    assert resource["fitted_parameter_count"] == 0
    assert resource["closed_form_solve_count"] == 0
    assert resource["estimated_gate_macs_per_query"] == 0
    assert resource["new_score_additions_per_query"] == 0


def test_inference_is_exactly_row_independent() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(5, seed=4)
    state = fit_support_evidence_gate(scores, labels, ranks, classes, old_count)
    rows = np.asarray(
        (
            (5.0, 1.0, 0.5, 4.0, 3.0, 2.0),
            (1.0, 3.0, 2.0, 4.0, 4.5, 2.5),
            (3.0, 3.5, 1.0, 2.0, 1.5, 2.5),
        ),
        dtype=np.float32,
    )
    together = apply_support_evidence_gate(state, rows)
    separately = np.concatenate(
        [apply_support_evidence_gate(state, rows[index : index + 1]) for index in range(3)],
        axis=0,
    )
    permuted = apply_support_evidence_gate(state, rows[[2, 0, 1]])

    assert np.array_equal(together, separately)
    assert np.array_equal(permuted, together[[2, 0, 1]])


def test_fit_and_configuration_fail_closed_on_protocol_drift() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(3, seed=5)
    with pytest.raises(SupportEvidenceGateError, match="K>=5"):
        fit_support_evidence_gate(scores, labels, ranks, classes, old_count)

    scores5, labels5, ranks5, classes5, old_count5 = _support_scores(5, seed=6)
    bad_ranks = ranks5.copy()
    bad_ranks[1] = bad_ranks[0]
    with pytest.raises(SupportEvidenceGateError, match="shot ranks"):
        fit_support_evidence_gate(
            scores5, labels5, bad_ranks, classes5, old_count5
        )
    nonfinite = scores5.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(SupportEvidenceGateError, match="finite"):
        fit_support_evidence_gate(
            nonfinite, labels5, ranks5, classes5, old_count5
        )
    with pytest.raises(SupportEvidenceGateError, match="lambda grid"):
        SupportEvidenceGateConfig(ridge_lambdas=(1.0,))
    with pytest.raises(SupportEvidenceGateError, match="alpha"):
        SupportEvidenceGateConfig(alpha=-1.0)
    with pytest.raises(SupportEvidenceGateError, match="delta"):
        SupportEvidenceGateConfig(delta=0.0)


def test_oof_no_gain_and_ineffective_features_fall_back_to_safe_passthrough() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(5, seed=8)
    perfect_scores = scores.copy()
    for row_index, class_name in enumerate(labels.tolist()):
        class_index = classes.index(class_name)
        if class_index >= old_count:
            perfect_scores[row_index, class_index - old_count] = np.float32(4.2)
    no_gain_state = fit_support_evidence_gate(
        perfect_scores, labels, ranks, classes, old_count
    )
    no_gain_audit = json.loads(no_gain_state.audit_json)

    assert no_gain_state.enabled is False
    assert no_gain_state.k_shot == 5
    assert no_gain_state.coefficients.shape == (0,)
    assert no_gain_audit["selection_policy"] == (
        "oof_identity_safety_failed_disabled_passthrough"
    )
    assert apply_support_evidence_gate(
        no_gain_state, perfect_scores
    ).tobytes() == perfect_scores.tobytes()

    constant_scores = np.tile(
        np.asarray((4.0, 3.0, 2.0, 4.0, 3.0, 2.0), dtype=np.float32),
        (len(scores), 1),
    )
    ineffective_state = fit_support_evidence_gate(
        constant_scores, labels, ranks, classes, old_count
    )
    ineffective_audit = json.loads(ineffective_state.audit_json)
    assert ineffective_state.enabled is False
    assert all(
        candidate["valid"] is False
        and "std<1e-4" in candidate["failure_reason"]
        for candidate in ineffective_audit["lambda_evidence"]
    )
    assert apply_support_evidence_gate(
        ineffective_state, constant_scores
    ).tobytes() == constant_scores.tobytes()


def test_resource_and_public_api_keep_query_protocol_closed() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(5, seed=7)
    state = fit_support_evidence_gate(
        scores,
        labels,
        ranks,
        classes,
        old_count,
        config=SupportEvidenceGateConfig(delta=1.0),
    )
    resource = state.resource_audit()
    signatures = "\n".join(
        (
            str(inspect.signature(fit_support_evidence_gate)),
            str(inspect.signature(apply_support_evidence_gate)),
            str(inspect.signature(predict_with_support_evidence_gate)),
        )
    ).lower()

    assert resource["fitted_parameter_count"] == 6
    assert resource["normalization_scalar_count"] == 10
    assert resource["total_fitted_state_scalar_count"] == 16
    assert resource["normalization_state_bytes"] == 40
    assert resource["persistent_state_cap_pass"] is True
    assert resource["gradient_trainable_parameter_count"] == 0
    assert resource["ridge_lambda_candidate_count"] == 3
    assert resource["closed_form_solve_count"] == 16
    assert resource["estimated_gate_macs_per_query"] == 6
    assert resource["normalization_subtractions_per_query"] == 5
    assert resource["normalization_divisions_per_query"] == 5
    assert resource["clip_scalar_count_per_query"] == 1
    assert resource["estimated_gate_temporary_bytes"] > 0
    assert resource["new_score_additions_per_query"] == len(classes) - old_count
    assert resource["old_score_writes_per_query"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_labels_used_for_fit"] is False
    assert resource["query_role_oracle_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert resource["query_batch_statistics_used"] is False
    assert resource["row_local_inference"] is True
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False
    assert all(
        forbidden not in signatures
        for forbidden in ("query", "truth", "role", "quota", "batch")
    )
