from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from cvsrffi.stage2_classwise_safe_release import (
    ClasswiseSafeReleaseConfig,
    ClasswiseSafeReleaseError,
    ClasswiseSafeReleaseState,
    apply_classwise_safe_release,
    fit_classwise_safe_release,
    predict_with_classwise_safe_release,
)


def _support_scores(
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int]:
    """Build a registry with one near-boundary and K-1 hard new rows/class."""

    classes = ("old_a", "old_b", "new_c", "new_d")
    old_count = 2
    rows: list[np.ndarray] = []
    labels: list[str] = []
    ranks: list[int] = []
    for class_index, class_name in enumerate(classes):
        for rank in range(k_shot):
            row = np.zeros(len(classes), dtype=np.float32)
            if class_index < old_count:
                row[class_index] = np.float32(5.0)
                row[1 - class_index] = np.float32(1.0)
            else:
                # Raw D27 predicts old_a.  Rank zero is only 0.05 below the
                # boundary; all other shots have a 1.0 deficit.  The latter
                # set the release width and allow the former to be corrected.
                row[0] = np.float32(3.0)
                row[class_index] = np.float32(2.95 if rank == 0 else 2.0)
                row[old_count + (1 - (class_index - old_count))] = np.float32(0.0)
            rows.append(row)
            labels.append(class_name)
            ranks.append(rank)
    return (
        np.stack(rows),
        np.asarray(labels),
        np.asarray(ranks, dtype=np.int64),
        classes,
        old_count,
    )


def test_k5_oof_release_strictly_improves_new_floor_without_old_regression() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(5)
    state = fit_classwise_safe_release(
        scores,
        labels,
        ranks,
        classes,
        old_count,
        config=ClasswiseSafeReleaseConfig(
            safety_budget=1.0,
            objective="floor_first",
            coordinate_passes=2,
        ),
    )
    audit = json.loads(state.audit_json)
    adjusted = apply_classwise_safe_release(state, scores)
    raw_predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    adjusted_predictions = np.asarray(classes)[np.argmax(adjusted, axis=1)]
    old_mask = np.isin(labels, np.asarray(classes[:old_count]))
    new_mask = ~old_mask

    assert state.enabled is True
    assert np.all(state.widths > 0.0)
    assert np.all(state.amplitudes > 0.0)
    assert audit["oof_evidence"]["strict_new_improvement"] is True
    assert audit["full_support_evidence"]["strict_new_improvement"] is True
    assert np.array_equal(adjusted_predictions[old_mask], labels[old_mask])
    assert np.mean(adjusted_predictions[new_mask] == labels[new_mask]) > np.mean(
        raw_predictions[new_mask] == labels[new_mask]
    )
    assert adjusted[:, :old_count].tobytes() == scores[:, :old_count].tobytes()
    assert adjusted.dtype == np.float32
    assert not adjusted.flags.writeable
    assert predict_with_classwise_safe_release(state, scores).tolist() == (
        adjusted_predictions.tolist()
    )


def test_k1_is_exact_disabled_passthrough_and_k2_to_k4_fail_closed() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(1)
    state = fit_classwise_safe_release(
        scores, labels, ranks, classes, old_count
    )
    adjusted = apply_classwise_safe_release(state, scores)
    audit = json.loads(state.audit_json)

    assert state.enabled is False
    assert np.count_nonzero(state.widths) == 0
    assert np.count_nonzero(state.amplitudes) == 0
    assert adjusted.tobytes() == scores.tobytes()
    assert audit["selection_policy"] == "k1_disabled_exact_d27_passthrough"
    assert state.resource_audit()["fitted_parameter_count"] == 0

    for k_shot in (2, 3, 4):
        bad_scores, bad_labels, bad_ranks, bad_classes, bad_old_count = (
            _support_scores(k_shot)
        )
        with pytest.raises(ClasswiseSafeReleaseError, match="K>=5"):
            fit_classwise_safe_release(
                bad_scores,
                bad_labels,
                bad_ranks,
                bad_classes,
                bad_old_count,
            )


def test_inference_is_row_local_and_permutation_equivariant() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(5)
    state = fit_classwise_safe_release(
        scores,
        labels,
        ranks,
        classes,
        old_count,
        config=ClasswiseSafeReleaseConfig(safety_budget=1.0),
    )
    rows = np.asarray(
        (
            (4.0, 1.0, 3.9, 0.2),
            (1.0, 4.0, 0.1, 3.7),
            (2.5, 2.0, 2.45, 2.4),
        ),
        dtype=np.float32,
    )
    together = apply_classwise_safe_release(state, rows)
    separately = np.concatenate(
        [
            apply_classwise_safe_release(state, rows[index : index + 1])
            for index in range(len(rows))
        ]
    )
    permuted = apply_classwise_safe_release(state, rows[[2, 0, 1]])

    assert np.array_equal(together, separately)
    assert np.array_equal(permuted, together[[2, 0, 1]])


def test_full_refit_without_strict_gain_atomically_falls_back() -> None:
    """OOF gain alone must not leave a nominally enabled deployment state."""

    rng = np.random.default_rng(123)
    classes = ("old_0", "old_1", "new_0", "new_1")
    labels = np.repeat(np.asarray(classes), 5)
    ranks = np.tile(np.arange(5, dtype=np.int64), len(classes))
    # The second deterministic draw has OOF strict gain, while its deployable
    # full-support refit has no strict new-class gain.
    for _ in range(2):
        scores = rng.normal(0.0, 1.0, (20, 4)).astype(np.float32)
        for row_index, class_name in enumerate(labels.tolist()):
            class_index = classes.index(class_name)
            if class_index < 2:
                scores[row_index, class_index] += np.float32(2.5)
            else:
                scores[row_index, 0] += np.float32(1.4)
                scores[row_index, class_index] += np.float32(
                    1.2 + rng.normal(0.0, 0.8)
                )

    state = fit_classwise_safe_release(
        scores,
        labels,
        ranks,
        classes,
        2,
        config=ClasswiseSafeReleaseConfig(
            safety_budget=1.0,
            objective="floor_first",
        ),
    )
    audit = json.loads(state.audit_json)

    assert audit["oof_evidence"]["strict_new_improvement"] is True
    assert audit["full_refit_pre_disable_evidence"][
        "strict_new_improvement"
    ] is False
    assert audit["full_fit_fallback_reason"] == "full_fit_no_strict_new_gain"
    assert state.enabled is False
    assert np.count_nonzero(state.widths) == 0
    assert np.count_nonzero(state.amplitudes) == 0
    assert audit["selected_policy_by_new_class"] == {
        "new_0": "disabled",
        "new_1": "disabled",
    }
    assert apply_classwise_safe_release(state, scores).tobytes() == scores.tobytes()


def test_full_refit_safety_failure_atomically_falls_back() -> None:
    rng = np.random.default_rng(456)
    classes = ("old_0", "old_1", "new_0", "new_1")
    labels = np.repeat(np.asarray(classes), 5)
    ranks = np.tile(np.arange(5, dtype=np.int64), len(classes))
    # The fourth deterministic draw passes OOF selection but its joint full
    # refit is unsafe.  This is a data-dependent safe fallback, not malformed
    # protocol input, so it must not abort the experiment row.
    for _ in range(4):
        scores = rng.normal(0.0, 1.0, (20, 4)).astype(np.float32)
        for row_index, class_name in enumerate(labels.tolist()):
            class_index = classes.index(class_name)
            if class_index < 2:
                scores[row_index, class_index] += np.float32(2.5)
            else:
                scores[row_index, 0] += np.float32(1.4)
                scores[row_index, class_index] += np.float32(
                    1.2 + rng.normal(0.0, 0.8)
                )

    state = fit_classwise_safe_release(
        scores,
        labels,
        ranks,
        classes,
        2,
        config=ClasswiseSafeReleaseConfig(
            safety_budget=1.0,
            objective="floor_first",
        ),
    )
    audit = json.loads(state.audit_json)

    assert audit["full_refit_pre_disable_evidence"]["safe"] is False
    assert audit["full_fit_fallback_reason"] == "full_fit_safety_failed"
    assert state.enabled is False
    assert np.count_nonzero(state.widths) == 0
    assert np.count_nonzero(state.amplitudes) == 0
    assert apply_classwise_safe_release(state, scores).tobytes() == scores.tobytes()


def test_twenty_new_classes_have_two_scalars_each_and_bounded_state() -> None:
    classes = tuple(f"old_{index}" for index in range(2)) + tuple(
        f"new_{index}" for index in range(20)
    )
    state = ClasswiseSafeReleaseState(
        schema="cvs.phase2.d29_classwise_safe_release.v1",
        registered_classes=classes,
        old_class_count=2,
        k_shot=10,
        enabled=True,
        widths=np.ones(20, dtype=np.float32),
        amplitudes=np.ones(20, dtype=np.float32),
        audit_json="{}",
        config=ClasswiseSafeReleaseConfig(),
    )
    resource = state.resource_audit()

    assert resource["fitted_parameter_count"] == 40
    assert resource["width_scalar_count"] == 20
    assert resource["amplitude_scalar_count"] == 20
    assert resource["deployable_predictor_state_bytes"] == 192
    assert resource["estimated_release_scalar_ops_per_query"] == 80
    assert resource["persistent_state_cap_pass"] is True
    assert resource["gradient_trainable_parameter_count"] == 0
    assert resource["dense_query_graph_bytes"] == 0


def test_protocol_surface_and_invalid_inputs_fail_closed() -> None:
    scores, labels, ranks, classes, old_count = _support_scores(5)
    signatures = "\n".join(
        (
            str(inspect.signature(fit_classwise_safe_release)),
            str(inspect.signature(apply_classwise_safe_release)),
            str(inspect.signature(predict_with_classwise_safe_release)),
        )
    ).lower()
    state = fit_classwise_safe_release(
        scores,
        labels,
        ranks,
        classes,
        old_count,
        config=ClasswiseSafeReleaseConfig(safety_budget=1.0),
    )
    resource = state.resource_audit()

    assert all(
        forbidden not in signatures
        for forbidden in ("query", "truth", "role", "quota", "batch")
    )
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_labels_used_for_fit"] is False
    assert resource["query_features_used_for_fit"] is False
    assert resource["query_role_oracle_access"] is False
    assert resource["query_true_batch_class_count_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert resource["row_local_inference"] is True
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False

    nonfinite = scores.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ClasswiseSafeReleaseError, match="finite"):
        fit_classwise_safe_release(
            nonfinite, labels, ranks, classes, old_count
        )
    bad_ranks = ranks.copy()
    bad_ranks[1] = bad_ranks[0]
    with pytest.raises(ClasswiseSafeReleaseError, match="shot ranks"):
        fit_classwise_safe_release(
            scores, labels, bad_ranks, classes, old_count
        )
    with pytest.raises(ClasswiseSafeReleaseError, match="budget"):
        ClasswiseSafeReleaseConfig(safety_budget=0.0)
    with pytest.raises(ClasswiseSafeReleaseError, match="objective"):
        ClasswiseSafeReleaseConfig(objective="query_selected")
    with pytest.raises(ClasswiseSafeReleaseError, match="coordinate"):
        ClasswiseSafeReleaseConfig(coordinate_passes=3)
    for invalid_passes in (1.5, "1", True):
        with pytest.raises(ClasswiseSafeReleaseError, match="coordinate"):
            ClasswiseSafeReleaseConfig(coordinate_passes=invalid_passes)  # type: ignore[arg-type]
    with pytest.raises(ClasswiseSafeReleaseError, match="counts"):
        ClasswiseSafeReleaseState(
            schema="cvs.phase2.d29_classwise_safe_release.v1",
            registered_classes=classes,
            old_class_count=2.5,  # type: ignore[arg-type]
            k_shot=5,
            enabled=False,
            widths=np.zeros(2, dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            audit_json="{}",
            config=ClasswiseSafeReleaseConfig(),
        )
    with pytest.raises(ClasswiseSafeReleaseError, match="state"):
        ClasswiseSafeReleaseState(
            schema="cvs.phase2.d29_classwise_safe_release.v1",
            registered_classes=classes,
            old_class_count=2,
            k_shot=5,
            enabled="false",  # type: ignore[arg-type]
            widths=np.zeros(2, dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            audit_json="{}",
            config=ClasswiseSafeReleaseConfig(),
        )
