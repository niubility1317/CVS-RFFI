from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from cvsrffi.stage2_max_envelope_calibration import (
    MaxEnvelopeCalibrationConfig,
    MaxEnvelopeCalibrationError,
    MaxEnvelopeCalibrationState,
    apply_max_envelope_calibration,
    audit_envelope_confusions,
    fit_max_envelope_calibration,
    predict_with_max_envelope_calibration,
)


def _support(
    k_shot: int, *, already_correct: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int]:
    classes = ("old_a", "old_b", "new_weak", "new_strong", "new_other")
    old_count = 2
    rows: list[np.ndarray] = []
    labels: list[str] = []
    ranks: list[int] = []
    for class_index, class_name in enumerate(classes):
        for rank in range(k_shot):
            row = np.full(len(classes), -2.0, dtype=np.float32)
            if class_index < old_count:
                row[class_index] = np.float32(5.0)
                row[1 - class_index] = np.float32(0.0)
                row[old_count:] = np.asarray((0.4, 0.3, 0.2), dtype=np.float32)
            elif class_name == "new_weak":
                row[:old_count] = np.asarray((0.2, 0.1), dtype=np.float32)
                row[2:] = np.asarray(
                    (1.4 if already_correct else 1.0, 1.3, 0.0),
                    dtype=np.float32,
                )
            elif class_name == "new_strong":
                row[:old_count] = np.asarray((0.2, 0.1), dtype=np.float32)
                row[2:] = np.asarray((0.0, 2.5, -0.5), dtype=np.float32)
            else:
                row[:old_count] = np.asarray((0.2, 0.1), dtype=np.float32)
                row[2:] = np.asarray((-0.5, 0.0, 2.5), dtype=np.float32)
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


def test_k5_oof_bias_fixes_new_new_floor_and_preserves_old_envelope() -> None:
    scores, labels, ranks, classes, old_count = _support(5)
    state = fit_max_envelope_calibration(
        scores,
        labels,
        ranks,
        classes,
        old_count,
        config=MaxEnvelopeCalibrationConfig(
            objective="floor_first", coordinate_passes=2
        ),
    )
    adjusted = apply_max_envelope_calibration(state, scores)
    raw_predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    adjusted_predictions = predict_with_max_envelope_calibration(state, scores)
    audit = json.loads(state.audit_json)
    old_mask = np.isin(labels, np.asarray(classes[:old_count]))
    new_mask = ~old_mask

    assert state.enabled is True
    assert np.count_nonzero(state.biases) > 0
    assert audit["oof_evidence"]["strict_new_improvement"] is True
    assert audit["full_support_evidence"]["strict_new_improvement"] is True
    assert adjusted[:, :old_count].tobytes() == scores[:, :old_count].tobytes()
    assert np.array_equal(
        np.max(adjusted[:, old_count:], axis=1),
        np.max(scores[:, old_count:], axis=1),
    )
    assert np.array_equal(
        adjusted_predictions[old_mask], raw_predictions[old_mask]
    )
    assert np.mean(adjusted_predictions[new_mask] == labels[new_mask]) > np.mean(
        raw_predictions[new_mask] == labels[new_mask]
    )
    assert adjusted.dtype == np.float32
    assert not adjusted.flags.writeable


def test_manual_bias_changes_new_identity_but_not_group_or_old_first_tie() -> None:
    classes = ("old_a", "old_b", "new_a", "new_b")
    state = MaxEnvelopeCalibrationState(
        schema="cvs.phase2.d30_max_envelope_calibration.v1",
        registered_classes=classes,
        old_class_count=2,
        k_shot=5,
        enabled=True,
        biases=np.asarray((0.0, 0.2), dtype=np.float32),
        audit_json="{}",
        config=MaxEnvelopeCalibrationConfig(),
    )
    scores = np.asarray(
        (
            (0.5, 0.1, 1.0, 0.9),  # new_a -> new_b, envelope 1.0
            (1.0, 0.1, 1.0, 0.9),  # exact old/new tie remains old_a
            (1.1, 0.1, 1.0, 0.9),  # new->old error cannot be repaired
        ),
        dtype=np.float32,
    )
    adjusted = apply_max_envelope_calibration(state, scores)
    raw = np.asarray(classes)[np.argmax(scores, axis=1)]
    calibrated = np.asarray(classes)[np.argmax(adjusted, axis=1)]

    assert raw.tolist() == ["new_a", "old_a", "old_a"]
    assert calibrated.tolist() == ["new_b", "old_a", "old_a"]
    assert np.array_equal(
        np.max(adjusted[:, 2:], axis=1), np.max(scores[:, 2:], axis=1)
    )


def test_k1_is_exact_passthrough_and_k2_to_k4_fail_closed() -> None:
    scores, labels, ranks, classes, old_count = _support(1)
    state = fit_max_envelope_calibration(
        scores, labels, ranks, classes, old_count
    )
    adjusted = apply_max_envelope_calibration(state, scores)
    audit = json.loads(state.audit_json)

    assert state.enabled is False
    assert np.count_nonzero(state.biases) == 0
    assert adjusted.tobytes() == scores.tobytes()
    assert audit["selection_policy"] == "k1_disabled_exact_passthrough"
    assert state.resource_audit()["optimizer_steps"] == 0

    for k_shot in (2, 3, 4):
        values = _support(k_shot)
        with pytest.raises(MaxEnvelopeCalibrationError, match="K>=5"):
            fit_max_envelope_calibration(*values)


def test_no_strict_oof_gain_atomically_disables_and_returns_raw_bytes() -> None:
    scores, labels, ranks, classes, old_count = _support(
        5, already_correct=True
    )
    state = fit_max_envelope_calibration(
        scores, labels, ranks, classes, old_count
    )
    audit = json.loads(state.audit_json)

    assert state.enabled is False
    assert np.count_nonzero(state.biases) == 0
    assert audit["fallback_reason"] == "oof_no_strict_new_gain"
    assert apply_max_envelope_calibration(state, scores).tobytes() == scores.tobytes()


def test_confusion_audit_exposes_reachable_ceiling_per_new_class() -> None:
    classes = ("old_a", "old_b", "new_a", "new_b")
    labels = np.asarray(("new_a", "new_a", "new_a", "new_b"))
    scores = np.asarray(
        (
            (2.0, 0.0, 1.0, 0.5),  # new_a old_win: unreachable
            (0.0, -1.0, 2.0, 1.0),  # new_a correct
            (0.0, -1.0, 1.0, 2.0),  # new_a wrong-new: repairable pool
            (0.0, -1.0, 0.0, 2.0),  # new_b correct
        ),
        dtype=np.float32,
    )
    audit = audit_envelope_confusions(scores, labels, classes, 2)
    weak = audit["per_new_class"]["new_a"]

    assert weak["old_win"] == 1
    assert weak["new_correct"] == 1
    assert weak["new_wrong"] == 1
    assert weak["current_accuracy"] == pytest.approx(1 / 3)
    assert weak["reachable_ceiling_accuracy"] == pytest.approx(2 / 3)
    assert weak["repairable_gain_ceiling"] == pytest.approx(1 / 3)


def test_inference_is_row_local_permutation_equivariant_and_gauge_invariant() -> None:
    classes = ("old_a", "old_b", "new_a", "new_b")
    base = MaxEnvelopeCalibrationState(
        schema="cvs.phase2.d30_max_envelope_calibration.v1",
        registered_classes=classes,
        old_class_count=2,
        k_shot=10,
        enabled=True,
        biases=np.asarray((-0.1, 0.1), dtype=np.float32),
        audit_json="{}",
        config=MaxEnvelopeCalibrationConfig(),
    )
    shifted = MaxEnvelopeCalibrationState(
        schema="cvs.phase2.d30_max_envelope_calibration.v1",
        registered_classes=classes,
        old_class_count=2,
        k_shot=10,
        enabled=True,
        biases=np.asarray((2.9, 3.1), dtype=np.float32),
        audit_json="{}",
        config=MaxEnvelopeCalibrationConfig(),
    )
    rows = np.asarray(
        ((0.5, 0.2, 1.0, 0.9), (1.2, 0.1, 1.0, 0.8), (0.0, 0.1, 0.4, 0.6)),
        dtype=np.float32,
    )
    together = apply_max_envelope_calibration(base, rows)
    separate = np.concatenate(
        [apply_max_envelope_calibration(base, rows[i : i + 1]) for i in range(3)]
    )
    permuted = apply_max_envelope_calibration(base, rows[[2, 0, 1]])
    gauge = apply_max_envelope_calibration(shifted, rows)

    assert np.array_equal(together, separate)
    assert np.array_equal(permuted, together[[2, 0, 1]])
    assert np.array_equal(np.argmax(gauge, axis=1), np.argmax(together, axis=1))
    assert np.allclose(gauge, together, rtol=0.0, atol=3.0e-7)
    assert np.array_equal(
        np.max(gauge[:, 2:], axis=1), np.max(together[:, 2:], axis=1)
    )


def test_per_new_class_drop_forces_atomic_oof_passthrough() -> None:
    """Aggregate gain must not hide a drop of one registered new class."""

    classes = (
        "old_a",
        "old_b",
        "new_dead",
        "new_weak",
        "new_strong",
        "new_other",
    )
    old_count = 2
    rows: list[np.ndarray] = []
    labels: list[str] = []
    ranks: list[int] = []
    for class_index, class_name in enumerate(classes):
        for rank in range(5):
            row = np.full(len(classes), -2.0, dtype=np.float32)
            if class_index < old_count:
                row[class_index] = np.float32(5.0)
                row[old_count:] = np.float32(0.2)
            elif class_name == "new_dead":
                row[:old_count] = np.asarray((2.0, 0.0), dtype=np.float32)
                row[2:] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
            elif class_name == "new_weak":
                row[:old_count] = np.asarray((0.0, -1.0), dtype=np.float32)
                row[2:] = np.asarray((0.0, 1.0, 1.3, 0.0), dtype=np.float32)
            elif class_name == "new_strong":
                row[:old_count] = np.asarray((0.0, -1.0), dtype=np.float32)
                if rank == 0:
                    row[2:] = np.asarray((0.0, 1.0, 1.2, 0.0), dtype=np.float32)
                else:
                    row[2:] = np.asarray((0.0, 0.0, 2.5, 0.0), dtype=np.float32)
            else:
                row[:old_count] = np.asarray((0.0, -1.0), dtype=np.float32)
                row[2:] = np.asarray((0.0, 0.0, 0.0, 2.5), dtype=np.float32)
            rows.append(row)
            labels.append(class_name)
            ranks.append(rank)
    scores = np.stack(rows)
    state = fit_max_envelope_calibration(
        scores,
        np.asarray(labels),
        np.asarray(ranks, dtype=np.int64),
        classes,
        old_count,
    )
    audit = json.loads(state.audit_json)

    assert audit["oof_evidence"][
        "new_floor_worst20_and_overall_non_degradation"
    ] is True
    assert audit["oof_evidence"]["strict_new_improvement"] is True
    assert audit["oof_evidence"][
        "per_new_class_accuracy_non_degradation"
    ] is False
    assert audit["fallback_reason"] == "oof_safety_failed"
    assert state.enabled is False
    assert apply_max_envelope_calibration(state, scores).tobytes() == scores.tobytes()


def test_twenty_new_classes_use_one_fp32_scalar_each_and_bounded_state() -> None:
    classes = tuple(f"old_{i}" for i in range(2)) + tuple(
        f"new_{i}" for i in range(20)
    )
    state = MaxEnvelopeCalibrationState(
        schema="cvs.phase2.d30_max_envelope_calibration.v1",
        registered_classes=classes,
        old_class_count=2,
        k_shot=10,
        enabled=True,
        biases=np.linspace(-0.5, 0.5, 20, dtype=np.float32),
        audit_json="{}",
        config=MaxEnvelopeCalibrationConfig(),
    )
    resource = state.resource_audit()

    assert resource["fitted_parameter_count"] == 20
    assert resource["bias_scalar_count"] == 20
    assert resource["deployable_predictor_state_bytes"] == 112
    assert resource["estimated_extra_macs_per_query"] == 0
    assert resource["estimated_scalar_ops_per_query"] == 98
    assert resource["scratch_bytes_per_query"] == 80
    assert resource["persistent_state_cap_pass"] is True
    assert resource["dense_query_graph_bytes"] == 0


def test_protocol_surface_and_invalid_inputs_fail_closed() -> None:
    scores, labels, ranks, classes, old_count = _support(5)
    signatures = "\n".join(
        (
            str(inspect.signature(fit_max_envelope_calibration)),
            str(inspect.signature(apply_max_envelope_calibration)),
            str(inspect.signature(predict_with_max_envelope_calibration)),
        )
    ).lower()
    state = fit_max_envelope_calibration(
        scores, labels, ranks, classes, old_count
    )
    resource = state.resource_audit()

    assert all(
        forbidden not in signatures
        for forbidden in ("query", "truth", "role", "quota", "batch")
    )
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_role_oracle_access"] is False
    assert resource["query_true_batch_class_count_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False

    nonfinite = scores.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(MaxEnvelopeCalibrationError, match="finite"):
        fit_max_envelope_calibration(
            nonfinite, labels, ranks, classes, old_count
        )
    bad_ranks = ranks.copy()
    bad_ranks[1] = bad_ranks[0]
    with pytest.raises(MaxEnvelopeCalibrationError, match="shot ranks"):
        fit_max_envelope_calibration(
            scores, labels, bad_ranks, classes, old_count
        )
    with pytest.raises(MaxEnvelopeCalibrationError, match="objective"):
        MaxEnvelopeCalibrationConfig(objective="query_selected")
    for invalid in (0, 3, 1.5, "1", True):
        with pytest.raises(MaxEnvelopeCalibrationError, match="coordinate"):
            MaxEnvelopeCalibrationConfig(coordinate_passes=invalid)  # type: ignore[arg-type]
    with pytest.raises(MaxEnvelopeCalibrationError, match="counts"):
        MaxEnvelopeCalibrationState(
            schema="cvs.phase2.d30_max_envelope_calibration.v1",
            registered_classes=classes,
            old_class_count=2.5,  # type: ignore[arg-type]
            k_shot=5,
            enabled=False,
            biases=np.zeros(3, dtype=np.float32),
            audit_json="{}",
            config=MaxEnvelopeCalibrationConfig(),
        )
