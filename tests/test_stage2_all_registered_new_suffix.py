from __future__ import annotations

import inspect
import json

import numpy as np

from cvsrffi.stage2_all_registered_new_suffix import (
    D31_NEW_CVAR_FLOOR,
    D31_OLD_MARGIN_PROTECTION,
    D31_PLAIN_BALANCED_CE,
    D31Stage2CConfig,
    MAX_PEAK_TRAINABLE_PARAMETERS,
    append_stage2c_all_registered_new_suffix,
    predict_all_registered,
    score_all_registered,
)
from cvsrffi.stage2_multimodal_compact_diag import (
    D26CompactDiagConfig,
    fit_stage2b_compact_diag,
    score_all_registered as score_d26,
)


DIM = 288


def _support(
    classes: tuple[str, ...],
    k: int,
    *,
    seed: int,
    starts: tuple[int, ...] | None = None,
    noise: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    offsets = starts or tuple(range(len(classes)))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(classes):
        center = np.zeros(DIM, dtype=np.float32)
        center[offsets[class_index]] = 1.0
        center[(offsets[class_index] + 31) % DIM] = 0.25
        for _ in range(k):
            rows.append(
                center + rng.normal(0.0, noise, size=DIM).astype(np.float32)
            )
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _base(
    old_classes: tuple[str, ...] = ("o0", "o1", "o2"),
    k: int = 5,
) -> tuple[object, np.ndarray, np.ndarray]:
    old_rows, old_labels = _support(old_classes, k, seed=11)
    fit = fit_stage2b_compact_diag(
        old_rows,
        old_labels,
        old_classes,
        config=D26CompactDiagConfig(stage2b_steps=2, stage2c_steps=0),
    )
    return fit.state, old_rows, old_labels


def _append(
    method: str = D31_PLAIN_BALANCED_CE,
    *,
    new_k: int = 5,
):
    base, old_rows, old_labels = _base()
    new_classes = ("n0", "n1", "n2")
    # n0 is deliberately close to o0, creating meaningful old-negative evidence.
    new_rows, new_labels = _support(
        new_classes, new_k, seed=29, starts=(0, 7, 8), noise=0.04
    )
    result = append_stage2c_all_registered_new_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=D31Stage2CConfig(method),
    )
    return base, old_rows, old_labels, new_rows, new_labels, result


def test_old_diag_and_weights_are_bitwise_frozen_and_gate_is_classwise_safe() -> None:
    base, old_rows, _, _, _, result = _append(D31_OLD_MARGIN_PROTECTION)
    state = result.state
    assert state.log_diag.tobytes() == base.log_diag.tobytes()
    assert state.weights[: len(base.classes)].tobytes() == base.weights.tobytes()
    assert state.base_old_lock_sha256 == base.old_lock_sha256
    assert score_all_registered(state, old_rows)[:, : len(base.classes)].tobytes() == (
        score_d26(base, old_rows).tobytes()
    )
    gate = json.loads(state.support_gate_json)
    assert gate["old_support_gate_pass"] is True
    assert gate["full_refit_gate_pass"] is True
    assert gate["pre_registration_correct_rows_preserved"] is True
    assert gate["per_old_class_non_degradation"] is True
    assert gate["new_class_biases_non_positive"] is True
    assert gate["old_first_exact_tie_policy_preserved"] is True
    assert np.all(state.new_class_biases <= np.float32(0.0))
    # At least one loose positive raw cap is clamped to exactly zero: D31 never
    # turns support safety slack into a positive boost over old-first ties.
    assert float(np.max(state.new_class_biases)) == 0.0
    assert all(
        gate["after_per_old_class_accuracy"][label]
        >= gate["before_per_old_class_accuracy"][label]
        for label in base.classes
    )


def test_every_step_uses_old_and_new_support_and_old_negatives_change_gradient_path() -> None:
    base, old_rows, old_labels = _base()
    new_classes = ("n0", "n1")
    new_rows, new_labels = _support(
        new_classes, 5, seed=41, starts=(0, 9), noise=0.035
    )
    near = append_stage2c_all_registered_new_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=D31Stage2CConfig(D31_PLAIN_BALANCED_CE),
    )
    # Rotate only old registered support away from n0. New support and the frozen
    # old head remain identical, so a changed suffix proves old rows enter loss.
    far_old = np.roll(old_rows, 80, axis=1).copy()
    far = append_stage2c_all_registered_new_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        far_old,
        old_labels,
        config=D31Stage2CConfig(D31_PLAIN_BALANCED_CE),
    )
    assert near.state.weights[len(base.classes) :].tobytes() != (
        far.state.weights[len(base.classes) :].tobytes()
    )
    assert near.loss_trace[1]["old_registered_support_rows_in_loss"] == len(old_rows)
    assert near.loss_trace[1]["all_registered_support_rows_in_loss"] == (
        len(old_rows) + len(new_rows)
    )
    assert near.loss_trace[1]["gradient_norm"] > 0.0


def test_cvar_floor_and_old_margin_method_locks_are_visible_in_complete_trace() -> None:
    _, _, _, _, _, cvar = _append(D31_NEW_CVAR_FLOOR)
    _, _, _, _, _, protected = _append(D31_OLD_MARGIN_PROTECTION)
    assert len(cvar.loss_trace) == 11
    assert len(protected.loss_trace) == 16
    assert [row["step"] for row in cvar.loss_trace] == list(range(11))
    for row in cvar.loss_trace:
        assert row["new_class_cvar_weight"] == 0.35
        assert row["new_class_cvar_loss"] >= 0.0
        assert 0.0 <= row["new_support_class_floor"] <= 1.0
        assert row["old_margin_protection_weight"] == 0.0
    for row in protected.loss_trace:
        assert row["new_class_cvar_weight"] == 0.25
        assert row["old_margin_protection_weight"] == 0.75
        assert row["old_margin_protection_loss"] >= 0.0
    cvar_lock = cvar.state.resource_audit()["method_lock"]
    protected_lock = protected.state.resource_audit()["method_lock"]
    assert cvar_lock["new_class_cvar_tail_fraction"] == 0.20
    assert cvar_lock["centroid_anchor_weight"] == 0.02
    assert protected_lock["new_class_cvar_tail_fraction"] == 0.20
    assert protected_lock["old_margin"] == 0.90
    assert protected_lock["centroid_anchor_weight"] == 0.05
    assert protected_lock["optimizer_steps"] == 15


def test_row_permutation_is_bitwise_invariant() -> None:
    base, old_rows, old_labels = _base()
    new_classes = ("n0", "n1", "n2")
    new_rows, new_labels = _support(new_classes, 5, seed=67, starts=(0, 6, 7))
    config = D31Stage2CConfig(D31_OLD_MARGIN_PROTECTION)
    first = append_stage2c_all_registered_new_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=config,
    )
    old_order = np.asarray([9, 1, 13, 4, 7, 0, 14, 2, 11, 5, 8, 3, 10, 6, 12])
    new_order = np.asarray([14, 0, 6, 12, 3, 9, 1, 7, 13, 4, 10, 2, 8, 5, 11])
    second = append_stage2c_all_registered_new_suffix(
        base,
        new_rows[new_order],
        new_labels[new_order],
        new_classes,
        old_rows[old_order],
        old_labels[old_order],
        config=config,
    )
    assert second.state.weights.tobytes() == first.state.weights.tobytes()
    assert second.state.new_class_biases.tobytes() == first.state.new_class_biases.tobytes()
    assert second.state.support_gate_json == first.state.support_gate_json
    assert second.loss_trace == first.loss_trace


def test_k1_centroid_only_safe_bypass_has_zero_optimizer_updates() -> None:
    base, old_rows, old_labels, new_rows, _, result = _append(new_k=1)
    assert result.state.stage2c_optimizer_steps == 0
    assert len(result.loss_trace) == 1
    assert result.loss_trace[0]["gradient_norm"] == 0.0
    audit = result.state.resource_audit()
    assert audit["k1_safe_bypass"] is True
    gate = json.loads(result.state.support_gate_json)
    assert gate["k1_centroid_only_safe_bypass"] is True
    assert gate["full_refit_attempted"] is False
    assert len(predict_all_registered(result.state, new_rows)) == len(new_rows)
    assert result.state.weights[: len(base.classes)].tobytes() == base.weights.tobytes()


def test_twenty_new_classes_stay_under_peak_2016_and_protocol_surface_is_closed() -> None:
    old_classes = tuple(f"o{index}" for index in range(6))
    base, old_rows, old_labels = _base(old_classes=old_classes, k=2)
    new_classes = tuple(f"n{index:02d}" for index in range(20))
    new_rows, new_labels = _support(
        new_classes, 2, seed=101, starts=tuple(range(20, 40)), noise=0.02
    )
    result = append_stage2c_all_registered_new_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=D31Stage2CConfig(D31_PLAIN_BALANCED_CE),
    )
    audit = result.state.resource_audit()
    assert audit["peak_trainable_parameters"] == MAX_PEAK_TRAINABLE_PARAMETERS
    assert audit["stage2c_active_trainable_parameters_per_step_max"] == (
        MAX_PEAK_TRAINABLE_PARAMETERS
    )
    assert audit["stage2c_total_new_weight_state_scalars"] == 20 * DIM
    assert audit["stage2c_optimizer_steps"] == 10
    assert audit["estimated_adaptation_macs"] == (
        audit["estimated_stage2b_adaptation_macs"]
        + audit["estimated_stage2c_adaptation_macs"]
    )
    assert audit["estimated_stage2b_adaptation_macs"] > 0
    assert audit["persistent_state_cap_pass"] is True
    assert audit["persistent_state_bytes"] == audit["deployable_predictor_state_bytes"]
    assert audit["support_gate_external_evidence_bytes"] > 0
    assert len(audit["support_gate_external_evidence_sha256"]) == 64
    assert audit["support_gate_external_evidence_excluded_from_deployment_state"] is True
    assert audit["dense_query_graph_bytes"] == 0
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_labels_used_for_fit"] is False
    assert audit["query_role_oracle_access"] is False
    assert audit["query_true_batch_class_count_access"] is False
    assert audit["query_class_quota_access"] is False
    assert audit["query_batch_global_assignment"] is False
    assert audit["clean_sample_access"] is False
    assert audit["source_sample_access"] is False
    parameter_names = set(
        inspect.signature(append_stage2c_all_registered_new_suffix).parameters
    )
    assert not any(
        forbidden in name
        for name in parameter_names
        for forbidden in ("query", "clean", "source", "quota", "role")
    )


def test_k10_five_new_stage2c_is_1440_but_full_pipeline_peak_is_2016() -> None:
    old_classes = tuple(f"o{index}" for index in range(6))
    base, old_rows, old_labels = _base(old_classes=old_classes, k=10)
    new_classes = tuple(f"n{index}" for index in range(5))
    new_rows, new_labels = _support(
        new_classes, 10, seed=151, starts=tuple(range(20, 25)), noise=0.02
    )
    result = append_stage2c_all_registered_new_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=D31Stage2CConfig(D31_PLAIN_BALANCED_CE),
    )
    audit = result.state.resource_audit()
    assert audit["stage2c_active_trainable_parameters_per_step_max"] == 5 * DIM
    assert audit["stage2c_active_trainable_parameters_per_step_max"] == 1_440
    assert audit["stage2b_trainable_parameters"] == 2_016
    assert audit["peak_trainable_parameters"] == 2_016
    assert audit["formal_total_optimizer_step_cap_pass"] is True
