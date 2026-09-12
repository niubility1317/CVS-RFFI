"""Cross-fitted atomic-safe Stage2 lifecycle row replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class D70LifecycleError(RuntimeError):
    """Raised when D70 support or lifecycle evidence is invalid."""


Fit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _validate_support(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(
            int(np.sum(y == class_index)) != int(k_shot)
            for class_index in range(int(class_count))
        )
    ):
        raise D70LifecycleError("D70 requires finite exact symmetric support")
    return x, y


def twofold_rank_partitions(
    labels: np.ndarray, class_count: int, k_shot: int
) -> list[np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    indices = [np.flatnonzero(y == index) for index in range(int(class_count))]
    if any(len(item) != int(k_shot) for item in indices):
        raise D70LifecycleError("D70 partition requires symmetric support")
    rank_groups = [np.arange(0, int(k_shot), 2), np.arange(1, int(k_shot), 2)]
    held = [
        np.concatenate([item[ranks] for item in indices]).astype(np.int64)
        for ranks in rank_groups
        if len(ranks)
    ]
    flat = np.concatenate(held) if held else np.empty(0, dtype=np.int64)
    if (
        len(held) != min(2, int(k_shot))
        or len(flat) != len(y)
        or not np.array_equal(np.sort(flat), np.arange(len(y)))
        or len(np.unique(flat)) != len(flat)
    ):
        raise D70LifecycleError("D70 held partition exact-once drift")
    return held


def _counts(scores: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (len(logits),):
        raise D70LifecycleError("D70 gate score shape drift")
    predicted = np.argmax(logits, axis=1)
    class_count = int(logits.shape[1])
    positive = np.zeros(class_count, dtype=np.int64)
    false_positive = np.zeros(class_count, dtype=np.int64)
    for class_index in range(class_count):
        mask = targets == class_index
        positive[class_index] = int(np.sum(predicted[mask] == class_index))
        false_positive[class_index] = int(
            np.sum(predicted[~mask] == class_index)
        )
    return positive, false_positive


def atomic_old_row_gate(
    base_scores: np.ndarray,
    before_old_scores: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    base = np.asarray(base_scores, dtype=np.float64)
    before = np.asarray(before_old_scores, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.int64)
    if (
        base.ndim != 2
        or before.ndim != 2
        or len(base) != len(before)
        or targets.shape != (len(base),)
        or before.shape[1] <= 0
        or before.shape[1] >= base.shape[1]
        or not np.isfinite(base).all()
        or not np.isfinite(before).all()
    ):
        raise D70LifecycleError("D70 atomic gate evidence drift")
    class_count, old_count = int(base.shape[1]), int(before.shape[1])
    base_positive, base_fp = _counts(base, targets)
    coordinate_positive = np.zeros(old_count, dtype=np.int64)
    coordinate_fp = np.zeros(old_count, dtype=np.int64)
    for class_index in range(old_count):
        hybrid = base.copy()
        hybrid[:, class_index] = before[:, class_index]
        positive, false_positive = _counts(hybrid, targets)
        coordinate_positive[class_index] = positive[class_index]
        coordinate_fp[class_index] = false_positive[class_index]
    initial = (
        (coordinate_positive >= base_positive[:old_count])
        & (coordinate_fp <= base_fp[:old_count])
        & (
            (coordinate_positive > base_positive[:old_count])
            | (coordinate_fp < base_fp[:old_count])
        )
    )
    joint_scores = base.copy()
    accepted_indices = np.flatnonzero(initial)
    joint_scores[:, accepted_indices] = before[:, accepted_indices]
    joint_positive, joint_fp = _counts(joint_scores, targets)
    atomic_safe = bool(
        np.all(joint_positive >= base_positive) and np.all(joint_fp <= base_fp)
    )
    final = initial if atomic_safe else np.zeros(old_count, dtype=bool)
    if np.any(final):
        status = "crossfitted_atomic_lifecycle_rows_active"
    elif np.any(initial):
        status = "joint_atomic_failure_exact_d62_fallback"
    else:
        status = "no_row_accepted_exact_d62_fallback"
    return {
        "class_count": class_count,
        "old_count": old_count,
        "base_positive": base_positive,
        "base_false_positive": base_fp,
        "coordinate_positive": coordinate_positive,
        "coordinate_false_positive": coordinate_fp,
        "joint_positive": joint_positive,
        "joint_false_positive": joint_fp,
        "initial_accept": initial,
        "final_accept": final,
        "atomic_safe": atomic_safe,
        "status": status,
        "exact_fallback": not bool(np.any(final)),
    }


@dataclass
class _Pending:
    rows: np.ndarray
    labels: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    class_count: int
    k_shot: int


class AtomicLifecycleRowReplacement:
    """Wrap one symmetric support fit function with the locked D70 lifecycle."""

    def __init__(self, base_fit: Fit) -> None:
        self._base_fit = base_fit
        self._pending: _Pending | None = None
        self.completed_pairs = 0
        self.records: list[dict[str, Any]] = []
        self.inner_fit_count = 0

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def __call__(
        self,
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x, y = _validate_support(rows, labels, class_count, k_shot)
        base_coefficient, base_intercept, base_audit = self._base_fit(
            x, y, int(class_count), int(k_shot)
        )
        joint_coefficient = np.asarray(base_coefficient, dtype=np.float32)
        joint_intercept = np.asarray(base_intercept, dtype=np.float32)
        if joint_coefficient.shape != (int(class_count), x.shape[1]) or joint_intercept.shape != (
            int(class_count),
        ):
            raise D70LifecycleError("D70 D62 affine shape drift")

        if self._pending is None:
            coefficient = joint_coefficient.copy()
            intercept = joint_intercept.copy()
            old_count, appended_count = int(class_count), 0
            mask = np.zeros(old_count, dtype=bool)
            gate = None
            partitions: list[dict[str, Any]] = []
            phase = "stage2b_exact_d62_and_freeze_candidate"
            self._pending = _Pending(
                rows=x.copy(),
                labels=y.copy(),
                coefficient=coefficient.copy(),
                intercept=intercept.copy(),
                class_count=old_count,
                k_shot=int(k_shot),
            )
        else:
            pending = self._pending
            old_count = int(pending.class_count)
            appended_count = int(class_count) - old_count
            old_rows = x[: old_count * int(k_shot)]
            old_labels = y[: old_count * int(k_shot)]
            if (
                int(k_shot) != pending.k_shot
                or appended_count <= 0
                or not np.array_equal(old_rows, pending.rows)
                or not np.array_equal(old_labels, pending.labels)
            ):
                raise D70LifecycleError("D70 Stage2-B/Stage2-C lifecycle order drift")
            partitions = []
            if int(k_shot) == 1:
                mask = np.zeros(old_count, dtype=bool)
                gate = {
                    "base_positive": np.zeros(int(class_count), dtype=np.int64),
                    "base_false_positive": np.zeros(int(class_count), dtype=np.int64),
                    "coordinate_positive": np.zeros(old_count, dtype=np.int64),
                    "coordinate_false_positive": np.zeros(old_count, dtype=np.int64),
                    "joint_positive": np.zeros(int(class_count), dtype=np.int64),
                    "joint_false_positive": np.zeros(int(class_count), dtype=np.int64),
                    "initial_accept": mask.copy(),
                    "final_accept": mask.copy(),
                    "atomic_safe": True,
                    "status": "k1_exact_d62_fallback",
                    "exact_fallback": True,
                }
            else:
                base_scores: list[np.ndarray] = []
                before_scores: list[np.ndarray] = []
                truths: list[np.ndarray] = []
                for fold_index, held in enumerate(
                    twofold_rank_partitions(y, int(class_count), int(k_shot))
                ):
                    all_mask = np.ones(len(x), dtype=bool)
                    all_mask[held] = False
                    held_old = held[held < old_count * int(k_shot)]
                    old_mask = np.ones(len(pending.rows), dtype=bool)
                    old_mask[held_old] = False
                    train_old_x, train_old_y = pending.rows[old_mask], pending.labels[old_mask]
                    train_all_x, train_all_y = x[all_mask], y[all_mask]
                    old_train_k = int(len(train_old_x) // old_count)
                    all_train_k = int(len(train_all_x) // int(class_count))
                    before_coef, before_bias, _ = self._base_fit(
                        train_old_x, train_old_y, old_count, old_train_k
                    )
                    final_coef, final_bias, _ = self._base_fit(
                        train_all_x, train_all_y, int(class_count), all_train_k
                    )
                    self.inner_fit_count += 2
                    held_x = x[held]
                    base_scores.append(
                        held_x @ np.asarray(final_coef, dtype=np.float64).T
                        + np.asarray(final_bias, dtype=np.float64)[None, :]
                    )
                    before_scores.append(
                        held_x @ np.asarray(before_coef, dtype=np.float64).T
                        + np.asarray(before_bias, dtype=np.float64)[None, :]
                    )
                    truths.append(y[held])
                    partitions.append(
                        {
                            "fold_index": fold_index,
                            "held_indices": held.tolist(),
                            "held_old_indices": held_old.tolist(),
                            "train_all_count": int(np.sum(all_mask)),
                            "train_old_count": int(np.sum(old_mask)),
                            "old_train_k": old_train_k,
                            "all_train_k": all_train_k,
                            "train_held_overlap_count": int(
                                np.intersect1d(np.flatnonzero(all_mask), held).size
                            ),
                        }
                    )
                gate = atomic_old_row_gate(
                    np.concatenate(base_scores),
                    np.concatenate(before_scores),
                    np.concatenate(truths),
                )
                mask = np.asarray(gate["final_accept"], dtype=bool)
            coefficient = joint_coefficient.copy()
            intercept = joint_intercept.copy()
            accepted_indices = np.flatnonzero(mask)
            coefficient[accepted_indices] = pending.coefficient[accepted_indices]
            intercept[accepted_indices] = pending.intercept[accepted_indices]
            phase = "stage2c_atomic_old_row_replacement"
            self._pending = None
            self.completed_pairs += 1

        scores = x.astype(np.float32) @ coefficient.T + intercept[None, :]
        audit = dict(base_audit)
        audit.update(
            {
                "d70_formula": "twofold support-held atomic old-row replacement on D62 final joint head",
                "d70_phase": phase,
                "d70_actual_k": int(k_shot),
                "d70_class_count": int(class_count),
                "d70_old_class_count": old_count,
                "d70_appended_class_count": appended_count,
                "d70_partition_audit": partitions,
                "d70_base_positive_by_class": None if gate is None else gate["base_positive"].tolist(),
                "d70_base_false_positive_by_class": None if gate is None else gate["base_false_positive"].tolist(),
                "d70_coordinate_positive_by_old_class": None if gate is None else gate["coordinate_positive"].tolist(),
                "d70_coordinate_false_positive_by_old_class": None if gate is None else gate["coordinate_false_positive"].tolist(),
                "d70_joint_positive_by_class": None if gate is None else gate["joint_positive"].tolist(),
                "d70_joint_false_positive_by_class": None if gate is None else gate["joint_false_positive"].tolist(),
                "d70_initial_accept_mask": mask.tolist() if gate is None else gate["initial_accept"].tolist(),
                "d70_final_accept_mask": mask.tolist(),
                "d70_joint_atomic_safe": True if gate is None else gate["atomic_safe"],
                "d70_gate_status": "stage2b_not_applicable" if gate is None else gate["status"],
                "d70_exact_d62_fallback": not bool(np.any(mask)),
                "d70_compiled_support_accuracy": float(np.mean(np.argmax(scores, axis=1) == y)),
                "d70_new_rows_match_joint_d62": bool(
                    np.array_equal(coefficient[old_count:], joint_coefficient[old_count:])
                    and np.array_equal(intercept[old_count:], joint_intercept[old_count:])
                ),
                "d70_class_id_specific_formula": False,
                "d70_old_new_role_specific_query_branch": False,
                "d70_scene_receiver_handle_specific_branch": False,
                "d70_uses_outer_held_or_query": False,
                "d70_query_joint_optimization": False,
                "d70_hyperparameter_count": 0,
                "d70_ground_component_input_count": 0,
                "d70_single_affine_state_only": True,
                "d70_base_joint_coefficient_fp32": joint_coefficient.tolist(),
                "d70_base_joint_intercept_fp32": joint_intercept.tolist(),
                "d70_actual_coefficient_fp32": coefficient.tolist(),
                "d70_actual_intercept_fp32": intercept.tolist(),
            }
        )
        self.records.append(
            {
                "phase": phase,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "accepted_old_row_count": int(np.sum(mask)),
                "gate_status": audit["d70_gate_status"],
                "atomic_safe": audit["d70_joint_atomic_safe"],
                "partition_count": len(partitions),
            }
        )
        return coefficient, intercept, audit
