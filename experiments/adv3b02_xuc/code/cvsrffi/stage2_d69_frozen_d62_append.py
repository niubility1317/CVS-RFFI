"""Frozen Stage2-B D62 rows with append-only Stage2-C D62 new rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import numpy as np


class D69LifecycleError(RuntimeError):
    """Raised when the Stage2-B/Stage2-C registration lifecycle is invalid."""


Fit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _sha256_rows(coefficient: np.ndarray, intercept: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coefficient, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(intercept, dtype=np.float32).tobytes())
    return digest.hexdigest()


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
        raise D69LifecycleError("D69 requires finite exact symmetric support")
    return x, y


@dataclass
class _Pending:
    coefficient: np.ndarray
    intercept: np.ndarray
    class_count: int
    k_shot: int
    row_sha256: str


class FrozenD62AppendLifecycle:
    """Pair Stage2-B and Stage2-C fits without changing either D62 formula."""

    def __init__(self, base_fit: Fit) -> None:
        self._base_fit = base_fit
        self._pending: _Pending | None = None
        self.completed_pairs = 0
        self.records: list[dict[str, Any]] = []

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
        d62_coefficient, d62_intercept, d62_audit = self._base_fit(
            x, y, int(class_count), int(k_shot)
        )
        joint_coefficient = np.asarray(d62_coefficient, dtype=np.float32)
        joint_intercept = np.asarray(d62_intercept, dtype=np.float32)
        if joint_coefficient.shape != (int(class_count), x.shape[1]) or joint_intercept.shape != (
            int(class_count),
        ):
            raise D69LifecycleError("D69 D62 affine shape drift")
        if not np.isfinite(joint_coefficient).all() or not np.isfinite(joint_intercept).all():
            raise D69LifecycleError("D69 D62 affine state became non-finite")

        if self._pending is None:
            coefficient = joint_coefficient.copy()
            intercept = joint_intercept.copy()
            before_sha = _sha256_rows(coefficient, intercept)
            self._pending = _Pending(
                coefficient=coefficient.copy(),
                intercept=intercept.copy(),
                class_count=int(class_count),
                k_shot=int(k_shot),
                row_sha256=before_sha,
            )
            phase = "stage2b_d62_fit_and_freeze"
            old_count, new_count = int(class_count), 0
            old_unchanged = True
            new_matches_joint = True
        else:
            pending = self._pending
            old_count = int(pending.class_count)
            new_count = int(class_count) - old_count
            old_labels = y[: old_count * int(k_shot)]
            new_labels = y[old_count * int(k_shot) :]
            if (
                int(k_shot) != pending.k_shot
                or new_count <= 0
                or not np.array_equal(np.unique(old_labels), np.arange(old_count))
                or not np.array_equal(
                    np.unique(new_labels), np.arange(old_count, int(class_count))
                )
            ):
                raise D69LifecycleError("D69 Stage2-B/Stage2-C lifecycle order drift")
            coefficient = np.concatenate(
                [pending.coefficient, joint_coefficient[old_count:]], axis=0
            ).astype(np.float32, copy=False)
            intercept = np.concatenate(
                [pending.intercept, joint_intercept[old_count:]], axis=0
            ).astype(np.float32, copy=False)
            old_unchanged = bool(
                np.array_equal(coefficient[:old_count], pending.coefficient)
                and np.array_equal(intercept[:old_count], pending.intercept)
            )
            new_matches_joint = bool(
                np.array_equal(coefficient[old_count:], joint_coefficient[old_count:])
                and np.array_equal(intercept[old_count:], joint_intercept[old_count:])
            )
            if not old_unchanged or not new_matches_joint:
                raise D69LifecycleError("D69 append-only row identity drift")
            before_sha = pending.row_sha256
            phase = "stage2c_append_d62_joint_new_rows"
            self._pending = None
            self.completed_pairs += 1

        actual_sha = _sha256_rows(coefficient, intercept)
        joint_sha = _sha256_rows(joint_coefficient, joint_intercept)
        scores = x.astype(np.float32) @ coefficient.T + intercept[None, :]
        audit = dict(d62_audit)
        audit.update(
            {
                "d69_formula": "freeze D62 Stage2-B old rows and append D62 Stage2-C joint new rows",
                "d69_phase": phase,
                "d69_actual_k": int(k_shot),
                "d69_class_count": int(class_count),
                "d69_old_class_count": old_count,
                "d69_appended_class_count": new_count,
                "d69_before_old_row_sha256": before_sha,
                "d69_joint_d62_row_sha256": joint_sha,
                "d69_actual_row_sha256": actual_sha,
                "d69_old_row_fp32_bitwise_unchanged": old_unchanged,
                "d69_new_row_fp32_matches_joint_d62": new_matches_joint,
                "d69_compiled_support_accuracy": float(
                    np.mean(np.argmax(scores, axis=1) == y)
                ),
                "d69_class_id_specific_formula": False,
                "d69_old_new_role_specific_query_branch": False,
                "d69_scene_receiver_handle_specific_branch": False,
                "d69_uses_outer_held_or_query": False,
                "d69_query_joint_optimization": False,
                "d69_hyperparameter_count": 0,
                "d69_ground_component_input_count": 0,
                "d69_single_affine_state_only": True,
                "d69_joint_d62_coefficient_fp32": joint_coefficient.tolist(),
                "d69_joint_d62_intercept_fp32": joint_intercept.tolist(),
                "d69_actual_coefficient_fp32": coefficient.tolist(),
                "d69_actual_intercept_fp32": intercept.tolist(),
            }
        )
        self.records.append(
            {
                "phase": phase,
                "class_count": int(class_count),
                "old_class_count": old_count,
                "appended_class_count": new_count,
                "before_old_row_sha256": before_sha,
                "joint_d62_row_sha256": joint_sha,
                "actual_row_sha256": actual_sha,
                "old_row_fp32_bitwise_unchanged": old_unchanged,
                "new_row_fp32_matches_joint_d62": new_matches_joint,
            }
        )
        return coefficient, intercept, audit
