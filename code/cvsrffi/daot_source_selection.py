from __future__ import annotations

import math
from typing import Sequence


def build_receiver_holdout_folds(receiver_ids: Sequence[int]) -> list[dict[str, object]]:
    receivers = tuple(dict.fromkeys(int(value) for value in receiver_ids))
    if len(receivers) < 3:
        raise ValueError("receiver-held-out selection requires at least three source receivers")
    return [
        {
            "holdout_receiver": holdout,
            "train_receivers": tuple(value for value in receivers if value != holdout),
        }
        for holdout in receivers
    ]


def source_only_selection_score(
    *,
    cvar20: float,
    receiver_floor: float,
    clean_accuracy: float,
    leo_weak_mean: float,
    receiver_probe: float,
    relative_cost: float,
) -> float:
    if min(clean_accuracy, leo_weak_mean, cvar20, receiver_floor) < 0.0 or float(relative_cost) <= 0.0:
        raise ValueError("invalid source-only selection metrics")
    harmonic = 2.0 * float(clean_accuracy) * float(leo_weak_mean) / max(
        float(clean_accuracy) + float(leo_weak_mean), 1e-12
    )
    return (
        0.30 * float(cvar20)
        + 0.30 * float(receiver_floor)
        + 0.30 * harmonic
        - 0.05 * float(receiver_probe)
        - 0.05 * math.log(float(relative_cost))
    )


def allocate_structured_batch(*, batch_size: int, receiver_count: int) -> dict[str, int]:
    if int(receiver_count) < 3:
        raise ValueError("structured DAOT batches require at least three receivers")
    if int(batch_size) < 10:
        raise ValueError("structured DAOT batch is too small")
    cross_rx = int(round(0.30 * int(batch_size)))
    balanced_u = int(round(0.45 * int(batch_size)))
    hard = int(round(0.15 * int(batch_size)))
    return {
        "cross_rx_labeled": cross_rx,
        "balanced_unlabeled": balanced_u,
        "hard_group": hard,
        "base_remainder": int(batch_size) - cross_rx - balanced_u - hard,
        "receiver_count": int(receiver_count),
    }
