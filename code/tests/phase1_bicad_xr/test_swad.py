from __future__ import annotations

import math

import pytest
import torch

from cvsrffi.phase1_bicad_xr.swad import SWADAccumulator


def _state(weight: float, batches: int) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([weight, weight + 2.0], dtype=torch.float32),
        "num_batches_tracked": torch.tensor(batches, dtype=torch.int64),
    }


def test_swad_admits_only_near_best_score_without_floor_regression() -> None:
    accumulator = SWADAccumulator()

    assert accumulator.consider(
        _state(1.0, 10),
        score=0.800,
        clean_floor=0.700,
        leo_floor=0.600,
        receiver_floor=0.650,
    )
    assert accumulator.consider(
        _state(3.0, 12),
        score=0.797,
        clean_floor=0.696,
        leo_floor=0.596,
        receiver_floor=0.646,
    )
    assert not accumulator.consider(
        _state(9.0, 14),
        score=0.794,
        clean_floor=0.700,
        leo_floor=0.600,
        receiver_floor=0.650,
    )
    assert not accumulator.consider(
        _state(9.0, 16),
        score=0.799,
        clean_floor=0.700,
        leo_floor=0.594,
        receiver_floor=0.650,
    )

    assert accumulator.window_size == 2


def test_swad_averages_floating_tensors_and_preserves_latest_integer_buffer() -> None:
    accumulator = SWADAccumulator()
    for state, score in ((_state(1.0, 10), 0.800), (_state(3.0, 12), 0.799)):
        assert accumulator.consider(
            state,
            score=score,
            clean_floor=0.700,
            leo_floor=0.600,
            receiver_floor=0.650,
        )

    averaged = accumulator.averaged_state_dict()

    assert torch.equal(averaged["weight"], torch.tensor([2.0, 4.0]))
    assert averaged["weight"].dtype == torch.float32
    assert torch.equal(averaged["num_batches_tracked"], torch.tensor(12))
    assert averaged["num_batches_tracked"].dtype == torch.int64


def test_swad_rejects_nonfinite_metrics_and_tensors() -> None:
    accumulator = SWADAccumulator()

    with pytest.raises(ValueError, match="finite"):
        accumulator.consider(
            _state(1.0, 10),
            score=math.nan,
            clean_floor=0.7,
            leo_floor=0.6,
            receiver_floor=0.65,
        )

    bad_state = _state(1.0, 10)
    bad_state["weight"][0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        accumulator.consider(
            bad_state,
            score=0.8,
            clean_floor=0.7,
            leo_floor=0.6,
            receiver_floor=0.65,
        )


def test_empty_swad_window_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        SWADAccumulator().averaged_state_dict()
