from __future__ import annotations

import pytest
import torch

from cvsrffi.slow_fast_adapter import (
    SlowFastAdapterState,
    SlowFastCandidate,
    apply_slow_fast,
)


def test_common_shift_subtracts_hand_computed_domain_offset() -> None:
    state = SlowFastAdapterState(
        candidate=SlowFastCandidate.COMMON_SHIFT_R4,
        slow_u=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        common_coeff=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    actual = apply_slow_fast(torch.tensor([[2.0, 2.0]]), state)

    expected = torch.tensor([[1.0, 2.0]])
    expected = expected / torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("candidate", "expected_fast_parameters"),
    [
        (SlowFastCandidate.FAST_FILM_R8, 16),
        (SlowFastCandidate.FAST_LOWRANK_R8, 24),
    ],
)
def test_rank8_candidates_expose_only_the_preregistered_fast_parameters(
    candidate: SlowFastCandidate,
    expected_fast_parameters: int,
) -> None:
    state = SlowFastAdapterState(
        candidate=candidate,
        slow_u=torch.eye(8, dtype=torch.float32).repeat(20, 1),
        slow_v=torch.flip(torch.eye(8, dtype=torch.float32), dims=(1,)).repeat(20, 1),
        rho=0.1,
        gamma=torch.zeros(8),
        beta=torch.zeros(8),
        direction_gate=torch.zeros(8)
        if candidate is SlowFastCandidate.FAST_LOWRANK_R8
        else None,
    )

    assert state.feature_dim == 160
    assert state.rank == 8
    assert state.fast_parameter_count == expected_fast_parameters
    output = apply_slow_fast(torch.randn(3, 160), state)
    assert output.shape == (3, 160)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=1), torch.ones(3))


def test_adapter_rejects_feature_width_that_differs_from_frozen_prototypes() -> None:
    state = SlowFastAdapterState(
        candidate=SlowFastCandidate.COMMON_SHIFT_R4,
        slow_u=torch.zeros(160, 4),
        common_coeff=torch.zeros(4),
    )

    with pytest.raises(ValueError, match="feature width"):
        apply_slow_fast(torch.zeros(2, 256), state)
