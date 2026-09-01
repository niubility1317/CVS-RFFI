from __future__ import annotations

import pytest
import torch

from cvsrffi.nmfdu_multiburst import aggregate_independent_bursts


def test_multiburst_uses_q_weighted_fingerprint_and_additive_fisher() -> None:
    fingerprints = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    q_sample = torch.tensor([0.8, 0.2, 0.0])
    fisher = {
        "raw": torch.stack([torch.eye(2), 2.0 * torch.eye(2), 3.0 * torch.eye(2)]),
        "pa": torch.stack(
            [torch.diag(torch.tensor([1.0, 0.0])), torch.eye(2), torch.zeros(2, 2)]
        ),
    }

    result = aggregate_independent_bursts(
        fingerprints,
        q_sample,
        physical_sample_ids=["burst-a", "burst-b", "burst-c"],
        branch_fisher=fisher,
    )

    assert torch.allclose(result["fingerprint"], torch.tensor([0.8, 0.2]))
    assert torch.allclose(result["branch_fisher"]["raw"], 6.0 * torch.eye(2))
    assert torch.allclose(
        result["branch_fisher"]["pa"], torch.diag(torch.tensor([2.0, 1.0]))
    )
    assert result["physical_burst_count"] == 3
    assert result["effective_burst_count"] == pytest.approx(1.0)


def test_multiburst_rejects_duplicate_physical_ids_even_for_distinct_views() -> None:
    with pytest.raises(ValueError, match="independent physical samples"):
        aggregate_independent_bursts(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0.5, 0.5]),
            physical_sample_ids=["same-iq", "same-iq"],
        )


def test_multiburst_does_not_recover_a_shared_structural_null_direction() -> None:
    rank_one = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    result = aggregate_independent_bursts(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.ones(3),
        physical_sample_ids=["a", "b", "c"],
        branch_fisher={"raw": torch.stack([rank_one, rank_one, rank_one])},
    )

    eigenvalues = torch.linalg.eigvalsh(result["branch_fisher"]["raw"])
    assert torch.allclose(eigenvalues, torch.tensor([0.0, 6.0]))
    assert result["branch_rank"]["raw"] == 1


def test_multiburst_rejects_invalid_shapes_or_quality() -> None:
    with pytest.raises(ValueError, match="q_sample"):
        aggregate_independent_bursts(
            torch.ones(2, 3),
            torch.tensor([0.5]),
            physical_sample_ids=["a", "b"],
        )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        aggregate_independent_bursts(
            torch.ones(2, 3),
            torch.tensor([0.5, 1.2]),
            physical_sample_ids=["a", "b"],
        )
