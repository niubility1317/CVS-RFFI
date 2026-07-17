from __future__ import annotations

import math

import pytest
import torch

from cvsrffi.phase1_geometry_streaming import (
    Phase1GeometryStreaming,
    Phase1GeometryStreamingError,
)


def _unit_rows(degrees: list[float]) -> torch.Tensor:
    radians = torch.tensor(degrees, dtype=torch.float64) * math.pi / 180.0
    return torch.stack((torch.cos(radians), torch.sin(radians)), dim=1).to(torch.float32)


def _feed_two_pass(
    aggregator: Phase1GeometryStreaming,
    rows: torch.Tensor,
    classes: torch.Tensor,
    domains: torch.Tensor,
) -> None:
    aggregator.update_first_pass(rows[:3], classes[:3], domains[:3])
    aggregator.update_first_pass(rows[3:], classes[3:], domains[3:])
    aggregator.begin_second_pass()
    aggregator.update_second_pass(rows[::2], classes[::2], domains[::2])
    aggregator.update_second_pass(rows[1::2], classes[1::2], domains[1::2])


def test_two_pass_centroids_and_p90_use_bounded_aggregate_state() -> None:
    class0 = _unit_rows([-20.0, -10.0, 0.0, 10.0, 20.0])
    class1 = _unit_rows([70.0, 80.0, 90.0, 100.0, 110.0])
    rows = torch.cat((class0, class1), dim=0)
    classes = torch.tensor([0] * 5 + [1] * 5, dtype=torch.int64)
    domains = torch.zeros(10, dtype=torch.int64)
    aggregator = Phase1GeometryStreaming(
        num_domains=1,
        num_classes=2,
        feature_dim=2,
        min_samples_per_cell=2,
        radius_histogram_bins=4096,
    )

    before = aggregator.bounded_accumulator_bytes
    _feed_two_pass(aggregator, rows, classes, domains)
    during = aggregator.bounded_accumulator_bytes
    result = aggregator.finalize()

    assert aggregator.state == "finalized"
    assert during > before
    assert result.domain_class_centroids.shape == (1, 2, 2)
    assert torch.allclose(
        result.domain_class_centroids[0],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        atol=1.0e-6,
    )
    exact_nearest_rank_p90 = 1.0 - math.cos(math.radians(20.0))
    observed = result.radius_p90_cosine_distance[0]
    assert bool(torch.all(observed >= exact_nearest_rank_p90))
    assert bool(
        torch.all(
            observed - exact_nearest_rank_p90
            <= result.radius_resolution_upper_bound + 1.0e-7
        )
    )
    assert torch.equal(result.domain_class_counts, torch.tensor([[5, 5]]))
    assert not hasattr(result, "sample_features")


def test_required_mask_allows_empty_slots_but_rejects_unauthorized_samples() -> None:
    required = torch.tensor([[True, True], [False, False]])
    aggregator = Phase1GeometryStreaming(
        num_domains=2,
        num_classes=2,
        feature_dim=2,
        required_cell_mask=required,
    )
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    classes = torch.tensor([0, 0])
    domains = torch.tensor([1, 1])
    with pytest.raises(Phase1GeometryStreamingError, match="non-authorized"):
        aggregator.update_first_pass(rows, classes, domains)


def test_begin_second_pass_fails_closed_on_missing_cell_coverage() -> None:
    aggregator = Phase1GeometryStreaming(
        num_domains=1, num_classes=2, feature_dim=2, min_samples_per_cell=2
    )
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    aggregator.update_first_pass(
        rows, torch.tensor([0, 0]), torch.tensor([0, 0])
    )
    with pytest.raises(Phase1GeometryStreamingError, match="lack minimum coverage"):
        aggregator.begin_second_pass()


@pytest.mark.parametrize(
    ("rows", "classes", "domains", "message"),
    [
        (
            torch.ones(2, 3),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            "shape",
        ),
        (
            torch.tensor([[1.0, 0.0], [float("nan"), 0.0]]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            "finite",
        ),
        (
            torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            "unit L2 norm",
        ),
        (
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0, 0]),
            "integer dtypes",
        ),
        (
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([0, 2]),
            torch.tensor([0, 0]),
            "outside the opaque registry",
        ),
    ],
)
def test_first_pass_rejects_invalid_batch_contract(
    rows: torch.Tensor,
    classes: torch.Tensor,
    domains: torch.Tensor,
    message: str,
) -> None:
    aggregator = Phase1GeometryStreaming(
        num_domains=1, num_classes=2, feature_dim=2
    )
    with pytest.raises(Phase1GeometryStreamingError, match=message):
        aggregator.update_first_pass(rows, classes, domains)


def test_state_machine_rejects_out_of_order_updates() -> None:
    aggregator = Phase1GeometryStreaming(
        num_domains=1, num_classes=1, feature_dim=2
    )
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0])
    with pytest.raises(Phase1GeometryStreamingError, match="frozen centroids"):
        aggregator.update_second_pass(rows, labels, labels)
    aggregator.update_first_pass(rows, labels, labels)
    aggregator.begin_second_pass()
    with pytest.raises(Phase1GeometryStreamingError, match="no longer allowed"):
        aggregator.update_first_pass(rows, labels, labels)


def test_finalize_rejects_second_pass_count_or_content_drift() -> None:
    aggregator = Phase1GeometryStreaming(
        num_domains=1, num_classes=1, feature_dim=2
    )
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0])
    aggregator.update_first_pass(first, labels, labels)
    aggregator.begin_second_pass()
    changed = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    aggregator.update_second_pass(changed, labels, labels)
    with pytest.raises(Phase1GeometryStreamingError, match="aggregate sums"):
        aggregator.finalize()


def test_finalize_rejects_incomplete_second_pass() -> None:
    aggregator = Phase1GeometryStreaming(
        num_domains=1, num_classes=1, feature_dim=2
    )
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0])
    aggregator.update_first_pass(rows, labels, labels)
    aggregator.begin_second_pass()
    aggregator.update_second_pass(rows[:1], labels[:1], labels[:1])
    with pytest.raises(Phase1GeometryStreamingError, match="cell counts"):
        aggregator.finalize()

