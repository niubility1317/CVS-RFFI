import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.ccoi_losses import (  # noqa: E402
    ccoi_supcon_loss,
    challenge_pair_masks,
    conditional_distance_diagnostics,
)


def test_challenge_masks_select_cross_domain_positive_and_same_domain_negative():
    y = torch.tensor([0, 0, 1, 1])
    domain = torch.tensor([0, 1, 0, 1])
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    masks = challenge_pair_masks(q, y, domain, min_cosine=0.99)

    assert masks.positive[0, 1]
    assert masks.negative[0, 2]
    assert not masks.negative[0, 3]


def test_supcon_empty_pairs_returns_differentiable_zero_and_counts():
    theta = torch.randn(3, 5, requires_grad=True)
    y = torch.tensor([0, 1, 2])
    domain = torch.zeros(3, dtype=torch.long)
    q = torch.eye(3)

    result = ccoi_supcon_loss(theta, challenge_pair_masks(q, y, domain, min_cosine=0.99))
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.positive_count == 0
    assert theta.grad is not None


def test_three_distance_diagnostic_recovers_literal_ordering():
    response = torch.tensor([[0.0], [0.1], [1.0], [3.0]])
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = torch.tensor([0, 0, 0, 1])
    domain = torch.tensor([0, 1, 1, 0])

    diag = conditional_distance_diagnostics(response, q, y, domain, min_cosine=0.99)

    assert diag["d1_count"] > 0 and diag["d2_count"] > 0 and diag["d3_count"] > 0
    assert diag["d1"] < diag["d2"] < diag["d3"]
    assert not math.isnan(diag["d1"])
