import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import energy_in_out_loss, negative_entropy_or_margin_loss, reject_negative_loss  # noqa: E402


def test_reject_energy_losses_support_empty_negative_batches():
    known = torch.tensor([[5.0, 1.0], [4.0, 0.5]], requires_grad=True)
    empty = known.new_zeros((0, 2), requires_grad=True)

    loss, metrics = energy_in_out_loss(known, empty, m_in=-2.0, m_out=2.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["negative_count"] == 0.0
    assert known.grad is not None


def test_reject_negative_loss_and_margin_loss_penalize_known_confident_negatives():
    neg_logits = torch.tensor([[4.0, 0.1, -1.0], [0.1, 4.0, -1.0]], requires_grad=True)
    reject_logits = torch.tensor([[0.1, 0.2, 4.0], [0.2, 0.1, 4.0]], requires_grad=True)

    bad_loss, _ = reject_negative_loss(neg_logits, reject_class_index=2)
    good_loss, _ = reject_negative_loss(reject_logits, reject_class_index=2)
    entropy_loss, _ = negative_entropy_or_margin_loss(neg_logits[:, :2], max_known_prob=0.4)
    (bad_loss + entropy_loss).backward()

    assert bad_loss > good_loss
    assert entropy_loss > 0.0
    assert neg_logits.grad is not None

