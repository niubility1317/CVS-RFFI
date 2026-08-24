import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.ccoi_pa import (  # noqa: E402
    PAChallengeEncoder,
    codebook_balance_regularizer,
    challenge_pretrain_losses,
    fixed_content_statistics,
    make_dual_iq_views,
    tokenize_iq,
)


def test_tokenization_has_thirteen_tokens_and_fingerprint_is_unchanged():
    x = torch.randn(3, 2, 256)

    content, fingerprint = make_dual_iq_views(x)

    assert torch.equal(fingerprint, x)
    assert tokenize_iq(content, 64, 16).shape == (3, 13, 2, 64)


def test_fixed_content_targets_are_detached_and_finite():
    x = torch.randn(2, 2, 256, requires_grad=True)

    stats = fixed_content_statistics(tokenize_iq(x, 64, 16))

    assert not stats.requires_grad
    assert stats.shape[:2] == (2, 13)
    assert torch.isfinite(stats).all()


def test_challenge_pretraining_reaches_encoder_but_not_fixed_targets():
    encoder = PAChallengeEncoder(token_length=64, stride=16, q_dim=32, codebook_size=48)
    clean = torch.randn(4, 2, 256)
    satellite = clean + 0.05 * torch.randn_like(clean)

    out = encoder(make_dual_iq_views(clean)[0])
    losses = challenge_pretrain_losses(encoder, clean, satellite)
    losses["total"].backward()

    assert out.q.shape == (4, 13, 32)
    assert out.code_prob.shape == (4, 13, 48)
    assert not out.content_stats.requires_grad
    assert torch.isfinite(losses["total"])
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in encoder.parameters())
    assert encoder.code_head.weight.grad is not None
    assert torch.count_nonzero(encoder.code_head.weight.grad)
    assert "code_consistency" in losses and "code_utilization" in losses


def test_codebook_balance_regularizer_targets_collapse_without_forcing_exact_uniformity():
    collapsed = torch.zeros(4, 13, 48)
    collapsed[..., 0] = 1.0
    broad = torch.zeros(4, 13, 48)
    broad[..., :36] = 1.0 / 36.0

    collapsed_loss, collapsed_stats = codebook_balance_regularizer(collapsed)
    broad_loss, broad_stats = codebook_balance_regularizer(broad)

    assert collapsed_loss > broad_loss
    assert collapsed_stats["effective_codes"] < 2.0
    assert broad_stats["effective_codes"] >= 35.9
    assert broad_loss.item() == 0.0
