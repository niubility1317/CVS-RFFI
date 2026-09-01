from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from cvsrffi.phase1_fcr_types import FCRConfig


def test_content_encoder_produces_temporal_tokens_complex_reconstruction_and_confidence() -> None:
    """Stride-four content factors keep 64 tokens before any summary is formed."""

    from cvsrffi.phase1_fcr_factors import ContentFactorEncoder

    config = FCRConfig()
    content_model = ContentFactorEncoder(config)
    output = content_model(torch.randn(3, 2, config.input_len))

    assert output.z_s.shape == (3, 64, 32)
    assert output.s_hat.shape == (3, config.input_len)
    assert output.s_hat.dtype == torch.complex64
    assert torch.isfinite(output.content_confidence).all()
    assert torch.all((output.content_confidence >= 0.0) & (output.content_confidence <= 1.0))


def test_masked_content_reconstruction_remains_differentiable() -> None:
    """Masking source samples must not sever reconstruction gradients."""

    from cvsrffi.phase1_fcr_factors import ContentFactorEncoder

    config = FCRConfig()
    content_model = ContentFactorEncoder(config)
    input_iq = torch.randn(2, 2, config.input_len)
    mask = torch.zeros(2, config.input_len, dtype=torch.bool)
    mask[:, ::5] = True

    output = content_model(input_iq, mask=mask)
    target = torch.complex(input_iq[:, 0], input_iq[:, 1])
    loss = (output.s_hat[mask] - target[mask]).abs().square().mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in content_model.parameters())


def test_identity_input_is_detached_by_default() -> None:
    """Identity CE may read the content summary without training the content path."""

    from cvsrffi.phase1_fcr_factors import ContentFactorEncoder

    config = FCRConfig()
    content_model = ContentFactorEncoder(config, detach_identity_input=True)
    output = content_model(torch.randn(4, 2, config.input_len))
    tx_head = nn.Linear(config.content_dim, 3)
    labels = torch.tensor([0, 1, 2, 1])

    tx_ce = functional.cross_entropy(tx_head(content_model.identity_input(output.z_s)), labels)
    tx_ce.backward()

    assert all(parameter.grad is None for parameter in content_model.parameters())
