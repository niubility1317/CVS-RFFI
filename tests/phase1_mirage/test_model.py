"""Behavioral checks for the lightweight MIRAGE IQ encoder boundary."""

from __future__ import annotations

import dataclasses
import importlib

import pytest
import torch
from torch import nn


def _model_api():
    """Import the Task 4 API inside tests so RED proves the missing module."""

    try:
        module = importlib.import_module("cvsrffi.phase1_mirage.model")
    except ModuleNotFoundError as error:
        if error.name == "cvsrffi.phase1_mirage.model":
            pytest.fail("missing lightweight MIRAGE IQ encoder module")
        raise
    return module.MIRAGEConfig, module.MIRAGEEncoder, module.preprocess_iq


def test_encoder_outputs_finite_normalized_features_under_budget():
    """Catch a missing/non-normalized encoder or a deployment-budget regression."""

    MIRAGEConfig, MIRAGEEncoder, _ = _model_api()
    config = MIRAGEConfig()
    model = MIRAGEEncoder(config)

    out = model(torch.randn(4, 2, 256))

    assert out.z_id.shape == (4, 160)
    assert out.z_dom.shape == (4, 32)
    assert out.quality.shape == (4,)
    assert out.tokens.shape == (4, 15, 192)
    assert torch.allclose(out.z_id.norm(dim=1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(out.z_id).all()
    assert torch.isfinite(out.z_dom).all()
    assert torch.isfinite(out.quality).all()
    assert torch.isfinite(out.tokens).all()
    assert torch.all((out.quality >= 0.0) & (out.quality <= 1.0))
    assert sum(parameter.numel() for parameter in model.parameters()) <= 3_000_000


def test_encoder_supports_cpu_length_512_with_the_expected_patch_count():
    """Catch a patching path that only works for the shortest deployment window."""

    MIRAGEConfig, MIRAGEEncoder, _ = _model_api()
    config = MIRAGEConfig()
    model = MIRAGEEncoder(config).cpu()

    out = model(torch.randn(2, 2, 512, device="cpu"))

    expected_tokens = (512 - config.patch_kernel) // config.patch_stride + 1
    assert out.tokens.shape == (2, expected_tokens, config.token_dim)
    assert torch.allclose(out.z_id.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(out.quality).all()


def test_preprocess_sanitizes_nonfinite_values_at_the_external_iq_boundary():
    """Catch omission of the sole permitted input-boundary numeric cleanup."""

    _, _, preprocess_iq = _model_api()
    iq = torch.tensor(
        [[[0.0, float("nan"), float("inf"), -1.0], [1.0, float("-inf"), 2.0, 3.0]]]
    )

    normalized, quality_aux = preprocess_iq(iq)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(quality_aux).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"patch_kernel": 0}, "patch_kernel"),
        ({"patch_stride": 0}, "patch_stride"),
        ({"token_dim": 190, "transformer_heads": 4}, "divisible"),
        ({"transformer_layers": 0}, "transformer_layers"),
        ({"z_id_dim": 0}, "z_id_dim"),
        ({"z_dom_dim": 0}, "z_dom_dim"),
    ),
)
def test_invalid_encoder_configuration_fails_closed(overrides: dict[str, int], message: str):
    """Catch malformed frozen architecture settings before a model can be created."""

    MIRAGEConfig, _, _ = _model_api()

    with pytest.raises(ValueError, match=message):
        MIRAGEConfig(**overrides)


def test_encoder_configuration_is_immutable_after_construction():
    """Catch mutable deployment architecture settings after they have been frozen."""

    MIRAGEConfig, _, _ = _model_api()
    config = MIRAGEConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.token_dim = 64


@pytest.mark.parametrize(
    ("iq", "error_type", "message"),
    (
        (torch.randn(2, 256), ValueError, r"\[B, 2, T\]"),
        (torch.randn(2, 1, 256), ValueError, r"\[B, 2, T\]"),
        (torch.randn(0, 2, 256), ValueError, "non-empty"),
        (torch.randn(2, 2, 31), ValueError, "patch_kernel"),
        (torch.ones(2, 2, 256, dtype=torch.int64), TypeError, "floating"),
    ),
)
def test_invalid_iq_inputs_fail_closed(iq: torch.Tensor, error_type: type[Exception], message: str):
    """Catch malformed IQ tensors before preprocessing or learned computation begins."""

    MIRAGEConfig, MIRAGEEncoder, _ = _model_api()
    model = MIRAGEEncoder(MIRAGEConfig())

    with pytest.raises(error_type, match=message):
        model(iq)


class _InjectNonFiniteIdentityHead(nn.Module):
    """Controlled test double that makes the formal finite-output guard observable."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (features.shape[0], 160), float("inf"), dtype=features.dtype, device=features.device
        )


def test_formal_mode_raises_for_a_controlled_internal_nonfinite_output():
    """Catch a formal-mode regression that silently passes a corrupt internal tensor."""

    MIRAGEConfig, MIRAGEEncoder, _ = _model_api()
    model = MIRAGEEncoder(MIRAGEConfig(formal_mode=True))
    model.identity_head = _InjectNonFiniteIdentityHead()

    with pytest.raises(FloatingPointError, match="identity_head"):
        model(torch.randn(2, 2, 256))
