from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_fcr_types import FCRDecodeOutput  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


FEATURE_SCHEMA = "ADV3B02:FCR:z_f_id:unit_l2:160:v1"


def _small_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        **kwargs,
    )


def test_fcr_enabled_single_view_outputs_exact_feature_contract() -> None:
    """Missing or misrouted FCR composition must break the public aux schema."""

    torch.manual_seed(1909)
    legacy = _small_model(use_fcr=False)
    torch.manual_seed(1909)
    model = _small_model(use_fcr=True)
    legacy.eval()
    model.eval()

    assert model.fcr is not None
    assert model.fcr_config is not None
    assert model.fcr_config.input_len == 64
    assert model.fcr_config.content_stride == 4
    assert model.fcr_config.content_dim == 32
    assert model.fcr_config.tx_state_dim == 16
    assert any(key.startswith("fcr.") for key in model.state_dict())
    assert "clean_companion" not in inspect.signature(model.forward).parameters
    assert "clean_companion" not in inspect.signature(model.fcr.forward).parameters

    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        legacy_out = legacy(x, return_aux=True)
        assert legacy_out["z_id"].shape == (2, 160)
        out = model(x, return_aux=True)

    torch.testing.assert_close(out["tx_logits"], legacy_out["tx_logits"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(out["z_id"], legacy_out["z_id"], rtol=0.0, atol=0.0)
    assert out["z_id_raw"] is out["z_id"]
    assert out["z_id_raw"].shape == (2, 160)
    assert out["z_f_id"].shape == (2, 160)
    assert out["z_tx_state"].shape == (2, 16)
    assert out["z_s"].shape == (2, 16, 32)
    assert set(out["z_n"]) == {"channel", "receiver", "sync", "gain"}
    assert out["z_n"]["channel"].shape == (2, 16)
    assert out["z_n"]["receiver"].shape == (2, 8)
    assert out["z_n"]["sync"].shape == (2, 6)
    assert out["z_n"]["gain"].shape == (2, 3)
    assert isinstance(out["fcr_decode"], FCRDecodeOutput)
    assert out["fcr_decode"].mu_iq.shape == (2, 2, 64)
    assert out["fcr_decode"].log_variance.shape == (2, 64)
    assert out["fcr_decode"].delta_f.shape == (2, 64)
    assert out["feature_schema"] == FEATURE_SCHEMA

    torch.testing.assert_close(
        out["z_f_id"].norm(dim=1), torch.ones(2), rtol=1e-5, atol=1e-6
    )
    for value in (
        out["z_f_id"],
        out["z_tx_state"],
        out["z_s"],
        *out["z_n"].values(),
        out["fcr_decode"].mu_iq,
        out["fcr_decode"].log_variance,
        out["fcr_decode"].delta_f.real,
        out["fcr_decode"].delta_f.imag,
        *out["fcr_quality"].values(),
    ):
        assert torch.isfinite(value).all()


def test_fcr_amp_precision_island_keeps_complex_path_complex64() -> None:
    """Removing the precision island must expose unsupported low-precision complex ops."""

    torch.manual_seed(3909)
    model = _small_model(use_fcr=True)
    assert model.fcr is not None
    model.fcr.train()
    x = torch.randn(2, 2, 64)
    id_feature_raw = torch.randn(2, 160, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        aggregate = model.fcr(x, id_feature_raw)

    assert aggregate.content.s_hat.dtype == torch.complex64
    assert aggregate.factors.response_coef.dtype == torch.complex64
    assert aggregate.decode.delta_f.dtype == torch.complex64
    assert aggregate.decode.mu_iq.dtype == torch.float32
    objective = (
        aggregate.decode.mu_iq.square().mean()
        + aggregate.factors.z_f_id.square().mean()
        + aggregate.factors.z_s.square().mean()
    )
    objective.backward()
    assert id_feature_raw.grad is not None
    assert torch.isfinite(id_feature_raw.grad).all()


def test_fcr_direct_cross_decode_keeps_complex_path_complex64_under_autocast() -> None:
    """Pair-objective direct module calls must not bypass the AMP precision boundary."""

    torch.manual_seed(4909)
    model = _small_model(use_fcr=True)
    assert model.fcr is not None
    x = torch.randn(2, 2, 64)
    id_feature_raw = torch.randn(2, 160, requires_grad=True)
    aggregate = model.fcr(x, id_feature_raw)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        response = model.fcr.fingerprint_operator(
            aggregate.content.s_hat.detach(), aggregate.fingerprint
        )
        decoded = model.fcr.decoder(
            aggregate.content.s_hat, response.delta_f, aggregate.nuisance
        )

    assert response.response_coef.dtype == torch.complex64
    assert response.delta_f.dtype == torch.complex64
    assert decoded.delta_f.dtype == torch.complex64
    assert decoded.mu_iq.dtype == torch.float32
    decoded.mu_iq.square().mean().backward()
    gradients = [
        parameter.grad
        for module in (model.fcr.fingerprint_operator, model.fcr.decoder)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_fcr_enabled_backward_reaches_fcr_and_raw_identity_path() -> None:
    """Disconnected FCR factors or an overwritten raw identity path must fail."""

    torch.manual_seed(2909)
    model = _small_model(use_fcr=True)
    model.train()
    x = torch.randn(2, 2, 64)
    out = model(x, return_aux=True)
    out["z_id_raw"].retain_grad()
    objective = (
        out["z_f_id"][:, 0].mean()
        + out["z_tx_state"].square().mean()
        + out["z_s"].square().mean()
        + out["fcr_decode"].mu_iq.square().mean()
        + out["fcr_decode"].log_variance.square().mean()
    )
    objective.backward()

    assert out["z_id_raw"].grad is not None
    assert torch.isfinite(out["z_id_raw"].grad).all()
    assert out["z_id_raw"].grad.abs().sum() > 0
    fcr_gradients = [
        parameter.grad
        for parameter in model.fcr.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert fcr_gradients
    assert all(torch.isfinite(gradient).all() for gradient in fcr_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in fcr_gradients) > 0.0
    upstream_gradients = [
        parameter.grad
        for parameter in model.id_backbone.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert upstream_gradients
    assert all(torch.isfinite(gradient).all() for gradient in upstream_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in upstream_gradients) > 0.0
