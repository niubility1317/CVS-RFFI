import math

import pytest
import torch

from model import SincConv1d


def _legacy_fp32_filters(layer: SincConv1d) -> torch.Tensor:
    t = layer.t_.float()
    window = layer.window_.float()
    nyq = layer.sample_rate / 2.0
    low = layer.min_low_hz + layer.low_hz_.float().abs()
    band = layer.min_band_hz + layer.band_hz_.float().abs()
    low = low.clamp(layer.min_low_hz, nyq - layer.min_band_hz - 1.0)
    high = torch.maximum(low + band, low + layer.min_band_hz)
    high = torch.minimum(high, torch.full_like(high, nyq - 1.0))
    numerator = torch.sin(2.0 * math.pi * high * t) - torch.sin(
        2.0 * math.pi * low * t
    )
    denominator = math.pi * t
    center = layer.kernel_size // 2
    safe_denominator = denominator.clone()
    safe_denominator[:, center] = 1.0
    bandpass = numerator / safe_denominator
    bandpass[:, center] = (2.0 * (high - low)).squeeze(1)
    bandpass = bandpass * window
    bandpass = bandpass / (bandpass.abs().amax(dim=1, keepdim=True) + 1e-8)
    return bandpass.view(layer.out_channels, 1, layer.kernel_size)


def test_sinc_filters_match_legacy_fp32_away_from_singular_center():
    layer = SincConv1d(out_channels=4, kernel_size=31, sample_rate=20_000.0)

    actual = layer._filters(torch.device("cpu"), torch.float32)
    expected = _legacy_fp32_filters(layer)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sinc_forward_and_parameter_gradients_are_finite_for_low_precision_inputs(dtype):
    layer = SincConv1d(out_channels=3, kernel_size=31, sample_rate=20_000.0)
    with torch.no_grad():
        layer.low_hz_[0] = 0.0
        layer.band_hz_[0] = 1.0e9
        layer.low_hz_[1] = 1.0e9
        layer.band_hz_[1] = 0.0
    x = torch.randn(2, 1, 96, dtype=dtype, requires_grad=True)

    y = layer(x)
    y.float().square().mean().backward()

    assert y.dtype == dtype
    assert torch.isfinite(y.float()).all()
    assert torch.isfinite(layer.low_hz_.grad).all()
    assert torch.isfinite(layer.band_hz_.grad).all()


def test_sinc_fp32_remains_finite_for_1000_optimizer_steps():
    torch.manual_seed(7)
    layer = SincConv1d(out_channels=2, kernel_size=15, sample_rate=20_000.0)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0e-3)
    x = torch.randn(1, 1, 32)

    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        loss = layer(x).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(layer.low_hz_.grad).all()
        assert torch.isfinite(layer.band_hz_.grad).all()
        optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GradScaler test")
def test_sinc_amp_high_grad_scaler_keeps_parameter_gradients_finite():
    layer = SincConv1d(out_channels=3, kernel_size=31, sample_rate=20_000.0).cuda()
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=65_536.0)
    x = (0.01 * torch.randn(2, 1, 96, device="cuda")).half()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = layer(x).float().square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    assert torch.isfinite(loss)
    assert torch.isfinite(layer.low_hz_.grad).all()
    assert torch.isfinite(layer.band_hz_.grad).all()
    scaler.step(optimizer)
    scaler.update()
