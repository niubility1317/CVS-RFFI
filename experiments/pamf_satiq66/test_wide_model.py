import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_satiq66 import wide_model


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_raw_tokens_cover_both_ends_of_each_symbol_in_order():
    encoder = wide_model.RawSymbolEncoder()
    raw = torch.zeros(16, 2, 8000)
    symbols = torch.arange(8).repeat_interleave(2)
    offsets = torch.tensor([0, 999]).repeat(8)
    raw[torch.arange(16), 0, 1000 * symbols + offsets] = 10

    reference = encoder.symbol_tokens(torch.zeros_like(raw))
    tokens = encoder.symbol_tokens(raw)
    affected = (tokens - reference).abs().amax(-1) > 1e-5
    assert torch.equal(affected, F.one_hot(symbols, num_classes=8).bool())


def test_paper_raw_tokens_keep_positions_within_all_eleven_symbols():
    encoder = wide_model.RawSymbolEncoder(symbol_count=11, bounded=False, time_bins=5)
    raw = torch.zeros(11, 2, 11000)
    symbols = torch.arange(11)
    raw[symbols, 0, 1000 * symbols + 970] = 10
    tokens = encoder.symbol_tokens(raw)
    assert tokens.shape == (11, 11, 320)
    assert torch.equal((tokens - encoder.symbol_tokens(torch.zeros_like(raw))).abs().amax(-1) > 1e-5,
                       F.one_hot(symbols, num_classes=11).bool())
    shifted = torch.roll(raw, -900, dims=2)
    assert not torch.allclose(tokens, encoder.symbol_tokens(shifted))
    assert encoder(raw).shape == (11, 160)


class _SyntheticNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(512, 160)
        self.head = nn.Linear(160, 66, bias=False)

    def features(self, coarse):
        return self.projection(coarse.flatten(1))

    def classify_features(self, features):
        return self.head(features)

    def forward(self, coarse):
        return self.classify_features(self.features(coarse))


def test_wide_fused_head_trains_raw_encoder_and_retains_native_branches(monkeypatch):
    # Stand in for the immutable historical PAMF release in this unit test.
    native = _SyntheticNative()

    def capture(model, coarse):
        features = model.features(coarse)
        branches = torch.stack([features + i for i in range(5)])
        return model.classify_features(features), features, branches

    monkeypatch.setattr(wide_model, "build_model", lambda runtime, seed, representation: native)
    monkeypatch.setattr(wide_model, "capture_native", capture)
    raw = torch.randn(2, 2, 8000, requires_grad=True)
    coarse = torch.randn(2, 2, 256)
    control = wide_model.build_wide_model({}, 17, use_raw=False)
    model = wide_model.build_wide_model({}, 17, use_raw=True)

    baseline_logits, baseline_features, baseline_branches = control(raw, coarse, capture=True)
    logits, fused, branches = model(raw, coarse, capture=True)
    assert (logits.shape, fused.shape, branches.shape) == ((2, 66), (2, 160), (5, 2, 160))
    torch.testing.assert_close(baseline_logits, native(coarse))
    torch.testing.assert_close(control(-raw, coarse), baseline_logits)
    torch.testing.assert_close(branches, baseline_branches)
    torch.testing.assert_close(logits, native.classify_features(fused))
    torch.testing.assert_close(logits, model(raw, coarse))
    torch.testing.assert_close((fused - baseline_features).square().mean().sqrt(), torch.tensor(.05), atol=1e-4, rtol=0)
    F.cross_entropy(logits, torch.tensor([0, 65])).backward()
    assert raw.grad is not None and raw.grad.square().sum() > 0
    assert model.raw_encoder.frontend[0].weight.grad.square().sum() > 0


def test_wide_views_share_phase_and_physical_noise_in_both_sample_grids():
    # Doubling the source input for the second packet exercises the common RMS scale.
    raw = torch.tensor([[[2.0], [1.0]], [[4.0], [2.0]]]).expand(2, 2, 8000)
    coarse = raw[:, :, :256]
    strength = .25
    ordinary_raw, ordinary_coarse, physical_raw, physical_coarse = wide_model.wide_views(
        raw, coarse, torch.Generator().manual_seed(19), strength
    )
    torch.testing.assert_close(ordinary_raw[:, :, :256], ordinary_coarse)
    torch.testing.assert_close(ordinary_raw.square().sum(1), raw.square().sum(1))
    assert not torch.equal(ordinary_raw, raw)

    perturbation_raw = physical_raw - raw
    perturbation_coarse = physical_coarse - coarse
    raw_coefficients = _fourier_coefficients(perturbation_raw)
    coarse_coefficients = _fourier_coefficients(perturbation_coarse)
    torch.testing.assert_close(raw_coefficients[:, :33], coarse_coefficients[:, :33], atol=1e-7, rtol=0)
    torch.testing.assert_close(raw_coefficients[:, -32:], coarse_coefficients[:, -32:], atol=1e-7, rtol=0)
    assert raw_coefficients[:, 33:-32].abs().max() < 1e-7
    assert coarse_coefficients[:, 33:-32].abs().max() < 1e-7
    snr = 10 * (raw.square().sum(1).mean(1) / (perturbation_raw / strength).square().sum(1).mean(1)).log10()
    assert ((5 <= snr) & (snr <= 15)).all()


def _fourier_coefficients(iq):
    return torch.fft.fft(torch.complex(iq[:, 0], iq[:, 1]), norm="forward")


def test_wide_views_return_inputs_exactly_at_zero_strength():
    raw = torch.randn(2, 2, 8000)
    coarse = torch.randn(2, 2, 256)
    generated = wide_model.wide_views(raw, coarse, torch.Generator().manual_seed(19), 0.0)
    assert all(torch.equal(observed, original) for observed, original in zip(generated, (raw, coarse, raw, coarse)))


def test_paper_pamf_branch_normalization_uses_fixed_source_statistics_at_inference(monkeypatch):
    from types import SimpleNamespace

    from experiments.pamf_satiq66 import paper_pamf

    native = _SyntheticNative()
    native.core = SimpleNamespace(id_backbone=SimpleNamespace(cls_head=native.head))
    monkeypatch.setattr(paper_pamf, "build_model", lambda runtime, seed, representation: native)
    model = paper_pamf.PaperPAMF({}, 17, classes=12, raw_time_bins=5, feature_batch_norm=True)
    raw = torch.randn(4, 2, 11000, generator=torch.Generator().manual_seed(42))
    fused, logits = model(raw)
    assert fused.shape == (4, 320)
    assert logits.shape == (4, 12)
    F.cross_entropy(logits, torch.tensor([0, 0, 5, 5])).backward()
    assert model.original_rate.frontend[0].weight.grad.square().sum() > 0
    assert model.common_norm.num_batches_tracked.item() == 1
    assert model.full_norm.num_batches_tracked.item() == 1

    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.embed(raw[:1]), model.embed(raw)[0:1])
    assert model.common_norm.num_batches_tracked.item() == 1
    assert model.full_norm.num_batches_tracked.item() == 1
