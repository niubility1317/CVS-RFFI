import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import w24_formal
from experiments.pamf_lora_main.w24_packet_dura import (
    PACKET, NativePacketDura, native_additive_pair, native_band_risks, source_objective,
)
from experiments.pamf_lora_main.w24_third_prior import (
    NATIVE_ADDITIVE_ARMS, NATIVE_PACKET_ARMS, NATIVE_SPECTRAL_ARMS, fit_inputs, trial,
)


def test_native_dura_restores_and_trains_over_the_entire_capture():
    torch.manual_seed(71)
    model = NativePacketDura()
    wave = torch.randn(2, PACKET, dtype=torch.complex64)
    clean = wave * .97
    stages, _ = model(wave)
    for out in stages:
        assert out.dtype == wave.dtype and out.shape == wave.shape and torch.isfinite(out).all()
    change = stages[-1] - wave
    assert change[:, :24].abs().max() > 0 and change[:, -24:].abs().max() > 0
    loss, parts = source_objective(model, wave, clean, with_identity=False)
    assert parts.shape == (7,) and torch.isfinite(parts).all()
    loss.backward()
    for module in (model.coarse, *model.priors):
        assert any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())


def test_full_capture_reaches_the_original_w24_front_end(monkeypatch):
    class Dummy:
        def restore_native(self, raw):
            return raw + (2 + 3j)

    raw = np.full((2, PACKET), 1 + 2j, np.complex64)
    original = torch.zeros((2, 8, 2, 432))
    seen = []

    def aperture(packets, device):
        seen.append(packets.copy())
        return torch.ones_like(original)

    monkeypatch.setattr(w24_formal, 'received_aperture', aperture)
    control, whole = w24_formal.native_readouts(Dummy(), raw, original, .5)
    np.testing.assert_array_equal(seen[0], raw + (1 + 1.5j))
    torch.testing.assert_close(control[..., :384], whole[..., :384])
    torch.testing.assert_close(control[..., 384:], original[..., 384:])
    assert (whole[..., 384:] == 1).all()
    with pytest.raises(ValueError):
        w24_formal.native_readouts(Dummy(), raw[:, :-1], original, .5)


def test_only_the_whole_packet_model_is_fitted_and_only_u_is_admitted():
    spec = trial('native-packet-dura')
    assert spec.steps == 24000 and spec.arms == NATIVE_PACKET_ARMS
    assert spec.routes == (w24_formal.FULL,) and spec.fit_arms == NATIVE_PACKET_ARMS[1:]
    rt = dict(source_root='U', run_root='runs/new', classifier_root='F', native_target_root='target')
    assert fit_inputs(rt, spec.fit_arms[0]) == dict(source_root='U', run_root='runs/new/native_dura_whole')
    with pytest.raises(ValueError):
        fit_inputs(rt, NATIVE_PACKET_ARMS[0])


def test_native_additive_reference_changes_only_the_supervision():
    raw = torch.randn(8, PACKET, dtype=torch.complex64, generator=torch.Generator().manual_seed(20))
    pair = lambda: native_additive_pair(raw, torch.Generator().manual_seed(23), with_identity=True)
    received, (original, after_action) = pair()
    received_again, references_again = pair()
    assert torch.equal(received_again, received)
    assert all(torch.equal(a, b) for a, b in zip(references_again, (original, after_action), strict=True))
    assert torch.equal(received[-2:], original[-2:]) and torch.equal(received[-2:], after_action[-2:])
    assert (received[:-2] - after_action[:-2]).abs().mean() > 0
    assert (original[:-2] - after_action[:-2]).abs().mean() > 0
    model = NativePacketDura().eval()
    with torch.no_grad():
        loss0, parts0 = source_objective(model, received, original, with_identity=True,
                                         audit_references=(original, after_action))
        loss1, parts1 = source_objective(model, received, after_action, with_identity=True,
                                         audit_references=(original, after_action))
    assert loss0 != loss1 and parts0.shape == (9,)
    torch.testing.assert_close(parts0[-2:], parts1[-2:])
    torch.testing.assert_close(parts0[0], parts0[-2])
    torch.testing.assert_close(parts1[0], parts1[-1])


def test_native_additive_predicts_two_whole_packet_models(monkeypatch):
    spec = trial('native-additive')
    assert spec.steps == 24000 and spec.arms == spec.fit_arms == NATIVE_ADDITIVE_ARMS
    assert spec.routes == (w24_formal.FULL,)
    rt = dict(source_root='U', run_root='runs/new', classifier_root='F', native_target_root='target')
    for arm in spec.fit_arms:
        assert fit_inputs(rt, arm) == dict(source_root='U', run_root=f'runs/new/{arm}')

    class Offset:
        def __init__(self, value):
            self.value = value

        def restore_native(self, raw):
            return raw + self.value

    raw = np.zeros((2, PACKET), np.complex64)
    seen = []

    def aperture(packets, _device):
        seen.append(packets.copy())
        return torch.full((2, 8, 2, 432), float(packets[0, 0].real))

    monkeypatch.setattr(w24_formal, 'received_aperture', aperture)
    monkeypatch.setattr(w24_formal, 'classify_aperture', lambda _head, x: x[:, -1, -1, -1].numpy())
    labels, count = w24_formal.prior_labels([object()], [Offset(2), Offset(4)], None, raw, 'cpu')
    assert count == 2
    np.testing.assert_array_equal(labels[0], [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(np.stack(seen)[:, 0, :], np.broadcast_to(np.arange(3)[:, None], (3, PACKET)))


def test_band_risk_separates_common_and_private_near_carrier_noise():
    grid = torch.arange(1024, dtype=torch.float32)
    common = torch.exp(2j * torch.pi * (5 / 1024) * grid)
    private = torch.exp(-2j * torch.pi * (18 / 1024) * grid)
    signs = torch.tensor([1, -1] * 4, dtype=torch.float32)[:, None]
    edit = ((common + signs * private) / 32.).flatten()[None]
    frame = torch.ones((1, 8, 1024), dtype=torch.complex64)
    powers = native_band_risks(edit, torch.zeros_like(edit), frame)
    assert powers[0] > 0.1 and powers[3] > 0.1
    assert (powers[[1, 2, 4, 5]] < 1e-8).all()


def test_spectral_pair_logs_the_same_terms_and_changes_optimizer_gradient():
    spec = trial('native-band-risk')
    assert spec.arms == spec.fit_arms == NATIVE_SPECTRAL_ARMS
    assert spec.steps == 24000 and spec.routes == (w24_formal.FULL,)
    model = NativePacketDura()
    raw = torch.randn(2, PACKET, dtype=torch.complex64)
    received, references = native_additive_pair(raw, torch.Generator().manual_seed(21), with_identity=False)
    results = [source_objective(model, received, references[1], with_identity=False,
                                audit_references=references, spectral_weight=w) for w in (None, 0., .50)]
    (legacy, _), (control, parts0), (candidate, parts1) = results
    assert torch.equal(legacy, control) and candidate > control
    assert parts0.shape == (15,)
    torch.testing.assert_close(parts0, parts1)
    param = model.priors[-1].read[-1].weight
    g0 = torch.autograd.grad(control, param)[0]
    g1 = torch.autograd.grad(candidate, param)[0]
    assert torch.isfinite(g1).all() and not torch.allclose(g0, g1)
