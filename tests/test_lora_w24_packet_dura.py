import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import w24_formal
from experiments.pamf_lora_main.w24_packet_dura import PACKET, NativePacketDura, source_objective
from experiments.pamf_lora_main.w24_third_prior import NATIVE_PACKET_ARMS, fit_inputs, trial


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
