import numpy as np
import torch

from experiments.pamf_lora_main.dura_w24 import apertures, lift, readouts
from experiments.pamf_lora_main.dura_w24_model import DuraW24, WideTracker, objective
from experiments.pamf_lora_main.dura_w24_train import RECIPE


def test_lift_preserves_complex_tone_on_the_same_time_grid():
    wave = torch.exp((2j * torch.pi * -17 / 256) * torch.arange(256, dtype=torch.float64))
    outcome = lift(torch.stack((wave.real, wave.imag))[None, None])
    reference = torch.exp((2j * torch.pi * -17 / 384) * torch.arange(384, dtype=torch.float64))
    torch.testing.assert_close(torch.complex(outcome[0, 0, 0], outcome[0, 0, 1]), reference)


def test_w24_retains_raw_edges_and_never_edits_bypassed_packets():
    rng = np.random.default_rng(33)
    native = (rng.normal(size=(2, 8192)) + 1j*rng.normal(size=(2, 8192))).astype(np.complex64)
    narrow, wide = map(torch.from_numpy, apertures(native))
    assert wide.shape == (2, 8, 2, 432)
    edges = torch.complex(wide[:, :, 0, 384:], wide[:, :, 1, 384:]).numpy()
    raw = native.reshape(2, 8, 1024)[..., np.r_[0:24, 1000:1024]]
    np.testing.assert_allclose(edges / edges[:, :1, :1], raw / raw[:, :1, :1], rtol=1e-6)

    reference, candidate = readouts(wide, narrow, lambda x: x+1, torch.tensor([True, False]))
    for output in (reference, candidate):
        torch.testing.assert_close(output[..., 384:], wide[..., 384:], rtol=0, atol=0)
        torch.testing.assert_close(output[1], wide[1], rtol=0, atol=0)
    torch.testing.assert_close(reference[0, (0, 7)], wide[0, (0, 7)], rtol=0, atol=0)
    torch.testing.assert_close(candidate[0, :, :, :384], wide[0, :, :, :384] + 1)


def test_training_front_end_matches_w24_and_joint_fit_uses_boundary_fields():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(7)
    raw = torch.randn((3, 8192), generator=rng, dtype=torch.complex64)
    tracker, model = WideTracker(), DuraW24()
    view = tracker.aperture(raw)
    _, np_result = apertures(raw.numpy())
    rms = view[..., :384].abs().square().mean((1, 2), keepdim=True).sqrt()
    torch_view = view/rms
    expected = torch.stack((torch_view.real, torch_view.imag), dim=2)
    torch.testing.assert_close(expected, torch.from_numpy(np_result), atol=2e-6, rtol=1e-5)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for step in (1, 2):
        loss, values = objective(model, tracker, raw, rng, with_identity=True, recipe=RECIPE, step=step)
        assert torch.isfinite(values).all()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True) > 0
        opt.step()
    weights = model.coarse.encoder.blocks[0][0].net[0].weight
    assert weights.grad[:, -12:].abs().sum() > 0

    with torch.no_grad():
        predicted, active = model.restore(tracker, expected, -float('inf'))
    assert active.all()
    torch.testing.assert_close(predicted[..., 384:], expected[..., 384:], atol=0, rtol=0)
