import torch

from experiments.pamf_lora_main.dura_apertures import ApertureTrackers, DualApertureDura, RECIPE, calibrate, objective
from experiments.pamf_lora_main.dura_repair import real_iq
from experiments.pamf_lora_main.dura_w24 import apertures


def test_apertures_share_weights_and_preserve_native_output():
    rng = torch.Generator().manual_seed(32)
    raw = torch.randn(4, 8192, dtype=torch.complex64, generator=rng)
    trackers, model = ApertureTrackers(), DualApertureDura()
    narrow, wide = trackers.narrow.aperture(raw), trackers.wide.aperture(raw)
    rms = wide[..., :384].abs().square().mean((1, 2), keepdim=True).sqrt()
    np_narrow, np_wide = apertures(raw.numpy())
    torch.testing.assert_close(real_iq(narrow/rms), torch.from_numpy(np_narrow)[:, 1:7], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(real_iq(wide/rms), torch.from_numpy(np_wide), atol=2e-6, rtol=1e-5)

    loss, values = objective(model, trackers, raw, rng, with_identity=True, recipe=RECIPE, step=1)
    assert values.shape == (18,) and torch.isfinite(values).all()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    assert model.priors[0].out.weight.grad.abs().sum() > 0
    assert model.priors[1].out.weight.grad.abs().sum() > 0

    threshold, _ = calibrate(model, trackers, raw)
    with torch.no_grad():
        for observed in (real_iq(narrow), real_iq(wide)):
            restored, mask = model.restore(trackers, observed, threshold)
            assert not mask.any() and restored is observed
            restored, mask = model.restore(trackers, observed, -1.)
            assert mask.all() and restored.shape == observed.shape and torch.isfinite(restored).all()
            torch.testing.assert_close(restored[..., 384:], observed[..., 384:], atol=0, rtol=0)
