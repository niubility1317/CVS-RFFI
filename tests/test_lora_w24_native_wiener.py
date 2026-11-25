import torch

from experiments.pamf_lora_main.w24_native_wiener import denoise, noise_profile, pilot_rows


def test_native_wiener_has_exact_inactive_path_and_finite_edits():
    n = torch.arange(8192, dtype=torch.float32)
    raw = torch.exp(.02j * n)[None]
    rng = torch.Generator().manual_seed(4)
    disturbed = raw + .7 * torch.randn(raw.shape, dtype=torch.complex64, generator=rng)
    x = torch.cat((raw, disturbed))
    ratio, _ = noise_profile(x)
    assert ratio[0] < .01 and ratio[1] > .05
    y, active = denoise(x, .05)
    assert active.tolist() == [False, True]
    assert torch.equal(y[0], raw[0])
    assert torch.isfinite(y).all()
    assert (y[1] - raw[0]).abs().square().mean() < (x[1] - raw[0]).abs().square().mean()


def test_pilot_rows_are_fixed_sorted_and_receiver_specific():
    a, b = pilot_rows(0), pilot_rows(1)
    assert len(a) == len(set(a)) == 200
    assert (a == sorted(a)).all() and (a == pilot_rows(0)).all()
    assert (a != b).any()
