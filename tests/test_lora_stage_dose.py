import torch

from experiments.pamf_lora_main.stage_dose import edit_controls


def test_direction_dose_controls_and_tangent_have_expected_geometry():
    rng = torch.Generator().manual_seed(47)
    coarse = torch.randn(3, 6, 256, generator=rng, dtype=torch.complex64)
    fine = torch.randn(3, 6, 256, generator=rng, dtype=torch.complex64)
    rms = torch.rand(3, 6, 1, generator=rng) + .2
    edits, error = edit_controls(coarse, fine, rms)
    assert float(error.max()) < 2e-6
    for value in edits:
        assert float(value.mean(-1).abs().max()) < 1e-6
    assert not torch.equal(edits[0], edits[-1])


def test_tangent_control_preserves_coarse_direction_on_collinear_inputs():
    t = torch.arange(256) / 256
    signal = torch.exp(2j * torch.pi * 8 * t)[None, None].expand(1, 6, -1)
    coarse = 2 * signal
    rms = torch.ones(1, 6, 1)
    edits, error = edit_controls(coarse, -3 * coarse, rms)
    torch.testing.assert_close(edits[0], -coarse, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(edits[1], 3 * coarse, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(edits[-1], coarse, atol=3e-5, rtol=1e-5)
    assert error.isfinite().all()


def test_missing_direction_uses_the_supported_edit_and_no_carrier():
    coarse = torch.zeros(1, 6, 256, dtype=torch.complex64)
    t = torch.arange(256) / 256
    fine = torch.exp(2j * torch.pi * 19 * t)[None, None].expand_as(coarse)
    rms = torch.ones(1, 6, 1)
    edits, error = edit_controls(coarse, fine, rms)
    assert not edits[0].count_nonzero() and not edits[-1].count_nonzero()
    assert error.isfinite().all() and float(error.max()) < 1e-6
    torch.testing.assert_close(edits[1], fine)
