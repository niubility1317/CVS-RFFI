import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_probe_transport import phase_probe, probe_views
from experiments.pamf_lora_main.w24_third_prior import UNET_BASELINE, fit_inputs, trial


class TestAperture:
    def aperture(self, raw):
        return raw.unflatten(-1, (8, 1024))[..., :432]


def condition(raw, rng):
    # A generic randomized linear action and additive sample noise.
    a = torch.rand((len(raw), 1, 1), generator=rng)
    e = torch.randn(raw.shape, dtype=raw.dtype, generator=rng)
    return 1j * a * raw + .2 * e


def test_probe_preserves_capture_power_and_has_zero_phase_at_boundaries():
    torch.manual_seed(27)
    raw = torch.randn(3, 8192, dtype=torch.complex64)
    weak = torch.randn(3, 8, 384, dtype=torch.complex64)
    chirps = raw.unflatten(-1, (8, 1024))
    probe = phase_probe(raw, weak)
    torch.testing.assert_close(probe.abs(), chirps.abs(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(probe[..., [0, -1]], chirps[..., [0, -1]], rtol=0, atol=0)
    assert (probe - chirps).abs().sum() > 0


def test_paired_transport_reuses_first_channel_draw_and_keeps_base_stream():
    torch.manual_seed(29)
    raw = torch.randn(4, 8192, dtype=torch.complex64)
    weak = .11j * torch.ones(4, 8, 384, dtype=raw.dtype)
    tracker = TestAperture()
    clean = tracker.aperture(raw)
    generator_after = torch.Generator().manual_seed(70)
    generator_through = torch.Generator().manual_seed(70)
    replay = torch.Generator().manual_seed(70)
    after, target = probe_views(raw, clean, tracker, condition, generator_after, weak,
                                through_channel=False, with_identity=True)
    through, propagated_target = probe_views(raw, clean, tracker, condition, generator_through, weak,
                                             through_channel=True, with_identity=True)
    expected = tracker.aperture(condition(phase_probe(raw, weak), replay).flatten(-2))
    torch.testing.assert_close(after[:, :2], through[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(target, propagated_target, rtol=0, atol=0)
    torch.testing.assert_close(after[-2:, 0], clean[-2:], rtol=0, atol=0)
    torch.testing.assert_close(through[-2:, 2], target[-2:], rtol=0, atol=0)
    torch.testing.assert_close(after[:2, 2], after[:2, 0] + (target[:2] - clean[:2]))
    torch.testing.assert_close(through[:2, 2], expected[:2], rtol=0, atol=0)
    assert (after[:2, 2] - through[:2, 2]).abs().max() > .01
    assert torch.equal(generator_after.get_state(), generator_through.get_state())


def test_probe_training_paths_include_only_verified_source_parent():
    spec = trial('probe-transport')
    assert spec.steps == 12000 and spec.routes == (FULL,)
    assert UNET_BASELINE in spec.source_modes
    assert set(cases('probe-transport')) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='source', run_root='runs/new', unet_parent='frozen source',
                   classifier_root='forbidden F', source_parent='unrelated', native_target_root='forbidden')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='source', run_root=f'runs/new/{arm}',
                                                unet_parent='frozen source')
