import torch

from experiments.pamf_lora_main.coherent_interval import CoherentInterval, edit_interval, initialize, objective, spectral_budget
from experiments.pamf_lora_main.coherent_interval_source import MODES, RECIPES, render
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq


def test_shared_interval_and_per_bin_budget():
    waveform_edit = torch.tensor([-2. + 3j, 2. - 3j])[None, None].expand(1, 6, 2)
    amplitude = torch.ones(1, 1, 2)
    bounded = edit_interval(waveform_edit, torch.ones_like(amplitude), amplitude, .9 * amplitude, amplitude)
    torch.testing.assert_close(bounded.real, torch.tensor([-.10, .02])[None, None].expand_as(bounded.real))
    assert (bounded.imag.abs() < .3).all()
    scalar = spectral_budget(waveform_edit, bounded)
    torch.testing.assert_close(scalar.abs().square().sum(1), bounded.abs().square().sum(1))
    torch.testing.assert_close(edit_interval(waveform_edit, amplitude, amplitude, amplitude, 0. * amplitude),
                               waveform_edit, atol=0, rtol=0)


def test_matched_real_shape_gradients_and_clean_bypass():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        tracker = PacketRestorer('consensus').eval()
        rng = torch.Generator().manual_seed(224)
        source = 1 + .03 * torch.randn(100, 6, 256, generator=rng, dtype=torch.complex64)
        clean = real_iq(source * tracker.reference.chirp)
        models = []
        for recipe in RECIPES.values():
            model = recipe.model()
            initialize(model, tracker, clean[:50], clean[50:])
            values_loss, terms = objective(model, tracker, clean[:24], rng, with_identity=True,
                                           recipe=recipe, step=8000)
            assert terms.numel() == 7 and torch.isfinite(terms).all()
            values_loss.backward()
            assert 0 < torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            models.append(model.eval())

        disturbed = real_iq(2.0 * torch.randn(3, 6, 256, generator=rng, dtype=torch.complex64))
        iq = torch.cat((clean[:1], disturbed))
        output, active = render(iq, None, lambda x: (x, None), tracker, models, threshold=.06)
        assert active == 3 and output.shape == (len(MODES), *iq.shape)
        assert torch.equal(output[:, 0], iq[0].expand_as(output[:, 0]))
        change = (output[2:] - iq).square().sum((2, 3, 4))
        torch.testing.assert_close(change[6], change[7], atol=1e-5, rtol=1e-6)
        assert (change <= .04001 * iq.square().sum((1, 2, 3))).all()
    finally:
        torch.set_num_threads(threads)


def test_low_cross_group_support_does_not_invent_confidence():
    model = CoherentInterval(True)
    spectrum = torch.zeros(2, 6, 256, dtype=torch.complex64)
    spectrum[:, :, 0] = 16.
    spectrum[:, :, 4] = .5j
    spectrum[1, 1::2, 4] *= -1
    y = torch.fft.ifft(spectrum, norm='ortho')
    (_, _, _, confidence), _ = model.spectral_evidence(y, torch.ones(2, 6, 1))
    assert confidence[0, 0, 4] > .9 and confidence[1, 0, 4] == 0
    assert (confidence[..., 0] == 0).all()
