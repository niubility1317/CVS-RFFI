import torch

from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.reference_cancel import ReferenceCancel, initialize, objective, pair_spectrum
from experiments.pamf_lora_main.reference_cancel_source import MODES, RECIPES, render


def simulated_iq(tracker, count):
    rng = torch.Generator().manual_seed(256)
    z = 1 + .08 * torch.randn(count, 6, 256, generator=rng, dtype=torch.complex64)
    return real_iq(z * tracker.reference.chirp)


def test_application_pair_does_not_determine_its_fit_or_dose():
    model = ReferenceCancel(guarded=True)
    generator = torch.Generator().manual_seed(13)
    wave = torch.randn(4, 6, 256, generator=generator, dtype=torch.complex64)
    base = model.band_prediction(pair_spectrum(wave), 'middle')
    changed = wave.clone()
    changed[:, :2] *= 2 + 1j
    altered = model.band_prediction(pair_spectrum(changed), 'middle')
    for original, actual in zip(base[1:], altered[1:], strict=True):
        torch.testing.assert_close(actual[:, 0], original[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(altered[0][:, 0], base[0][:, 0] * (2 + 1j))


def test_two_arms_update_and_guarded_render_preserves_rows():
    saved_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        tracker = PacketRestorer('consensus').eval()
        data = simulated_iq(tracker, 80)
        models = [recipe.model() for recipe in RECIPES.values()]
        for model, recipe in zip(models, RECIPES.values(), strict=True):
            initialize(model, tracker, data[:40], data[40:])
            rng = torch.Generator().manual_seed(58)
            loss, values = objective(model, tracker, data[:24], rng, with_identity=True, recipe=recipe, step=1)
            assert values.shape == (7,) and torch.isfinite(values).all()
            loss.backward()
            assert 0 < torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            model.eval()

        clean = data[:1]
        disturbed = real_iq(torch.randn(1, 6, 256, generator=torch.Generator().manual_seed(9),
                                      dtype=torch.complex64))
        iq = torch.cat((clean, disturbed))
        y, rotation, _, level = coordinates(tracker, iq)
        threshold = float(models[0].training_gate)
        assert corruption_score(y)[0] <= threshold < corruption_score(y)[1]
        waves, active = render(iq, None, lambda observed: (observed, None), tracker, models, threshold=threshold)
        assert active == 1 and waves.shape == (len(MODES), *iq.shape)
        assert torch.equal(waves[:, 0], clean.expand_as(waves[:, 0]))
        learned = waves[4:]
        edit = learned - iq
        basis = rotation * level
        complex_edit = torch.complex(edit[..., 0, :], edit[..., 1, :])
        torch.testing.assert_close((complex_edit / basis).mean(-1),
                                  torch.zeros_like(complex_edit[..., 0]), atol=1e-6, rtol=0)
        assert (edit.square().sum((2, 3, 4)) <= .010001 * iq.square().sum((1, 2, 3))).all()
    finally:
        torch.set_num_threads(saved_threads)
