import torch

from experiments.pamf_lora_main.chirp_spectrum_recovery import ChirpSpectrumRecovery, components
from experiments.pamf_lora_main.chirp_spectrum_source import RECIPES
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.natural_detail import natural_objective, paired_echo_inputs


def test_reconstruction_preserves_carrier_and_respects_chirp_order():
    generator = torch.Generator().manual_seed(123)
    observed = torch.randn(2, 6, 256, dtype=torch.complex64, generator=generator)
    coherence = torch.rand(2, 6, 1, generator=generator)
    permutation = torch.tensor([2, 5, 1, 0, 4, 3])
    for separate in (False, True):
        model = ChirpSpectrumRecovery(separate)
        output = model(observed, coherence)
        permuted = model(observed[:, permutation], coherence[:, permutation])
        for wave, reordered in zip(output, permuted, strict=True):
            torch.testing.assert_close(wave.mean(-1), observed.mean(-1))
            torch.testing.assert_close(reordered, wave[:, permutation], atol=1e-6, rtol=1e-5)
            assert torch.isfinite(wave).all()


def test_split_protection_keeps_coherent_complex_values_without_keeping_innovation():
    model = ChirpSpectrumRecovery(separate=True)
    observed = torch.tensor([[[2+3j], [4-7j]]])
    common, distinct = components(observed)
    shrink = torch.tensor([[[0.], [1.]]])
    pooled = model.reconstruct(common, distinct, shrink, protection=1.)
    torch.testing.assert_close(pooled, common.expand_as(observed))
    control = ChirpSpectrumRecovery(separate=False)
    torch.testing.assert_close(control.reconstruct(common, distinct, shrink, protection=1.), .5 * observed)


def test_both_stages_learn_from_same_source_iq_objective():
    generator = torch.Generator().manual_seed(42)
    iq = real_iq(torch.randn(4, 6, 256, dtype=torch.complex64, generator=generator))
    tracker = PacketRestorer('consensus')
    for recipe in RECIPES.values():
        model = recipe.model()
        rng = torch.Generator().manual_seed(7)
        loss, parts = natural_objective(model, tracker, iq, rng, with_identity=True, recipe=recipe,
                                       step=1, preserve=.35, make_pair=paired_echo_inputs)
        assert torch.isfinite(parts).all()
        loss.backward()
        assert model.coarse[-1].weight.grad.norm() > 0
        assert model.fine[-1].weight.grad.norm() > 0
