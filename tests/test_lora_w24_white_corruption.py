import torch

from experiments.pamf_lora_main.w24_white_corruption import training_observations
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


def test_whitening_matches_center_energy_and_keeps_control_draws_aligned():
    raw = torch.ones(3, 8192, dtype=torch.complex64)
    def aperture(value):
        return value.unflatten(-1, (8, 1024))[..., :432]

    def generic(value, rng):
        draw = torch.rand(3, 1, 1, generator=rng)
        return value + (draw + .5) * (1 + 1j)

    clean = aperture(raw)
    control_rng = torch.Generator().manual_seed(31)
    treat_rng = torch.Generator().manual_seed(31)
    ordinary = training_observations(raw, clean, aperture, generic, control_rng, white=False)
    white = training_observations(raw, clean, aperture, generic, treat_rng, white=True)

    assert torch.equal(control_rng.get_state(), treat_rng.get_state())
    for a, b in zip(ordinary, white, strict=True):
        assert not torch.equal(a, b)
        energy = lambda x: (x - clean)[..., :384].abs().square().sum((-2, -1))
        torch.testing.assert_close(energy(a), energy(b))


def test_white_corruption_preserves_an_unperturbed_pair_without_nan():
    raw = torch.zeros(2, 8192, dtype=torch.complex64)
    aperture = lambda value: value.unflatten(-1, (8, 1024))[..., :432]
    clean = aperture(raw)
    for received in training_observations(raw, clean, aperture, lambda x, _: x,
                                         torch.Generator().manual_seed(37), white=True):
        assert torch.equal(received, clean)


def test_both_arms_have_identical_source_only_scope():
    spec = trial('white-corruption')
    assert spec.steps == 1800
    assert spec.source_modes[-1] == 'outer_parent_half'
    assert set(cases('white-corruption')) == {f'{FULL}-{i}' for i in range(3)}
    rt = dict(source_root='/U', source_parent='/G', run_root='/out',
              classifier_root='/F', native_target_root='/T')
    for arm in spec.arms:
        assert fit_inputs(rt, arm) == dict(source_root='/U', source_parent='/G', run_root=f'/out/{arm}')
