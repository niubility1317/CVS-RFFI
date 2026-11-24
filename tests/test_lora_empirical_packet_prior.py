import torch

from experiments.pamf_lora_main.empirical_packet_prior import (
    bank_summary, detail_bins, observation_bins, posterior, split,
)


def test_conditional_private_uses_same_common_and_marginal_values():
    reference = torch.zeros((2, 6, 64), dtype=torch.complex128)
    reference[0] += 1
    reference[1] -= 1
    reference[:, 0, 1] += torch.tensor([2., -2.])
    reference[:, 1, 1] -= torch.tensor([2., -2.])
    observation = reference[:1].clone()
    observation[..., 1] = observation[..., 1].mean(1)
    keys = observation_bins(detail_bins(reference.device))
    mean, power = bank_summary(reference)
    independent, joint, weights = posterior(observation, torch.full((1, 6), .3),
                                            reference, mean, power, keys)
    torch.testing.assert_close(split(independent)[0], split(joint)[0])
    assert weights[0, 0] > .99
    assert split(independent)[1].count_nonzero() == 0
    assert split(joint)[1].abs().max() > .01


def test_broad_likelihood_returns_the_same_private_mean_as_control():
    rng = torch.Generator().manual_seed(2)
    bank = torch.randn(9, 6, 64, dtype=torch.complex128, generator=rng)
    observation = torch.randn(2, 6, 64, dtype=torch.complex128, generator=rng)
    keys = observation_bins(detail_bins(bank.device))
    independent, linked, weights = posterior(observation, torch.full((2, 6), 1e20),
                                             bank, *bank_summary(bank), keys)
    torch.testing.assert_close(weights, torch.full_like(weights, 1/len(bank)))
    torch.testing.assert_close(independent, linked)
    assert independent.isfinite().all()


def test_likelihood_keys_stay_inside_registered_near_carrier_bins():
    bins = detail_bins('cpu')
    selected = bins[observation_bins(bins)]
    assert bins.unique().numel() == 64
    assert selected.tolist() == [*range(1, 9), *range(248, 256)]
