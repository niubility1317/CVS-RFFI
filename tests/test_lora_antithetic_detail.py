import torch

from experiments.pamf_lora_main.antithetic_detail import (
    antithetic_inputs, component_risk, pair_terms,
)
from experiments.pamf_lora_main.crossfit_pilot import apply_transfer, generic_taps
from experiments.pamf_lora_main.rf_noise_transport import colored_observation


def test_same_action_opposite_noise_and_matched_clean_secant():
    clean = torch.randn(4, 6, 256, dtype=torch.complex64)
    change = .05 * torch.roll(clean, 1, dims=-1)
    seed = 11
    rng = torch.Generator().manual_seed(seed)
    output = antithetic_inputs(clean, change, colored_observation, rng, with_identity=True)
    independent = torch.Generator().manual_seed(seed)
    taps = generic_taps(clean, independent)
    taps *= torch.rand(4, 1, generator=independent) < .5
    mean, odd = pair_terms(output)
    torch.testing.assert_close(mean[:2, 0], apply_transfer(clean, taps)[:2])
    torch.testing.assert_close(mean[:2, 1] - mean[:2, 0], apply_transfer(change, taps)[:2])
    torch.testing.assert_close(odd[:, 0], odd[:, 1])
    assert odd[:2].abs().max() > 0
    assert not odd[-2:].count_nonzero()
    torch.testing.assert_close(mean[-2:, 0], clean[-2:])
    torch.testing.assert_close(mean[-2:, 1], clean[-2:] + change[-2:])


def test_even_odd_risk_decomposition_retains_nonzero_noise_penalty():
    rng = torch.Generator().manual_seed(49)
    # A balanced pair has zero bias and a nonzero noise response.
    clean = torch.randn(2, 2, 6, 256, generator=rng, dtype=torch.complex64)
    noise = .2 * torch.randn(clean.shape, generator=rng, dtype=clean.dtype)
    mixed = torch.stack((clean + noise, clean - noise), dim=2).flatten(1, 2)
    rebuilt, residual = pair_terms(mixed)
    torch.testing.assert_close(rebuilt, clean)
    torch.testing.assert_close(residual, noise)
    assert component_risk(rebuilt - clean, clean) < 1e-8
    assert component_risk(residual, clean) > .005
    nonzero = mixed.clone().requires_grad_()
    mean, odd = pair_terms(nonzero)
    loss = component_risk(mean - clean, clean) + component_risk(odd, clean)
    loss.backward()
    assert nonzero.grad.isfinite().all() and nonzero.grad.abs().sum() > 0
