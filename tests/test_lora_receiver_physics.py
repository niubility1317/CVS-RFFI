from functools import partial

import torch

from experiments.pamf_lora_main.antithetic_detail import antithetic_inputs, pair_terms
from experiments.pamf_lora_main.receiver_physics import apply_generic_physics, fractional_warp
from experiments.pamf_lora_main.rf_noise_transport import colored_observation


def test_interpolation_preserves_integer_coordinates_and_reflects_boundaries():
    wave = torch.randn(2, 6, 256, dtype=torch.complex64)
    index = torch.arange(256).expand_as(wave).float()
    torch.testing.assert_close(fractional_warp(wave, index), wave)
    shifted = fractional_warp(wave, index - 1)
    torch.testing.assert_close(shifted[..., 1:], wave[..., :-1])
    torch.testing.assert_close(shifted[..., 0], wave[..., 1])
    flat = torch.full_like(wave, 1 + 2j)
    torch.testing.assert_close(fractional_warp(flat, index + .42), flat)


def test_matched_weak_secant_uses_the_same_receiver_impairments():
    rng = torch.Generator().manual_seed(997)
    base, change = torch.randn(2, 8, 6, 256, generator=rng, dtype=torch.complex64)
    change *= .05
    seed = 361
    base_out, shifted_out = apply_generic_physics(base, base + change,
                                                  generator=rng.manual_seed(seed))
    change_out, _ = apply_generic_physics(change, change, generator=rng.manual_seed(seed))
    torch.testing.assert_close(shifted_out - base_out, change_out, atol=1e-6, rtol=1e-5)
    assert (base_out - base).abs().mean() > .02


def test_extra_physics_keeps_original_draw_order_and_clean_exposure():
    wave, change = torch.randn(2, 8, 6, 256, dtype=torch.complex64,
                               generator=torch.Generator().manual_seed(65))
    change *= .08
    rng = torch.Generator().manual_seed(84)
    before = rng.get_state()
    plain = antithetic_inputs(wave, change, colored_observation, rng, with_identity=True)
    state_after_plain = rng.get_state()
    rng.set_state(before)
    physics = partial(apply_generic_physics, generator=torch.Generator().manual_seed(361))
    changed = antithetic_inputs(wave, change, colored_observation, rng,
                                with_identity=True, transform=physics)
    assert torch.equal(rng.get_state(), state_after_plain)
    assert (changed[:-2] - plain[:-2]).abs().mean() > .02
    assert torch.equal(changed[-2:], plain[-2:])
    _, noise = pair_terms(changed)
    torch.testing.assert_close(noise[:, 0], noise[:, 1])
