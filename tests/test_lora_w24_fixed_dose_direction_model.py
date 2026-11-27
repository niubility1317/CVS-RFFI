import torch

from experiments.pamf_lora_main.w24_fixed_dose_direction_model import (
    ApertureDirection,
    DirectionHead,
    direction_objective,
    matched_aperture_edit,
)
from experiments.pamf_lora_main.w24_native_center import CENTER, NATIVE_LENGTH


def complex_noise(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def test_projected_direction_preserves_two_independent_packet_doses():
    torch.manual_seed(14)
    base = complex_noise(4, 8, 432)
    proposal = complex_noise(4, 8, 432)
    base[0, :, :CENTER] = 0
    proposal[1, :, CENTER:] = 0
    base[2, :, CENTER:] = 0
    proposal[3, :, :CENTER] *= 5

    actual = matched_aperture_edit(base, proposal)

    for region in (slice(None, CENTER), slice(CENTER, None)):
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual[..., region], dim=(-2, -1)),
            torch.linalg.vector_norm(base[..., region], dim=(-2, -1)),
        )
    torch.testing.assert_close(actual[0, :, :CENTER], base[0, :, :CENTER], atol=0, rtol=0)
    torch.testing.assert_close(actual[1, :, CENTER:], base[1, :, CENTER:], atol=0, rtol=0)
    torch.testing.assert_close(actual[2, :, CENTER:], base[2, :, CENTER:], atol=0, rtol=0)
    assert not torch.allclose(actual[3, :, :CENTER], base[3, :, :CENTER])


def test_source_objective_has_finite_nonzero_gradient_after_matching_dose():
    torch.manual_seed(17)
    observed = complex_noise(2, NATIVE_LENGTH)
    base = 0.12 * complex_noise(2, NATIVE_LENGTH)
    target = observed + 0.18 * complex_noise(2, NATIVE_LENGTH)
    target[-1] = observed[-1]
    clean = torch.tensor([False, True])
    model = DirectionHead()
    aperture = ApertureDirection()
    proposal = model(observed, base)

    loss, metrics = direction_objective(aperture, observed, base, proposal, target, clean_mask=clean)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) and not value.requires_grad for value in metrics.values())
    gm = [p.grad for p in model.parameters() if p.grad is not None]
    assert gm and all(torch.isfinite(g).all() for g in gm)
    assert any(torch.count_nonzero(g) for g in gm)
    for view in aperture(observed, base, proposal):
        assert view.shape == (2, 8, 2, 432) and torch.isfinite(view).all()
        torch.testing.assert_close(view[..., :CENTER].square().sum(2).mean((1, 2)), torch.ones(2))


def test_zero_edit_keeps_clean_packet_and_finite_gradients():
    observed = complex_noise(1, NATIVE_LENGTH)
    base = torch.zeros_like(observed)
    model = DirectionHead()
    aperture = ApertureDirection()
    proposal = model(observed, base)

    direct, frozen, candidate = aperture(observed, base, proposal)
    loss, metrics = direction_objective(aperture, observed, base, proposal, observed)
    loss.backward()

    torch.testing.assert_close(proposal, base, rtol=0, atol=0)
    torch.testing.assert_close(candidate, frozen, rtol=0, atol=0)
    torch.testing.assert_close(candidate, direct, rtol=0, atol=0)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
