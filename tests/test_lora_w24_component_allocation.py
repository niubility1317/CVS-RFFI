import torch

from experiments.pamf_lora_main.w24_component_allocation import components, counterfactual_edits
from experiments.pamf_lora_main.w24_practical_direction_factor import ALLOCATION_MODES


def test_fixed_dose_factorial_independently_replaces_component_norm_and_complex_direction():
    gen = torch.Generator().manual_seed(2026112559)
    shape = (2, 8, 384)
    estimate, exact = (torch.randn(shape, dtype=torch.complex64, generator=gen) for _ in range(2))
    phase = torch.exp(1j * torch.randn(shape, generator=gen))
    norm_only, direction_only, both, exact_center = counterfactual_edits(estimate, exact, phase)
    assert ALLOCATION_MODES[2:] == ('component_norms', 'component_directions', 'component_both', 'exact_center')
    torch.testing.assert_close(exact_center, exact, rtol=0, atol=0)

    def parts_and_norm(edit):
        parts, _ = components(edit, phase)
        return parts, torch.linalg.vector_norm(parts, dim=(-2, -1), keepdim=True)

    estimated_parts, estimated_norm = parts_and_norm(estimate)
    exact_parts, exact_norm = parts_and_norm(exact)
    total = lambda n: n.square().sum(1, keepdim=True).sqrt()
    new_norm = exact_norm * (total(estimated_norm) / total(exact_norm))
    torch.testing.assert_close(parts_and_norm(norm_only)[0], new_norm * estimated_parts / estimated_norm)
    torch.testing.assert_close(parts_and_norm(direction_only)[0], estimated_norm * exact_parts / exact_norm)
    torch.testing.assert_close(parts_and_norm(both)[0], new_norm * exact_parts / exact_norm)


def test_clean_and_unsupported_direction_do_not_borrow_an_oracle_edit():
    estimate = torch.ones((2, 8, 384), dtype=torch.complex64)
    exact = torch.zeros_like(estimate)
    exact[1, 0, 3] = 2j
    exact[1, 1, 3] = -2j
    rows = counterfactual_edits(estimate, exact, torch.ones_like(estimate))
    for row in rows:
        torch.testing.assert_close(row[0], estimate[0], rtol=0, atol=0)
    torch.testing.assert_close(rows[0][1], estimate[1])
    torch.testing.assert_close(rows[-1][1], exact[1], rtol=0, atol=0)
