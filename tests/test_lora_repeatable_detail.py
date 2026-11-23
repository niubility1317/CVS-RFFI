import torch

from experiments.pamf_lora_main.repeatable_detail import estimate_basis, equal_dose


def test_basis_prefers_clean_variation_over_independent_draw_noise():
    rng = torch.Generator().manual_seed(31)
    clean = torch.randn(3000, 4, generator=rng) * torch.tensor([2., .5, .5, .5])
    sigma = torch.tensor([.05, 4., 3., 2.])
    first, second = (clean + torch.randn(clean.shape, generator=rng) * sigma for _ in range(2))
    basis, values = estimate_basis(clean, first, second, rank=1)
    assert basis.shape == (4, 1)
    assert basis[0, 0].abs() > .95 and values[0] > values[1]
    torch.testing.assert_close(basis.T @ basis, torch.ones(1, 1, dtype=basis.dtype))


def test_equal_dose_preserves_each_band_and_outside_including_zero_rows():
    y = torch.complex(torch.arange(36.).reshape(2, 3, 6), torch.ones(2, 3, 6))
    y[0, 0] = 0
    x = torch.randn_like(y)
    x[1, 1] = 0
    near = torch.tensor([False, True, True, False, False, False])
    mid = torch.tensor([False, False, False, True, True, False])
    answer = equal_dose(y, x, (near, mid))
    assert torch.equal(answer[..., ~(near | mid)], y[..., ~(near | mid)])
    assert torch.equal(answer[0, 0], y[0, 0]) and torch.equal(answer[1, 1], y[1, 1])
    for band in (near, mid):
        torch.testing.assert_close(answer[..., band].abs().square().sum(-1),
                                   y[..., band].abs().square().sum(-1))
