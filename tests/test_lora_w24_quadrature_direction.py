import torch

from experiments.pamf_lora_main.w24_quadrature_direction import rotated_proposals


def test_quadrature_average_commutes_with_a_quarter_turn():
    gen = torch.Generator().manual_seed(17)
    obs = torch.randn(3, 32, dtype=torch.complex64, generator=gen)
    edit = torch.randn(3, 32, dtype=torch.complex64, generator=gen)

    def arbitrary_head(y, u):
        return u + .2 * y.conj() + .1 * y.square() + .1j

    original, half, quarter = rotated_proposals(arbitrary_head, obs, edit)
    torch.testing.assert_close(original, arbitrary_head(obs, edit))
    torch.testing.assert_close(rotated_proposals(arbitrary_head, -obs, -edit)[1], -half)
    rotated = rotated_proposals(arbitrary_head, 1j * obs, 1j * edit)[2]
    torch.testing.assert_close(rotated, 1j * quarter, atol=1e-6, rtol=1e-6)


def test_averaging_preserves_an_equivariant_complex_head():
    obs = torch.tensor([[2.+1j, 3.-2j], [-2.+3j, 6.-1j]])
    edit = obs.conj()

    def linear(y, u):
        return (.2+.3j)*y - .1j*u

    raw, half, quarter = rotated_proposals(linear, obs, edit)
    torch.testing.assert_close(half, raw)
    torch.testing.assert_close(quarter, raw)
