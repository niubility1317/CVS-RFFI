import torch

from experiments.pamf_lora_main.action_support import confirmation_scores, support_gain


def test_held_chirp_gain_matches_pair_of_squared_error_reductions():
    rng = torch.Generator().manual_seed(7)
    a = torch.randn((4, 16), dtype=torch.complex64, generator=rng)
    b = torch.randn((4, 16), dtype=torch.complex64, generator=rng)
    corrected_a = a + .25 * (b - a)
    corrected_b = b + .5 * (a - b)
    bands = torch.stack((torch.arange(16)<8,torch.arange(16)>=8)).float()
    result = support_gain(a,b,corrected_a,corrected_b,bands)
    torch.testing.assert_close(result,torch.full_like(result,.4375),atol=2e-7,rtol=1e-6)
    same = support_gain(a,a,a,a,bands)
    assert torch.equal(same,torch.zeros_like(same))
    unilateral = support_gain(a,b,corrected_a,b,bands)
    assert torch.equal(unilateral,torch.zeros_like(unilateral))


def test_near_check_does_not_use_outer_bands():
    gains=torch.tensor([[.2,.4,.6,-1.]])
    result=confirmation_scores(gains)
    torch.testing.assert_close(result,torch.tensor([[.4,.05]]))
