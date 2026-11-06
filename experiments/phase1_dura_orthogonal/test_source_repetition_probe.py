import torch

from experiments.phase1_dura_orthogonal.source_repetition_probe import phase_aligned_direction, positive_source_oracle


def test_phase_aligned_periodic_signal_has_zero_direction():
    torch.manual_seed(1)
    block = torch.randn(2, 2, 16)
    repeated = block.repeat(1, 1, 16)
    torch.testing.assert_close(phase_aligned_direction(repeated, 16), torch.zeros_like(repeated), atol=1e-6, rtol=0)


def test_source_oracle_matches_known_positive_and_negative_steps():
    torch.manual_seed(2)
    anchor = torch.randn(2, 2, 256)
    direction = phase_aligned_direction(anchor, 20)
    clean = anchor + torch.tensor([.3, -.2])[:, None, None]*direction
    alpha, baseline, corrected = positive_source_oracle(anchor, clean, direction)
    torch.testing.assert_close(alpha, torch.tensor([.3, -.2]), atol=1e-6, rtol=0)
    assert corrected[0] < baseline[0]*1e-10
    torch.testing.assert_close(corrected[1], baseline[1])
