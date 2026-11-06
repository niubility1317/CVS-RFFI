import torch
from experiments.phase1_dura_orthogonal.source_posterior_teacher import source_positive_teacher


def test_teacher_uses_source_clean_and_clips_negative_direction():
    anchor = torch.ones(2, 2, 256)
    clean = torch.stack((2*anchor[0], .5*anchor[1]))
    posterior = 2*anchor
    teacher, alpha = source_positive_teacher(anchor, clean, posterior)
    torch.testing.assert_close(alpha, torch.tensor([.5, 0.]))
    torch.testing.assert_close(teacher[0], 1.5*anchor[0])
    torch.testing.assert_close(teacher[1], anchor[1])
