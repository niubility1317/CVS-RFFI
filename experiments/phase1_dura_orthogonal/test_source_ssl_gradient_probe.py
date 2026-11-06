import torch
from experiments.phase1_dura_orthogonal.source_ssl_gradient_probe import alignment, measure


def test_alignment_detects_opposed_and_zero_gradients():
    feature = torch.tensor([[[1., 0.]], [[-2., 0.]], [[0., 0.]]])
    iq = torch.tensor([[[1., 0.]], [[1., 0.]], [[1., 0.]]])
    cosine, ratio, valid = alignment(feature, iq)
    torch.testing.assert_close(cosine, torch.tensor([1., -1., 0.]))
    torch.testing.assert_close(ratio, torch.tensor([.05, .1, 0.]))
    assert valid.tolist() == [True, True, False]


def test_measure_matches_known_normalized_iq_gradient():
    clean = torch.ones(2, 2, 256)
    restored = 2*clean
    measured = measure(torch.nn.Flatten(), restored, clean)
    torch.testing.assert_close(measured[:, 0], torch.ones(2))
    torch.testing.assert_close(measured[:, 1], torch.full((2,), 25.6))
    torch.testing.assert_close(measured[:, 2], torch.ones(2))
