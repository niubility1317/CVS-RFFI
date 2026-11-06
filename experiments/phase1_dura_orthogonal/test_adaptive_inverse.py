import torch
from experiments.phase1_dura_orthogonal.adaptive_inverse import AdaptiveInverseRecovery, widely_linear_filter


def test_widely_linear_filter_matches_complex_formula_and_is_per_record():
    x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]], [[-1., 1., 0.], [2., 0., -2.]]])
    taps = torch.tensor([[[2.], [3.], [4.], [5.]], [[-1.], [2.], [0.5], [-3.]]])
    z = torch.complex(x[:, 0], x[:, 1])
    main = torch.complex(taps[:, 0, 0], taps[:, 1, 0])[:, None]
    image = torch.complex(taps[:, 2, 0], taps[:, 3, 0])[:, None]
    expected = main*z + image*z.conj()
    actual = widely_linear_filter(x, taps)
    torch.testing.assert_close(torch.complex(actual[:, 0], actual[:, 1]), expected)


def test_direct_model_initially_identity_and_receives_gradients():
    torch.manual_seed(31)
    model = AdaptiveInverseRecovery(width=8, taps=5)
    x = torch.randn(3, 2, 256)
    output = model(x)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    (output-x.roll(1, -1)).square().mean().backward()
    assert model.wave.weight.grad is not None and torch.isfinite(model.wave.weight.grad).all()
    assert model.filter_head.weight.grad is not None and torch.isfinite(model.filter_head.weight.grad).all()
    assert torch.count_nonzero(model.filter_head.weight.grad)


def test_grouped_filter_equals_individual_filters_for_multiple_taps():
    torch.manual_seed(72)
    x = torch.randn(4, 2, 16)
    taps = torch.randn(4, 4, 5)
    grouped = widely_linear_filter(x, taps)
    individual = torch.cat([widely_linear_filter(x[n:n+1], taps[n:n+1]) for n in range(len(x))])
    torch.testing.assert_close(grouped, individual)
