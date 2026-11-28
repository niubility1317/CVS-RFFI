import torch

from experiments.pamf_lora_main.shared_boundedprior_run import bound_prior, make_eval_gain


def test_prior_cap_and_gradient_both_packet_sizes():
    for length in (256, 8192):
        reference = torch.ones(2, length, dtype=torch.complex64)
        physical = (reference * torch.tensor([1.005, 1.2])[:, None]).requires_grad_()
        result = bound_prior(reference, physical)
        rms = (result-reference).abs().square().mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.tensor([.005, .01]), atol=2e-6, rtol=0)
        gradient, = torch.autograd.grad(result.real.sum(), physical)
        assert torch.isfinite(gradient).all()
        assert gradient[0].norm() > 0


def test_wisig_prediction_uses_bounded_trained_prior():
    class Action:
        def __call__(self, x):
            return x

        def adjoint(self, x):
            return x
    y = torch.ones(1, 256, dtype=torch.complex64)
    action = Action()
    model = make_eval_gain('wisig', lambda x: (bound_prior(x, 2*x), action))(32)
    torch.testing.assert_close(model(y, y, y*7, action), y*1.01)
