import pytest
import torch

from experiments.pamf_lora_main.shared_phasetrack_run import SharedPhaseTrackGain, make_eval_gain


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


@pytest.mark.parametrize('width', [32, 1024])
def test_tracker_preserves_anchor_initially_and_has_bounded_phase_equivariance(width):
    torch.manual_seed(10)
    y = torch.randn(2, 8*width, dtype=torch.complex64)
    a, b = y*.8, y*1.1
    model = SharedPhaseTrackGain(width, anchor_real=True, correction_cap=.05)
    torch.testing.assert_close(model(y, a, b, IdentityAction()), a, rtol=0, atol=0)
    with torch.no_grad():
        model.phase_net[-1].weight.normal_(0, .05)
        model.net[-1].weight.normal_(0, .05)
    output = model(y, a, b, IdentityAction())
    ratio = (output-a).abs().square().mean(-1).sqrt()/a.abs().square().mean(-1).sqrt()
    assert torch.all(ratio <= .05001)
    phase = torch.exp(torch.tensor(.7j))
    torch.testing.assert_close(model(y*phase, a*phase, b*phase, IdentityAction()), output*phase, rtol=3e-5, atol=3e-5)


def test_tracker_gets_gradient_and_predict_uses_updated_prior():
    torch.manual_seed(12)
    y = torch.randn(2, 256, dtype=torch.complex64)
    action = IdentityAction()
    model = SharedPhaseTrackGain(32, anchor_real=False)
    result = model(y, y*.8, y*1.2, action)
    target = y*torch.exp(1j*torch.sin(torch.arange(256)/17))
    grad, = torch.autograd.grad((result-target).abs().square().mean(), model.phase_net[-1].weight)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    predict = make_eval_gain('wisig', lambda x: (x*1.1, action))(32)
    torch.testing.assert_close(predict(y, y, y*7, action), y*1.1, rtol=0, atol=0)
