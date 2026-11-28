import pytest
import torch

from experiments.pamf_lora_main.shared_fullspectrum_run import SharedFullSpectrumGain, make_eval_gain


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


@pytest.mark.parametrize('width', [32, 1024])
def test_native_full_spectrum_initialization_cap_and_phase_equivariance(width):
    torch.manual_seed(7)
    y = torch.randn(2, 8*width, dtype=torch.complex64)
    a, b = y*.8, y*1.1
    action = IdentityAction()
    model = SharedFullSpectrumGain(width, anchor_real=True, correction_cap=.05)
    torch.testing.assert_close(model(y, a, b, action), a)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, .1)
    result = model(y, a, b, action)
    ratio = (result-a).abs().square().mean(-1).sqrt()/a.abs().square().mean(-1).sqrt()
    assert torch.all(ratio <= .05001)
    phase = torch.exp(torch.tensor(0.73j))
    torch.testing.assert_close(model(y*phase, a*phase, b*phase, action), result*phase, rtol=2e-5, atol=2e-5)


def test_wisig_prediction_uses_updated_physical_prior():
    y = torch.randn(2, 256, dtype=torch.complex64)
    action = IdentityAction()
    model = make_eval_gain('wisig', lambda x: (x*1.1, action))(32)
    torch.testing.assert_close(model(y, y, y*9, action), y*1.1)
