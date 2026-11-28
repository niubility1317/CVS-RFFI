import pytest
import torch

from experiments.pamf_lora_main.shared_trainprior_run import make_eval_gain, restore
from experiments.pamf_lora_main.shared_dura_realgain import SharedBoundedGain


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


def test_wisig_inference_uses_trained_prior_and_strict_gain_weights():
    action = IdentityAction()
    packet = torch.randn(2, 256, dtype=torch.complex64)
    model = SharedBoundedGain(32, anchor_real=False, correction_cap=.05)
    evaluation = make_eval_gain('wisig', lambda x: (x*2, action))(32)
    evaluation.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(evaluation(packet, packet, packet*7, action), packet*2)


def test_restore_routes_gradient_to_prior_and_gain():
    action = IdentityAction()
    packet = torch.randn(2, 256, dtype=torch.complex64)
    prior = torch.nn.Parameter(torch.tensor(1.3))
    model = SharedBoundedGain(32, anchor_real=False, correction_cap=.05)
    out, _, _ = restore(packet, lambda y: (y*prior, action), lambda y: y*.7, model)
    loss = (out-packet*.9).abs().square().mean()
    gradients = torch.autograd.grad(loss, (prior, model.net[-1].weight))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)


def test_unknown_packet_length_is_rejected():
    with pytest.raises(ValueError):
        SharedBoundedGain(31, anchor_real=False)
