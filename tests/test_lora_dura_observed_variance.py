import torch

from experiments.pamf_lora_main.dura_observed_variance_source import ObservedVarianceDura, receive_variance
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms


def test_receive_variance_has_per_chirp_real_component_units_and_floor():
    torch.manual_seed(920)
    packet = torch.randn((2,6,256), dtype=torch.complex64)
    power = observed_noise_rms(packet).square().reshape(-1)
    variance = receive_variance(packet)
    assert variance.shape == (12,)
    torch.testing.assert_close(2*variance, power)
    assert torch.all(receive_variance(torch.zeros_like(packet)) == 1e-8)


def test_measured_variance_matches_train_and_eval_and_keeps_joint_gradients():
    torch.manual_seed(921)
    model = ObservedVarianceDura()
    y = 1 + .1 * torch.randn((1,6,256), dtype=torch.complex64)
    rotation = torch.polar(torch.ones(256), torch.linspace(0.,4.,256)).expand_as(y)
    native = torch.stack(((y*rotation).real,(y*rotation).imag), dim=-2).flatten(0,1)
    for training in (True,False):
        model.train(training)
        stages, used, _, coarse = model.solve(native, y)
        torch.testing.assert_close(used, receive_variance(y)[:,None].expand_as(used), atol=0., rtol=0.)
        assert len(stages) == 2 and torch.isfinite(stages[-1]).all()
    model.train()
    coarse, fine = model(y, None, rotation)
    desired = torch.randn_like(y)
    ((coarse-desired).abs().square().mean() + (fine-desired).abs().square().mean()).backward()
    for module in (model.restorer.frozen.coarse, model.restorer.frozen.estimator, *model.restorer.prior):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(g.isfinite().all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads)
