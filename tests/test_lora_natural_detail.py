from functools import partial

import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.natural_detail import (
    detail_bands, natural_objective, paired_inputs, transfer_risk, weak_difference,
)


def test_balanced_transfer_penalizes_the_same_loss_of_gain_at_weak_signal_levels():
    time = torch.arange(256)/256.
    expected = .006*(torch.polar(torch.ones_like(time), 2.*torch.pi*4.*time)
                      + torch.polar(torch.ones_like(time), 2.*torch.pi*18.*time))
    expected = expected.expand(3, 6, -1)
    full_risk = transfer_risk(.5*expected, expected, balance=True)
    weak = .25*expected
    weak_risk = transfer_risk(.5*weak, weak, balance=True)
    torch.testing.assert_close(weak_risk, full_risk)
    assert transfer_risk(.5*weak, weak) < weak_risk
    assert transfer_risk(weak, weak, balance=True) == 0.
from experiments.pamf_lora_main.natural_detail_source import RECIPES, render


def test_natural_difference_uses_measured_complex_band_and_shared_noise():
    spectrum = torch.zeros(4, 6, 256, dtype=torch.complex64)
    spectrum[:, :, 0] = 15
    scale = torch.arange(1, 5, dtype=torch.float32)[:, None]
    spectrum[:, :, 4] = scale * (1.+.5j)
    spectrum[:, :, -18] = scale * (.8-1.5j)
    spectrum[:, :, 41] = scale * 3j
    clean = torch.fft.ifft(spectrum, norm='ortho')
    delta = weak_difference(clean)
    wanted = torch.zeros_like(spectrum)
    wanted[..., (4, -18)] = .5 * (spectrum.roll(1, dims=0)-spectrum)[..., (4, -18)]
    torch.testing.assert_close(torch.fft.fft(delta, norm='ortho'), wanted, atol=2e-6, rtol=1e-6)
    generator = torch.Generator().manual_seed(11)
    obs = paired_inputs(clean, RECIPES['detail'].perturb, generator, delta, with_identity=True)
    torch.testing.assert_close(obs[:, 2]-(clean+delta), obs[:, 0]-clean)
    assert torch.count_nonzero(obs[:2, 0]-obs[:2, 1])
    assert torch.equal(obs[-2:, 0], clean[-2:])
    assert transfer_risk(.5*delta, delta) > transfer_risk(delta, delta)


def test_natural_objective_has_transfer_gradient_and_receive_only_render():
    tracker = PacketRestorer('consensus')
    t = torch.arange(256)
    carrier = tracker.reference.chirp
    shape = torch.stack((1.+.02j*torch.cos(2*torch.pi*t/32),
                         1.+.03*torch.sin(2*torch.pi*t/16)))[:, None].expand(-1, 6, -1)
    iq = real_iq(shape*carrier)
    recipe = RECIPES['detail']
    model = recipe.model()
    objective = partial(natural_objective, model, tracker, iq, with_identity=False, recipe=recipe, step=1)
    _, first = objective(torch.Generator().manual_seed(7), preserve=0.)
    loss, second = objective(torch.Generator().manual_seed(7), preserve=.35)
    torch.testing.assert_close(first, second)
    assert torch.isfinite(loss)
    second[-1].backward()
    assert model.prior.out.weight.grad.norm() > 0.
    baseline = lambda x: (x, None)
    active, count = render(iq, None, baseline, tracker, (model.eval(), model), threshold=-1.)
    assert count == len(iq) and torch.isfinite(active).all()
    clean, count = render(iq, None, baseline, tracker, (model, model), threshold=10.)
    assert count == 0 and torch.equal(clean, iq.expand_as(clean))
    assert [int(mask.sum()) for mask in detail_bands(shape)] == [16, 48]
