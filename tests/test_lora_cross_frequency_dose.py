import torch

from experiments.pamf_lora_main.cross_frequency_dose import CrossFrequencyDose, initialize, objective
from experiments.pamf_lora_main.cross_frequency_dose_source import MODES, RECIPES, recover
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq


def test_equal_rereadout_uses_the_same_physical_edit_budget_and_fixed_prior():
    torch.manual_seed(41)
    y = torch.randn(18, 6, 256, dtype=torch.complex64)
    model = CrossFrequencyDose(separate=True).eval()
    model.prior.calibrate([y])
    observed = y[:4]
    rms = torch.linspace(.7, 1.3, 6)[None, :, None].expand(4, -1, -1)
    state = model.observe(observed, torch.ones_like(rms), rms)
    analytic = model.apply(observed, rms, state[1], torch.ones(4, 2))
    same_prior = model.prior.changes(observed, rms)[0][2]
    torch.testing.assert_close(analytic, same_prior, atol=1e-6, rtol=1e-5)
    assert (analytic.mean(-1).abs() < 1e-7).all()
    iq = real_iq(observed)
    tracker = PacketRestorer('consensus').eval()
    modes, enabled, information = recover(iq, tracker, [model, model], 0., compare=True)
    assert (len(modes), enabled, len(information)) == (len(MODES) - 2, 4, 4)
    changes = torch.complex(modes[6:8, ..., 0, :], modes[6:8, ..., 1, :]) - observed
    actual = changes.abs().square().mean((2, 3)).sqrt()
    torch.testing.assert_close(actual[0], actual[1], atol=1e-7, rtol=1e-5)
    bypass, active, _ = recover(iq, tracker, [model, model], float('inf'), compare=True)
    assert active == 0 and torch.equal(bypass, iq.expand_as(bypass))


def test_both_stages_fit_one_legal_source_objective():
    torch.manual_seed(61)
    tracker = PacketRestorer('consensus').eval()
    model = CrossFrequencyDose(separate=True)
    source = torch.randn(18, 6, 2, 256)
    initialize(model, tracker, source, None)
    loss, values = objective(model, tracker, source[:5], torch.Generator().manual_seed(73),
                             with_identity=True, recipe=RECIPES['separate'], step=95)
    assert values.shape == (6,) and torch.isfinite(values).all()
    loss.backward()
    for network in (model.coarse, model.fine):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
        assert sum(p.grad.abs().sum() for p in network.parameters()) > 0
    assert all(not buffer.requires_grad for buffer in model.prior.buffers())
    tied = model.dose(torch.tensor([[-1., 2.]]), separate=False)
    assert torch.equal(tied[:, 0], tied[:, 1])
