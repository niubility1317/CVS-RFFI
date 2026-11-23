import torch

from experiments.pamf_unrolled_restore.wisig_repeat_drift import RepeatDrift, objective, polar_edit
from experiments.pamf_unrolled_restore.wisig_repeat_drift_source import BASE_READOUTS, DRIFT_READOUTS, proposals


def period():
    random = torch.Generator().manual_seed(2941)
    return torch.randn(20, dtype=torch.complex64, generator=random).repeat(13)[:256][None]


def test_affine_drift_untouched_and_curvature_improves_visible_section():
    model = RepeatDrift(True)
    y = period()
    time = model.time.float()
    original = polar_edit(y, .03 + .001*time, .5+.052*time)
    log_gain, phase, confidence = model.estimate_drift(original)
    assert confidence.item() > .99
    torch.testing.assert_close(log_gain, torch.zeros_like(log_gain), atol=2e-6, rtol=0)
    torch.testing.assert_close(phase, torch.zeros_like(phase), atol=2e-6, rtol=0)

    shape = .6 * torch.cos((time-99.5)/30)
    shape = (shape - (shape @ model.remove_affine.mT) @ model.affine.mT) * model.edge_taper
    warped = polar_edit(y, .4 * shape, shape)
    log_gain, phase, _ = model.estimate_drift(warped)
    assert (phase-shape).square().mean() < .7 * shape.square().mean()
    assert not phase[:, :40].any() and not phase[:, 160:].any()
    torch.testing.assert_close(phase @ model.affine, torch.zeros(1, 2), atol=1e-5, rtol=0)


def test_batch24_gradients_and_polar_readouts():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        rng = torch.Generator().manual_seed(12345)
        baseline = RepeatDrift(False)
        model = RepeatDrift(True)
        model.load_state_dict(baseline.state_dict())
        waves = (model.canonical[None,:] +
                 .045*torch.randn(24,256,dtype=torch.complex64,generator=rng))
        for arm in (model,baseline):
            loss, metrics = objective(arm, waves, rng, identity=True)
            assert metrics.shape == (7,) and torch.isfinite(metrics).all()
            loss.backward()
            assert all(net[-1].weight.grad.norm()>0 for net in (arm.timing,arm.coarse,arm.refine))
        model.eval()
        baseline.eval()
        raw = polar_edit(waves, .15*torch.cos(model.time/22), .30*torch.cos(model.time/17))
        values, active, _ = proposals(model,(1000.,1000.),raw)
        assert len(values) == len(DRIFT_READOUTS) and not active.any()
        assert torch.equal(values[4],raw) and torch.equal(values[5],raw) and torch.equal(values[6],raw)
        values, active, _ = proposals(model,(-1.,-1.),raw)
        assert active.all()
        torch.testing.assert_close(values[0].abs(),values[5].abs(),atol=2e-6,rtol=1e-6)
        assert len(proposals(baseline,(1000.,1000.),raw)[0]) == len(BASE_READOUTS)
        torch.testing.assert_close(values[6], proposals(baseline,(-1.,-1.),raw)[0][-1])
    finally:
        torch.set_num_threads(old_threads)
