import torch

from experiments.pamf_repl_view_shuffle.observable_action import objective
from experiments.pamf_repl_view_shuffle.preamble_phase import ScratchUnroll, ShortFieldPhase
from experiments.pamf_repl_view_shuffle.receiver_pilot import phase
from experiments.pamf_repl_view_shuffle.receiver_timing import preamble_template


def test_split_stf_phase_removes_constant_rotation_and_preserves_slope():
    gauge = ShortFieldPhase()
    t = torch.arange(256, dtype=torch.float32)
    delay = torch.tensor([0, 5, 17])
    center = -.6
    truth = preamble_template(t[None] - 200 - delay[:, None]) * phase(center + .008 * (t - 120))
    angle = torch.tensor([.08, -1.1, 1.9])
    received = truth * phase(angle[:, None])
    rotation, active, _ = gauge.rotation(received, center, .2)
    torch.testing.assert_close(received[1:] * rotation[1:, None], truth[1:], atol=2e-5, rtol=2e-5)
    assert active.tolist() == [False, True, True]


def test_split_stf_refuses_incoherent_or_empty_signal():
    gauge = ShortFieldPhase()
    _, active, _ = gauge.rotation(torch.zeros(2, 256, dtype=torch.complex64), .8, .2)
    assert not active.any()


def test_scratch_train_updates_coarse_and_final_prior_without_parent_weights():
    torch.manual_seed(87)
    model = ScratchUnroll(canonical=True, center=-.6, limit=.3)
    assert sum(p.numel() for p in model.parameters()) == 266951
    original = torch.randn(8, 256, dtype=torch.complex64)
    loss, values = objective(model, original, torch.Generator().manual_seed(13))
    assert torch.isfinite(values).all()
    loss.backward()
    for head in (model.coarse, model.fine):
        assert torch.isfinite(head.out.weight.grad).all()
        assert head.out.weight.grad.norm() > 0
