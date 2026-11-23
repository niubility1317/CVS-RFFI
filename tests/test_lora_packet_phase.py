import torch

from experiments.pamf_lora_main.packet_phase import PacketPhaseTracker


def setup(phases):
    batch = phases.shape[0]
    wave = torch.polar(torch.ones_like(phases), phases)
    orientation = torch.ones_like(wave)
    q = torch.ones(batch, 6)
    return wave, orientation, q, q.median(1).values, torch.zeros(batch)


def test_refinement_preserves_independent_chirp_slopes_and_original_iq():
    tracker = PacketPhaseTracker()
    slope = torch.linspace(-.32, .27, 6).reshape(1, 6, 1)
    gamma = .42
    old = setup(slope * tracker.time + gamma * tracker.quadratic)
    estimated = tracker.curvature(old[0], old[2])
    torch.testing.assert_close(estimated, torch.tensor([[gamma]]), atol=.012, rtol=0)
    new = tracker.refine(old, estimated)
    torch.testing.assert_close(new[0] * new[1], old[0] * old[1], atol=1e-6, rtol=1e-6)
    relative = new[1][..., -1] * new[1][..., 0].conj()
    torch.testing.assert_close(relative.angle() / 2, slope[..., 0], atol=.015, rtol=0)


def test_disagreeing_split_and_all_zero_obs_use_no_shared_curvature():
    tracker = PacketPhaseTracker()
    signs = torch.tensor([1., -1., 1., -1., 1., -1.])
    signal = setup(.45 * signs[None, :, None] * tracker.quadratic)
    fit = tracker.curvature(signal[0], signal[2])
    assert torch.count_nonzero(fit) == 0
    null = tracker.curvature(torch.zeros_like(signal[0]), torch.zeros_like(signal[2]))
    assert torch.isfinite(null).all() and torch.count_nonzero(null) == 0
