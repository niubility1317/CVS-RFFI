import torch

from experiments.pamf_lora_main.weak_guard import GuardFloor, proposed_doses


def test_spectral_guard_is_finite_and_independent_of_each_chirp_rms():
    rng = torch.Generator().manual_seed(21)
    iq = torch.randn((4, 6, 2, 256), generator=rng)
    scale = torch.linspace(.3, 2.2, 6)[None, :, None, None]
    guard = GuardFloor()
    observations = guard(iq)
    assert observations.shape == (4, 8) and torch.isfinite(observations).all()
    torch.testing.assert_close(observations, guard(iq * scale), atol=3e-6, rtol=3e-6)
    assert torch.isfinite(guard(torch.zeros_like(iq))).all()


def test_extensions_touch_only_newly_detected_packets():
    rms = torch.tensor([.020, .025, .070, .080])
    old = torch.tensor([False, False, True, True])
    additional = torch.tensor([True, False, True, False])
    actual = proposed_doses(rms, .05, old, additional)
    expected = torch.tensor([[0., 0., .5, .5],
                             [.25, 0., .5, .5],
                             [.50, 0., .5, .5],
                             [.08, 0., .5, .5]])
    torch.testing.assert_close(actual, expected)


def test_recover_broadcasts_real_per_packet_doses_over_complex_chirps(monkeypatch):
    from experiments.pamf_lora_main import weak_guard_source as source
    from experiments.pamf_lora_main.dura_repair import complex_iq

    iq = torch.zeros((3, 6, 2, 256))
    y = complex_iq(iq)
    phase = torch.ones_like(y)
    rms = torch.ones_like(y.real[..., :1])
    monkeypatch.setattr(source, 'coordinates', lambda *_: (y, phase, rms, rms))
    monkeypatch.setattr(source, 'corruption_score', lambda _: torch.tensor([.07, .02, .02]))

    def model(batch, _quality, _phase):
        return batch, batch + (1 + 2j)

    guard = lambda _: torch.tensor([True, True, False])
    outputs, active = source.recover(iq, None, model, guard, .05)
    assert active == 2
    expected = torch.tensor([[.50, 0., 0.], [.50, .25, 0.],
                             [.50, .50, 0.], [.50, .08, 0.]])
    rows = torch.stack(outputs)
    torch.testing.assert_close(rows[:, :, 0, 0, 0], expected)
    torch.testing.assert_close(rows[:, :, 0, 1, 0], 2 * expected)
    assert torch.equal(rows[:, 2], torch.zeros_like(rows[:, 2]))
