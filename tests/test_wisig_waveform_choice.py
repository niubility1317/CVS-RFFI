import torch

from experiments.pamf_repl_view_shuffle.waveform_choice import (
    PROPOSALS, WaveformChoice, objective, select, waveform_risk,
)


def test_identity_choice_is_exact_and_margin_blocks_uncertain_recovery():
    candidates = torch.arange(3*6*4, dtype=torch.float32).reshape(3, 6, 4).to(torch.complex64)
    logits = torch.tensor([[.3, .4, 0., 0., 0., 0.],
                           [0., -.4, .8, 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.]])
    y, index, _ = select(candidates, logits, minimum_margin=.2)
    assert index.tolist() == [0, 2, 0]
    torch.testing.assert_close(y, candidates[torch.arange(3), index], rtol=0, atol=0)


def test_waveform_only_risk_distinguishes_amplitude_loss_and_quadrature():
    clean = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(torch.complex64)
    candidates = torch.stack((clean, .4 * clean, 1j * clean), dim=1)
    risk = waveform_risk(candidates, clean)
    assert torch.equal(risk[:, 0], torch.zeros(3))
    assert (risk[:, 2] > risk[:, 1]).all() and (risk[:, 1] > risk[:, 0]).all()


def test_real_choice_model_learns_both_stages_and_selector_together():
    rng = torch.Generator().manual_seed(29)
    clean = torch.randn(8, 256, dtype=torch.complex64, generator=rng)
    model = WaveformChoice()
    candidates, logits = model(clean)
    assert candidates.shape == (8, len(PROPOSALS), 256)
    assert logits.shape == (8, len(PROPOSALS))
    assert torch.equal(candidates[:, 0], clean)
    loss, parts = objective(model, clean, rng)
    assert torch.isfinite(parts).all()
    loss.backward()
    for p in (model.coarse.out.weight, model.fine.out.weight, model.selector[-1].weight):
        assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
