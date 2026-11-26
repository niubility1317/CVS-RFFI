import torch

from experiments.pamf_lora_main.w24_edit_budget_audit import CAP, oracle_edits, rms


def test_noise_direction_oracles_respect_dose_and_projected_carrier():
    phase = torch.complex(torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
    noise = torch.tensor([[[2., 0., 2., 0.]]], dtype=torch.complex64) * phase
    original = torch.full_like(noise, 2.)
    learned = torch.full_like(noise, .1j)
    same_dose, cap, projected_exact, exact = oracle_edits(original, noise, learned, phase)
    torch.testing.assert_close(rms(same_dose), rms(learned))
    torch.testing.assert_close(rms(cap), CAP * rms(original))
    torch.testing.assert_close((cap * phase.conj()).mean(-1), torch.zeros(1, 1, dtype=cap.dtype))
    torch.testing.assert_close(exact, -noise)
    assert (rms(noise + cap) < rms(noise)).all()
    assert (rms(projected_exact) > rms(cap)).all()
    large = learned * 30
    exact_dose = oracle_edits(original, noise, large, phase)[0]
    torch.testing.assert_close(rms(exact_dose), rms(large))


def test_zero_noise_leaves_every_oracle_control_at_identity():
    received = torch.ones(2, 8, 384, dtype=torch.complex64)
    zeros = torch.zeros_like(received)
    for edit in oracle_edits(received, zeros, received * .2, received):
        assert torch.equal(edit, zeros)
