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


def test_full_input_replay_uses_registered_full_trace_only(monkeypatch):
    import numpy as np
    from experiments.pamf_lora_main import w24_edit_budget_audit as audit

    packet = np.zeros(8192, dtype=np.complex64)
    observed, signal = packet + 1j, packet + 1
    seen = []

    def trace(raw, scene, receiver, row, namespace):
        assert raw is packet
        seen.append((scene, receiver, row, namespace))
        return observed, signal, observed-signal

    monkeypatch.setattr(audit, 'trace_components', trace)
    monkeypatch.setattr(audit, 'native_full_view', lambda *args: observed)
    clean, ref = audit.full_components('clean', (42, 'rtl_1', packet), verify=True)
    assert clean is ref is packet and not seen
    wave, ref = audit.full_components('practical_low_urban', (42, 'rtl_1', packet), verify=True)
    assert wave is observed and np.array_equal(ref, signal)
    assert seen == [('practical_low_urban', 'rtl_1', 42, audit.WAVEFORM_NAMESPACE)]
