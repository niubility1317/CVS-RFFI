import torch

from experiments.pamf_lora_main.w24_noise_components import component_edits, power


def test_frequency_and_chirp_components_close_exact_noise_in_receive_scale():
    torch.manual_seed(98)
    source = torch.randn(4, 8, 384, dtype=torch.complex64)
    angle = torch.randn(4, 8, 384)
    phase = torch.polar(torch.ones_like(angle), angle)
    near, far, common, private, far_plus, near_plus, common_equal, private_equal, exact = (
        component_edits(source, phase)
    )
    torch.testing.assert_close(near + far, exact, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(common + private, exact, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(power(common) + power(private), power(source))
    torch.testing.assert_close(power(near) + power(far), power(source))
    torch.testing.assert_close(power(common_equal), power(private_equal), atol=1e-7, rtol=1e-5)
    private_frame = (private * phase.conj()).mean(-2)
    torch.testing.assert_close(private_frame, torch.zeros_like(private_frame), atol=3e-7, rtol=0)
    kept_common_near = torch.fft.fft((source + far_plus) * phase.conj(), norm='ortho')
    kept_common_far = torch.fft.fft((source + near_plus) * phase.conj(), norm='ortho')
    freqs = (384 * torch.fft.fftfreq(384)).round().abs()
    assert kept_common_near[..., freqs > 32].abs().max() < 2e-6
    assert kept_common_far[..., freqs <= 32].abs().max() < 2e-6


def test_clean_trace_produces_exact_identity_deltas():
    noise = torch.zeros(2, 8, 384, dtype=torch.complex64)
    edits = component_edits(noise, torch.ones_like(noise))
    assert all(torch.equal(x, noise) for x in edits)
