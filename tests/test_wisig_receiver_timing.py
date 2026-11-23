import torch

from experiments.pamf_repl_view_shuffle.receiver_pilot import phase
from experiments.pamf_repl_view_shuffle.receiver_timing import Timing, preamble_template


def test_joint_timing_preserves_received_overlap_and_complex_scale():
    method = Timing('cpu')
    position = torch.arange(256).float()
    delta = torch.tensor([-36, -25, -12, 0, 11, 19, 29, 37])
    relative = position[None, :, None] - 200 - delta[:, None, None] - torch.arange(-2, 3)
    impulse = torch.tensor([.04+.03j, .19-.05j, 1+.3j, -.29+.11j, .08+.14j])
    iq = (preamble_template(relative) @ impulse) * phase(.013 * position)
    thresholds = dict(boundary=0., joint=0.)

    output, diagnostic = method.restore(iq, thresholds)
    assert torch.equal(diagnostic['joint_shift'], delta)
    assert torch.equal(diagnostic['boundary_shift'], delta)
    aligned = output['joint_protocol']
    for sample, shift in enumerate(delta.tolist()):
        output_overlap = slice(max(-shift, 0), min(256-shift, 256))
        input_overlap = slice(max(shift, 0), min(256+shift, 256))
        assert torch.equal(aligned[sample, output_overlap], iq[sample, input_overlap])

    scaled, estimate = method.restore(iq * (2-3j), thresholds)
    assert torch.equal(estimate['joint_shift'], delta)
    torch.testing.assert_close(scaled['joint_protocol'], aligned * (2-3j), atol=1e-5, rtol=1e-5)


def test_timing_threshold_and_zero_signal_keep_original_samples():
    method = Timing('cpu')
    observed = torch.stack((
        preamble_template(torch.arange(256).float() - 218),
        torch.zeros(256, dtype=torch.complex64),
    ))
    output, diagnostics = method.restore(observed, dict(boundary=10., joint=10.))
    assert all(torch.equal(wave, observed) for wave in output.values())
    assert all(torch.isfinite(stat).all() for stat in diagnostics.values())
