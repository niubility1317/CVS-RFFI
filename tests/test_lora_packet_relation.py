import torch

from experiments.pamf_lora_main.packet_relation import anchor_risk, packet_relation_risk


def test_complex_relations_match_identity_and_respond_to_amplitude_and_phase_loss():
    rng = torch.Generator().manual_seed(43)
    reference = torch.randn(8, 6, 256, generator=rng, dtype=torch.complex64)
    pair = reference[:, None].expand(-1, 2, -1, -1)
    assert packet_relation_risk(pair, reference).abs().max() == 0
    assert (packet_relation_risk(.5 * pair, reference) > 0).all()
    assert (packet_relation_risk(pair * 1j, reference) > 0).all()


def test_relations_ignore_packet_order_and_common_translation():
    rng = torch.Generator().manual_seed(73)
    reference = torch.randn(7, 6, 256, generator=rng, dtype=torch.complex64)
    changed = torch.randn(7, 2, 6, 256, generator=rng, dtype=torch.complex64)
    shift = torch.randn(1, 2, 6, 256, generator=rng, dtype=torch.complex64)
    order = torch.randperm(7, generator=rng)
    reference_risk = packet_relation_risk(changed, reference)
    torch.testing.assert_close(packet_relation_risk((changed + shift)[order], reference[order]),
                               reference_risk)


def test_no_self_anchor_and_finite_weak_band_gradient():
    reference = 1e-7 * torch.ones(1, 8, dtype=torch.complex64)
    prediction = torch.ones(1, 2, 8, dtype=torch.complex64)
    assert anchor_risk(prediction, reference) == 0
    rng = torch.Generator().manual_seed(19)
    clean = 1e-5 * torch.randn(4, 6, 256, generator=rng, dtype=torch.complex64)
    restored = torch.randn(4, 2, 6, 256, generator=rng, dtype=torch.complex64).requires_grad_()
    value = packet_relation_risk(restored, clean)
    value.sum().backward()
    assert value.isfinite().all() and restored.grad.isfinite().all()
