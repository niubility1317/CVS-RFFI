import torch

from ntrs import (
    BoundedWidelyLinearCorrector,
    FastSlowContext,
    NTRSAdapterOnlyResidual,
    NTRSRobustifier,
    NuisanceTangentBasis,
    compute_grouped_physical_descriptors,
    ntrs_stage_scale,
)


def test_adapter_only_residual_trains_q_without_backpropagating_into_anchor():
    module = NTRSAdapterOnlyResidual(
        embedding_dim=16,
        q_dim=8,
        rank=4,
        alpha_max=0.05,
    )
    z_anchor = torch.randn(6, 16, requires_grad=True)
    q = torch.randn(6, 8, requires_grad=True)

    output = module(z_anchor, q, epoch=1)
    loss = output.z_rob.square().mean()
    loss.backward()

    assert q.grad is not None
    assert float(q.grad.abs().sum()) > 0.0
    assert z_anchor.grad is None
    assert torch.all(output.alpha <= 0.05 + 1e-6)


def test_adapter_only_residual_is_q_only_low_rank_without_layernorm():
    module = NTRSAdapterOnlyResidual(
        embedding_dim=12,
        q_dim=5,
        rank=3,
        alpha_max=0.02,
    )
    assert not any(isinstance(child, torch.nn.LayerNorm) for child in module.modules())
    assert module.basis.shape == (12, 3)

    q = torch.randn(4, 5)
    anchor_a = torch.randn(4, 12)
    anchor_b = torch.randn(4, 12) * 7.0
    out_a = module(anchor_a, q, epoch=1)
    out_b = module(anchor_b, q, epoch=1)

    direction_a = torch.nn.functional.normalize(out_a.coefficients, dim=1)
    direction_b = torch.nn.functional.normalize(out_b.coefficients, dim=1)
    assert torch.allclose(direction_a, direction_b, atol=1e-6, rtol=1e-6)
    assert torch.all(out_a.alpha <= 0.02 + 1e-6)
    assert torch.all(out_b.alpha <= 0.02 + 1e-6)


def test_grouped_physical_descriptors_are_finite_detached_and_40_dimensional():
    x = torch.randn(4, 2, 256, requires_grad=True)
    x = x.clone()
    x[0, 0, 0] = float("nan")
    x[1, 1, 1] = float("inf")

    descriptors = compute_grouped_physical_descriptors(x)

    assert descriptors.shape == (4, 40)
    assert torch.isfinite(descriptors).all()
    assert not descriptors.requires_grad


def test_fast_slow_context_updates_only_training_source_domains():
    module = FastSlowContext(
        descriptor_dim=40,
        q_dim=16,
        fast_dim=8,
        slow_dim=8,
        metadata_dim=3,
        num_domains=3,
        slow_ema_decay=0.95,
    )
    x = torch.randn(6, 2, 256)
    domains = torch.tensor([0, 0, 1, 1, 2, 2])
    metadata = torch.randn(6, 3)
    valid = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)

    module.train()
    out = module(
        x,
        metadata=metadata,
        metadata_valid=valid,
        domains=domains,
        update_slow=True,
    )
    counts_after_source = module.slow_counts.clone()

    assert out.q.shape == (6, 16)
    assert out.q_fast.shape == (6, 8)
    assert out.q_slow.shape == (6, 8)
    assert out.q_meta.shape == (6, 8)
    assert out.uncertainty.shape == (6,)
    assert torch.equal(counts_after_source, torch.tensor([2, 2, 2]))

    module.eval()
    _ = module(
        torch.randn(3, 2, 256),
        metadata=None,
        metadata_valid=None,
        domains=torch.tensor([0, 1, 2]),
        update_slow=True,
    )
    assert torch.equal(module.slow_counts, counts_after_source)


def test_bounded_widely_linear_corrector_is_identity_initialized_and_finite():
    corrector = BoundedWidelyLinearCorrector(q_dim=12, taps=3)
    x = torch.randn(3, 2, 256)
    q = torch.randn(3, 12)

    identity = corrector(x, q, stage_scale=1.0)
    assert identity.corrected.shape == x.shape
    assert torch.allclose(identity.corrected, x, atol=1e-6, rtol=1e-6)
    assert torch.equal(identity.energy, torch.zeros_like(identity.energy))

    with torch.no_grad():
        corrector.parameter_head.bias.fill_(0.35)
        corrector.gate_head.bias.fill_(2.0)
    perturbed = corrector(x, q, stage_scale=1.0)
    assert torch.isfinite(perturbed.corrected).all()
    assert torch.all(perturbed.gate >= 0.0)
    assert torch.all(perturbed.gate <= 1.0)
    assert torch.all(perturbed.energy > 0.0)

    invalid = x.clone()
    invalid[0, 0, 0] = float("nan")
    invalid[1, 1, 1] = float("inf")
    safe = corrector(invalid, q, stage_scale=1.0)
    assert torch.isfinite(safe.corrected).all()


def test_tangent_basis_is_orthonormal_and_projection_has_no_off_subspace_energy():
    basis = NuisanceTangentBasis(embedding_dim=12, rank=3, momentum=0.5)
    clean = torch.randn(32, 12)
    directions = torch.randn(3, 12)
    coefficients = torch.randn(32, 3)
    satellite = clean + coefficients @ directions

    basis.train()
    basis.update(clean, satellite)
    gram = basis.basis.transpose(0, 1) @ basis.basis
    projected = basis.project(torch.randn(7, 3))

    assert torch.allclose(gram, torch.eye(3), atol=1e-5, rtol=1e-5)
    assert projected.shape == (7, 12)
    assert torch.max(basis.off_subspace_energy(projected)).item() < 1e-5
    assert int(basis.update_count.item()) == 1


def test_ntrs_stage_schedule_matches_s1_s2a_s2b_s3_contract():
    assert ntrs_stage_scale(1) == 0.0
    assert ntrs_stage_scale(16) == 0.0
    assert 0.0 < ntrs_stage_scale(17) < ntrs_stage_scale(40) == 0.5
    assert 0.5 < ntrs_stage_scale(41) < ntrs_stage_scale(68) == 1.0
    assert ntrs_stage_scale(69) == 1.0
    assert ntrs_stage_scale(200) == 1.0


def test_robustifier_is_zero_in_s1_and_bounded_inside_rank8_subspace_after_ramp():
    module = NTRSRobustifier(
        embedding_dim=16,
        q_dim=8,
        rank=8,
        alpha_max=0.20,
        support_domains=2,
    )
    with torch.no_grad():
        module.coefficient_head[-1].bias.copy_(torch.linspace(-0.5, 0.5, 8))
        module.alpha_head.bias.fill_(2.0)
        module.correctability_head[-1].bias.fill_(2.0)
    z_anchor = torch.randn(5, 16)
    z_phys = z_anchor + 0.1 * torch.randn(5, 16)
    q = torch.randn(5, 8)
    uncertainty = torch.full((5,), 0.1)
    raw_margin = torch.full((5,), 0.4)
    domains = torch.tensor([0, 0, 1, 1, 1])

    module.train()
    s1 = module(
        z_anchor,
        z_phys,
        q,
        uncertainty=uncertainty,
        raw_margin=raw_margin,
        epoch=16,
        update_source_support=True,
        source_domains=domains,
    )
    active = module(
        z_anchor,
        z_phys,
        q,
        uncertainty=uncertainty,
        raw_margin=raw_margin,
        epoch=68,
        update_source_support=False,
        source_domains=domains,
    )

    assert torch.equal(s1.gate, torch.zeros_like(s1.gate))
    assert torch.equal(s1.correction_energy, torch.zeros_like(s1.correction_energy))
    assert torch.all(active.alpha >= 0.0)
    assert torch.all(active.alpha <= 0.20)
    assert torch.all(active.gate >= 0.0)
    assert torch.all(active.gate <= 1.0)
    assert torch.max(module.tangent.off_subspace_energy(active.correction)).item() < 1e-5
    assert torch.max(active.subspace_residual).item() < 1e-5
    assert torch.isfinite(active.z_rob).all()
