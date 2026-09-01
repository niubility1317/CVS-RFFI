from __future__ import annotations

import torch

from cvsrffi.fisher_gate import (
    FisherDiscriminabilityUncertaintyGate,
    NormalizedFiveBranchFusion,
)


def _evidence(batch: int = 1):
    return {
        "I": torch.full((batch, 5), 0.01),
        "D": torch.full((batch, 5), 0.01),
        "S": torch.full((batch, 5), 0.01),
        "U": torch.ones(batch, 5),
    }


def test_null_route_dominates_when_all_physical_evidence_is_invalid() -> None:
    gate = FisherDiscriminabilityUncertaintyGate(
        branch_count=5, correction_dim=4, delta_max=0.2
    ).eval()
    result = gate(_evidence(), correction_context=torch.zeros(1, 5, 4))

    assert result["null_weight"].item() > 0.9
    assert result["q_sample"].item() < 0.1
    torch.testing.assert_close(
        result["weights"].sum(dim=-1) + result["null_weight"], torch.ones(1)
    )


def test_reliable_branch_wins_and_learned_correction_is_bounded_and_detached() -> None:
    gate = FisherDiscriminabilityUncertaintyGate(
        branch_count=5, correction_dim=4, delta_max=0.15
    ).eval()
    evidence = _evidence()
    for key in ("I", "D", "S"):
        evidence[key][:, 2] = 1.0
    evidence["U"][:, 2] = 0.0
    context = torch.randn(1, 5, 4, requires_grad=True)
    result = gate(evidence, correction_context=context)

    assert result["weights"].argmax(dim=-1).item() == 2
    assert result["weights"][0, 2].item() > 0.5
    assert result["correction"].abs().max().item() <= 0.15 + 1e-7
    result["weights"].sum().backward()
    assert context.grad is None


def test_physical_only_mode_has_exactly_zero_learned_correction() -> None:
    gate = FisherDiscriminabilityUncertaintyGate(
        branch_count=5,
        correction_dim=4,
        delta_max=0.15,
        use_learned_correction=False,
    )
    result = gate(_evidence(), correction_context=torch.randn(1, 5, 4))
    assert torch.equal(result["correction"], torch.zeros_like(result["correction"]))


def test_normalized_fusion_equalizes_branch_norms_before_weighting() -> None:
    torch.manual_seed(5)
    fusion = NormalizedFiveBranchFusion(
        branch_names=("raw", "hom", "phase", "pa", "hos"),
        input_dim=8,
        output_dim=16,
    ).eval()
    branches = {
        "raw": torch.randn(2, 8) * 1000.0,
        "hom": torch.randn(2, 8) * 0.001,
        "phase": torch.randn(2, 8),
        "pa": torch.randn(2, 8) * 10.0,
        "hos": torch.randn(2, 8) * 0.1,
    }
    weights = torch.full((2, 5), 0.2)
    fused, diagnostics = fusion(branches, weights)

    assert fused.shape == (2, 16)
    torch.testing.assert_close(fused.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        diagnostics["projected_norms"], torch.ones(2, 5), atol=1e-5, rtol=1e-5
    )
