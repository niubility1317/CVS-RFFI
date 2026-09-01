from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

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


def test_null_quality_changes_fused_direction_instead_of_being_normalized_away() -> None:
    torch.manual_seed(55)
    fusion = NormalizedFiveBranchFusion(
        branch_names=("raw", "hom", "phase", "pa", "hos"),
        input_dim=8,
        output_dim=8,
    )
    branches = {
        name: torch.randn(2, 8) for name in ("raw", "hom", "phase", "pa", "hos")
    }
    conditional = torch.full((2, 5), 0.2)
    high_quality, _ = fusion(branches, conditional * 0.95)
    low_quality, _ = fusion(branches, conditional * 0.05)
    assert not torch.allclose(high_quality, low_quality, atol=1e-4, rtol=0.0)
    assert F.cosine_similarity(high_quality, low_quality).max().item() < 0.999


@pytest.mark.parametrize(
    ("mode", "ignored"),
    [
        ("i_only", ("D", "S", "U")),
        ("i_d", ("S", "U")),
        ("i_d_s", ("U",)),
    ],
)
def test_factor_ladder_ignores_only_the_omitted_physical_terms(
    mode: str, ignored: tuple[str, ...]
) -> None:
    gate = FisherDiscriminabilityUncertaintyGate(
        branch_count=5, correction_dim=4
    ).eval()
    evidence = {
        "I": torch.full((2, 5), 0.8, requires_grad=True),
        "D": torch.full((2, 5), 0.7, requires_grad=True),
        "S": torch.full((2, 5), 0.6, requires_grad=True),
        "U": torch.full((2, 5), 0.2, requires_grad=True),
    }
    changed = {key: value.clone() for key, value in evidence.items()}
    for key in ignored:
        changed[key].fill_(0.95 if key != "U" else 0.9)
    first = gate(evidence, evidence_mode=mode, enable_correction=False)
    second = gate(changed, evidence_mode=mode, enable_correction=False)
    torch.testing.assert_close(first["physical_logits"], second["physical_logits"])
    torch.testing.assert_close(first["q_sample"], second["q_sample"])


def test_fixed_full_physical_mode_uses_unit_coefficients() -> None:
    gate = FisherDiscriminabilityUncertaintyGate(
        branch_count=5, correction_dim=4
    ).eval()
    evidence = {
        "I": torch.full((2, 5), 0.8),
        "D": torch.full((2, 5), 0.7),
        "S": torch.full((2, 5), 0.6),
        "U": torch.full((2, 5), 0.2),
    }
    output = gate(
        evidence, evidence_mode="full_fixed", enable_correction=False
    )
    expected = (
        torch.log(evidence["I"] + 1e-6)
        + torch.log(evidence["D"] + 1e-6)
        + torch.log(evidence["S"] + 1e-6)
        - evidence["U"]
    )
    torch.testing.assert_close(output["physical_logits"], expected)


@pytest.mark.parametrize(
    ("mode", "active"),
    [
        ("i_only", ("I",)),
        ("i_d", ("I", "D")),
        ("i_d_s", ("I", "D", "S")),
        ("full_fixed", ("I", "D", "S", "U")),
    ],
)
def test_fixed_factor_ladder_has_exact_logits_and_no_coefficient_gradient(
    mode: str, active: tuple[str, ...]
) -> None:
    gate = FisherDiscriminabilityUncertaintyGate(
        branch_count=5, correction_dim=4
    ).train()
    evidence = {
        "I": torch.full((2, 5), 0.8, requires_grad=True),
        "D": torch.full((2, 5), 0.7, requires_grad=True),
        "S": torch.full((2, 5), 0.6, requires_grad=True),
        "U": torch.full((2, 5), 0.2, requires_grad=True),
    }
    output = gate(evidence, evidence_mode=mode, enable_correction=False)
    expected = torch.zeros_like(evidence["I"])
    for key in ("I", "D", "S"):
        if key in active:
            expected = expected + torch.log(evidence[key] + 1e-6)
    if "U" in active:
        expected = expected - evidence["U"]
    torch.testing.assert_close(output["physical_logits"], expected)
    output["physical_logits"].sum().backward()
    assert gate.log_coefficients.grad is None
