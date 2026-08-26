from __future__ import annotations

import json

import pytest
import torch
from torch.nn import functional as F

from cvsrffi.factored_slow_fast import (
    FactoredSlowFastState,
    apply_factored_context,
    basis_scene_diagnostics,
    fit_factored_state,
    solve_factored_context,
    support_safety_diagnostics,
)
from cvsrffi.slow_fast_cache import GroundFeatureCache


def _cache(*, outer_delta: float = 0.0) -> GroundFeatureCache:
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    receivers: list[str] = []
    days: list[str] = []
    scenes: list[str] = []
    physical_ids: list[str] = []
    views: list[str] = []
    roles: list[str] = []
    class_centers = F.normalize(
        torch.tensor(
            [[0.0, 0.0, 1.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 1.0, 0.2, 0.0]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    rx_shift = {
        "r0": torch.tensor([0.12, 0.00, 0.0, 0.0, 0.0, 0.0]),
        "r1": torch.tensor([-0.08, 0.04, 0.0, 0.0, 0.0, 0.0]),
        "r2": torch.tensor([outer_delta, -outer_delta, 0.0, 0.0, 0.0, 0.0]),
    }
    leo_shift = {
        "leo_clear_weak": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.08, 0.00]),
        "leo_low_elev_weak": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.02, 0.10]),
        "leo_rain_weak": torch.tensor([0.0, 0.0, 0.0, 0.0, -0.06, 0.09]),
    }
    for receiver in ("r0", "r1", "r2"):
        for class_id in (0, 1):
            for sample in range(3):
                pid = f"{receiver}-{class_id}-{sample}"
                clean = F.normalize(
                    class_centers[class_id]
                    + rx_shift[receiver]
                    + torch.tensor([0.0, 0.0, 0.0, 0.0, sample * 0.002, 0.0]),
                    dim=0,
                )
                for view in ("clean", *leo_shift):
                    feature = clean if view == "clean" else F.normalize(clean + leo_shift[view], dim=0)
                    rows.append(feature)
                    labels.append(class_id)
                    receivers.append(receiver)
                    days.append("d0")
                    scenes.append(view)
                    physical_ids.append(pid)
                    views.append(view)
                    roles.append("L_s")
    return GroundFeatureCache(
        features=torch.stack(rows),
        labels=torch.tensor(labels),
        receivers=tuple(receivers),
        days=tuple(days),
        scenes=tuple(scenes),
        physical_sample_ids=tuple(physical_ids),
        views=tuple(views),
        roles=tuple(roles),
    )


def test_outer_receiver_never_changes_factored_slow_state() -> None:
    prototypes = F.normalize(torch.tensor([[0.0, 0.0, 1.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 1.0, 0.2, 0.0]]), dim=1)
    left, left_audit = fit_factored_state(
        _cache(outer_delta=0.0), prototypes, torch.tensor([0, 1]), excluded_receiver="r2", rank_rx=2, rank_leo=2
    )
    right, right_audit = fit_factored_state(
        _cache(outer_delta=0.9), prototypes, torch.tensor([0, 1]), excluded_receiver="r2", rank_rx=2, rank_leo=2
    )

    assert torch.allclose(left.geometric_centers, right.geometric_centers)
    assert torch.allclose(left.receiver_basis, right.receiver_basis)
    assert torch.allclose(left.leo_basis, right.leo_basis)
    assert left_audit["fit_receivers"] == right_audit["fit_receivers"] == ["r0", "r1"]


def test_closed_form_domain_code_uses_only_registered_old_classes() -> None:
    state = FactoredSlowFastState(
        receiver_basis=torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        leo_basis=torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        geometric_centers=torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        decision_prototypes=torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        class_ids=torch.tensor([10, 20]),
        ridge_receiver=0.01,
        ridge_leo=0.01,
    )
    support = F.normalize(
        torch.tensor([[0.2, 0.1, 1.0, 0.0], [0.2, 0.1, 1.0, 0.0], [0.2, 0.1, 0.0, 1.0], [0.2, 0.1, 0.0, 1.0]]),
        dim=1,
    )
    context, audit = solve_factored_context(support, torch.tensor([10, 10, 20, 20]), state)

    assert tuple(context.shape) == (2,)
    assert context[0] > 0.0 and context[1] > 0.0
    assert audit["old_class_count"] == 2
    assert audit["fast_parameter_count"] == 2
    with pytest.raises(ValueError, match="registered old classes"):
        solve_factored_context(support[:2], torch.tensor([10, 99]), state)


def test_zero_context_is_exact_da0_and_query_never_updates_state() -> None:
    state = FactoredSlowFastState(
        receiver_basis=torch.eye(5)[:, :2],
        leo_basis=torch.eye(5)[:, 2:4],
        geometric_centers=F.normalize(torch.randn(2, 5, generator=torch.Generator().manual_seed(2)), dim=1),
        decision_prototypes=F.normalize(torch.randn(2, 5, generator=torch.Generator().manual_seed(3)), dim=1),
        class_ids=torch.tensor([0, 1]),
    )
    query = torch.randn(4, 5, generator=torch.Generator().manual_seed(4))
    before = state.receiver_basis.clone(), state.leo_basis.clone()
    adapted = apply_factored_context(query, state, torch.zeros(4))

    assert torch.allclose(adapted, F.normalize(query, dim=1))
    assert torch.equal(before[0], state.receiver_basis)
    assert torch.equal(before[1], state.leo_basis)


def test_full_support_guard_blocks_correct_to_wrong_and_reports_soft_tails() -> None:
    prototypes = torch.eye(2)
    baseline = F.normalize(torch.tensor([[1.0, 0.1], [0.1, 1.0], [0.8, 0.2], [0.2, 0.8]]), dim=1)
    harmful = F.normalize(torch.tensor([[0.1, 1.0], [0.1, 1.0], [0.6, 0.4], [0.4, 0.6]]), dim=1)
    audit = support_safety_diagnostics(
        baseline, harmful, torch.tensor([0, 1, 0, 1]), prototypes, coverage=0.8, disagreement=0.1,
        min_coverage=0.2, max_disagreement=1.0, min_correct_margin_q10=0.5, min_wrong_margin_median=0.0,
        min_class_margin_cvar=-0.1,
    )

    assert audit["correct_to_wrong_flips"] == 1
    assert audit["safe_to_commit"] is False
    assert "CORRECT_TO_WRONG" in audit["rejection_reasons"]
    assert "correct_margin_ratio_q10" in audit
    assert "worst_class_margin_cvar20" in audit


def test_support_safety_audit_serializes_undefined_subgroup_metrics_as_null() -> None:
    prototypes = torch.eye(2)
    all_correct = F.normalize(
        torch.tensor([[1.0, 0.1], [0.1, 1.0], [0.8, 0.2], [0.2, 0.8]]), dim=1
    )
    audit = support_safety_diagnostics(
        all_correct,
        all_correct,
        torch.tensor([0, 1, 0, 1]),
        prototypes,
        coverage=0.8,
        disagreement=0.1,
        min_coverage=0.2,
        max_disagreement=1.0,
        min_correct_margin_q10=0.5,
        min_wrong_margin_median=0.0,
        min_class_margin_cvar=-0.1,
    )

    assert audit["wrong_margin_delta_median"] is None
    json.dumps(audit, allow_nan=False)


def test_scene_basis_diagnostics_report_angles_and_explained_ratio() -> None:
    state, _audit = fit_factored_state(
        _cache(), F.normalize(torch.tensor([[0.0, 0.0, 1.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 1.0, 0.2, 0.0]]), dim=1),
        torch.tensor([0, 1]), excluded_receiver="r2", rank_rx=2, rank_leo=2,
    )
    result = basis_scene_diagnostics(_cache(), state, excluded_receiver="r2")

    assert set(result["explained_ratio_by_scene"]) == {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
    assert set(result["principal_angle_deg"]) == {"leo_clear_weak__leo_low_elev_weak", "leo_clear_weak__leo_rain_weak", "leo_low_elev_weak__leo_rain_weak"}
    assert all(0.0 <= value <= 1.0 for value in result["explained_ratio_by_scene"].values())
