from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from cvsrffi.fisher_gate_diagnostics import (
    controlled_receiver_probe,
    collect_model_gate_diagnostics,
    detect_gate_collapse,
    expected_behavior_checks,
    summarize_gate_by_group,
)
from model_dual_cvsincnet import build_dual_model


BRANCHES = ("raw", "hom", "phase", "pa", "hos")


def _diagnostics(weights: torch.Tensor, null_weight: torch.Tensor):
    batch, branches = weights.shape
    quality = (1.0 - null_weight).unsqueeze(1).expand(batch, branches)
    return {
        "weights": weights,
        "null_weight": null_weight,
        "q_sample": 1.0 - null_weight,
        "entropy": -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(1),
        "I": quality,
        "D": torch.full_like(weights, 0.6),
        "S": torch.full_like(weights, 0.7),
        "U": 1.0 - quality,
    }


def test_group_summary_keeps_scenario_rows_and_quality_quantiles_together() -> None:
    weights = torch.tensor(
        [
            [0.30, 0.20, 0.20, 0.20, 0.10],
            [0.25, 0.25, 0.20, 0.20, 0.10],
            [0.08, 0.07, 0.05, 0.05, 0.05],
            [0.06, 0.06, 0.04, 0.04, 0.04],
        ]
    )
    diagnostics = _diagnostics(weights, torch.tensor([0.0, 0.0, 0.70, 0.76]))
    summary = summarize_gate_by_group(
        diagnostics,
        groups=["clean", "clean", "leo_rain_weak", "leo_rain_weak"],
        branch_names=BRANCHES,
    )
    assert tuple(summary) == ("clean", "leo_rain_weak")
    assert summary["clean"]["count"] == 2
    assert summary["leo_rain_weak"]["null_mean"] > summary["clean"]["null_mean"]
    assert summary["leo_rain_weak"]["q_p05"] <= summary["leo_rain_weak"]["q_p50"]
    assert set(summary["clean"]["weight_mean"]) == set(BRANCHES)
    assert set(summary["clean"]["I_mean"]) == set(BRANCHES)


def test_collapse_detector_separates_balanced_soft_routing_from_starvation() -> None:
    balanced = _diagnostics(torch.full((20, 5), 0.18), torch.full((20,), 0.10))
    balanced_result = detect_gate_collapse(balanced, branch_names=BRANCHES)
    assert balanced_result["collapsed"] is False
    assert balanced_result["warnings"] == []

    starved_weights = torch.zeros(20, 5)
    starved_weights[:, 0] = 0.98
    starved_weights[:, 1:] = 0.005
    starved = _diagnostics(starved_weights, torch.zeros(20))
    starved_result = detect_gate_collapse(starved, branch_names=BRANCHES)
    assert starved_result["collapsed"] is True
    assert "BRANCH_STARVATION" in starved_result["warnings"]
    assert "OVER_HARD_ROUTING" in starved_result["warnings"]


def test_receiver_probe_controls_quality_before_attributing_receiver_shortcut() -> None:
    torch.manual_seed(91)
    receivers = torch.arange(180) % 3
    snr = receivers.float().unsqueeze(1) * 2.0 + 0.05 * torch.randn(180, 1)
    bandwidth = 0.3 * snr + 0.05 * torch.randn(180, 1)
    sync = -0.2 * snr + 0.05 * torch.randn(180, 1)
    controls = torch.cat([snr, bandwidth, sync], dim=1)
    quality_only_gate = torch.softmax(
        torch.cat([snr, -snr, bandwidth, sync, torch.zeros_like(snr)], dim=1),
        dim=1,
    )
    controlled = controlled_receiver_probe(
        quality_only_gate,
        receivers,
        controls,
        folds=5,
        ridge=1e-2,
    )
    assert controlled["raw_accuracy"] > controlled["majority_accuracy"]
    assert controlled["controlled_excess_accuracy"] < 0.15

    shortcut_gate = quality_only_gate.clone()
    shortcut_gate[torch.arange(180), receivers] += 1.0
    shortcut = controlled_receiver_probe(
        shortcut_gate,
        receivers,
        controls,
        folds=5,
        ridge=1e-2,
    )
    assert shortcut["controlled_excess_accuracy"] > 0.30


def test_expected_behavior_checks_encode_report_examples_without_hard_routing() -> None:
    weights = torch.tensor(
        [
            [0.20, 0.15, 0.25, 0.10, 0.10],  # BPSK
            [0.25, 0.10, 0.15, 0.25, 0.15],  # rich QAM
            [0.04, 0.04, 0.03, 0.03, 0.02],  # low SNR
            [0.20, 0.18, 0.04, 0.18, 0.10],  # cycle slip
            [0.18, 0.16, 0.22, 0.16, 0.08],  # stable phase reference
        ]
    )
    diagnostics = _diagnostics(weights, torch.tensor([0.20, 0.10, 0.84, 0.30, 0.20]))
    checks = expected_behavior_checks(
        diagnostics,
        conditions=["bpsk", "rich_qam", "low_snr", "cycle_slip", "phase_stable"],
        branch_names=BRANCHES,
    )
    assert checks["low_snr_null_above_clean"]["passed"] is True
    assert checks["rich_qam_iq_pa_above_bpsk"]["passed"] is True
    assert checks["cycle_slip_phase_below_stable"]["passed"] is True


def test_collector_keeps_physical_diagnostics_separate_from_labels_and_outcomes() -> None:
    torch.manual_seed(92)
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        model_variant="lite_h",
        input_len=96,
        fast_infer_when_no_aux=False,
        physical_gate_variant="nmfdu_v1",
    )
    x = torch.randn(4, 2, 96)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    result = collect_model_gate_diagnostics(
        model,
        loader,
        torch.device("cpu"),
    )
    assert result["branch_names"] == BRANCHES
    assert result["diagnostics"]["weights"].shape == (4, 5)
    assert result["correct"].shape == (4,)
    assert result["labels"].shape == (4,)
    assert model.training is True
