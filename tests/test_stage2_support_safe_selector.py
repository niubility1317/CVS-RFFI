from __future__ import annotations

import inspect

import numpy as np
import pytest

import cvsrffi.stage2_support_safe_selector as d6


SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


def _hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _deletion(
    *,
    state: str,
    scenario: str,
    correctness: tuple[int, ...],
    protocol: str,
) -> d6.SupportDeletionEvidence:
    labels = tuple(
        label
        for label in ("a", "b")
        for _ in range(len(correctness) // 2)
    )
    predictions = tuple(
        label if correct else ("b" if label == "a" else "a")
        for label, correct in zip(labels, correctness)
    )
    ids = tuple(
        f"{state}-{scenario}-sample-{index}" for index in range(len(labels))
    )
    return d6.SupportDeletionEvidence(
        protocol=protocol,
        physical_sample_ids=ids,
        parent_received_iq_sha256=tuple(_hash(value) for value in ids),
        labels=labels,
        predicted_labels=predictions,
        margins=tuple(0.5 if value else -0.5 for value in correctness),
    )


def _candidate(
    name: str,
    *,
    view_count: int,
    loo_correct: tuple[int, ...],
    stability_correct: tuple[int, ...] | None = None,
) -> d6.SupportCandidateEvidence:
    stability = stability_correct or loo_correct
    slices = []
    for state in ("before", "after"):
        for scenario in SCENARIOS:
            slices.append(
                d6.SupportSliceEvidence(
                    registration_state=state,
                    scenario=scenario,
                    loo=_deletion(
                        state=state,
                        scenario=scenario,
                        correctness=loo_correct,
                        protocol="support_leave_one_physical_sample_out",
                    ),
                    stability=_deletion(
                        state=state,
                        scenario=scenario,
                        correctness=stability,
                        protocol="support_leave_two_physical_samples_out",
                    ),
                )
            )
    return d6.SupportCandidateEvidence(
        name=name,
        view_count=view_count,
        slices=tuple(slices),
    )


def test_wilson_and_beta_smooth_small_floor_counts():
    assert d6.wilson_lower_bound(1, 10) > 0.0
    assert d6.wilson_lower_bound(1, 10) < 0.10
    assert d6.wilson_lower_bound(0, 10) == 0.0
    assert d6.beta_smoothed_accuracy(1, 10) == pytest.approx(2.0 / 12.0)
    assert d6.beta_smoothed_accuracy(0, 10) == pytest.approx(1.0 / 12.0)


def test_stratified_leave_two_covers_each_physical_support_once():
    labels = ("a",) * 6 + ("b",) * 6 + ("c",) * 6
    folds = d6.stratified_deletion_folds(labels, width_per_class=2)
    assert len(folds) == 3
    assert all(len(fold) == 6 for fold in folds)
    assert sorted(index for fold in folds for index in fold) == list(
        range(len(labels))
    )


def test_collect_deletion_evidence_is_support_only_and_leave_two():
    labels = ("a",) * 4 + ("b",) * 4
    ids = tuple(f"id-{index}" for index in range(8))

    def predictor(train: np.ndarray, held: np.ndarray):
        assert set(train.tolist()).isdisjoint(set(held.tolist()))
        return [labels[index] for index in held], [0.25] * len(held)

    evidence = d6.collect_deletion_evidence(
        labels=labels,
        physical_sample_ids=ids,
        parent_received_iq_sha256=tuple(_hash(value) for value in ids),
        predictor=predictor,
        width_per_class=2,
    )
    assert evidence.protocol == "support_leave_two_physical_samples_out"
    assert evidence.predicted_labels == labels
    assert d6.public_interface_is_support_only()
    signatures = " ".join(
        inspect.signature(function).__str__()
        for function in (
            d6.collect_deletion_evidence,
            d6.select_support_safe_candidate,
        )
    ).lower()
    assert "query" not in signatures
    assert "role" not in signatures
    assert "quota" not in signatures


def test_overall_drop_gate_rejects_anonymous_degraded_candidate():
    baseline = _candidate(
        "reference",
        view_count=1,
        loo_correct=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    )
    degraded = _candidate(
        "anonymous-arm-7",
        view_count=3,
        loo_correct=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
    )
    result = d6.select_support_safe_candidate(
        (baseline, degraded),
        baseline_name="reference",
    )
    assert result["selected_candidate"] == "reference"
    rejected = {row["name"]: row for row in result["rejected"]}
    assert rejected["anonymous-arm-7"]["loo_overall_noninferiority_pass"] is False
    assert "cfo" not in degraded.name.lower()


def test_stability_and_wilson_floor_gate_block_loo_lucky_candidate():
    baseline = _candidate(
        "base",
        view_count=1,
        loo_correct=(1, 1, 1, 1, 1, 1, 1, 0, 1, 0),
        stability_correct=(1, 1, 1, 1, 1, 1, 1, 0, 1, 0),
    )
    lucky = _candidate(
        "lucky-floor",
        view_count=2,
        loo_correct=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0),
        stability_correct=(1, 1, 1, 0, 0, 1, 1, 1, 0, 0),
    )
    result = d6.select_support_safe_candidate(
        (baseline, lucky),
        baseline_name="base",
    )
    assert result["selected_candidate"] == "base"
    rejected = {row["name"]: row for row in result["rejected"]}
    assert (
        rejected["lucky-floor"]["stability_overall_noninferiority_pass"]
        is False
    )


def test_exact_metric_tie_prefers_single_view():
    base = _candidate(
        "base",
        view_count=1,
        loo_correct=(1, 1, 1, 1, 1, 1, 1, 0, 1, 0),
    )
    same_three_view = _candidate(
        "same-metrics",
        view_count=3,
        loo_correct=(1, 1, 1, 1, 1, 1, 1, 0, 1, 0),
    )
    result = d6.select_support_safe_candidate(
        (same_three_view, base),
        baseline_name="base",
    )
    assert result["selected_candidate"] == "base"
    assert result["complexity_tie_break"] == "lower_view_count"


def test_cross_scenario_physical_reuse_and_view_as_k_fail_closed():
    base = _candidate(
        "base",
        view_count=1,
        loo_correct=(1, 1, 1, 1, 1, 1, 1, 0, 1, 0),
    )
    slices = list(base.slices)
    first = slices[0]
    second = slices[1]
    reused_loo = d6.SupportDeletionEvidence(
        protocol=second.loo.protocol,
        physical_sample_ids=first.loo.physical_sample_ids,
        parent_received_iq_sha256=first.loo.parent_received_iq_sha256,
        labels=second.loo.labels,
        predicted_labels=second.loo.predicted_labels,
        margins=second.loo.margins,
    )
    reused_stability = d6.SupportDeletionEvidence(
        protocol=second.stability.protocol,
        physical_sample_ids=first.stability.physical_sample_ids,
        parent_received_iq_sha256=first.stability.parent_received_iq_sha256,
        labels=second.stability.labels,
        predicted_labels=second.stability.predicted_labels,
        margins=second.stability.margins,
    )
    slices[1] = d6.SupportSliceEvidence(
        registration_state=second.registration_state,
        scenario=second.scenario,
        loo=reused_loo,
        stability=reused_stability,
    )
    reused = d6.SupportCandidateEvidence(
        name="base",
        view_count=1,
        slices=tuple(slices),
    )
    with pytest.raises(d6.SupportSafeSelectorError, match="cross-scenario"):
        d6.select_support_safe_candidate((reused,), baseline_name="base")

    invalid_view = d6.SupportCandidateEvidence(
        name="invalid",
        view_count=3,
        slices=base.slices,
        views_count_as_additional_physical_samples=True,
    )
    with pytest.raises(d6.SupportSafeSelectorError, match="fixed-received-IQ"):
        d6.select_support_safe_candidate(
            (base, invalid_view),
            baseline_name="base",
        )
