"""Behavioral checks for MIRAGE same-row source scoring and non-compensatory Gates."""

from __future__ import annotations

import importlib

import pytest

from cvsrffi.phase1_mirage.protocol import ProxyRole, SourcePartition


def _apis():
    """Import lazily so RED demonstrates missing scoring rather than a test typo."""

    calibration = importlib.import_module("cvsrffi.phase1_mirage.calibration")
    scoring = importlib.import_module("cvsrffi.phase1_mirage.scoring")
    return calibration, scoring


def _known_rows(calibration, *, fold: int, candidate_shift: float = 0.0):
    return (
        calibration.KnownScoreRow(
            physical_id=f"known-{fold}-0",
            query_id=f"known-query-{fold}-0",
            quality=0.90,
            unknown_risk=0.10,
            inside_registered_support=True,
            predicted_class=0,
            true_class=0,
            receiver="rx-a",
            day=0,
            scene="clear",
            fold=fold,
        ),
        calibration.KnownScoreRow(
            physical_id=f"known-{fold}-1",
            query_id=f"known-query-{fold}-1",
            quality=0.90,
            unknown_risk=0.10 + candidate_shift,
            inside_registered_support=True,
            predicted_class=0,
            true_class=1,
            receiver="rx-b",
            day=1,
            scene="rain",
            fold=fold,
        ),
    )


def _proxy_rows(calibration, *, fold: int):
    return (
        calibration.ProxyScoreRow(
            physical_id=f"proxy-{fold}-0",
            query_id=f"proxy-query-{fold}-0",
            quality=0.90,
            unknown_risk=0.90,
            inside_registered_support=False,
            predicted_class=0,
            fold=fold,
        ),
        calibration.ProxyScoreRow(
            physical_id=f"proxy-{fold}-1",
            query_id=f"proxy-query-{fold}-1",
            quality=0.90,
            unknown_risk=0.10,
            inside_registered_support=True,
            predicted_class=0,
            fold=fold,
        ),
        calibration.ProxyScoreRow(
            physical_id=f"proxy-{fold}-2",
            query_id=f"proxy-query-{fold}-2",
            quality=0.20,
            unknown_risk=0.50,
            inside_registered_support=False,
            predicted_class=1,
            fold=fold,
        ),
    )


def _selection_decisions(calibration, *, fold: int = 0, candidate_id: str = "A"):
    known = calibration.KnownScoreTable(
        role=SourcePartition.V_SELECT,
        rows=_known_rows(calibration, fold=fold),
        update_count=0,
    )
    proxy = calibration.ProxyScoreTable(
        role=ProxyRole.P_SELECT,
        rows=_proxy_rows(calibration, fold=fold),
        update_count=0,
    )
    thresholds = importlib.import_module("cvsrffi.phase1_mirage.head").DecisionThresholds(
        tau_q=0.50,
        tau_reg=0.20,
        tau_unk=0.80,
    )
    return calibration.freeze_selection_decisions(known, proxy, thresholds, candidate_id=candidate_id)


def test_same_row_metrics_are_computed_from_one_frozen_selection_decision_table():
    """Catch assembled marginal maxima, treating defer as rejection, or proxy false-accept loss."""

    calibration, scoring = _apis()
    metrics = scoring.score_same_row(_selection_decisions(calibration))

    assert metrics.candidate_id == "A"
    assert metrics.fold == 0
    assert metrics.macro_accuracy == pytest.approx(0.50)
    assert metrics.per_class_accuracy == {0: 1.0, 1: 0.0}
    assert metrics.min_class_accuracy == 0.0
    assert metrics.receiver_accuracy == {"rx-a": 1.0, "rx-b": 0.0}
    assert metrics.day_accuracy == {0: 1.0, 1: 0.0}
    assert metrics.scene_accuracy == {"clear": 1.0, "rain": 0.0}
    assert metrics.worst_scene_accuracy == 0.0
    assert metrics.known_frr == 0.0
    assert metrics.proxy_explicit_rejection == pytest.approx(1.0 / 3.0)
    assert metrics.proxy_false_accept == pytest.approx(1.0 / 3.0)
    assert metrics.proxy_defer == pytest.approx(1.0 / 3.0)
    assert metrics.proxy_coverage == pytest.approx(2.0 / 3.0)
    assert metrics.proxy_auroc == pytest.approx(5.0 / 6.0)
    assert metrics.proxy_update_count == 0


def test_same_row_macro_and_per_class_metrics_weight_scenes_equally_inside_a_fold():
    """Catch a crowded clear scene overwhelming a sparse rain scene in source selection."""

    calibration, scoring = _apis()
    known_rows = tuple(
        calibration.KnownScoreRow(
            physical_id=f"scene-known-{index}",
            query_id=f"scene-known-query-{index}",
            quality=0.90,
            unknown_risk=0.10,
            inside_registered_support=True,
            predicted_class=0 if index < 3 else 1,
            true_class=0,
            receiver="rx",
            day=0,
            scene="clear" if index < 3 else "rain",
            fold=0,
        )
        for index in range(4)
    )
    known = calibration.KnownScoreTable(role=SourcePartition.V_SELECT, rows=known_rows, update_count=0)
    proxy = calibration.ProxyScoreTable(role=ProxyRole.P_SELECT, rows=_proxy_rows(calibration, fold=0), update_count=0)
    thresholds = importlib.import_module("cvsrffi.phase1_mirage.head").DecisionThresholds(
        tau_q=0.50,
        tau_reg=0.20,
        tau_unk=0.80,
    )

    metrics = scoring.score_same_row(
        calibration.freeze_selection_decisions(known, proxy, thresholds, candidate_id="scene-balanced")
    )

    assert metrics.scene_accuracy == {"clear": 1.0, "rain": 0.0}
    assert metrics.macro_accuracy == 0.50
    assert metrics.per_class_accuracy == {0: 0.50}


def _metric(scoring, *, arm: str, fold: int, macro: float, minimum: float, worst: float, auc: float, frr: float = 0.05):
    return scoring.SameRowMetrics(
        candidate_id=arm,
        fold=fold,
        macro_accuracy=macro,
        per_class_accuracy={0: macro, 1: minimum},
        min_class_accuracy=minimum,
        receiver_accuracy={"rx": macro},
        day_accuracy={0: macro},
        scene_accuracy={"scene": worst},
        worst_scene_accuracy=worst,
        known_frr=frr,
        proxy_explicit_rejection=0.80,
        proxy_false_accept=0.10,
        proxy_defer=0.10,
        proxy_coverage=0.90,
        proxy_auroc=auc,
        proxy_update_count=0,
        decision_table_id=f"{arm}-{fold}",
    )


def _summary(scoring, *, arm: str, values):
    return scoring.aggregate_sixfold(arm, tuple(values))


def _gate1(scoring):
    receipts = tuple(
        scoring.TrainerFoldReceipt(
            fold=fold,
            receipt={
                "schema": "phase1_mirage_completion_receipt_v1",
                "status": "COMPLETED",
                "checkpoint_sha256": "a" * 64,
                "epochs_completed": 200,
                "selected_epoch": 100,
                "selection_source": "V_select",
                "v_select_known_macro": 0.70,
                "v_select_worst_scene": 0.60,
            },
        )
        for fold in range(6)
    )
    return scoring.Gate1Evidence(
        split_receipt_valid=True,
        receiver_tx_disjoint=True,
        proxy_origin_valid=True,
        target_training_access_count=0,
        target_calibration_access_count=0,
        target_selection_access_count=0,
        checkpoint_forward_complete=True,
        trainer_receipts=receipts,
    )


def test_gate_failure_cannot_be_compensated_by_other_metrics():
    """Catch a promotion that lets excellent Gate2 compensate for failed Gate3 proxy readiness."""

    _, scoring = _apis()
    baseline = _summary(
        scoring,
        arm="B0",
        values=tuple(
            _metric(scoring, arm="B0", fold=fold, macro=0.80, minimum=0.70, worst=0.60, auc=0.80)
            for fold in range(6)
        ),
    )
    great_macro_bad_proxy = _summary(
        scoring,
        arm="A",
        values=tuple(
            _metric(scoring, arm="A", fold=fold, macro=0.82, minimum=0.71, worst=0.60, auc=0.84)
            for fold in range(6)
        ),
    )

    receipt = scoring.evaluate_source_gates(
        candidate=great_macro_bad_proxy,
        baseline=baseline,
        gate1=_gate1(scoring),
    )

    assert receipt.gate1_pass
    assert receipt.gate2_pass
    assert not receipt.gate3_pass
    assert not receipt.promoted


def test_gate2_requires_exactly_six_folds_and_the_5_of_6_boundary_is_not_rounded_up():
    """Catch choosing a best fold, accepting five-of-five, or accepting only four non-degraded folds."""

    _, scoring = _apis()
    baseline = _summary(
        scoring,
        arm="B0",
        values=tuple(
            _metric(scoring, arm="B0", fold=fold, macro=0.80, minimum=0.70, worst=0.60, auc=0.80)
            for fold in range(6)
        ),
    )
    five_of_six = _summary(
        scoring,
        arm="A",
        values=tuple(
            _metric(
                scoring,
                arm="A",
                fold=fold,
                macro=0.826 if fold < 5 else 0.794,
                minimum=0.716 if fold < 5 else 0.689,
                worst=0.606 if fold < 5 else 0.594,
                auc=0.90,
            )
            for fold in range(6)
        ),
    )
    four_of_six = _summary(
        scoring,
        arm="B",
        values=tuple(
            _metric(
                scoring,
                arm="B",
                fold=fold,
                macro=0.832 if fold < 4 else 0.794,
                minimum=0.721 if fold < 4 else 0.689,
                worst=0.610 if fold < 4 else 0.594,
                auc=0.90,
            )
            for fold in range(6)
        ),
    )

    five_receipt = scoring.evaluate_source_gates(five_of_six, baseline, _gate1(scoring))
    assert five_receipt.gate2_pass
    assert five_receipt.gate2_checks["fold_nondegrade_5_of_6"] is True
    four_receipt = scoring.evaluate_source_gates(four_of_six, baseline, _gate1(scoring))
    assert not four_receipt.gate2_pass
    assert four_receipt.gate2_checks["fold_nondegrade_5_of_6"] is False
    with pytest.raises(scoring.ScoringProtocolError, match="exactly six"):
        _summary(
            scoring,
            arm="too-few",
            values=tuple(
                _metric(scoring, arm="too-few", fold=fold, macro=0.82, minimum=0.71, worst=0.60, auc=0.90)
                for fold in range(5)
            ),
        )


def test_gate4_only_scores_two_sealed_target_summaries_and_requires_all_four_unknown_scenes():
    """Catch target selection feedback or a Gate4 pass with an omitted deployment scene."""

    _, scoring = _apis()
    b0 = scoring.SealedTargetSummary(
        arm_id="B0*",
        seal_id="sealed-b0",
        known_macro_accuracy=0.80,
        min_class_accuracy=0.70,
        worst_scene_accuracy=0.60,
        explicit_unknown_rejection={"global": 0.0, "clear": 0.0, "low_elev": 0.0, "rain": 0.0},
        known_frr=0.10,
    )
    candidate = scoring.SealedTargetSummary(
        arm_id="M*",
        seal_id="sealed-m",
        known_macro_accuracy=0.82,
        min_class_accuracy=0.70,
        worst_scene_accuracy=0.60,
        explicit_unknown_rejection={"global": 0.70, "clear": 0.70, "low_elev": 0.70, "rain": 0.70},
        known_frr=0.10,
    )

    receipt = scoring.evaluate_gate4(candidate, b0)
    assert receipt.passed
    assert receipt.checks["all_unknown_scenes"] is True
    with pytest.raises(scoring.ScoringProtocolError, match="exactly"):
        scoring.SealedTargetSummary(
            arm_id="bad",
            seal_id="sealed-bad",
            known_macro_accuracy=0.82,
            min_class_accuracy=0.70,
            worst_scene_accuracy=0.60,
            explicit_unknown_rejection={"global": 0.70, "clear": 0.70, "rain": 0.70},
            known_frr=0.10,
        )


def test_unique_arm_selection_uses_only_promoted_source_receipts_and_records_stable_tie_break():
    """Catch target-backed selection or an ambiguous all-equal source arm choice."""

    _, scoring = _apis()
    baseline = _summary(
        scoring,
        arm="B0",
        values=tuple(
            _metric(scoring, arm="B0", fold=fold, macro=0.80, minimum=0.70, worst=0.60, auc=0.80)
            for fold in range(6)
        ),
    )
    candidates = []
    for arm in ("C", "A"):
        summary = _summary(
            scoring,
            arm=arm,
            values=tuple(
                _metric(scoring, arm=arm, fold=fold, macro=0.83, minimum=0.72, worst=0.61, auc=0.90)
                for fold in range(6)
            ),
        )
        candidates.append(
            scoring.ArmSelectionCandidate(
                arm_id=arm,
                gate_receipt=scoring.evaluate_source_gates(summary, baseline, _gate1(scoring)),
                bundle_bytes=1_024,
            )
        )

    selection = scoring.select_unique_arm(tuple(candidates))
    assert selection.selected_arm_id == "A"
    assert selection.tie_break == "stable_arm_id"
    assert selection.used_target_data is False


def test_unique_arm_selection_uses_normalized_weakest_gate_slack_not_raw_unit_margins():
    """Catch an arm ranking that reverses when accuracy, AUROC, and fold-count units are mixed."""

    _, scoring = _apis()
    baseline = _summary(
        scoring,
        arm="B0",
        values=tuple(
            _metric(scoring, arm="B0", fold=fold, macro=0.80, minimum=0.70, worst=0.60, auc=0.80)
            for fold in range(6)
        ),
    )
    arm_a = _summary(
        scoring,
        arm="A",
        values=tuple(
            _metric(scoring, arm="A", fold=fold, macro=0.825, minimum=0.712, worst=0.605, auc=0.90, frr=0.08)
            for fold in range(6)
        ),
    )
    arm_b = _summary(
        scoring,
        arm="B",
        values=tuple(
            _metric(scoring, arm="B", fold=fold, macro=0.823, minimum=0.714, worst=0.606, auc=0.88, frr=0.085)
            for fold in range(6)
        ),
    )
    receipt_a = scoring.evaluate_source_gates(arm_a, baseline, _gate1(scoring))
    receipt_b = scoring.evaluate_source_gates(arm_b, baseline, _gate1(scoring))

    assert receipt_a.promoted and receipt_b.promoted
    assert min(receipt_a.gate_margins.values()) < min(receipt_b.gate_margins.values())
    assert receipt_a.normalized_gate_slacks["gate2_minimum_delta"] == pytest.approx(0.20)
    assert receipt_b.normalized_gate_slacks["gate3_known_frr"] == pytest.approx(0.15)
    assert receipt_a.weakest_gate_margin == pytest.approx(0.20)
    assert receipt_b.weakest_gate_margin == pytest.approx(0.15)

    selection = scoring.select_unique_arm(
        (
            scoring.ArmSelectionCandidate("B", receipt_b, bundle_bytes=1_024),
            scoring.ArmSelectionCandidate("A", receipt_a, bundle_bytes=1_024),
        )
    )

    assert selection.selected_arm_id == "A"
    assert selection.tie_break == "weakest_gate_margin"
