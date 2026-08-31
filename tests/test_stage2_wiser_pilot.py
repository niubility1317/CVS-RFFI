from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_wiser_pilot import (
    ARMS,
    P3_ARMS,
    formal_promotion_decision,
    formal_p3_primary_decision,
    load_query_package,
    normalize_p3_arms,
    select_p3_primary_champion,
)


def test_wiser_matrix_contains_baseline_and_all_abc_arms() -> None:
    assert ARMS == ("B0", "A", "B", "C", "ABC")


def test_p3_pilot_registry_is_n0_through_n6_and_isolated_from_legacy() -> None:
    assert P3_ARMS == ("N0", "N1", "N2", "N3", "N4", "N5", "N6")
    assert normalize_p3_arms(("N4",)) == ("N0", "N4")
    assert normalize_p3_arms(("N6", "N2")) == ("N0", "N6", "N2")
    with pytest.raises(ValueError, match="mixed"):
        normalize_p3_arms(("N1", "A"))
    with pytest.raises(ValueError, match="duplicate"):
        normalize_p3_arms(("N2", "N2"))


def _paired_p3_rows(
    *,
    arm: str = "N6",
    p3_ba_delta_pp: tuple[float, float, float] = (4.0, 3.0, 3.5),
    p3_floor_delta_pp: tuple[float, float, float] = (0.0, 1.0, 0.0),
    net_help: tuple[int, int, int] = (3, 2, -1),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, ba, floor, net in zip(
        ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"),
        p3_ba_delta_pp,
        p3_floor_delta_pp,
        net_help,
    ):
        rows.append(
            {
                "schema": "cvs.phase2.wiser_rf.paired_query_delta.v1",
                "control_arm": "N0",
                "candidate_arm": arm,
                "outer_key": "rx_3_19__seed_713102__k_10__new_5",
                "capsule_id": "capsule",
                "split_id": "split",
                "receiver": "rx_3_19",
                "scenario": scenario,
                "probes": {
                    "P1_SOURCE_HEAD": {"balanced_accuracy_delta_pp": -1.0},
                    "P2_SOURCE_PROTOTYPE": {"balanced_accuracy_delta_pp": -1.0},
                    "P3_OLD_D92": {
                        "balanced_accuracy_delta_pp": ba,
                        "floor_delta_pp": floor,
                        "help_count": max(net, 0),
                        "harm_count": max(-net, 0),
                        "net_help_minus_harm": net,
                    },
                },
                "candidate_training_audit": {
                    "final_zero_identity_count": 0,
                    "baseline_joint_condition_number": 2.0,
                    "final_joint_condition_number": 3.0,
                },
            }
        )
    return rows


def test_p3_gate_requires_cross_scene_floor_flip_and_support_safety() -> None:
    decision = formal_p3_primary_decision(_paired_p3_rows(), arm="N6")
    assert decision["passed"] is True
    assert decision["median_p3_ba_delta_pp"] == pytest.approx(3.5)

    unsafe_floor = formal_p3_primary_decision(
        _paired_p3_rows(p3_floor_delta_pp=(0.0, -1.0, 0.0)), arm="N6"
    )
    assert unsafe_floor["passed"] is False
    unsafe_condition = _paired_p3_rows()
    unsafe_condition[0]["candidate_training_audit"] = {
        "final_zero_identity_count": 0,
        "baseline_joint_condition_number": 2.0,
        "final_joint_condition_number": 4.1,
    }
    assert formal_p3_primary_decision(unsafe_condition, arm="N6")["passed"] is False


def test_p3_gate_accepts_every_inclusive_threshold_boundary() -> None:
    rows = _paired_p3_rows(
        p3_ba_delta_pp=(3.0, 3.0, -0.5),
        p3_floor_delta_pp=(0.0, 0.0, 0.0),
        net_help=(1, 1, -1),
    )
    for row in rows:
        row["probes"]["P1_SOURCE_HEAD"]["balanced_accuracy_delta_pp"] = -2.0
        row["probes"]["P2_SOURCE_PROTOTYPE"]["balanced_accuracy_delta_pp"] = -2.0
        row["candidate_training_audit"] = {
            "final_zero_identity_count": 0,
            "baseline_joint_condition_number": 2.0,
            "final_joint_condition_number": 4.0,
        }

    decision = formal_p3_primary_decision(rows, arm="N6")

    assert decision["passed"] is True
    assert decision["median_p3_ba_delta_pp"] == pytest.approx(3.0)
    assert decision["worst_scene_p3_ba_delta_pp"] == pytest.approx(-0.5)
    assert decision["median_p3_floor_delta_pp"] == pytest.approx(0.0)
    assert decision["leo_low_elev_p3_floor_delta_pp"] == pytest.approx(0.0)
    assert decision["condition_ratios"] == [pytest.approx(2.0)] * 3


@pytest.mark.parametrize(
    "audit",
    [
        {"final_zero_identity_count": 1, "baseline_joint_condition_number": 2.0, "final_joint_condition_number": 2.0},
        {"final_zero_identity_count": 0, "baseline_joint_condition_number": 2.0, "final_joint_condition_number": 4.000001},
    ],
)
def test_p3_gate_rejects_nonzero_identity_or_just_over_two_x_condition(
    audit: dict[str, float]
) -> None:
    rows = _paired_p3_rows()
    rows[0]["candidate_training_audit"] = audit

    assert formal_p3_primary_decision(rows, arm="N6")["passed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("p3_ba", (2.99, 2.99, 3.01)),
        ("p3_ba", (-0.51, 3.0, 3.0)),
        ("p3_floor", (-0.01, -0.01, 0.0)),
        ("p3_floor", (0.0, -0.01, 0.0)),
        ("p1", (-2.01, -1.0, -1.0)),
        ("p2", (-2.01, -1.0, -1.0)),
        ("net", (1, 0, 0)),
    ],
)
def test_p3_gate_fails_at_each_formal_threshold(field: str, value: tuple[float, float, float]) -> None:
    rows = _paired_p3_rows()
    if field == "p3_ba":
        for row, item in zip(rows, value):
            row["probes"]["P3_OLD_D92"]["balanced_accuracy_delta_pp"] = item
    elif field == "p3_floor":
        for row, item in zip(rows, value):
            row["probes"]["P3_OLD_D92"]["floor_delta_pp"] = item
    elif field in {"p1", "p2"}:
        probe = "P1_SOURCE_HEAD" if field == "p1" else "P2_SOURCE_PROTOTYPE"
        for row, item in zip(rows, value):
            row["probes"][probe]["balanced_accuracy_delta_pp"] = item
    else:
        for row, item in zip(rows, value):
            row["probes"]["P3_OLD_D92"]["help_count"] = max(int(item), 0)
            row["probes"]["P3_OLD_D92"]["harm_count"] = max(-int(item), 0)
            row["probes"]["P3_OLD_D92"]["net_help_minus_harm"] = int(item)
    assert formal_p3_primary_decision(rows, arm="N6")["passed"] is False


def test_p3_selection_rejects_n1_and_breaks_multiple_passing_arms_deterministically() -> None:
    n1 = formal_p3_primary_decision(_paired_p3_rows(arm="N1"), arm="N1")
    n2 = formal_p3_primary_decision(_paired_p3_rows(arm="N2"), arm="N2")
    n3 = formal_p3_primary_decision(_paired_p3_rows(arm="N3"), arm="N3")

    assert n1["passed"] is False
    assert select_p3_primary_champion({"N1": n1, "N3": n3, "N2": n2}) == "N2"
    assert select_p3_primary_champion({"N1": n1}) is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "binding"])
def test_p3_gate_fails_closed_for_incomplete_or_unbound_paired_evidence(mutation: str) -> None:
    rows = _paired_p3_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "nonfinite":
        rows[0]["probes"] = dict(rows[0]["probes"])
        rows[0]["probes"]["P3_OLD_D92"] = dict(rows[0]["probes"]["P3_OLD_D92"])
        rows[0]["probes"]["P3_OLD_D92"]["balanced_accuracy_delta_pp"] = float("nan")
    else:
        rows[0]["split_id"] = "wrong"
    decision = formal_p3_primary_decision(rows, arm="N6")
    assert decision["passed"] is False


def test_query_package_rejects_label_or_truth_members(tmp_path: Path) -> None:
    path = tmp_path / "query.npz"
    np.savez_compressed(
        path,
        query_leo_weak_iq=np.zeros((2, 2, 256), np.float32),
        query_tokens=np.asarray(["q0", "q1"]),
        query_labels=np.asarray([0, 1]),
    )

    with pytest.raises(ValueError, match="forbidden"):
        load_query_package(path)


def test_formal_promotion_ignores_c_and_requires_all_preregistered_gates() -> None:
    rows = []
    for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        baseline = {
            "P1_SOURCE_HEAD": {"balanced_accuracy": 0.50, "floor": 0.30},
            "P2_SOURCE_PROTOTYPE": {"balanced_accuracy": 0.50, "floor": 0.30},
            "P3_OLD_D92": {"balanced_accuracy": 0.50, "floor": 0.30},
        }
        promoted = {
            "P1_SOURCE_HEAD": {"balanced_accuracy": 0.54, "floor": 0.31},
            "P2_SOURCE_PROTOTYPE": {"balanced_accuracy": 0.52, "floor": 0.31},
            "P3_OLD_D92": {"balanced_accuracy": 0.54, "floor": 0.31},
        }
        rows.extend(
            [
                {"arm": "B0", "scenario": scenario, "probes": baseline, "geometry": {"within_trace": 2.0, "between_within_ratio": 1.0}},
                {"arm": "B", "scenario": scenario, "probes": promoted, "geometry": {"within_trace": 1.8, "between_within_ratio": 1.1}},
                {"arm": "C", "scenario": scenario, "probes": promoted, "geometry": {"within_trace": 1.0, "between_within_ratio": 2.0}},
            ]
        )

    decision = formal_promotion_decision(rows, arm="B")

    assert decision["passed"] is True
    assert decision["formal_arm"] == "B"
    assert decision["scenario_count"] == 3
    assert decision["c_diagnostic_rows_used"] == 0
