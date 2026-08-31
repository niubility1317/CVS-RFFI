"""Minimal historical-pilot lifecycle helpers for WISER-RF."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("B0", "A", "B", "C", "ABC")
P3_ARMS = ("N0", "N1", "N2", "N3", "N4", "N5", "N6")


@dataclass(frozen=True)
class WISERSupportPackage:
    iq: np.ndarray
    labels: np.ndarray
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class WISERQueryPackage:
    iq: np.ndarray
    tokens: tuple[str, ...]


def _load_npz_subset(
    path: str | Path,
    *,
    required: frozenset[str],
    forbidden_tokens: tuple[str, ...],
) -> dict[str, np.ndarray]:
    package_path = Path(path)
    try:
        with np.load(package_path, allow_pickle=False) as arrays:
            members = frozenset(arrays.files)
            if not required.issubset(members):
                raise ValueError(f"WISER package members missing: {package_path}")
            lowered = tuple(name.lower() for name in members)
            if any(token in name for name in lowered for token in forbidden_tokens):
                raise ValueError(f"WISER package exposes forbidden member: {package_path}")
            return {name: np.asarray(arrays[name]) for name in required}
    except OSError as exc:
        raise ValueError(f"cannot load WISER package: {package_path}") from exc


def load_support_package(path: str | Path) -> WISERSupportPackage:
    data = _load_npz_subset(
        path,
        required=frozenset(
            {"support_leo_weak_iq", "support_class_indices", "support_tokens"}
        ),
        forbidden_tokens=("query", "truth", "role", "quota"),
    )
    iq = np.asarray(data["support_leo_weak_iq"], dtype=np.float32)
    labels = np.asarray(data["support_class_indices"], dtype=np.int64)
    tokens = tuple(np.asarray(data["support_tokens"]).astype(str).tolist())
    if iq.ndim != 3 or iq.shape[1:] != (2, 256) or labels.shape != (len(iq),):
        raise ValueError("WISER support package geometry drift")
    if len(tokens) != len(iq) or not np.isfinite(iq).all():
        raise ValueError("WISER support package token/value drift")
    return WISERSupportPackage(iq=iq, labels=labels, tokens=tokens)


def load_query_package(path: str | Path) -> WISERQueryPackage:
    data = _load_npz_subset(
        path,
        required=frozenset({"query_leo_weak_iq", "query_tokens"}),
        forbidden_tokens=("label", "truth", "role", "quota", "class_count"),
    )
    iq = np.asarray(data["query_leo_weak_iq"], dtype=np.float32)
    tokens = tuple(np.asarray(data["query_tokens"]).astype(str).tolist())
    if iq.ndim != 3 or iq.shape[1:] != (2, 256) or len(tokens) != len(iq):
        raise ValueError("WISER query package geometry drift")
    if len(set(tokens)) != len(tokens) or not np.isfinite(iq).all():
        raise ValueError("WISER query package token/value drift")
    return WISERQueryPackage(iq=iq, tokens=tokens)


def normalize_p3_arms(values: Sequence[str]) -> tuple[str, ...]:
    """Return a P3-only arm subset with the frozen N0 baseline first."""

    selected = tuple(str(value).upper() for value in values)
    if not selected:
        raise ValueError("P3 arm subset is empty")
    if len(set(selected)) != len(selected):
        raise ValueError("P3 arm subset contains a duplicate")
    p3_names = set(P3_ARMS)
    if any(name not in p3_names for name in selected):
        if any(name in ARMS for name in selected):
            raise ValueError("P3 arm subset contains mixed legacy and N-series arms")
        raise ValueError("P3 arm subset contains an unknown arm")
    return ("N0",) + tuple(name for name in selected if name != "N0")


def _p3_failed_decision(arm: str, reason: str) -> Mapping[str, Any]:
    return {
        "schema": "cvs.phase2.wiser_rf.p3_primary_gate.v1",
        "candidate_arm": arm,
        "status": "PENDING_EVIDENCE",
        "evidence_complete": False,
        "reason": reason,
        "gates": {},
        "passed": False,
        "eligible_for_champion": arm in {"N2", "N3", "N4", "N5", "N6"},
    }


def _finite_p3_value(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("P3 paired evidence is nonfinite")
    return result


def _p3_probe_delta(row: Mapping[str, Any], probe: str, field: str) -> float:
    probes = row.get("probes")
    if not isinstance(probes, Mapping) or not isinstance(probes.get(probe), Mapping):
        raise ValueError("P3 paired probe evidence is missing")
    return _finite_p3_value(probes[probe].get(field))


def _p3_audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = row.get("candidate_training_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("P3 candidate support audit is missing")
    return audit


def formal_p3_primary_decision(
    rows: Sequence[Mapping[str, Any]], *, arm: str
) -> Mapping[str, Any]:
    """Apply the truth-last, three-scene P3-primary gate to paired N0 evidence."""

    candidate_arm = str(arm).upper()
    if candidate_arm not in P3_ARMS or candidate_arm == "N0":
        raise ValueError("P3 formal decision requires an N1-N6 candidate arm")
    selected = [
        row
        for row in rows
        if str(row.get("candidate_arm")) == candidate_arm
    ]
    by_scenario: dict[str, Mapping[str, Any]] = {}
    try:
        if len(selected) != len(SCENARIOS):
            raise ValueError("P3 paired three-scene evidence is incomplete")
        for row in selected:
            if row.get("schema") != "cvs.phase2.wiser_rf.paired_query_delta.v1":
                raise ValueError("P3 paired schema drift")
            scenario = str(row.get("scenario"))
            if scenario not in SCENARIOS or scenario in by_scenario:
                raise ValueError("P3 paired scenario coverage is invalid")
            if str(row.get("control_arm")) != "N0":
                raise ValueError("P3 paired control must be N0")
            by_scenario[scenario] = row
        if set(by_scenario) != set(SCENARIOS):
            raise ValueError("P3 paired scenario coverage is incomplete")
        bindings = ("outer_key", "capsule_id", "split_id", "receiver")
        binding_values = {
            field: {str(row.get(field, "")) for row in by_scenario.values()}
            for field in bindings
        }
        if any("" in values or len(values) != 1 for values in binding_values.values()):
            raise ValueError("P3 paired binding drift")

        p1 = [_p3_probe_delta(by_scenario[scene], "P1_SOURCE_HEAD", "balanced_accuracy_delta_pp") for scene in SCENARIOS]
        p2 = [_p3_probe_delta(by_scenario[scene], "P2_SOURCE_PROTOTYPE", "balanced_accuracy_delta_pp") for scene in SCENARIOS]
        p3_ba = [_p3_probe_delta(by_scenario[scene], "P3_OLD_D92", "balanced_accuracy_delta_pp") for scene in SCENARIOS]
        p3_floor = [_p3_probe_delta(by_scenario[scene], "P3_OLD_D92", "floor_delta_pp") for scene in SCENARIOS]
        net_help: list[int] = []
        condition_ratios: list[float] = []
        support_safe = []
        for scene in SCENARIOS:
            row = by_scenario[scene]
            probes = row["probes"]
            p3 = probes["P3_OLD_D92"]
            help_count = int(p3.get("help_count"))
            harm_count = int(p3.get("harm_count"))
            net = int(p3.get("net_help_minus_harm"))
            if help_count < 0 or harm_count < 0 or net != help_count - harm_count:
                raise ValueError("P3 flip evidence is invalid")
            net_help.append(net)
            audit = _p3_audit(row)
            baseline_condition = _finite_p3_value(audit.get("baseline_joint_condition_number"))
            final_condition = _finite_p3_value(audit.get("final_joint_condition_number"))
            zero_count = int(audit.get("final_zero_identity_count"))
            if baseline_condition <= 0.0 or final_condition < 0.0 or zero_count < 0:
                raise ValueError("P3 support audit is invalid")
            condition_ratios.append(final_condition / baseline_condition)
            support_safe.append(zero_count == 0 and final_condition <= 2.0 * baseline_condition)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _p3_failed_decision(candidate_arm, str(error))

    median_ba = float(median(p3_ba))
    worst_ba = float(min(p3_ba))
    median_floor = float(median(p3_floor))
    low_elev_floor = float(p3_floor[SCENARIOS.index("leo_low_elev_weak")])
    median_condition_ratio = float(median(condition_ratios))
    total_net_help = int(sum(net_help))
    gates = {
        "eligible_p3_primary_arm": candidate_arm in {"N2", "N3", "N4", "N5", "N6"},
        "median_p3_ba_delta_ge_3pp": median_ba >= 3.0,
        "worst_scene_p3_ba_delta_ge_minus_0_5pp": worst_ba >= -0.5,
        "median_p3_floor_delta_ge_0pp": median_floor >= 0.0,
        "low_elev_p3_floor_delta_ge_0pp": low_elev_floor >= 0.0,
        "p1_ba_delta_ge_minus_2pp_all_scenes": min(p1) >= -2.0,
        "p2_ba_delta_ge_minus_2pp_all_scenes": min(p2) >= -2.0,
        "support_safety_all_scenes": all(support_safe),
        "positive_p3_net_help_at_least_2of3": sum(value > 0 for value in net_help) >= 2,
    }
    return {
        "schema": "cvs.phase2.wiser_rf.p3_primary_gate.v1",
        "candidate_arm": candidate_arm,
        "status": "ANALYZED",
        "evidence_complete": True,
        "outer_key": next(iter({str(row["outer_key"]) for row in by_scenario.values()})),
        "capsule_id": next(iter({str(row["capsule_id"]) for row in by_scenario.values()})),
        "split_id": next(iter({str(row["split_id"]) for row in by_scenario.values()})),
        "receiver": next(iter({str(row["receiver"]) for row in by_scenario.values()})),
        "scenario_count": len(SCENARIOS),
        "p1_ba_delta_pp": p1,
        "p2_ba_delta_pp": p2,
        "p3_ba_delta_pp": p3_ba,
        "p3_floor_delta_pp": p3_floor,
        "p3_net_help_minus_harm": net_help,
        "condition_ratios": condition_ratios,
        "median_p3_ba_delta_pp": median_ba,
        "worst_scene_p3_ba_delta_pp": worst_ba,
        "median_p3_floor_delta_pp": median_floor,
        "leo_low_elev_p3_floor_delta_pp": low_elev_floor,
        "total_p3_net_help_minus_harm": total_net_help,
        "median_condition_ratio": median_condition_ratio,
        "selection_key": (
            -median_ba,
            -worst_ba,
            -median_floor,
            -total_net_help,
            median_condition_ratio,
            int(candidate_arm[1:]),
        ),
        "gates": gates,
        "eligible_for_champion": gates["eligible_p3_primary_arm"],
        "passed": all(gates.values()),
    }


def select_p3_primary_champion(
    decisions: Mapping[str, Mapping[str, Any]]
) -> str | None:
    """Return the fixed-order unique P3 champion, or None without a pass."""

    eligible = [
        (str(arm), decision)
        for arm, decision in decisions.items()
        if str(arm) in {"N2", "N3", "N4", "N5", "N6"}
        and decision.get("passed") is True
        and decision.get("eligible_for_champion") is True
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: tuple(item[1]["selection_key"]))[0]


def formal_promotion_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> Mapping[str, Any]:
    """Apply representation-only gates to A or B; C rows never contribute."""

    formal_arm = str(arm).upper()
    if formal_arm not in {"A", "B"}:
        raise ValueError("only A or B can receive formal WISER promotion")
    selected = {str(row["scenario"]): row for row in rows if str(row["arm"]) == formal_arm}
    baseline = {str(row["scenario"]): row for row in rows if str(row["arm"]) == "B0"}
    if set(selected) != set(SCENARIOS) or set(baseline) != set(SCENARIOS):
        raise ValueError("WISER promotion requires matched three-scenario rows")

    deltas: dict[str, list[float]] = {
        "P1_SOURCE_HEAD": [],
        "P2_SOURCE_PROTOTYPE": [],
        "P3_OLD_D92": [],
    }
    floor_delta = []
    within_delta = []
    ratio_delta = []
    for scenario in SCENARIOS:
        current = selected[scenario]
        control = baseline[scenario]
        for probe in deltas:
            deltas[probe].append(
                float(current["probes"][probe]["balanced_accuracy"])
                - float(control["probes"][probe]["balanced_accuracy"])
            )
        floor_delta.append(
            float(current["probes"]["P3_OLD_D92"]["floor"])
            - float(control["probes"]["P3_OLD_D92"]["floor"])
        )
        within_delta.append(
            float(current["geometry"]["within_trace"])
            - float(control["geometry"]["within_trace"])
        )
        ratio_delta.append(
            float(current["geometry"]["between_within_ratio"])
            - float(control["geometry"]["between_within_ratio"])
        )

    positive_fractions = {
        probe: sum(value > 0.0 for value in values) / float(len(values))
        for probe, values in deltas.items()
    }
    gates = {
        "median_p1_delta_ge_3pp": median(deltas["P1_SOURCE_HEAD"]) >= 0.03,
        "median_p2_delta_gt_0": median(deltas["P2_SOURCE_PROTOTYPE"]) > 0.0,
        "median_p3_delta_ge_3pp": median(deltas["P3_OLD_D92"]) >= 0.03,
        "all_probe_positive_fraction_ge_2of3": all(
            value >= 2.0 / 3.0 for value in positive_fractions.values()
        ),
        "median_p3_floor_delta_ge_0": median(floor_delta) >= 0.0,
        "min_p1_p3_domain_delta_ge_minus_2pp": min(
            deltas["P1_SOURCE_HEAD"] + deltas["P3_OLD_D92"]
        )
        >= -0.02,
        "geometry_improved": median(within_delta) <= 0.0 or median(ratio_delta) >= 0.0,
    }
    return {
        "schema": "cvs.phase2.wiser_rf.representation_gate.v1",
        "formal_arm": formal_arm,
        "scenario_count": 3,
        "deltas": deltas,
        "positive_fractions": positive_fractions,
        "median_floor_delta": median(floor_delta),
        "median_within_delta": median(within_delta),
        "median_between_within_ratio_delta": median(ratio_delta),
        "gates": gates,
        "passed": all(gates.values()),
        "c_diagnostic_rows_used": 0,
    }


__all__ = [
    "ARMS",
    "P3_ARMS",
    "SCENARIOS",
    "WISERQueryPackage",
    "WISERSupportPackage",
    "formal_promotion_decision",
    "formal_p3_primary_decision",
    "load_query_package",
    "load_support_package",
    "normalize_p3_arms",
    "select_p3_primary_champion",
]
