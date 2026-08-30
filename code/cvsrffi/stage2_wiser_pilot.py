"""Minimal historical-pilot lifecycle helpers for WISER-RF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("B0", "A", "B", "C", "ABC")


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
    "SCENARIOS",
    "WISERQueryPackage",
    "WISERSupportPackage",
    "formal_promotion_decision",
    "load_query_package",
    "load_support_package",
]
