"""Role-aware offline J0 audits over closed JMRS01 prediction streams."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


REQUIRED_SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
REQUIRED_ROWS = ("M0", "S1", "R1", "R2", "D1", "P2")
DEFAULT_COMBINATIONS = (
    ("R1", "D1"),
    ("R1", "P2"),
    ("R2", "D1"),
    ("D1", "P2"),
    ("R1", "D1", "P2"),
)


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("scope", "held_audit")),
        str(row["scenario"]),
        int(row.get("held_receiver", row.get("receiver", -1))),
        int(row.get("receiver", -1)),
        int(row.get("day", -1)),
        int(row.get("base_index", -1)),
    )


def fisher_ratio(embedding: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Return trace(S_between)/trace(S_within), comparable across dimensions and scale."""
    x = np.asarray(embedding, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or y.size == 0:
        raise ValueError("Fisher inputs must be non-empty aligned rank-2/rank-1 arrays")
    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in np.unique(y):
        selected = x[y == label]
        center = selected.mean(axis=0)
        between += float(selected.shape[0]) * float(np.square(center - overall).sum())
        within += float(np.square(selected - center).sum())
    return float(between / max(float(eps), within))


def _accuracy(rows: Sequence[Mapping[str, Any]], field: str = "predicted_class") -> float:
    return float(np.mean([int(row[field]) == int(row["true_class"]) for row in rows])) if rows else 0.0


def _group_bootstrap_synergy(
    base: np.ndarray,
    members: Sequence[np.ndarray],
    groups: Sequence[tuple[Any, ...]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if not members or any(member.shape != base.shape for member in members) or len(groups) != base.size:
        raise ValueError("bootstrap correctness arrays and groups must align")
    group_to_indices: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[tuple(group)].append(index)
    keys = sorted(group_to_indices, key=str)

    counts = np.asarray([len(group_to_indices[key]) for key in keys], dtype=np.float64)
    base_sums = np.asarray(
        [float(np.sum(base[group_to_indices[key]])) for key in keys], dtype=np.float64
    )
    union_all = np.maximum.reduce([base, *members])
    union_sums = np.asarray(
        [float(np.sum(union_all[group_to_indices[key]])) for key in keys], dtype=np.float64
    )
    single_union_sums = np.asarray(
        [
            [
                float(np.sum(np.maximum(base, member)[group_to_indices[key]]))
                for key in keys
            ]
            for member in members
        ],
        dtype=np.float64,
    )

    def statistic(group_indices: np.ndarray) -> float:
        total = float(np.sum(counts[group_indices]))
        selected_base = float(np.sum(base_sums[group_indices]))
        combination_gain = (float(np.sum(union_sums[group_indices])) - selected_base) / total
        best_single = max(
            (float(np.sum(member_sums[group_indices])) - selected_base) / total
            for member_sums in single_union_sums
        )
        return combination_gain - best_single

    all_groups = np.arange(len(keys), dtype=np.int64)
    observed = statistic(all_groups)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(max(1, int(resamples))):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        draws.append(statistic(np.asarray(chosen, dtype=np.int64)))
    return {
        "difference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "group_count": len(keys),
        "resamples": max(1, int(resamples)),
        "aggregation": "preaggregated_receiver_day_scenario_group_counts",
    }


def _validate_and_align(rows: Sequence[Mapping[str, Any]]) -> tuple[
    dict[str, dict[tuple[Any, ...], Mapping[str, Any]]], list[tuple[Any, ...]]
]:
    held = [row for row in rows if str(row.get("scope", "held_audit")) == "held_audit"]
    by_row: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = defaultdict(dict)
    for row in held:
        name = str(row["row"])
        key = _key(row)
        if key in by_row[name]:
            raise ValueError(f"duplicate evaluation key for {name}: {key}")
        by_row[name][key] = row
    missing = sorted(set(REQUIRED_ROWS).difference(by_row))
    if missing:
        raise ValueError(f"missing preregistered rows: {missing}")
    base_keys = sorted(by_row["M0"], key=str)
    for name in REQUIRED_ROWS:
        if sorted(by_row[name], key=str) != base_keys:
            raise ValueError(f"row {name} does not align with M0 evaluation keys")
        scenarios = {str(row["scenario"]) for row in by_row[name].values()}
        if scenarios != set(REQUIRED_SCENARIOS):
            raise ValueError(f"scenario matrix is incomplete for {name}: {sorted(scenarios)}")
    return dict(by_row), base_keys


def _correctness(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]], keys: Sequence[tuple[Any, ...]]
) -> np.ndarray:
    return np.asarray(
        [int(lookup[key]["predicted_class"]) == int(lookup[key]["true_class"]) for key in keys],
        dtype=np.float64,
    )


def analyze_j0_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    combinations: Sequence[Sequence[str]] = DEFAULT_COMBINATIONS,
    bootstrap_resamples: int = 2000,
    seed: int = 20260826,
) -> dict[str, Any]:
    by_row, keys = _validate_and_align(rows)
    correctness = {name: _correctness(by_row[name], keys) for name in REQUIRED_ROWS}
    base = correctness["M0"]
    sham = correctness["S1"]

    semantic: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    cost: dict[str, Any] = {}
    clean_keys = [key for key in keys if key[1] == "clean"]
    receivers = sorted({int(key[3]) for key in clean_keys})
    for name in REQUIRED_ROWS:
        candidate_by_receiver: dict[int, float] = {}
        base_by_receiver: dict[int, float] = {}
        sham_by_receiver: dict[int, float] = {}
        for receiver in receivers:
            selected = [key for key in clean_keys if int(key[3]) == receiver]
            candidate_by_receiver[receiver] = _accuracy([by_row[name][key] for key in selected])
            base_by_receiver[receiver] = _accuracy([by_row["M0"][key] for key in selected])
            sham_by_receiver[receiver] = _accuracy([by_row["S1"][key] for key in selected])

        rescue_keys = [
            key
            for index, key in enumerate(keys)
            if base[index] == 0.0 and correctness[name][index] == 1.0
        ]
        positive_leo = {str(key[1]) for key in rescue_keys if key[1] != "clean"}
        fold_ids = sorted({int(key[2]) for key in keys})
        nonzero_folds = []
        for fold_id in fold_ids:
            alphas = {
                float(by_row[name][key].get("safe_alpha", 0.0))
                for key in keys
                if int(key[2]) == fold_id
            }
            if any(alpha > 0.0 for alpha in alphas):
                nonzero_folds.append(fold_id)
        nonzero_keys = [key for key in keys if int(key[2]) in nonzero_folds]
        safe_available = bool(nonzero_keys) and all(
            "safe_predicted_class" in by_row[name][key] for key in nonzero_keys
        )
        safe_accuracy = (
            _accuracy([by_row[name][key] for key in nonzero_keys], "safe_predicted_class")
            if safe_available
            else None
        )
        base_nonzero_accuracy = (
            _accuracy([by_row["M0"][key] for key in nonzero_keys]) if nonzero_keys else None
        )
        safe_utility = (
            float(safe_accuracy - base_nonzero_accuracy)
            if safe_accuracy is not None and base_nonzero_accuracy is not None
            else None
        )
        semantic[name] = {
            "family_sham_row": "S1",
            "family_sham_scope": "legacy_shared_capacity_control_not_family_specific",
            "nondegraded_receiver_count_vs_family_sham": sum(
                candidate_by_receiver[receiver] >= sham_by_receiver[receiver]
                for receiver in receivers
            ),
            "nondegraded_receiver_count_vs_M0": sum(
                candidate_by_receiver[receiver] >= base_by_receiver[receiver]
                for receiver in receivers
            ),
            "receiver_count": len(receivers),
            "breadth_vs_core90_rescue": {
                "receiver_count": len({int(key[3]) for key in rescue_keys}),
                "day_count": len({int(key[4]) for key in rescue_keys}),
                "leo_scenario_count": len(positive_leo),
                "rescue_count": len(rescue_keys),
            },
            "safe_gate": {
                "evaluable_nonzero_alpha": bool(safe_available),
                "nonzero_alpha_fold_count": len(nonzero_folds),
                "safe_accuracy_on_nonzero_alpha_folds": safe_accuracy,
                "base_accuracy_on_same_folds": base_nonzero_accuracy,
                "selected_weighted_utility": safe_utility,
                "passes": bool(safe_utility is not None and safe_utility > 0.0),
                "alpha_zero_is_not_a_pass": True,
            },
        }

        fold_ratios: dict[str, float] = {}
        for fold_id in fold_ids:
            selected = [key for key in clean_keys if int(key[2]) == fold_id]
            embeddings = np.asarray([by_row[name][key]["embedding"] for key in selected])
            labels = np.asarray([by_row[name][key]["true_class"] for key in selected])
            fold_ratios[str(fold_id)] = fisher_ratio(embeddings, labels)
        geometry[name] = {
            "fisher_ratio": float(np.mean(list(fold_ratios.values()))),
            "folds": fold_ratios,
            "metric": "trace_between_over_trace_within",
            "fold_local": True,
        }
        values = list(by_row[name].values())
        cost[name] = {
            "parameter_count": max(int(row.get("parameter_count", 0)) for row in values),
            "incremental_runtime_ms_per_sample": float(
                np.mean([float(row.get("runtime_ms_per_sample", 0.0)) for row in values])
            ),
            "runtime_scope": (
                "cached_core90_access_only" if name == "M0" else "post_cached_core90_branch_only"
            ),
            "full_system_runtime_ms_per_sample": None,
        }

    m0_fisher = geometry["M0"]["fisher_ratio"]
    for name in REQUIRED_ROWS:
        geometry[name]["fisher_ratio_retention_vs_M0"] = (
            float(geometry[name]["fisher_ratio"] / m0_fisher) if m0_fisher > 0.0 else None
        )

    s1_rescue = (base == 0.0) & (sham == 1.0)
    groups = [(int(key[3]), int(key[4]), str(key[1])) for key in keys]
    joint: dict[str, Any] = {}
    for combination_index, raw_members in enumerate(combinations):
        members = tuple(str(name) for name in raw_members)
        if not members or any(name not in correctness or name in {"M0", "S1"} for name in members):
            raise ValueError(f"invalid J0 combination: {members}")
        member_correctness = [correctness[name] for name in members]
        member_rescues = [(base == 0.0) & (member == 1.0) for member in member_correctness]
        rescue_union = np.logical_or.reduce(member_rescues)
        rescue_intersection = np.logical_and.reduce(member_rescues)
        oracle = np.maximum.reduce([base, *member_correctness])
        base_accuracy = float(np.mean(base))
        oracle_gain = float(np.mean(oracle) - base_accuracy)
        single_gains = {
            name: float(np.mean(np.maximum(base, correctness[name])) - base_accuracy)
            for name in members
        }
        best_single = max(single_gains.values())
        bootstrap = _group_bootstrap_synergy(
            base,
            member_correctness,
            groups,
            resamples=bootstrap_resamples,
            seed=seed + 101 * combination_index,
        )
        scenario_metrics: dict[str, Any] = {}
        for scenario in REQUIRED_SCENARIOS:
            mask = np.asarray([key[1] == scenario for key in keys], dtype=bool)
            scenario_base = float(np.mean(base[mask]))
            scenario_oracle = float(np.mean(oracle[mask]))
            scenario_single = max(
                float(np.mean(np.maximum(base[mask], correctness[name][mask])) - scenario_base)
                for name in members
            )
            scenario_metrics[scenario] = {
                "oracle_gain": scenario_oracle - scenario_base,
                "synergy": scenario_oracle - scenario_base - scenario_single,
            }
        member_unique = {}
        for member_index, name in enumerate(members):
            others = [mask for index, mask in enumerate(member_rescues) if index != member_index]
            other_union = np.logical_or.reduce(others) if others else np.zeros_like(rescue_union)
            member_unique[name] = int(np.sum(member_rescues[member_index] & ~other_union))
        union_count = int(np.sum(rescue_union))
        joint["+".join(members)] = {
            "members": list(members),
            "count": len(keys),
            "base_accuracy": base_accuracy,
            "oracle_accuracy": float(np.mean(oracle)),
            "oracle_gain": oracle_gain,
            "single_oracle_gain": single_gains,
            "best_single_oracle_gain": best_single,
            "synergy": oracle_gain - best_single,
            "synergy_pp": 100.0 * (oracle_gain - best_single),
            "synergy_group_bootstrap": bootstrap,
            "rescue_union_count": union_count,
            "rescue_intersection_count": int(np.sum(rescue_intersection)),
            "rescue_jaccard": float(np.sum(rescue_intersection) / union_count) if union_count else 0.0,
            "unique_rescue_vs_S1_count": int(np.sum(rescue_union & ~s1_rescue)),
            "member_unique_rescue_count": member_unique,
            "scenario_metrics": scenario_metrics,
        }

    passed = [
        name
        for name, value in joint.items()
        if value["synergy"] > 0.0
        and value["synergy_group_bootstrap"]["ci95_low"] > 0.0
    ]
    decision = {
        "status": "J0_SIGNAL" if passed else "J0_NO_SIGNAL",
        "passing_combinations": passed,
        "promotion_rule": "synergy_gt_0_and_group_bootstrap_ci95_low_gt_0",
        "next_stage": (
            "PROCEED_TO_ROLE_CORRECT_J1_SINGLE_MODULES"
            if passed
            else "STOP_JMRS02_JOINT_NO_J0_EVIDENCE"
        ),
        "direct_joint_training_authorized": False,
        "target_dg_claim_authorized": False,
    }
    return {
        "semantic_audit": semantic,
        "joint_rescue": joint,
        "identity_geometry": geometry,
        "cost_scope": cost,
        "decision": decision,
    }


__all__ = [
    "DEFAULT_COMBINATIONS",
    "REQUIRED_ROWS",
    "REQUIRED_SCENARIOS",
    "analyze_j0_rows",
    "fisher_ratio",
]
