#!/usr/bin/env python3
"""Independent truth-connected scorer for JMRS01 prediction streams."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REQUIRED_SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
REQUIRED_ARTIFACTS = (
    "mechanism_identity_stability.json",
    "mechanism_receiver_probe.json",
    "mechanism_loro_metrics.json",
    "mechanism_clean_sat_consistency.json",
    "mechanism_complementarity.json",
    "mechanism_observability.json",
    "mechanism_cost.json",
    "mechanism_decision.json",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _macro_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    classes = np.unique(truth)
    accuracy = float(np.mean(predicted == truth)) if truth.size else 0.0
    f1_values = []
    for label in classes:
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        f1_values.append(2.0 * tp / denominator if denominator else 0.0)
    return {
        "count": int(truth.size),
        "macro_accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
    }


def _balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [float(np.mean(predicted[truth == label] == label)) for label in np.unique(truth)]
    return float(np.mean(recalls)) if recalls else 0.0


def _group_bootstrap_difference(
    candidate: np.ndarray,
    control: np.ndarray,
    groups: Sequence[tuple[Any, ...]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if candidate.shape != control.shape or candidate.ndim != 1 or len(groups) != candidate.size:
        raise ValueError("paired bootstrap arrays and groups must align")
    group_to_indices: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[tuple(group)].append(index)
    keys = sorted(group_to_indices, key=str)
    if not keys:
        return {"difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "group_count": 0}
    observed = float(np.mean(candidate - control))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(1, int(resamples))):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        indices = [index for choice in chosen for index in group_to_indices[keys[int(choice)]]]
        draws.append(float(np.mean(candidate[indices] - control[indices])))
    return {
        "difference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "group_count": len(keys),
    }


def _join_truth(predictions: list[dict[str, Any]], truths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    truth_by_id: dict[str, int] = {}
    for row in truths:
        sample_id = str(row["sample_id"])
        if sample_id in truth_by_id:
            raise ValueError(f"duplicate truth sample_id: {sample_id}")
        truth_by_id[sample_id] = int(row["true_class"])
    seen: set[str] = set()
    joined = []
    for row in predictions:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        seen.add(sample_id)
        if sample_id not in truth_by_id:
            raise ValueError(f"prediction has no truth record: {sample_id}")
        joined.append({**row, "true_class": truth_by_id[sample_id]})
    if seen != set(truth_by_id):
        raise ValueError("prediction/truth closure mismatch")
    return joined


def _evaluation_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("scope", "held_audit"),
        row["scenario"],
        int(row.get("held_receiver", row.get("receiver", -1))),
        int(row.get("receiver", -1)),
        int(row.get("day", -1)),
        int(row.get("base_index", -1)),
    )


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _centroid_distances(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    clean = [row for row in rows if row["scenario"] == "clean" and "embedding" in row]
    if not clean:
        return {"D_TX": None, "D_RX": None, "D_day": None}
    grouped: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for row in clean:
        grouped[(int(row["true_class"]), int(row.get("receiver", -1)), int(row.get("day", -1)))].append(
            np.asarray(row["embedding"], dtype=np.float64)
        )
    centroids = {key: np.mean(value, axis=0) for key, value in grouped.items()}
    tx_terms: list[float] = []
    rx_terms: list[float] = []
    day_terms: list[float] = []
    items = list(centroids.items())
    for index, (left_key, left) in enumerate(items):
        for right_key, right in items[index + 1 :]:
            left_tx, left_rx, left_day = left_key
            right_tx, right_rx, right_day = right_key
            if left_rx == right_rx and left_day == right_day and left_tx != right_tx:
                tx_terms.append(_distance(left, right))
            if left_tx == right_tx and left_day == right_day and left_rx != right_rx:
                rx_terms.append(_distance(left, right))
            if left_tx == right_tx and left_rx == right_rx and left_day != right_day:
                day_terms.append(_distance(left, right))
    mean = lambda values: float(np.mean(values)) if values else None
    return {"D_TX": mean(tx_terms), "D_RX": mean(rx_terms), "D_day": mean(day_terms)}


def _receiver_probes(rows_by_row: Mapping[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
    except ImportError:
        return {row: {"status": "UNAVAILABLE_SKLEARN"} for row in rows_by_row}
    payload: dict[str, Any] = {}
    for row, rows in rows_by_row.items():
        fold_ids = sorted(
            {
                int(item["held_receiver"])
                for item in rows
                if item.get("scope") in {"probe_fit", "probe_eval"} and item["scenario"] == "clean"
            }
        )
        if not fold_ids:
            payload[row] = {"status": "UNAVAILABLE_NO_PROBE_SPLIT"}
            continue
        folds: dict[str, Any] = {}
        for fold_id in fold_ids:
            fit = [
                item
                for item in rows
                if item.get("scope") == "probe_fit"
                and item["scenario"] == "clean"
                and int(item["held_receiver"]) == fold_id
            ]
            evaluation = [
                item
                for item in rows
                if item.get("scope") == "probe_eval"
                and item["scenario"] == "clean"
                and int(item["held_receiver"]) == fold_id
            ]
            if not fit or not evaluation:
                continue
            x_fit = np.asarray([item["embedding"] for item in fit], dtype=np.float64)
            y_fit = np.asarray([item["receiver"] for item in fit], dtype=np.int64)
            x_eval = np.asarray([item["embedding"] for item in evaluation], dtype=np.float64)
            y_eval = np.asarray([item["receiver"] for item in evaluation], dtype=np.int64)
            if np.unique(y_fit).size < 2 or not set(np.unique(y_eval)).issubset(set(np.unique(y_fit))):
                continue
            models = {
                "linear": LogisticRegression(max_iter=300, random_state=seed + fold_id),
                "mlp": MLPClassifier(
                    hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed + fold_id
                ),
                "knn": KNeighborsClassifier(n_neighbors=min(5, len(fit))),
            }
            scores = {}
            for name, model in models.items():
                model.fit(x_fit, y_fit)
                scores[name] = _balanced_accuracy(y_eval, model.predict(x_eval))
            chance = 1.0 / float(np.unique(y_fit).size)
            folds[str(fold_id)] = {
                "balanced_accuracy": scores,
                "best_balanced_accuracy": max(scores.values()),
                "chance": chance,
                "fit_count": len(fit),
                "eval_count": len(evaluation),
            }
        if not folds:
            payload[row] = {"status": "UNAVAILABLE_INSUFFICIENT_RECEIVER_CLASSES"}
            continue
        score_names = ("linear", "mlp", "knn")
        scores = {
            name: float(np.mean([fold["balanced_accuracy"][name] for fold in folds.values()]))
            for name in score_names
        }
        chance = float(np.mean([fold["chance"] for fold in folds.values()]))
        best = max(scores.values())
        payload[row] = {
            "status": "COMPLETE",
            "folds": folds,
            "balanced_accuracy": scores,
            "best_balanced_accuracy": best,
            "chance": chance,
            "normalized_leakage": (best - chance) / max(1e-12, 1.0 - chance),
            "fit_count": sum(fold["fit_count"] for fold in folds.values()),
            "eval_count": sum(fold["eval_count"] for fold in folds.values()),
            "aggregation": "mean_of_fold_local_probes_no_cross_fold_coordinate_mixing",
        }
    baseline = payload.get("M0", {})
    baseline_leakage = baseline.get("normalized_leakage")
    for row, value in payload.items():
        leakage = value.get("normalized_leakage")
        if baseline_leakage is not None and leakage is not None and baseline_leakage > 0:
            value["relative_leakage_reduction_vs_M0"] = (baseline_leakage - leakage) / baseline_leakage
    return payload


def score_prediction_streams(
    prediction_path: Path,
    truth_path: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 20260826,
) -> dict[str, Any]:
    predictions = _read_jsonl(Path(prediction_path))
    truths = _read_jsonl(Path(truth_path))
    rows = _join_truth(predictions, truths)
    held_rows = [row for row in rows if row.get("scope", "held_audit") == "held_audit"]
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_row[str(row["row"])].append(row)
    held_by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in held_rows:
        held_by_row[str(row["row"])].append(row)
    for row, values in held_by_row.items():
        observed = {str(item["scenario"]) for item in values}
        missing = set(REQUIRED_SCENARIOS).difference(observed)
        if missing:
            raise ValueError(f"scenario matrix is incomplete for {row}: {sorted(missing)}")

    loro: dict[str, Any] = {}
    for row, values in held_by_row.items():
        loro[row] = {}
        for scenario in REQUIRED_SCENARIOS:
            selected = [item for item in values if item["scenario"] == scenario]
            truth = np.asarray([item["true_class"] for item in selected], dtype=np.int64)
            predicted = np.asarray([item["predicted_class"] for item in selected], dtype=np.int64)
            receiver_metrics = {}
            for receiver in sorted({int(item["receiver"]) for item in selected}):
                mask = np.asarray([int(item["receiver"]) == receiver for item in selected])
                receiver_metrics[str(receiver)] = _macro_metrics(truth[mask], predicted[mask])
            loro[row][scenario] = {
                **_macro_metrics(truth, predicted),
                "receiver_floor": min(
                    (value["macro_accuracy"] for value in receiver_metrics.values()), default=0.0
                ),
                "by_receiver": receiver_metrics,
            }

    identity: dict[str, Any] = {}
    consistency: dict[str, Any] = {}
    margins: dict[str, float | None] = {}
    for row, held_values in held_by_row.items():
        all_values = by_row[row]
        fold_ids = sorted({int(item["held_receiver"]) for item in held_values})
        fold_payload: dict[str, Any] = {}
        all_sat_terms: list[float] = []
        for fold_id in fold_ids:
            geometry_rows = [
                item
                for item in all_values
                if int(item.get("held_receiver", -1)) == fold_id
                and item.get("scope", "held_audit") in {"held_audit", "probe_eval"}
                and item["scenario"] == "clean"
            ]
            distances = _centroid_distances(geometry_rows)
            fold_held = [
                item
                for item in held_values
                if int(item.get("held_receiver", -1)) == fold_id and "embedding" in item
            ]
            keyed = {_evaluation_key(item): item for item in fold_held}
            sat_terms = []
            for item in fold_held:
                if item["scenario"] == "clean":
                    continue
                clean_key = list(_evaluation_key(item))
                clean_key[1] = "clean"
                clean = keyed.get(tuple(clean_key))
                if clean is not None:
                    sat_terms.append(
                        _distance(np.asarray(clean["embedding"]), np.asarray(item["embedding"]))
                    )
            d_sat = float(np.mean(sat_terms)) if sat_terms else None
            all_sat_terms.extend(sat_terms)
            denominator = sum(
                value for value in (distances["D_RX"], distances["D_day"], d_sat) if value is not None
            )
            stability = (
                float(distances["D_TX"] / (denominator + 1e-12))
                if distances["D_TX"] is not None and denominator > 0
                else None
            )
            fold_payload[str(fold_id)] = {
                **distances,
                "D_sat": d_sat,
                "identity_stability_ratio": stability,
                "sat_pair_count": len(sat_terms),
            }
        aggregate = {}
        for name in ("D_TX", "D_RX", "D_day", "D_sat", "identity_stability_ratio"):
            values = [value[name] for value in fold_payload.values() if value[name] is not None]
            aggregate[name] = float(np.mean(values)) if values else None
        margins[row] = aggregate["D_TX"]
        identity[row] = {
            **aggregate,
            "folds": fold_payload,
            "aggregation": "mean_of_fold_local_geometry",
        }
        consistency[row] = {
            "paired_embedding_distance": float(np.mean(all_sat_terms)) if all_sat_terms else None,
            "pair_count": len(all_sat_terms),
            "fold_local": True,
        }

    probes = _receiver_probes(by_row, seed)
    complementarity: dict[str, Any] = {}
    baseline_lookup = {_evaluation_key(item): item for item in held_by_row.get("M0", [])}
    sham_lookup = {_evaluation_key(item): item for item in held_by_row.get("S1", [])}
    for row, values in held_by_row.items():
        if row == "M0":
            continue
        mechanism_correct = []
        base_correct = []
        sham_correct = []
        groups = []
        for item in values:
            key = _evaluation_key(item)
            base = baseline_lookup.get(key)
            sham = sham_lookup.get(key)
            if base is None:
                continue
            mechanism_correct.append(int(item["predicted_class"] == item["true_class"]))
            base_correct.append(int(base["predicted_class"] == base["true_class"]))
            sham_correct.append(
                int(sham["predicted_class"] == sham["true_class"]) if sham is not None else 0
            )
            groups.append((int(item.get("receiver", -1)), int(item.get("day", -1))))
        mech = np.asarray(mechanism_correct, dtype=np.float64)
        base = np.asarray(base_correct, dtype=np.float64)
        sham = np.asarray(sham_correct, dtype=np.float64)
        base_errors = base == 0
        ci = _group_bootstrap_difference(
            mech, sham, groups, resamples=bootstrap_resamples, seed=seed + 31
        ) if mech.size else {"difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "group_count": 0}
        complementarity[row] = {
            "mechanism_correct_given_core90_wrong": float(np.mean(mech[base_errors])) if np.any(base_errors) else None,
            "oracle_accuracy": float(np.mean(np.maximum(mech, base))) if mech.size else 0.0,
            "oracle_gain": float(np.mean(np.maximum(mech, base)) - np.mean(base)) if mech.size else 0.0,
            "loro_vs_sham_group_bootstrap": ci,
        }

    observability: dict[str, Any] = {}
    for row, values in held_by_row.items():
        reliability = np.asarray([float(item.get("reliability", 0.0)) for item in values])
        correct = np.asarray([item["predicted_class"] == item["true_class"] for item in values], dtype=float)
        curve = []
        for fraction in (0.25, 0.50, 0.75, 1.00):
            keep = max(1, int(np.ceil(len(values) * fraction))) if values else 0
            chosen = np.argsort(-reliability)[:keep]
            curve.append(
                {
                    "coverage": float(keep / max(1, len(values))),
                    "accuracy": float(np.mean(correct[chosen])) if keep else 0.0,
                    "utility_vs_core90": float(
                        np.mean(
                            correct[chosen]
                            - np.asarray(
                                [
                                    baseline_lookup.get(_evaluation_key(values[index]), values[index]).get(
                                        "predicted_class"
                                    )
                                    == values[index]["true_class"]
                                    for index in chosen
                                ],
                                dtype=float,
                            )
                        )
                    ) if keep else 0.0,
                }
            )
        observability[row] = {
            "mean_reliability": float(np.mean(reliability)) if reliability.size else 0.0,
            "coverage_at_0_30": float(np.mean(reliability >= 0.30)) if reliability.size else 0.0,
            "accuracy_utility_coverage": curve,
        }

    cost = {}
    for row, values in held_by_row.items():
        cost[row] = {
            "parameter_count": max((int(item.get("parameter_count", 0)) for item in values), default=0),
            "runtime_ms_per_sample": float(
                np.mean([float(item.get("runtime_ms_per_sample", 0.0)) for item in values])
            ) if values else 0.0,
        }

    decisions: dict[str, Any] = {}
    m0_margin = margins.get("M0")
    for row, values in held_by_row.items():
        if row in {"M0", "S1"}:
            decisions[row] = {"role": "BASELINE" if row == "M0" else "CAPACITY_CONTROL"}
            continue
        comp = complementarity.get(row, {})
        probe = probes.get(row, {})
        margin = margins.get(row)
        ci_low = comp.get("loro_vs_sham_group_bootstrap", {}).get("ci95_low", -1.0)
        leakage_reduction = probe.get("relative_leakage_reduction_vs_M0")
        clean_values = [item for item in values if item["scenario"] == "clean"]
        safe_available = bool(clean_values) and all("safe_predicted_class" in item for item in clean_values)
        safe_accuracy = (
            float(np.mean([item["safe_predicted_class"] == item["true_class"] for item in clean_values]))
            if safe_available else None
        )
        base_clean = loro.get("M0", {}).get("clean", {}).get("macro_accuracy")
        clean_drop_pp = (
            100.0 * (base_clean - safe_accuracy)
            if base_clean is not None and safe_accuracy is not None else None
        )
        candidate_by_rx = loro[row]["clean"]["by_receiver"]
        sham_by_rx = loro.get("S1", {}).get("clean", {}).get("by_receiver", {})
        nondegraded = sum(
            value["macro_accuracy"] >= sham_by_rx.get(rx, {}).get("macro_accuracy", 1.0)
            for rx, value in candidate_by_rx.items()
        )
        positive_receivers: set[int] = set()
        positive_days: set[int] = set()
        positive_leo: set[str] = set()
        for item in values:
            sham = sham_lookup.get(_evaluation_key(item))
            if sham is None:
                continue
            gain = int(item["predicted_class"] == item["true_class"]) - int(
                sham["predicted_class"] == sham["true_class"]
            )
            if gain > 0:
                positive_receivers.add(int(item.get("receiver", -1)))
                positive_days.add(int(item.get("day", -1)))
                if item["scenario"] != "clean":
                    positive_leo.add(str(item["scenario"]))
        gates = {
            "loro_vs_sham_ci_low_gt_0": bool(ci_low > 0.0),
            "receiver_leakage_reduction_ge_20pct": bool(
                leakage_reduction is not None and leakage_reduction >= 0.20
            ),
            "between_tx_margin_retention_ge_90pct": bool(
                margin is not None and m0_margin is not None and m0_margin > 0 and margin / m0_margin >= 0.90
            ),
            "safe_clean_drop_le_0_30pp": bool(clean_drop_pp is not None and clean_drop_pp <= 0.30),
            "majority_receivers_nondegraded": bool(nondegraded >= 4),
            "oracle_gain_ge_0_30pp": bool(comp.get("oracle_gain", 0.0) >= 0.003),
            "coverage_ge_30pct": bool(observability[row]["coverage_at_0_30"] >= 0.30),
            "breadth_2rx_2day_2leo": bool(
                len(positive_receivers) >= 2 and len(positive_days) >= 2 and len(positive_leo) >= 2
            ),
        }
        decisions[row] = {
            "gates": gates,
            "all_gates_pass": all(gates.values()),
            "next_stage": "PROMOTE_TO_S1_POOL" if all(gates.values()) else "DO_NOT_PROMOTE",
            "observed": {
                "ci_low": ci_low,
                "receiver_leakage_reduction": leakage_reduction,
                "margin_retention": margin / m0_margin if margin is not None and m0_margin else None,
                "safe_clean_drop_pp": clean_drop_pp,
                "nondegraded_receiver_count": nondegraded,
                "oracle_gain": comp.get("oracle_gain"),
                "coverage": observability[row]["coverage_at_0_30"],
                "positive_receiver_count": len(positive_receivers),
                "positive_day_count": len(positive_days),
                "positive_leo_scenario_count": len(positive_leo),
            },
        }

    artifacts = {
        "mechanism_identity_stability.json": identity,
        "mechanism_receiver_probe.json": probes,
        "mechanism_loro_metrics.json": loro,
        "mechanism_clean_sat_consistency.json": consistency,
        "mechanism_complementarity.json": complementarity,
        "mechanism_observability.json": observability,
        "mechanism_cost.json": cost,
        "mechanism_decision.json": decisions,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in artifacts.items():
        _write_json(output / name, value)
    return {
        "status": "ANALYZED",
        "loro": loro,
        "decision": decisions,
        "artifact_count": len(artifacts),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score closed JMRS01 prediction and truth streams")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = score_prediction_streams(
        Path(args.predictions),
        Path(args.truth),
        Path(args.output_dir),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
