"""Publication runner for the proposed CVS Stage2 heads on frozen CVS features.

Stage2-B uses support-only prototype-Gaussian calibration (OPGAC). Stage2-C
uses the documented qKNNV42 int8 support-memory head. Target query labels are
used only after prediction to compute metrics and detailed result tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from paper_reproduction.common.config import load_json_config


METHOD_STAGE = {"cvs_opgac": "Stage2-B", "cvs_qknnv42": "Stage2-C"}
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EPS = 1.0e-8


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def _stable_rank(seed: int, *parts: object) -> int:
    raw = ":".join([str(seed), *(str(value) for value in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "features", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids",
            "dataset_role", "sat_scenarios",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"feature NPZ is missing keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files if key != "manifest_json"}


def _sample_id(arrays: dict[str, np.ndarray], index: int) -> str:
    return "|".join(
        str(arrays[key][index])
        for key in ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
    )


def _select_split(
    arrays: dict[str, np.ndarray], *, role: str, tx_labels: list[str], receiver: str,
    seed: int, k_shot: int, support_pool_max_k: int, query_per_tx: int,
) -> tuple[list[int], list[int]]:
    roles = arrays["dataset_role"].astype(str)
    tx = arrays["tx_ids"].astype(str)
    rx = arrays["rx_ids"].astype(str)
    support: list[int] = []
    query: list[int] = []
    for label in tx_labels:
        candidates = np.where((roles == role) & (tx == label) & (rx == receiver))[0].tolist()
        ordered = sorted(
            (int(i) for i in candidates),
            key=lambda i: _stable_rank(
                seed, role, label, receiver, arrays["day_ids"][i], arrays["eq_ids"][i], arrays["sig_ids"][i]
            ),
        )
        needed = int(support_pool_max_k) + int(query_per_tx)
        if len(ordered) < needed:
            raise ValueError(f"insufficient {role}/{label}/{receiver}: {len(ordered)} < {needed}")
        support.extend(ordered[: int(k_shot)])
        query.extend(ordered[int(support_pool_max_k) : needed])
    return support, query


def _class_scores(features: np.ndarray, labels: np.ndarray, query: np.ndarray) -> tuple[list[str], np.ndarray]:
    classes = sorted(set(labels.astype(str).tolist()))
    prototypes = np.vstack([_norm(features[labels.astype(str) == label].mean(axis=0, keepdims=True))[0] for label in classes])
    return classes, _norm(query) @ prototypes.T


def _opgac_predict(
    source_x: np.ndarray, source_y: np.ndarray, support_x: np.ndarray, support_y: np.ndarray,
    query_x: np.ndarray, *, shrinkage_kappa: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_x = _norm(source_x)
    support_x = _norm(support_x)
    query_x = _norm(query_x)
    classes, before_scores = _class_scores(source_x, source_y, query_x)
    before = np.asarray(classes, dtype=object)[np.argmax(before_scores, axis=1)]
    global_var = np.maximum(np.var(source_x, axis=0), 1.0e-4)
    score_columns: list[np.ndarray] = []
    compactness: list[float] = []
    for label in classes:
        src = source_x[source_y.astype(str) == label]
        sup = support_x[support_y.astype(str) == label]
        if sup.size == 0:
            raise ValueError(f"OPGAC missing labeled target support for {label}")
        ground_mean = _norm(src.mean(axis=0, keepdims=True))[0]
        target_mean = _norm(sup.mean(axis=0, keepdims=True))[0]
        alpha = len(sup) / (len(sup) + float(shrinkage_kappa))
        mean = _norm(((1.0 - alpha) * ground_mean + alpha * target_mean)[None, :])[0]
        local_var = np.var(sup, axis=0) if len(sup) > 1 else global_var
        diag = np.maximum((1.0 - alpha) * global_var + alpha * local_var, 1.0e-4)
        diff = query_x - mean[None, :]
        score_columns.append(-0.5 * (np.sum(diff * diff / diag[None, :], axis=1) + np.log(diag).sum()))
        compactness.append(float(np.mean(np.sum((sup - mean[None, :]) ** 2 / diag[None, :], axis=1))))
    scores = np.stack(score_columns, axis=1)
    predicted = np.asarray(classes, dtype=object)[np.argmax(scores, axis=1)]
    return predicted, before, {
        "adaptation_objective": "opgac_support_only_prototype_gaussian_calibration",
        "support_only": True,
        "query_update_forbidden": True,
        "loss": float(np.mean(compactness)),
        "class_count": len(classes),
    }


def _labelprop(
    support: np.ndarray, support_y: np.ndarray, query: np.ndarray, classes: list[str],
    *, neighbors: int = 10, alpha: float = 0.76, temperature: float = 0.05, rounds: int = 8,
) -> np.ndarray:
    x = _norm(np.vstack([support, query]))
    n_support = len(support)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(int(neighbors), max(1, len(x) - 1))
    positions = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    w = np.zeros_like(similarity)
    rows = np.arange(len(x))[:, None]
    logits = similarity[rows, positions] / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    w[rows, positions] = weights
    class_to_i = {label: i for i, label in enumerate(classes)}
    y = np.zeros((len(x), len(classes)), dtype=np.float64)
    for row, label in enumerate(support_y.astype(str).tolist()):
        y[row, class_to_i[label]] = 1.0
    f = y.copy()
    for _ in range(int(rounds)):
        f = float(alpha) * (w @ f) + (1.0 - float(alpha)) * y
        f[:n_support] = y[:n_support]
    q = f[n_support:]
    return np.clip((q - q.mean(axis=1, keepdims=True)) / np.maximum(q.std(axis=1, keepdims=True), 1.0e-6), -2.0, 2.0)


def _diag_whiten_fisher(
    support: np.ndarray, labels: np.ndarray, query: np.ndarray, *, strength: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    support = _norm(support)
    labels = labels.astype(str)
    center = support.mean(axis=0)
    centered = support - center
    classes = sorted(set(labels.tolist()))
    means = np.vstack([centered[labels == label].mean(axis=0) for label in classes])
    counts = np.asarray([np.sum(labels == label) for label in classes], dtype=np.float64)
    global_mean = np.average(means, axis=0, weights=counts)
    between = np.average((means - global_mean) ** 2, axis=0, weights=counts)
    within = np.concatenate([
        (centered[labels == label] - centered[labels == label].mean(axis=0, keepdims=True)) ** 2
        for label in classes
    ]).mean(axis=0)
    fisher = between / np.maximum(within, 1.0e-6)
    fisher /= max(float(np.median(fisher)), 1.0e-6)
    whiten = 1.0 / np.sqrt(np.maximum(centered.var(axis=0), 1.0e-5))
    scale = np.power(np.clip(fisher, 0.05, 20.0), float(strength))
    scale *= np.power(whiten / max(float(np.median(whiten)), 1.0e-6), 0.5)
    scale = np.clip(scale, 0.05, 20.0)
    return (
        _norm((support - center[None, :]) * scale[None, :]),
        _norm((_norm(query) - center[None, :]) * scale[None, :]),
        {"transform_scale_min": float(scale.min()), "transform_scale_max": float(scale.max()),
         "transform_scale_mean": float(scale.mean())},
    )


def _qknnv42_predict(
    support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray, *, old_labels: set[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    support, query, transform_info = _diag_whiten_fisher(support_x, support_y, query_x, strength=0.1)
    quantized = np.clip(np.rint(127.0 * support), -127, 127).astype(np.int8)
    restored = _norm(quantized.astype(np.float64) / 127.0)
    labels = support_y.astype(str)
    classes = sorted(set(labels.tolist()))
    score_columns: list[np.ndarray] = []
    for label in classes:
        class_support = restored[labels == label]
        similarity = query @ class_support.T
        knn = np.max(similarity, axis=1)
        prototype = _norm(class_support.mean(axis=0, keepdims=True))[0]
        score = 0.55 * knn + 0.45 * (query @ prototype)
        if label in old_labels:
            score = score + 0.001
        score_columns.append(score)
    scores = np.stack(score_columns, axis=1)
    scores += 0.025 * _labelprop(restored, labels, query, classes)
    predicted = np.asarray(classes, dtype=object)[np.argmax(scores, axis=1)]
    compactness = 1.0 - np.max(restored @ np.vstack([
        _norm(restored[labels == label].mean(axis=0, keepdims=True))[0] for label in classes
    ]).T, axis=1)
    return predicted, {
        "adaptation_objective": "qknnv42_int8_top1_proto45_old_anchor_labelprop",
        "transform_mode": "diag_whiten_fisher",
        "transform_strength": 0.1,
        **transform_info,
        "stored_quantized_support_code_count": int(len(restored)),
        "stored_raw_support_count": 0,
        "stored_class_prototype_count": len(classes),
        "feature_dim": int(restored.shape[1]),
        "query_labels_used_for_adaptation": False,
        "scenario_residual_weight": 0.5,
        "scenario_residual_applied": False,
        "scenario_residual_note": "zero_by_full_same_scenario_support_for_every_registered_class",
        "loss": float(np.mean(compactness)),
    }


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred.astype(str) == truth.astype(str)))


def _detail_rows(pred: np.ndarray, truth: np.ndarray, meta: list[dict[str, str]], scenario: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[int]] = {}
    for i, row in enumerate(meta):
        rx, tx, day, role = row["receiver_label"], row["transmitter_label"], row["day_i"], row["role"]
        for key in (("per_receiver", rx, "ALL", "ALL", role), ("per_transmitter", "ALL", tx, "ALL", role),
                    ("per_receiver_transmitter", rx, tx, "ALL", role),
                    ("per_receiver_transmitter_day", rx, tx, day, role)):
            groups.setdefault(key, []).append(i)
    rows: list[dict[str, Any]] = []
    for (kind, rx, tx, day, role), indices in sorted(groups.items()):
        confusion: dict[str, int] = {}
        for i in indices:
            key = f"{truth[i]}->{pred[i]}"
            confusion[key] = confusion.get(key, 0) + 1
        correct = sum(str(pred[i]) == str(truth[i]) for i in indices)
        rows.append({"scenario": scenario, "group_type": kind, "receiver_label": rx,
                     "transmitter_label": tx, "day": day, "role": role, "sample_count": len(indices),
                     "correct_count": correct, "accuracy": correct / len(indices),
                     "confusion_json": json.dumps(confusion, ensure_ascii=False, sort_keys=True)})
    return rows


def _meta(arrays: dict[str, np.ndarray], indices: list[int]) -> list[dict[str, str]]:
    return [{"sample_id": _sample_id(arrays, i), "receiver_label": str(arrays["rx_ids"][i]),
             "transmitter_label": str(arrays["tx_ids"][i]), "day_i": str(arrays["day_ids"][i]),
             "eq_i": str(arrays["eq_ids"][i]), "sig_i": str(arrays["sig_ids"][i]),
             "role": str(arrays["dataset_role"][i])} for i in indices]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config: dict[str, Any]) -> None:
    method = str(config.get("method", "")).lower()
    if method not in METHOD_STAGE:
        raise ValueError(f"method must be one of {sorted(METHOD_STAGE)}")
    if str(config.get("stage")) != METHOD_STAGE[method]:
        raise ValueError(f"{method} requires stage={METHOD_STAGE[method]}")
    if len(config.get("target_receiver_labels", [])) != 1:
        raise ValueError("each run must contain exactly one target receiver")
    if list(config.get("target_channel_scenarios", [])) != list(SCENARIOS):
        raise ValueError(f"formal tests must be exactly {list(SCENARIOS)}")
    mapping = config.get("feature_npz_by_scenario", {})
    if set(mapping) != set(SCENARIOS):
        raise ValueError("feature_npz_by_scenario must contain all formal LEO scenarios")
    if int(config.get("k_shot", 0)) <= 0 or int(config.get("support_pool_max_k", 0)) < int(config["k_shot"]):
        raise ValueError("invalid nested K-shot settings")
    if bool(config.get("unknown_rejection_enabled", False)) or config.get("target_unknown_tx_labels"):
        raise ValueError("Phase2 publication mainline excludes unknown rejection")


def run(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    validate_config(config)
    method = str(config["method"]).lower()
    stage = METHOD_STAGE[method]
    receiver = str(config["target_receiver_labels"][0])
    seed = int(config["split_seed"])
    old_labels = [str(v) for v in config["target_old_tx_labels"]]
    new_labels = [str(v) for v in config.get("target_new_tx_labels", [])] if stage == "Stage2-C" else []
    if stage == "Stage2-C" and not new_labels:
        raise ValueError("Stage2-C requires target-new labels")
    metrics_by_scenario: dict[str, dict[str, Any]] = {}
    score_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    manifest_splits: dict[str, Any] = {}
    for scenario in SCENARIOS:
        arrays = _load_npz(Path(config["feature_npz_by_scenario"][scenario]))
        roles = arrays["dataset_role"].astype(str)
        scenario_values = arrays["sat_scenarios"].astype(str)
        target_mask = np.isin(roles, ["target_old", "target_new"])
        if not np.all(scenario_values[target_mask] == scenario):
            raise ValueError(f"target rows in {scenario} cache are not all satellite-augmented with that scenario")
        old_support, old_query = _select_split(
            arrays, role="target_old", tx_labels=old_labels, receiver=receiver, seed=seed,
            k_shot=int(config["k_shot"]), support_pool_max_k=int(config["support_pool_max_k"]),
            query_per_tx=int(config["query_per_tx"]),
        )
        new_support: list[int] = []
        new_query: list[int] = []
        if stage == "Stage2-C":
            new_support, new_query = _select_split(
                arrays, role="target_new", tx_labels=new_labels, receiver=receiver, seed=seed,
                k_shot=int(config["k_shot"]), support_pool_max_k=int(config["support_pool_max_k"]),
                query_per_tx=int(config["query_per_tx"]),
            )
        support_idx = old_support + new_support
        query_idx = old_query + new_query
        support_x = arrays["features"][support_idx]
        support_y = arrays["tx_ids"][support_idx].astype(str)
        query_x = arrays["features"][query_idx]
        truth = arrays["tx_ids"][query_idx].astype(str)
        started = time.perf_counter()
        if method == "cvs_opgac":
            source_mask = (roles == "source") & np.isin(arrays["tx_ids"].astype(str), old_labels)
            predicted, before, info = _opgac_predict(
                arrays["features"][source_mask], arrays["tx_ids"][source_mask].astype(str),
                support_x, support_y, query_x, shrinkage_kappa=float(config.get("opgac_old_shrinkage_kappa", 3.0)),
            )
            metrics = {"target_old_accuracy": _accuracy(predicted, truth),
                       "target_old_accuracy_before_adaptation": _accuracy(before, truth)}
            metrics["target_old_accuracy_delta"] = metrics["target_old_accuracy"] - metrics["target_old_accuracy_before_adaptation"]
        else:
            old_pred, _ = _qknnv42_predict(arrays["features"][old_support], arrays["tx_ids"][old_support],
                                            arrays["features"][old_query], old_labels=set(old_labels))
            predicted, info = _qknnv42_predict(support_x, support_y, query_x, old_labels=set(old_labels))
            old_count = len(old_query)
            old_acc = _accuracy(predicted[:old_count], truth[:old_count])
            new_acc = _accuracy(predicted[old_count:], truth[old_count:])
            harmonic = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
            old_before = _accuracy(old_pred, arrays["tx_ids"][old_query].astype(str))
            metrics = {"old_acc": old_acc, "seen_new_acc": new_acc, "H_old_new": harmonic,
                       "old_acc_before_increment": old_before, "average_forgetting": old_before - old_acc,
                       "old_to_seen_new_rate": float(np.mean(np.isin(predicted[:old_count], new_labels))),
                       "seen_new_to_old_rate": float(np.mean(np.isin(predicted[old_count:], old_labels)))}
        elapsed = time.perf_counter() - started
        metrics.update({"adaptation_latency_sec": elapsed,
                        "latency_per_query_ms": elapsed * 1000.0 / len(query_idx), **info})
        metrics_by_scenario[scenario] = metrics
        trace.append({"method": method, "scenario": scenario, "phase": "support_only_fit", "step": 1,
                      "total_steps": 1, "loss": float(info["loss"]), "gradient_updates": 0})
        meta = _meta(arrays, query_idx)
        detailed.extend(_detail_rows(predicted, truth, meta, scenario))
        for row, true, pred in zip(meta, truth.tolist(), predicted.tolist()):
            score_rows.append({**row, "true_label": true, "predicted_label": pred,
                               "correct": int(str(true) == str(pred)), "scenario": scenario})
        support_ids = [_sample_id(arrays, i) for i in support_idx]
        query_ids = [_sample_id(arrays, i) for i in query_idx]
        if set(support_ids) & set(query_ids):
            raise ValueError("support/query overlap")
        manifest_splits[scenario] = {"support_sample_ids": support_ids, "query_sample_ids": query_ids,
                                     "support_count": len(support_ids), "query_count": len(query_ids)}
    aggregate_keys = ["target_old_accuracy", "target_old_accuracy_before_adaptation", "target_old_accuracy_delta"] \
        if stage == "Stage2-B" else ["old_acc", "seen_new_acc", "H_old_new", "average_forgetting"]
    aggregate = {key + "_mean": float(np.mean([row[key] for row in metrics_by_scenario.values()])) for key in aggregate_keys}
    manifest = {"stage": stage, "method": method, "cvs_proposed_method": True,
                "backbone": str(config.get("backbone_id", "ADV3B02_CORE90_SOFT_E200")),
                "target_receiver_labels": [receiver], "target_old_tx_labels": old_labels,
                "target_new_tx_labels": new_labels, "target_labels_scope": "registered_support_only",
                "target_query_used_for_training": False, "target_query_used_for_model_selection": False,
                "query_used_for_transductive_inference": method == "cvs_qknnv42",
                "support_query_overlap": False, "all_tests_satellite_augmented": True,
                "seed": int(config["seed"]), "split_seed": seed, "k_shot": int(config["k_shot"]),
                "support_pool_max_k": int(config["support_pool_max_k"]), "target_sample_strategy": "seeded_nested",
                "splits_by_scenario": manifest_splits, "unknown_rejection_enabled": False}
    result = {"experiment_id": config.get("experiment_id", f"{method}_{seed}"), "method": method,
              "stage": stage, "seed": int(config["seed"]), "target_receiver_label": receiver,
              "metrics": aggregate, "metrics_by_scenario": metrics_by_scenario,
              "detailed_result_rows": detailed, "split_manifest": manifest}
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", result), ("split_manifest.json", manifest),
                          ("resolved_config.json", config), ("detailed_metrics.json", detailed),
                          ("loss_trace.json", trace)):
        (run_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(run_dir / "score_table.csv", score_rows)
    _write_csv(run_dir / "detailed_metrics.csv", detailed)
    _write_csv(run_dir / "loss_trace.csv", trace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_STAGE), default=None)
    parser.add_argument("--target-receiver", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    for key, value in (("method", args.method), ("seed", args.seed), ("split_seed", args.split_seed),
                       ("k_shot", args.k_shot), ("experiment_id", args.experiment_id)):
        if value is not None:
            config[key] = value
    if args.target_receiver is not None:
        config["target_receiver_labels"] = [args.target_receiver]
    if args.method is not None:
        config["stage"] = METHOD_STAGE[args.method]
    validate_config(config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, sort_keys=True))
        return 0
    result = run(config, args.run_dir)
    print(json.dumps({"experiment_id": result["experiment_id"], "method": result["method"],
                      "metrics": result["metrics"], "run_dir": str(args.run_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
