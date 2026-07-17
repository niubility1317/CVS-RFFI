from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
METHODS = ("top1", "top2_trimmed", "medoid", "bagged2", "radius_standardized")
K_SHOT = 10
OLD_COUNT = 6
NEW_COUNT = 5
STATE_LIMIT_BYTES = 256 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("l5_formal_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load formal helper script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1.0e-8)


def _quantize_vectors(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(rows, dtype=np.float32)
    peak = np.max(np.abs(value), axis=1)
    scales = np.maximum(peak / 127.0, 1.0e-8).astype(np.float16)
    quantized = np.clip(np.rint(value / scales[:, None]), -127, 127).astype(np.int8)
    dequantized = _normalize(quantized.astype(np.float32) * scales.astype(np.float32)[:, None])
    return quantized, scales, dequantized.astype(np.float32)


def _build_state(method: str, support_by_class: list[np.ndarray]) -> dict[str, Any]:
    vectors: list[np.ndarray] = []
    owners: list[int] = []
    radii: list[tuple[float, float]] = []
    for class_index, raw_rows in enumerate(support_by_class):
        rows = _normalize(raw_rows)
        if method in {"top1", "top2_trimmed"}:
            chosen = rows
        elif method == "medoid":
            similarity = rows @ rows.T
            chosen = rows[[int(np.argmax(np.mean(similarity, axis=1)))]]
        elif method == "bagged2":
            groups = (rows[0::2], rows[1::2])
            chosen = np.concatenate(
                [_normalize(np.mean(group, axis=0))[0:1] for group in groups if len(group)],
                axis=0,
            )
        elif method == "radius_standardized":
            chosen = _normalize(np.mean(rows, axis=0))[0:1]
            distance = 1.0 - (rows @ chosen[0])
            center = float(np.median(distance))
            robust_scale = float(1.4826 * np.median(np.abs(distance - center)))
            radii.append((center, max(robust_scale, 0.005)))
        else:
            raise ValueError(method)
        vectors.append(chosen)
        owners.extend([class_index] * len(chosen))
    stacked = np.concatenate(vectors, axis=0).astype(np.float32)
    q, scales, dequantized = _quantize_vectors(stacked)
    owner_array = np.asarray(owners, dtype=np.uint16)
    radius_array = np.asarray(radii, dtype=np.float16)
    state_bytes = int(q.nbytes + scales.nbytes + owner_array.nbytes + radius_array.nbytes)
    return {
        "method": method,
        "vectors": dequantized,
        "owners": owner_array.astype(np.int64),
        "radius": radius_array.astype(np.float32),
        "state_bytes": state_bytes,
        "int8_vector_count": int(len(q)),
        "feature_dim": int(q.shape[1]),
    }


def _predict(state: dict[str, Any], query: np.ndarray, class_count: int) -> np.ndarray:
    """Truth-free, role-free, quota-free independent per-query prediction."""
    similarity = _normalize(query) @ state["vectors"].T
    scores = np.full((len(query), class_count), -np.inf, dtype=np.float32)
    method = str(state["method"])
    for class_index in range(class_count):
        member = state["owners"] == class_index
        values = similarity[:, member]
        if method == "top1" or method == "bagged2":
            scores[:, class_index] = np.max(values, axis=1)
        elif method == "top2_trimmed":
            # Discard the lower K-2 neighbours, then average the retained top two.
            take = min(2, values.shape[1])
            scores[:, class_index] = np.mean(np.partition(values, -take, axis=1)[:, -take:], axis=1)
        elif method == "medoid":
            scores[:, class_index] = values[:, 0]
        elif method == "radius_standardized":
            distance = 1.0 - values[:, 0]
            center, scale = state["radius"][class_index]
            scores[:, class_index] = -(distance - center) / max(float(scale), 0.005)
        else:
            raise ValueError(method)
    return np.argmax(scores, axis=1).astype(np.int64)


def _metrics(pred: np.ndarray, truth: np.ndarray, old_count: int) -> dict[str, Any]:
    old = truth < old_count
    new = ~old
    old_acc = float(np.mean(pred[old] == truth[old]))
    new_acc = float(np.mean(pred[new] == truth[new])) if np.any(new) else 0.0
    harmonic = 2.0 * old_acc * new_acc / (old_acc + new_acc) if old_acc + new_acc else 0.0
    per_class = {
        str(index): float(np.mean(pred[truth == index] == index))
        for index in sorted(set(truth.tolist()))
    }
    old_values = [per_class[str(index)] for index in range(old_count)]
    new_indices = [index for index in sorted(set(truth.tolist())) if index >= old_count]
    new_values = [per_class[str(index)] for index in new_indices]
    return {
        "old_accuracy": old_acc,
        "seen_new_accuracy": new_acc,
        "H_old_new": float(harmonic),
        "old_floor": float(min(old_values)),
        "seen_new_floor": float(min(new_values)) if new_values else 0.0,
        "per_class_index": per_class,
    }


def _loo(method: str, support_by_class: list[np.ndarray], old_count: int) -> dict[str, Any]:
    predictions: list[int] = []
    truth: list[int] = []
    state_sizes: list[int] = []
    for class_index, class_rows in enumerate(support_by_class):
        for held_index in range(len(class_rows)):
            fold = [
                np.delete(rows, held_index, axis=0) if index == class_index else rows
                for index, rows in enumerate(support_by_class)
            ]
            state = _build_state(method, fold)
            predictions.append(int(_predict(state, class_rows[held_index : held_index + 1], len(fold))[0]))
            truth.append(class_index)
            state_sizes.append(int(state["state_bytes"]))
    scored = _metrics(np.asarray(predictions), np.asarray(truth), old_count)
    scored["max_fold_state_bytes"] = max(state_sizes)
    return scored


def _pooled(rows: list[dict[str, Any]], old_count: int) -> dict[str, Any]:
    pred = np.concatenate([row.pop("_pred") for row in rows])
    truth = np.concatenate([row.pop("_truth") for row in rows])
    return _metrics(pred, truth, old_count)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M3稳健局部注册：formal K10/new5",
        "",
        "固定特征为`A0=normalize([normalize(z_id160),8*normalize(FFT96)])`。候选及排序均仅使用三场景K10 support LOO；query不参与超参数、方法选择或回滚。所有候选实际以逐向量对称int8状态解量化后预测。",
        "",
        f"support预锁定方法：`{payload['selector_lock']['selected_method']}`。",
        "",
        "## Support LOO预锁定",
        "",
        "|方法|old|new|H|old floor|new floor|遗忘|状态B|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = payload["support_loo"][method]["pooled_after"]
        info = payload["support_loo"][method]
        lines.append(
            f"|{method}|{row['old_accuracy']:.2%}|{row['seen_new_accuracy']:.2%}|{row['H_old_new']:.2%}|{row['old_floor']:.2%}|{row['seen_new_floor']:.2%}|{info['pooled_forgetting']:.2%}|{info['max_full_state_bytes']}|"
        )
    lines += [
        "",
        "排序键依次为：`min(old_floor,new_floor)`、`H_old_new`、`-forgetting`、`-state_bytes`；全局统一，不按场景或query选择。",
        "",
        "## Formal query隔离评分",
        "",
        "|方法|注册前old|注册后old|new|H|old floor|new floor|遗忘|状态B|dot MAC/query|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        info = payload["query_evaluation"][method]
        row = info["pooled_after"]
        lines.append(
            f"|{method}|{info['pooled_before_old_accuracy']:.2%}|{row['old_accuracy']:.2%}|{row['seen_new_accuracy']:.2%}|{row['H_old_new']:.2%}|{row['old_floor']:.2%}|{row['seen_new_floor']:.2%}|{info['pooled_forgetting']:.2%}|{info['max_full_state_bytes']}|{info['dot_macs_per_query']}|"
        )
    selected = payload["selector_lock"]["selected_method"]
    s = payload["query_evaluation"][selected]
    p = s["pooled_after"]
    lines += [
        "",
        "## 预锁定方法结论",
        "",
        f"`{selected}`在formal query上的注册前old={s['pooled_before_old_accuracy']:.2%}，注册后old={p['old_accuracy']:.2%}，seen-new={p['seen_new_accuracy']:.2%}，H={p['H_old_new']:.2%}，old floor={p['old_floor']:.2%}，new floor={p['seen_new_floor']:.2%}，遗忘={s['pooled_forgetting']:.2%}。",
        "",
        "本实验是单receiver×单seed开发筛选，不构成125确认矩阵或正式达标结论。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--helper-script", required=True)
    parser.add_argument("--formal-capsule-predictor-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receiver", default="20-1")
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    helper_path = Path(args.helper_script).resolve()
    helpers = _load_helpers(helper_path)
    runtime_path = Path(args.runtime).resolve()
    runtime = torch.jit.load(str(runtime_path), map_location=torch.device(args.device))
    scenario_data: dict[str, Any] = {}
    input_audit: list[dict[str, Any]] = []
    capsule_alignment: list[dict[str, Any]] = []
    old_names: list[str] | None = None
    new_names: list[str] | None = None
    for scenario in SCENARIOS:
        path = Path(args.input_dir).resolve() / f"{scenario}.npz"
        with np.load(path, allow_pickle=False) as archive:
            iq = np.asarray(archive["leo_weak_iq"], dtype=np.float32)
            tx = np.asarray(archive["tx_ids"]).astype(str)
            roles = np.asarray(archive["dataset_role"]).astype(str)
            sample_ids = np.asarray(archive["sample_ids"]).astype(str)
            rx = np.asarray(archive["rx_ids"]).astype(str)
            overlay = np.asarray(archive["overlay_applied"], dtype=bool)
            sat = np.asarray(archive["sat_scenarios"]).astype(str)
            overlay_ids = np.asarray(archive["overlay_ids"]).astype(str)
            post_channel_hashes = np.asarray(archive["post_channel_iq_sha256"]).astype(str)
        if not np.all(overlay) or set(sat.tolist()) != {scenario} or set(rx.tolist()) != {args.receiver}:
            raise RuntimeError("LEO_weak single-scenario/receiver guard failed")
        registry = helpers._ordered_registry(tx)
        current_old, current_new = registry[:OLD_COUNT], registry[OLD_COUNT : OLD_COUNT + NEW_COUNT]
        if old_names is None:
            old_names, new_names = current_old, current_new
        if current_old != old_names or current_new != new_names:
            raise RuntimeError("registry drift across scenarios")
        split = helpers._split_indices(
            tx, roles, sample_ids, registry, receiver=args.receiver, seed=args.seed,
            split_policy="somph_offline_split_v1",
        )
        classes = current_old + current_new
        support_indices = [split[name]["support_pool"][:K_SHOT] for name in classes]
        query_indices = np.concatenate([split[name]["query"] for name in classes])
        formal_root = Path(args.formal_capsule_predictor_root).resolve()
        with np.load(formal_root / "after" / "enrollment_only" / f"support_{scenario}.npz", allow_pickle=False) as formal:
            formal_support_iq = np.asarray(formal["support_leo_weak_iq"], dtype=np.float32)
            formal_support_hashes = np.asarray(formal["support_post_channel_iq_sha256"]).astype(str)
        with np.load(formal_root / "after" / "apply_only_staging" / f"query_{scenario}.npz", allow_pickle=False) as formal:
            formal_query_hashes = np.asarray(formal["query_post_channel_iq_sha256"]).astype(str)
        flat_support = np.concatenate(support_indices)
        if not np.array_equal(iq[flat_support], formal_support_iq):
            raise RuntimeError(f"formal support IQ mismatch: {scenario}")
        if not np.array_equal(post_channel_hashes[flat_support], formal_support_hashes):
            raise RuntimeError(f"formal support hash mismatch: {scenario}")
        if sorted(post_channel_hashes[query_indices].tolist()) != sorted(formal_query_hashes.tolist()):
            raise RuntimeError(f"formal query hash-set mismatch: {scenario}")
        capsule_alignment.append({"scenario": scenario, "support_iq_exact": True, "support_hash_order_exact": True, "query_hash_set_exact": True})
        z_id = helpers._runtime_features(runtime, iq, device=torch.device(args.device), batch_size=128)
        feature = helpers._fused_feature(z_id, iq)
        scenario_data[scenario] = {
            "support": [feature[index] for index in support_indices],
            "query": feature[query_indices],
            "truth": np.repeat(np.arange(len(classes), dtype=np.int64), 20),
            "tokens": np.asarray([hashlib.sha256(value.encode()).hexdigest() for value in overlay_ids[query_indices]]),
        }
        input_audit.append({"scenario": scenario, "path": str(path), "sha256": _sha256(path), "leo_weak_guard": True})

    support_loo: dict[str, Any] = {}
    for method in METHODS:
        scene_after: list[dict[str, Any]] = []
        scene_rows: list[dict[str, Any]] = []
        max_state = 0
        for scenario in SCENARIOS:
            support = scenario_data[scenario]["support"]
            after = _loo(method, support, OLD_COUNT)
            before = _loo(method, support[:OLD_COUNT], OLD_COUNT)
            forgetting = before["old_accuracy"] - after["old_accuracy"]
            scene_rows.append({"scenario": scenario, "before_old_accuracy": before["old_accuracy"], "after": after, "forgetting": forgetting})
            # Pooled support rows are reconstructed from per-class rates because K is balanced.
            pred_proxy, truth_proxy = [], []
            for key, accuracy in after["per_class_index"].items():
                correct = int(round(accuracy * K_SHOT))
                label = int(key)
                pred_proxy.extend([label] * correct + [999] * (K_SHOT - correct))
                truth_proxy.extend([label] * K_SHOT)
            scene_after.append({"_pred": np.asarray(pred_proxy), "_truth": np.asarray(truth_proxy)})
            max_state = max(max_state, _build_state(method, support)["state_bytes"])
        pooled_after = _pooled(scene_after, OLD_COUNT)
        pooled_before = float(np.mean([row["before_old_accuracy"] for row in scene_rows]))
        support_loo[method] = {
            "scenes": scene_rows,
            "pooled_after": pooled_after,
            "pooled_before_old_accuracy": pooled_before,
            "pooled_forgetting": pooled_before - pooled_after["old_accuracy"],
            "max_full_state_bytes": int(max_state),
        }
    ranking = sorted(
        METHODS,
        key=lambda method: (
            min(support_loo[method]["pooled_after"]["old_floor"], support_loo[method]["pooled_after"]["seen_new_floor"]),
            support_loo[method]["pooled_after"]["H_old_new"],
            -support_loo[method]["pooled_forgetting"],
            -support_loo[method]["max_full_state_bytes"],
            -METHODS.index(method),
        ),
        reverse=True,
    )
    selector_lock = {
        "selected_method": ranking[0],
        "ranking": ranking,
        "evidence": "three_scenario_k10_support_loo_only",
        "selection_key": ["max_min_old_new_floor", "max_H_old_new", "min_forgetting", "min_state_bytes", "fixed_order"],
        "query_used_for_selection": False,
    }
    (output_dir / "selector_lock.json").write_text(json.dumps(selector_lock, indent=2), encoding="utf-8")

    query_evaluation: dict[str, Any] = {}
    prediction_dir = output_dir / "sealed_predictions"
    prediction_dir.mkdir(exist_ok=True)
    for method in METHODS:
        scene_rows: list[dict[str, Any]] = []
        pooled_rows: list[dict[str, Any]] = []
        max_state = 0
        vector_count = 0
        for scenario in SCENARIOS:
            data = scenario_data[scenario]
            before_state = _build_state(method, data["support"][:OLD_COUNT])
            after_state = _build_state(method, data["support"])
            old_mask = data["truth"] < OLD_COUNT
            before_pred = _predict(before_state, data["query"][old_mask], OLD_COUNT)
            after_pred = _predict(after_state, data["query"], OLD_COUNT + NEW_COUNT)
            np.savez_compressed(
                prediction_dir / f"{method}__{scenario}.npz",
                query_token=data["tokens"], predicted_class_index=after_pred,
            )
            after = _metrics(after_pred, data["truth"], OLD_COUNT)
            before_old = float(np.mean(before_pred == data["truth"][old_mask]))
            scene_rows.append({"scenario": scenario, "before_old_accuracy": before_old, "after": after, "forgetting": before_old - after["old_accuracy"]})
            pooled_rows.append({"_pred": after_pred.copy(), "_truth": data["truth"].copy()})
            max_state = max(max_state, after_state["state_bytes"])
            vector_count = max(vector_count, after_state["int8_vector_count"])
        pooled_after = _pooled(pooled_rows, OLD_COUNT)
        pooled_before = float(np.mean([row["before_old_accuracy"] for row in scene_rows]))
        dot_macs = int(vector_count * 256)
        query_evaluation[method] = {
            "scenes": scene_rows,
            "pooled_after": pooled_after,
            "pooled_before_old_accuracy": pooled_before,
            "pooled_forgetting": pooled_before - pooled_after["old_accuracy"],
            "max_full_state_bytes": int(max_state),
            "state_within_256KB": bool(max_state <= STATE_LIMIT_BYTES),
            "int8_vector_count": int(vector_count),
            "dot_macs_per_query": dot_macs,
        }
    payload = {
        "status": "COMPLETE_DEVELOPMENT_SCREEN_NOT_FORMAL_CONFIRMATION",
        "receiver": args.receiver,
        "seed": args.seed,
        "k_shot": K_SHOT,
        "seen_new_count": NEW_COUNT,
        "feature": "A0_z_id160_plus_8x_fft96_normalized_dim256",
        "methods": list(METHODS),
        "selector_lock": selector_lock,
        "support_loo": support_loo,
        "query_evaluation": query_evaluation,
        "input_audit": input_audit,
        "formal_capsule_alignment": capsule_alignment,
        "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
        "helper_script": {"path": str(helper_path), "sha256": _sha256(helper_path)},
        "protocol": {
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "clean_sample_access": False,
            "query_decision": "per_sample_all_registered_classes",
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "support_loo_only_selection": True,
        },
        "resource": {"adapter_parameters": 0, "adapt_epochs": 0, "dense_query_graph": False, "state_limit_bytes": STATE_LIMIT_BYTES},
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "results.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"selected": ranking[0], "query": query_evaluation[ranking[0]], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
