from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
K_VALUES = (1, 5, 10, 20)
NEW_COUNTS = (5, 10, 20)
OLD_COUNT = 6
QUERY_PER_CLASS = 20
FFT_DIM = 96
FFT_WEIGHT = 8.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return (value / np.maximum(norm, 1.0e-8)).astype(np.float32)


def _fft96(rows: np.ndarray) -> np.ndarray:
    raw = np.asarray(rows, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, FFT_DIM, dtype=np.float64)
    output: list[np.ndarray] = []
    for row in raw:
        value = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        value -= np.mean(value)
        rms = float(np.sqrt(np.mean(np.abs(value) ** 2)))
        if rms > 1.0e-8:
            value /= rms
        window = np.hanning(value.size)
        spectrum = np.fft.fftshift(np.fft.fft(value * window))
        logmag = np.log1p(np.abs(spectrum))
        source_x = np.linspace(0.0, 1.0, logmag.size, dtype=np.float64)
        sketch = np.interp(target_x, source_x, logmag).astype(np.float32)
        sketch -= np.mean(sketch, dtype=np.float64).astype(np.float32)
        output.append(sketch)
    return _normalize(np.stack(output))


def _runtime_features(
    runtime: torch.jit.ScriptModule,
    iq: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    runtime.eval()
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            batch = torch.from_numpy(
                np.asarray(iq[start : start + batch_size], dtype=np.float32)
            ).to(device)
            output = runtime(batch)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("sealed runtime must return (z_id, logits)")
            z_id = output[0]
            if not torch.is_tensor(z_id) or z_id.ndim != 2 or z_id.shape[1] != 160:
                raise RuntimeError("sealed runtime z_id shape drift")
            chunks.append(z_id.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _fused_feature(z_id: np.ndarray, iq: np.ndarray) -> np.ndarray:
    z = _normalize(z_id)
    fft = _fft96(iq)
    return _normalize(np.concatenate((z, FFT_WEIGHT * fft), axis=1))


def _ordered_registry(tx_ids: np.ndarray) -> list[str]:
    return list(dict.fromkeys(np.asarray(tx_ids).astype(str).tolist()))


def _split_indices(
    tx_ids: np.ndarray,
    roles: np.ndarray,
    sample_ids: np.ndarray,
    registry: list[str],
    *,
    receiver: str,
    seed: int,
    split_policy: str,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    values = np.asarray(tx_ids).astype(str)
    role_values = np.asarray(roles).astype(str)
    sample_values = np.asarray(sample_ids).astype(str)
    for tx in registry:
        indices = np.flatnonzero(values == tx)
        if len(indices) != 40:
            raise RuntimeError(f"mother cell must contain 40 ordered rows for {tx}")
        role_set = set(role_values[indices].tolist())
        if len(role_set) != 1:
            raise RuntimeError(f"class role drift for {tx}")
        role = next(iter(role_set))
        if split_policy == "somph_offline_split_v1":
            ordered = sorted(
                (int(value) for value in indices.tolist()),
                key=lambda index: (
                    hashlib.sha256(
                        (
                            f"somph-offline-split-v1|{receiver}|{seed}|{role}|"
                            f"{tx}|{sample_values[index]}"
                        ).encode("utf-8")
                    ).hexdigest(),
                    str(sample_values[index]),
                ),
            )
            result[tx] = {
                "support_pool": np.asarray(ordered[:20], dtype=np.int64),
                "query": np.asarray(ordered[20:40], dtype=np.int64),
            }
        elif split_policy == "legacy_mother_array_order":
            result[tx] = {"support_pool": indices[:20], "query": indices[20:]}
        else:
            raise RuntimeError("unsupported split policy")
    return result


def _opaque_query_token(overlay_id: str) -> str:
    return "qid_" + hashlib.sha256(str(overlay_id).encode("utf-8")).hexdigest()


def _predict_all_registered(
    support_features: np.ndarray,
    support_class_indices: np.ndarray,
    query_features: np.ndarray,
    registered_classes: list[str],
) -> np.ndarray:
    """Truth-free per-query predictor; no labels/roles/quotas enter this function."""
    similarities = np.asarray(query_features, dtype=np.float32) @ np.asarray(
        support_features, dtype=np.float32
    ).T
    class_scores = np.full(
        (len(query_features), len(registered_classes)), -np.inf, dtype=np.float32
    )
    for class_index in range(len(registered_classes)):
        member = support_class_indices == class_index
        if not np.any(member):
            raise RuntimeError("registered class has no support")
        class_scores[:, class_index] = np.max(similarities[:, member], axis=1)
    predicted_indices = np.argmax(class_scores, axis=1)
    return np.asarray(
        [registered_classes[index] for index in predicted_indices], dtype=np.str_
    )


def _per_class_accuracy(
    predictions: np.ndarray, truth: np.ndarray, classes: list[str]
) -> dict[str, float]:
    return {
        tx: float(np.mean(predictions[truth == tx] == tx)) for tx in classes
    }


def _score_prediction_artifact(
    predictions: np.ndarray,
    truth: np.ndarray,
    old_classes: list[str],
    new_classes: list[str],
) -> dict[str, Any]:
    """Isolated scorer: query truth is first joined only inside this function."""
    old_mask = np.isin(truth, old_classes)
    new_mask = np.isin(truth, new_classes)
    old_per_class = _per_class_accuracy(predictions, truth, old_classes)
    new_per_class = _per_class_accuracy(predictions, truth, new_classes)
    old_acc = float(np.mean(predictions[old_mask] == truth[old_mask]))
    new_acc = float(np.mean(predictions[new_mask] == truth[new_mask]))
    harmonic = (
        float(2.0 * old_acc * new_acc / (old_acc + new_acc))
        if old_acc + new_acc > 0.0
        else 0.0
    )
    return {
        "old_accuracy": old_acc,
        "seen_new_accuracy": new_acc,
        "H_old_new": harmonic,
        "old_floor": min(old_per_class.values()),
        "seen_new_floor": min(new_per_class.values()),
        "old_per_class": old_per_class,
        "seen_new_per_class": new_per_class,
    }


def _resource(k: int, new_count: int) -> dict[str, Any]:
    before_support = OLD_COUNT * k
    after_support = (OLD_COUNT + new_count) * k
    feature_dim = 160 + FFT_DIM
    return {
        "adapter_trainable_parameters": 0,
        "adapt_epochs": 0,
        "feature_dim": feature_dim,
        "support_feature_storage_dtype": "float32",
        "before_support_count": before_support,
        "after_support_count": after_support,
        "before_persistent_state_bytes": before_support * (feature_dim * 4 + 2),
        "after_persistent_state_bytes": after_support * (feature_dim * 4 + 2),
        "before_head_dot_macs_per_query": before_support * feature_dim,
        "after_head_dot_macs_per_query": after_support * feature_dim,
        "after_class_max_comparisons_per_query": (OLD_COUNT + new_count) * (k - 1),
        "query_query_graph_used": False,
        "query_batch_state_bytes": 0,
        "fft96_transform_in_head_mac_count": False,
        "fft96_note": "deterministic 256-point received-IQ FFT/interpolation is computed once per row; reported dot MACs cover the exact KNN head only",
    }


def _markdown(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# D21 L5 fixed received-IQ快速dev矩阵",
        "",
        "本结果仅使用mother cell中已经叠加一次LEO_weak信道的固定IQ。每场景只提取一次`z_id160`和`FFT96`，没有clean样本、额外信道view、query标签拟合、角色Oracle、类别配额或query-query图。query truth只在独立评分函数中加入。",
        "",
        "特征：`g=normalize(concat(normalize(z_id160),8*normalize(FFT96)))`；分类：每类support top1 cosine，逐query对全部已注册类argmax。",
        "",
        "## 36-cell结果",
        "",
        "|场景|K|new|before old|after old|new|H|old floor|new floor|forgetting|K1 H gain|MAC/query|state B|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "|{scenario}|{k}|{new}|{before:.2%}|{after:.2%}|{new_acc:.2%}|{h:.2%}|{of:.2%}|{nf:.2%}|{forget:.2%}|{gain:+.2%}|{mac}|{state}|".format(
                scenario=row["scenario"],
                k=row["k_shot"],
                new=row["seen_new_count"],
                before=row["before"]["old_accuracy"],
                after=row["after"]["old_accuracy"],
                new_acc=row["after"]["seen_new_accuracy"],
                h=row["after"]["H_old_new"],
                of=row["after"]["old_floor"],
                nf=row["after"]["seen_new_floor"],
                forget=row["forgetting"],
                gain=row["k1_gain"]["H_old_new"],
                mac=row["resource"]["after_head_dot_macs_per_query"],
                state=row["resource"]["after_persistent_state_bytes"],
            )
        )
    lines += [
        "",
        "## 三场景均值",
        "",
        "|K|new|before old|after old|new|H|old floor worst|new floor worst|forgetting|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "|{k}|{new}|{before:.2%}|{after:.2%}|{new_acc:.2%}|{h:.2%}|{of:.2%}|{nf:.2%}|{forget:.2%}|".format(
                k=row["k_shot"], new=row["seen_new_count"],
                before=row["mean_before_old"], after=row["mean_after_old"],
                new_acc=row["mean_seen_new"], h=row["mean_H_old_new"],
                of=row["worst_old_floor"], nf=row["worst_seen_new_floor"],
                forget=row["mean_forgetting"],
            )
        )
    lines += [
        "",
        "## 资源口径",
        "",
        "`MAC/query`只统计精确top1-cosine KNN头的点积MAC，即`registered_classes*K*256`。FFT96是每个固定received-IQ样本一次性的256点FFT和96维插值，本表未把FFT实运算混入MAC，避免用未经锁定的FFT实现估算冒充精确MAC。持久状态按float32 support特征加uint16类索引计算；adapter参数和训练epoch均为0。",
        "",
        "before阶段只有6个旧类注册，因此before seen-new与before H按不可识别定义为0；after阶段追加新类support，旧support特征和值保持不变，forgetting=`before old-after old`反映纯竞争遗忘。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--split-policy",
        choices=("legacy_mother_array_order", "somph_offline_split_v1"),
        default="legacy_mother_array_order",
    )
    parser.add_argument("--receiver", default="20-1")
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--formal-capsule-predictor-root")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    runtime_path = Path(args.runtime).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runtime = torch.jit.load(str(runtime_path), map_location=device)
    runtime.eval()
    started = time.time()
    rows: list[dict[str, Any]] = []
    prediction_artifacts: list[dict[str, Any]] = []
    truth_sidecar: dict[str, dict[str, str]] = {}
    input_files: list[dict[str, Any]] = []
    registries: dict[str, Any] = {}
    capsule_alignment: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        path = input_dir / f"{scenario}.npz"
        input_files.append(
            {"scenario": scenario, "path": str(path), "sha256": _sha256(path)}
        )
        with np.load(path, allow_pickle=False) as archive:
            iq = np.array(archive["leo_weak_iq"], copy=True)
            tx_ids = np.asarray(archive["tx_ids"]).astype(str)
            roles = np.asarray(archive["dataset_role"]).astype(str)
            overlay_ids = np.asarray(archive["overlay_ids"]).astype(str)
            post_channel_hashes = np.asarray(
                archive["post_channel_iq_sha256"]
            ).astype(str)
            sample_ids = np.asarray(archive["sample_ids"]).astype(str)
            rx_ids = np.asarray(archive["rx_ids"]).astype(str)
            overlay_applied = np.asarray(archive["overlay_applied"], dtype=bool)
            sat_scenarios = np.asarray(archive["sat_scenarios"]).astype(str)
        if not np.all(overlay_applied) or set(sat_scenarios.tolist()) != {scenario}:
            raise RuntimeError("mother cell is not single-scenario LEO_weak-only")
        if set(rx_ids.tolist()) != {str(args.receiver)}:
            raise RuntimeError("mother cell receiver drift")
        registry = _ordered_registry(tx_ids)
        if len(registry) != 26:
            raise RuntimeError("expected 6 old plus 20 new classes")
        old_classes = registry[:OLD_COUNT]
        new_master = registry[OLD_COUNT:]
        if set(roles[np.isin(tx_ids, old_classes)].tolist()) != {"target_old"}:
            raise RuntimeError("old registry role drift")
        if set(roles[np.isin(tx_ids, new_master)].tolist()) != {"target_new"}:
            raise RuntimeError("new registry role drift")
        split = _split_indices(
            tx_ids,
            roles,
            sample_ids,
            registry,
            receiver=str(args.receiver),
            seed=int(args.seed),
            split_policy=str(args.split_policy),
        )
        if args.formal_capsule_predictor_root:
            formal_root = Path(args.formal_capsule_predictor_root).resolve()
            formal_classes = old_classes + new_master[:5]
            support_indices = np.concatenate(
                [split[tx]["support_pool"][:10] for tx in formal_classes]
            )
            query_indices_formal = np.concatenate(
                [split[tx]["query"] for tx in formal_classes]
            )
            support_path = (
                formal_root
                / "after"
                / "enrollment_only"
                / f"support_{scenario}.npz"
            )
            query_path = (
                formal_root
                / "after"
                / "apply_only_staging"
                / f"query_{scenario}.npz"
            )
            with np.load(support_path, allow_pickle=False) as formal_support:
                support_iq = np.asarray(formal_support["support_leo_weak_iq"], dtype=np.float32)
                support_hashes = np.asarray(
                    formal_support["support_post_channel_iq_sha256"]
                ).astype(str)
            with np.load(query_path, allow_pickle=False) as formal_query:
                query_iq = np.asarray(formal_query["query_leo_weak_iq"], dtype=np.float32)
                query_hashes = np.asarray(
                    formal_query["query_post_channel_iq_sha256"]
                ).astype(str)
            selected_support = np.asarray(iq[support_indices], dtype=np.float32)
            selected_query = np.asarray(iq[query_indices_formal], dtype=np.float32)
            if not np.array_equal(selected_support, support_iq):
                raise RuntimeError(f"formal support IQ mismatch for {scenario}")
            selected_support_hashes = post_channel_hashes[support_indices]
            selected_query_hashes = post_channel_hashes[query_indices_formal]
            if not np.array_equal(selected_support_hashes, support_hashes):
                raise RuntimeError(f"formal support IQ row-hash mismatch for {scenario}")
            if sorted(selected_query_hashes.tolist()) != sorted(query_hashes.tolist()):
                raise RuntimeError(f"formal query IQ hash-set mismatch for {scenario}")
            hash_to_index = {
                str(post_channel_hashes[index]): int(index)
                for index in query_indices_formal
            }
            if len(hash_to_index) != len(query_indices_formal):
                raise RuntimeError(f"duplicate formal query IQ hash for {scenario}")
            formal_order_indices = np.asarray(
                [hash_to_index[str(value)] for value in query_hashes], dtype=np.int64
            )
            selected_query_formal_order = np.asarray(
                iq[formal_order_indices], dtype=np.float32
            )
            if not np.array_equal(selected_query_formal_order, query_iq):
                raise RuntimeError(f"formal query IQ ordered mismatch for {scenario}")
            capsule_alignment.append(
                {
                    "scenario": scenario,
                    "support_iq_exact": True,
                    "query_iq_exact": True,
                    "support_count": int(len(selected_support)),
                    "query_count": int(len(selected_query)),
                    "support_iq_root_sha256": hashlib.sha256(
                        selected_support.tobytes(order="C")
                    ).hexdigest(),
                    "query_iq_root_sha256": hashlib.sha256(
                        selected_query_formal_order.tobytes(order="C")
                    ).hexdigest(),
                }
            )
        z_id = _runtime_features(
            runtime, iq, device=device, batch_size=int(args.batch_size)
        )
        feature = _fused_feature(z_id, iq)
        registries[scenario] = {
            "old_classes": old_classes,
            "nested_new_master": new_master,
            "support_pool_size_per_class": 20,
            "query_size_per_class": 20,
        }

        for new_count in NEW_COUNTS:
            new_classes = new_master[:new_count]
            after_classes = old_classes + new_classes
            query_classes = after_classes
            query_indices = np.concatenate([split[tx]["query"] for tx in query_classes])
            query_features = feature[query_indices]
            query_truth = tx_ids[query_indices]
            query_tokens = np.asarray(
                [_opaque_query_token(overlay_ids[index]) for index in query_indices]
            )
            for token, truth, role in zip(
                query_tokens,
                query_truth,
                roles[query_indices],
            ):
                truth_sidecar[str(token)] = {"truth_tx": str(truth), "role": str(role)}

            for k_shot in K_VALUES:
                old_support_indices = np.concatenate(
                    [split[tx]["support_pool"][:k_shot] for tx in old_classes]
                )
                before_support_labels = np.repeat(
                    np.arange(len(old_classes), dtype=np.int64), k_shot
                )
                before_predictions = _predict_all_registered(
                    feature[old_support_indices],
                    before_support_labels,
                    query_features,
                    old_classes,
                )
                before_scored = _score_prediction_artifact(
                    before_predictions, query_truth, old_classes, new_classes
                )
                before_scored["seen_new_accuracy"] = 0.0
                before_scored["H_old_new"] = 0.0
                before_scored["seen_new_floor"] = 0.0
                before_scored["seen_new_per_class"] = {
                    tx: 0.0 for tx in new_classes
                }

                after_support_indices = np.concatenate(
                    [split[tx]["support_pool"][:k_shot] for tx in after_classes]
                )
                after_support_labels = np.repeat(
                    np.arange(len(after_classes), dtype=np.int64), k_shot
                )
                after_predictions = _predict_all_registered(
                    feature[after_support_indices],
                    after_support_labels,
                    query_features,
                    after_classes,
                )
                after_scored = _score_prediction_artifact(
                    after_predictions, query_truth, old_classes, new_classes
                )
                cell_id = f"{scenario}__k{k_shot}__new{new_count}"
                row = {
                    "cell_id": cell_id,
                    "scenario": scenario,
                    "k_shot": k_shot,
                    "seen_new_count": new_count,
                    "old_classes": old_classes,
                    "new_classes": new_classes,
                    "before": before_scored,
                    "after": after_scored,
                    "forgetting": before_scored["old_accuracy"]
                    - after_scored["old_accuracy"],
                    "resource": _resource(k_shot, new_count),
                }
                rows.append(row)
                prediction_artifacts.append(
                    {
                        "cell_id": cell_id,
                        "query_schema": "unlabeled_per_sample_all_registered_classes_v1",
                        "query_tokens": query_tokens.tolist(),
                        "before_predictions": before_predictions.tolist(),
                        "after_predictions": after_predictions.tolist(),
                    }
                )

    k1 = {
        (row["scenario"], row["seen_new_count"]): row
        for row in rows
        if row["k_shot"] == 1
    }
    for row in rows:
        base = k1[(row["scenario"], row["seen_new_count"])]
        row["k1_gain"] = {
            "after_old": row["after"]["old_accuracy"]
            - base["after"]["old_accuracy"],
            "seen_new": row["after"]["seen_new_accuracy"]
            - base["after"]["seen_new_accuracy"],
            "H_old_new": row["after"]["H_old_new"]
            - base["after"]["H_old_new"],
            "old_floor": row["after"]["old_floor"] - base["after"]["old_floor"],
            "seen_new_floor": row["after"]["seen_new_floor"]
            - base["after"]["seen_new_floor"],
        }
    summary: list[dict[str, Any]] = []
    for k_shot in K_VALUES:
        for new_count in NEW_COUNTS:
            group = [
                row
                for row in rows
                if row["k_shot"] == k_shot
                and row["seen_new_count"] == new_count
            ]
            summary.append(
                {
                    "k_shot": k_shot,
                    "seen_new_count": new_count,
                    "mean_before_old": float(
                        np.mean([row["before"]["old_accuracy"] for row in group])
                    ),
                    "mean_after_old": float(
                        np.mean([row["after"]["old_accuracy"] for row in group])
                    ),
                    "mean_seen_new": float(
                        np.mean([row["after"]["seen_new_accuracy"] for row in group])
                    ),
                    "mean_H_old_new": float(
                        np.mean([row["after"]["H_old_new"] for row in group])
                    ),
                    "worst_old_floor": min(row["after"]["old_floor"] for row in group),
                    "worst_seen_new_floor": min(
                        row["after"]["seen_new_floor"] for row in group
                    ),
                    "mean_forgetting": float(
                        np.mean([row["forgetting"] for row in group])
                    ),
                }
            )
    payload = {
        "schema": "cvs.d21.l5_fast_dev_matrix.v1",
        "status": "DEVELOPMENT_ONLY",
        "method": "L5_concat_zid160_fft96x8_top1_support_cosine",
        "split_policy": str(args.split_policy),
        "receiver": str(args.receiver),
        "seed": int(args.seed),
        "formal_capsule_alignment": capsule_alignment,
        "input_files": input_files,
        "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
        "registries": registries,
        "grid": {
            "k_shot": list(K_VALUES),
            "seen_new_count": list(NEW_COUNTS),
            "scenarios": list(SCENARIOS),
            "cell_count": len(rows),
        },
        "query_decision_policy": "per_sample_all_registered_classes",
        "query_truth_used_by_predictor": False,
        "query_role_oracle_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions_truth_free.json").write_text(
        json.dumps(prediction_artifacts, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "scorer_truth_sidecar.json").write_text(
        json.dumps(truth_sidecar, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.md").write_text(
        _markdown(rows, summary), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "cells": len(rows), "output": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
