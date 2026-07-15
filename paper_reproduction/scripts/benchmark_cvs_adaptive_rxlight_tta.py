#!/usr/bin/env python
"""Benchmark fixed and adaptive rx_light5 with one deployed ADV3B02 state.

The benchmark is deliberately non-transductive.  Registered support builds one
prototype bank per receive view and leave-one-out support scores calibrate a
single preregistered 1->3->5 early-exit gate.  Query labels are loaded only
after the gate is frozen and are used exclusively for reporting metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from export_spaceborne_features import (
    _satellite_tta_views,
    _spectral_logmag_sketch_batch,
)
from paper_reproduction.cvs_aligned.adaptive_rxlight_tta import (
    RX_LIGHT5_ORDER,
    apply_adaptive_rxlight_tta,
    calibrate_adaptive_rxlight_tta,
)
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, _select_split
from paper_reproduction.cvs_aligned.extreme_light_adapter import (
    concatenate_registered_features,
)
from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
    _feature_forward,
    _json_safe,
    _load_npz,
    _numpy_to_tensor_compat,
    _sha256_file,
    _tensor_to_numpy_compat,
)
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    LATE_KEY_FT_TARGETS,
)


BASE_MARGIN_GRID = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)
SHIFT3_MARGIN_GRID = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
DISAGREEMENT_GRID = (0.0, 1.0 / 3.0, 2.0 / 3.0)


def _norm(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return _json_safe(value)


def _sample_id(arrays: dict[str, np.ndarray], index: int) -> str:
    return "|".join(
        str(arrays[key][index])
        for key in ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
    )


def _ids_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_fp16_checkpoint_delta(
    model: torch.nn.Module,
    delta_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Merge the exact six-tensor FP16 late-key patch into a strict checkpoint."""

    expected = {
        f"{module_name}.{suffix}"
        for module_name in LATE_KEY_FT_TARGETS
        for suffix in ("weight", "bias")
    }
    if set(delta_state) != expected:
        raise ValueError(
            "late-key delta key mismatch: "
            f"observed={sorted(delta_state)}, expected={sorted(expected)}"
        )
    parameters = dict(model.named_parameters())
    element_count = 0
    with torch.no_grad():
        for name in sorted(expected):
            if name not in parameters:
                raise ValueError(f"checkpoint model is missing delta parameter {name}")
            delta = delta_state[name].detach().cpu()
            parameter = parameters[name]
            if tuple(delta.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"delta shape mismatch for {name}: {tuple(delta.shape)} != "
                    f"{tuple(parameter.shape)}"
                )
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError(f"non-finite delta tensor: {name}")
            parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype))
            element_count += int(delta.numel())
    if element_count != 31_200:
        raise ValueError(f"late-key delta element budget drift: {element_count}")
    return {
        "format": "fp16_delta_from_strict_checkpoint",
        "tensor_count": len(expected),
        "element_count": int(element_count),
        "tensor_bytes_fp16": int(element_count * 2),
        "target_modules": list(LATE_KEY_FT_TARGETS),
        "merged_added_macs_per_query": 0,
    }


def build_view_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Return independently normalized prototypes shaped [5,C,D]."""

    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).astype(str)
    if features.ndim != 3 or features.shape[0] != 5 or features.shape[1] != len(labels):
        raise ValueError("support_features must have shape [5,N,D] aligned to labels")
    normalized = _norm(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    prototypes = []
    for view_index in range(5):
        view_rows = []
        for label in classes:
            selected = normalized[view_index, labels == str(label)]
            if len(selected) < 2:
                raise ValueError(f"view prototype requires >=2 support rows for {label}")
            view_rows.append(_norm(selected.mean(axis=0, keepdims=True))[0])
        prototypes.append(np.stack(view_rows).astype(np.float32))
    # The five banks are persisted in FP16 so the adapter patch plus prototype
    # state remains below the strict 128 KiB deployment cap.  Scoring promotes
    # the stored values to FP32.
    return np.stack(prototypes).astype(np.float16)


def score_views(query_features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Score matching query/prototype views and return [N,5,C]."""

    query = np.asarray(query_features, dtype=np.float32)
    banks = np.asarray(prototypes, dtype=np.float32)
    if query.ndim != 3 or query.shape[0] != 5 or banks.ndim != 3 or banks.shape[0] != 5:
        raise ValueError("query features and prototypes must each start with five views")
    if query.shape[2] != banks.shape[2]:
        raise ValueError("query/prototype feature dimensions differ")
    return np.stack(
        [_norm(query[v]) @ _norm(banks[v]).T for v in range(5)], axis=1
    ).astype(np.float32)


def leave_one_out_support_scores(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Build legal support-only calibration scores without self-prototypes."""

    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).astype(str)
    if features.ndim != 3 or features.shape[0] != 5 or features.shape[1] != len(labels):
        raise ValueError("support_features must have shape [5,N,D]")
    normalized = _norm(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    class_to_index = {str(label): index for index, label in enumerate(classes)}
    scores = np.empty((len(labels), 5, len(classes)), dtype=np.float32)
    for view_index in range(5):
        full_sums = {
            str(label): normalized[view_index, labels == str(label)].sum(axis=0)
            for label in classes
        }
        counts = {str(label): int(np.sum(labels == str(label))) for label in classes}
        if min(counts.values()) < 2:
            raise ValueError("leave-one-out calibration requires >=2 support rows per class")
        for row_index, row_label in enumerate(labels):
            banks = []
            for label in classes:
                label = str(label)
                total = full_sums[label]
                count = counts[label]
                if label == row_label:
                    total = total - normalized[view_index, row_index]
                    count -= 1
                banks.append(_norm((total / count)[None, :])[0])
            scores[row_index, view_index] = (
                normalized[view_index, row_index] @ np.stack(banks).T
            )
    if set(labels) - set(class_to_index):
        raise ValueError("support labels are outside the registered class list")
    return scores


@torch.no_grad()
def extract_joint_rxlight5(
    model: torch.nn.Module,
    raw_iq: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    fft_dim: int = 96,
    fft_weight: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract same-view ADV3B02+FFT features for exact rx_light5."""

    rows = _numpy_to_tensor_compat(
        raw_iq,
        numpy_dtype=np.dtype(np.float32),
        torch_dtype=torch.float32,
    )
    generated = _satellite_tta_views(rows, "rx_light5")
    names = tuple(name for name, _ in generated)
    if names != RX_LIGHT5_ORDER:
        raise ValueError(f"rx_light5 definition drift: {names} != {RX_LIGHT5_ORDER}")
    outputs: list[np.ndarray] = []
    timings: dict[str, float] = {}
    model.eval()
    for name, view_rows in generated:
        started = time.perf_counter()
        primary_parts: list[np.ndarray] = []
        for start in range(0, int(view_rows.shape[0]), int(batch_size)):
            batch = view_rows[start : start + int(batch_size)].to(device)
            primary, _ = _feature_forward(model, batch)
            primary_parts.append(
                _tensor_to_numpy_compat(primary, dtype=np.dtype(np.float32))
            )
        primary_np = np.concatenate(primary_parts, axis=0)
        raw_np = view_rows.detach().cpu().numpy().astype(np.float32)
        fft = _spectral_logmag_sketch_batch(raw_np, dim=int(fft_dim))
        outputs.append(
            concatenate_registered_features(
                primary_np,
                fft,
                auxiliary_weight=float(fft_weight),
            )
        )
        timings[name] = float(time.perf_counter() - started)
    return np.stack(outputs).astype(np.float32), {
        "view_names": list(names),
        "physical_rows": int(len(raw_iq)),
        "joint_feature_dim": int(outputs[0].shape[1]),
        "seconds_by_view": timings,
        "total_seconds": float(sum(timings.values())),
    }


def _split_indices(
    arrays: dict[str, np.ndarray], config: dict[str, Any], scenario: str
) -> tuple[list[int], list[int]]:
    common = {
        "receiver": str(config["target_receiver_labels"][0]),
        "seed": int(config["seed"]),
        "k_shot": int(config["k_shot"]),
        "support_pool_max_k": int(config["support_pool_max_k"]),
        "query_per_tx": int(config["query_per_tx"]),
        "scenario": str(scenario),
    }
    old_support, old_query = _select_split(
        arrays,
        role="target_old",
        tx_labels=[str(value) for value in config["target_old_tx_labels"]],
        **common,
    )
    new_support, new_query = _select_split(
        arrays,
        role="target_new",
        tx_labels=[str(value) for value in config["target_new_tx_labels"]],
        **common,
    )
    return old_support + new_support, old_query + new_query


def _metric_row(
    predictions: np.ndarray,
    truth: np.ndarray,
    roles: np.ndarray,
    old_labels: Sequence[str],
    new_labels: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pred = np.asarray(predictions).astype(str)
    y = np.asarray(truth).astype(str)
    role_values = np.asarray(roles).astype(str)
    per_class: list[dict[str, Any]] = []
    for label in list(old_labels) + list(new_labels):
        mask = y == str(label)
        per_class.append(
            {
                "tx_label": str(label),
                "evaluation_role": "target_old" if str(label) in set(old_labels) else "target_new",
                "query_count": int(mask.sum()),
                "accuracy": float(np.mean(pred[mask] == y[mask])),
            }
        )
    old_mask = role_values == "target_old"
    new_mask = role_values == "target_new"
    old_acc = float(np.mean(pred[old_mask] == y[old_mask]))
    new_acc = float(np.mean(pred[new_mask] == y[new_mask]))
    harmonic = float(2.0 * old_acc * new_acc / max(old_acc + new_acc, 1.0e-12))
    old_rows = [row for row in per_class if row["evaluation_role"] == "target_old"]
    new_rows = [row for row in per_class if row["evaluation_role"] == "target_new"]
    return {
        "old_accuracy": old_acc,
        "min_old_class_accuracy": float(min(row["accuracy"] for row in old_rows)),
        "new_accuracy": new_acc,
        "min_new_class_accuracy": float(min(row["accuracy"] for row in new_rows)),
        "harmonic_mean": harmonic,
        "overall_accuracy": float(np.mean(pred == y)),
    }, per_class


def _reference_parity(
    reference_path: Path | None,
    arrays: dict[str, np.ndarray],
    selected_indices: Sequence[int],
    generated_base: np.ndarray,
) -> dict[str, Any]:
    if reference_path is None:
        return {"checked": False}
    # Adapted feature exports intentionally omit raw_iq; the training loader
    # enforces raw_iq and therefore cannot be reused for this read-only parity
    # cache.  Keep this loader narrow and reject pickle/object arrays.
    with np.load(reference_path, allow_pickle=False) as payload:
        reference = {key: payload[key] for key in payload.files}
    required = {
        "features",
        "fft_logmag_features",
        "dataset_role",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "eq_ids",
        "sig_ids",
    }
    if not required.issubset(reference):
        raise KeyError(
            f"reference feature cache is missing keys: {sorted(required - set(reference))}"
        )
    expected_ids = [_sample_id(arrays, index) for index in selected_indices]
    reference_lookup = {_sample_id(reference, index): index for index in range(len(reference["tx_ids"]))}
    positions = [reference_lookup[value] for value in expected_ids]
    expected = concatenate_registered_features(
        reference["features"][positions],
        reference["fft_logmag_features"][positions],
        auxiliary_weight=2.0,
    )
    cosine = np.sum(_norm(generated_base) * _norm(expected), axis=1)
    return {
        "checked": True,
        "reference_path": str(reference_path),
        "row_count": int(len(positions)),
        "mean_cosine": float(np.mean(cosine)),
        "min_cosine": float(np.min(cosine)),
        "mean_absolute_difference": float(np.mean(np.abs(generated_base - expected))),
        "max_absolute_difference": float(np.max(np.abs(generated_base - expected))),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Raw-IQ protocol config")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--adapter_state", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--reference_config", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_accuracy_drop_pp", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 1 <= int(args.batch_size) <= 4096:
        raise ValueError("batch_size must be in [1,4096]")
    if not 0.0 <= float(args.max_accuracy_drop_pp) <= 3.0:
        raise ValueError("max_accuracy_drop_pp must be in [0,3]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if list(config["target_receiver_labels"]) != ["8-8"]:
        raise ValueError("this preregistered recovery check is fixed to receiver 8-8")
    if int(config["k_shot"]) != 10 or int(config["seed"]) != 713101:
        raise ValueError("this preregistered recovery check is fixed to K10/seed713101")
    if len(config["target_new_tx_labels"]) != 20:
        raise ValueError("this preregistered recovery check is fixed to 20 seen-new classes")
    reference_config = None
    if args.reference_config is not None:
        reference_config = json.loads(args.reference_config.read_text(encoding="utf-8-sig"))
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=256, device=device
    )
    delta_state = torch.load(args.adapter_state, map_location="cpu")
    if not isinstance(delta_state, dict):
        raise TypeError("adapter_state must be a tensor dictionary")
    delta_audit = apply_fp16_checkpoint_delta(model, delta_state)
    model.to(device).eval()

    old_labels = [str(value) for value in config["target_old_tx_labels"]]
    new_labels = [str(value) for value in config["target_new_tx_labels"]]
    classes = sorted(old_labels + new_labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    scenario_payloads: dict[str, dict[str, Any]] = {}
    calibration_scores: list[np.ndarray] = []
    calibration_labels: list[np.ndarray] = []
    protocol_audit: dict[str, Any] = {}
    extraction_audit: dict[str, Any] = {}
    parity_audit: dict[str, Any] = {}
    for scenario in SCENARIOS:
        raw_path = Path(config["feature_npz_by_scenario"][scenario])
        arrays, raw_manifest = _load_npz(raw_path)
        if "raw_iq" not in arrays:
            raise ValueError(f"raw-IQ cache is missing raw_iq: {raw_path}")
        support_idx, query_idx = _split_indices(arrays, config, scenario)
        selected_idx = support_idx + query_idx
        selected_features, extract = extract_joint_rxlight5(
            model,
            arrays["raw_iq"][selected_idx],
            batch_size=int(args.batch_size),
            device=device,
        )
        support_count = len(support_idx)
        support_features = selected_features[:, :support_count]
        query_features = selected_features[:, support_count:]
        support_y = arrays["tx_ids"][support_idx].astype(str)
        truth = arrays["tx_ids"][query_idx].astype(str)
        roles = arrays["dataset_role"][query_idx].astype(str)
        prototypes = build_view_prototypes(support_features, support_y, classes)
        query_scores = score_views(query_features, prototypes)
        loo_scores = leave_one_out_support_scores(support_features, support_y, classes)
        calibration_scores.append(loo_scores)
        calibration_labels.append(
            np.asarray([class_to_index[label] for label in support_y], dtype=np.int64)
        )
        reference_path = None
        if reference_config is not None:
            reference_path = Path(reference_config["feature_npz_by_scenario"][scenario])
        parity_audit[scenario] = _reference_parity(
            reference_path, arrays, selected_idx, selected_features[0]
        )
        if parity_audit[scenario]["checked"]:
            if (
                float(parity_audit[scenario]["mean_cosine"]) < 0.9999
                or float(parity_audit[scenario]["min_cosine"]) < 0.999
            ):
                raise ValueError(
                    "deployed FP16 delta base-view extraction does not reproduce "
                    f"the registered adapted cache for {scenario}: "
                    f"{parity_audit[scenario]}"
                )
        protocol_audit[scenario] = {
            "raw_cache": str(raw_path),
            "raw_cache_sha256": _sha256_file(raw_path),
            "raw_cache_manifest": raw_manifest,
            "support_count": int(len(support_idx)),
            "query_count": int(len(query_idx)),
            "support_ids_sha256": _ids_sha256([_sample_id(arrays, i) for i in support_idx]),
            "query_ids_sha256": _ids_sha256([_sample_id(arrays, i) for i in query_idx]),
            "support_query_overlap": int(len(set(support_idx) & set(query_idx))),
            "support_roles": sorted(set(arrays["dataset_role"][support_idx].astype(str).tolist())),
            "query_roles": sorted(set(roles.tolist())),
        }
        extraction_audit[scenario] = extract
        scenario_payloads[scenario] = {
            "scores": query_scores,
            "truth": truth,
            "roles": roles,
            "query_ids": [_sample_id(arrays, i) for i in query_idx],
        }

    calibration = calibrate_adaptive_rxlight_tta(
        np.concatenate(calibration_scores, axis=0),
        np.concatenate(calibration_labels, axis=0),
        base_margin_grid=BASE_MARGIN_GRID,
        shift3_margin_grid=SHIFT3_MARGIN_GRID,
        disagreement_grid=DISAGREEMENT_GRID,
        max_accuracy_drop_pp=float(args.max_accuracy_drop_pp),
    )
    thresholds = calibration["selected"]["thresholds"]
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"pred": [], "truth": [], "roles": []}
        for name in ("fixed1", "fixed3", "fixed5", "adaptive1to3to5")
    }
    for scenario, payload in scenario_payloads.items():
        scores = payload["scores"]
        truth = payload["truth"]
        roles = payload["roles"]
        fixed_indices = {
            "fixed1": np.argmax(scores[:, 0], axis=1),
            "fixed3": np.argmax(scores[:, :3].mean(axis=1), axis=1),
            "fixed5": np.argmax(scores.mean(axis=1), axis=1),
        }
        adaptive = apply_adaptive_rxlight_tta(scores, thresholds)
        fixed_indices["adaptive1to3to5"] = adaptive["predictions"]
        for method_name, indices in fixed_indices.items():
            predicted = np.asarray(classes, dtype=object)[indices].astype(str)
            metrics, class_rows = _metric_row(
                predicted, truth, roles, old_labels, new_labels
            )
            if method_name == "fixed1":
                resources = {"mean_backbone_forwards": 1.0, "p95_backbone_forwards": 1.0,
                             "view1_rate": 1.0, "view3_rate": 0.0, "view5_rate": 0.0}
            elif method_name == "fixed3":
                resources = {"mean_backbone_forwards": 3.0, "p95_backbone_forwards": 3.0,
                             "view1_rate": 0.0, "view3_rate": 1.0, "view5_rate": 0.0}
            elif method_name == "fixed5":
                resources = {"mean_backbone_forwards": 5.0, "p95_backbone_forwards": 5.0,
                             "view1_rate": 0.0, "view3_rate": 0.0, "view5_rate": 1.0}
            else:
                resources = {
                    "mean_backbone_forwards": float(adaptive["mean_backbone_forwards"]),
                    "p95_backbone_forwards": float(adaptive["p95_backbone_forwards"]),
                    **adaptive["trigger_rates"],
                }
            rows.append({"scenario": scenario, "method": method_name, **metrics, **resources})
            per_class_rows.extend(
                {"scenario": scenario, "method": method_name, **row}
                for row in class_rows
            )
            aggregate[method_name]["pred"].append(predicted)
            aggregate[method_name]["truth"].append(truth)
            aggregate[method_name]["roles"].append(roles)
            budgets = (
                adaptive["view_budgets"]
                if method_name == "adaptive1to3to5"
                else np.full(len(truth), int(method_name[-1]), dtype=np.int64)
            )
            prediction_rows.extend(
                {
                    "scenario": scenario,
                    "method": method_name,
                    "query_id": payload["query_ids"][index],
                    "truth": str(truth[index]),
                    "prediction": str(predicted[index]),
                    "evaluation_role": str(roles[index]),
                    "view_budget": int(budgets[index]),
                    "correct": int(predicted[index] == truth[index]),
                }
                for index in range(len(truth))
            )
    for method_name, blocks in aggregate.items():
        pred = np.concatenate(blocks["pred"])
        truth = np.concatenate(blocks["truth"])
        roles = np.concatenate(blocks["roles"])
        metrics, class_rows = _metric_row(pred, truth, roles, old_labels, new_labels)
        method_scenario_rows = [row for row in rows if row["method"] == method_name]
        resources = {
            key: float(np.mean([row[key] for row in method_scenario_rows]))
            for key in ("mean_backbone_forwards", "p95_backbone_forwards", "view1_rate", "view3_rate", "view5_rate")
        }
        rows.append({"scenario": "ALL", "method": method_name, **metrics, **resources})
        per_class_rows.extend(
            {"scenario": "ALL", "method": method_name, **row}
            for row in class_rows
        )

    fixed1_parity = [row for row in rows if row["scenario"] == "ALL" and row["method"] == "fixed1"][0]
    manifest = {
        "method": "support_prototype_adaptive_rxlight5_v1",
        "stage": "Stage2-C_recovery_diagnostic",
        "decision_rule": "per_sample_argmax_view_score_mean",
        "view_prototype_rule": "matching_view_support_prototype",
        "calibration": calibration,
        "calibration_scope": "registered_support_leave_one_out_across_three_scenarios_only",
        "query_labels_used_for_calibration": False,
        "query_features_used_for_calibration": False,
        "old_new_role_used_for_decision": False,
        "class_quota_used": False,
        "query_batch_state_required": False,
        "threshold_grid": {
            "base_margin": list(BASE_MARGIN_GRID),
            "shift3_margin": list(SHIFT3_MARGIN_GRID),
            "shift3_disagreement": list(DISAGREEMENT_GRID),
        },
        "classes": classes,
        "protocol_audit": protocol_audit,
        "extraction_audit": extraction_audit,
        "reference_base_view_parity": parity_audit,
        "fixed1_deployed_fp16_delta_metrics": fixed1_parity,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "checkpoint_load_audit": checkpoint_audit,
        "adapter_state": str(args.adapter_state),
        "adapter_state_sha256": _sha256_file(args.adapter_state),
        "adapter_delta_audit": delta_audit,
        "persistent_state": {
            "adapter_tensor_bytes_fp16": 62_400,
            "five_view_prototype_tensor_bytes_fp16": int(5 * 26 * 256 * 2),
            "threshold_bytes_fp32": 12,
            "total_bytes": int(62_400 + 5 * 26 * 256 * 2 + 12),
            "headroom_to_128kib_bytes": int(
                128 * 1024 - (62_400 + 5 * 26 * 256 * 2 + 12)
            ),
            "under_128kib": bool(
                62_400 + 5 * 26 * 256 * 2 + 12 <= 128 * 1024
            ),
            "class_label_strings_excluded_from_tensor_state_accounting": True,
        },
        "results": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_serializable(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for filename, fieldnames, data in (
        ("summary.csv", list(rows[0]), rows),
        ("per_class.csv", list(per_class_rows[0]), per_class_rows),
        ("predictions.csv", list(prediction_rows[0]), prediction_rows),
    ):
        with (args.out_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(_serializable({
        "out_dir": str(args.out_dir),
        "selected_calibration": calibration["selected"],
        "aggregate_results": [row for row in rows if row["scenario"] == "ALL"],
        "adapter_delta_audit": delta_audit,
    }), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
