"""Fit a tiny qKNN-side source-only ridge map and apply it to frozen z_id caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


POLICIES = ("none", "rx_shift3", "rx_cfo3", "rx_light5")
POLICY_VIEW_COUNTS = {"none": 1, "rx_shift3": 3, "rx_cfo3": 3, "rx_light5": 5}
KEY_FIELDS = (
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "dataset_role",
    "sat_scenarios",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(data: Any) -> dict[str, Any]:
    if "manifest_json" not in data.files:
        raise ValueError("feature cache has no manifest_json")
    return json.loads(str(np.asarray(data["manifest_json"]).item()))


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must contain 64 hex characters")
    return normalized


def _validate_teacher(
    path: Path, manifest: dict[str, Any], *, expected_sha256: str
) -> dict[str, Any]:
    expected = _validate_sha256(expected_sha256, field="expected_teacher_sha256")
    actual = _sha256(path).lower()
    adapter = dict(manifest.get("adapter", {}))
    model_adapter = dict(adapter.get("model_adapter", {}))
    load_audit = dict(manifest.get("checkpoint_load_audit", {}))
    checks = {
        "sha256": actual == expected,
        "payload_source": manifest.get("payload_source")
        == "phase1_model_feature_adapter_satonly_features_v29",
        "feature_name": manifest.get("feature_name") == "z_id",
        "tta_policy": manifest.get("satellite_tta_policy") == "rx_light5",
        "tta_view_count": int(manifest.get("satellite_tta_view_count", -1)) == 5,
        "adapter_epochs": int(adapter.get("epochs", -1)) == 60,
        "adapter_mode": model_adapter.get("mode") == "id_norm_late_feature",
        "adapter_parameters": int(model_adapter.get("trainable_parameters", -1)) == 289685,
        "checkpoint_load_strict": manifest.get("checkpoint_load_strict") is True,
        "checkpoint_load_audit": all(
            int(load_audit.get(key, -1)) == 0
            for key in ("missing_keys", "unexpected_keys", "skipped_mismatch")
        ),
        "no_target_labels_for_training": manifest.get("uses_target_labels_for_training")
        is False,
        "no_target_clean": manifest.get("uses_target_clean") is False,
        "no_unknown_query_threshold": manifest.get("uses_unknown_query_for_threshold")
        is False,
        "source_role": dict(manifest.get("source", {})).get("role") == "source",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"teacher cache failed strict adapter60 validation: {failed}")
    return {"teacher_sha256": actual, "checks": checks}


def _validate_frozen_source(
    manifest: dict[str, Any], *, expected_checkpoint_sha256: str,
    expected_policy: str, expected_tta_view_count: int,
) -> None:
    expected = _validate_sha256(
        expected_checkpoint_sha256, field="expected_checkpoint_sha256"
    )
    adapter = dict(manifest.get("adapter", {}))
    load_audit = dict(manifest.get("checkpoint_load_audit", {}))
    checks = {
        "payload_source": manifest.get("payload_source")
        == "qknnv42_frozen_adv3b02_identity_only_features_v1",
        "feature_name": manifest.get("feature_name") == "z_id",
        "checkpoint_sha256": str(manifest.get("source_checkpoint_sha256", "")).lower()
        == expected,
        "tta_policy": manifest.get("satellite_tta_policy") == str(expected_policy),
        "tta_view_count": int(manifest.get("satellite_tta_view_count", -1))
        == int(expected_tta_view_count),
        "skip_adapter_training": adapter.get("skip_adapter_training") is True,
        "adv3b02_gradient_updates": int(adapter.get("adv3b02_gradient_updates", -1)) == 0,
        "identity_only_forward": manifest.get("identity_only_forward") is True,
        "domain_branch_not_executed": manifest.get("domain_branch_executed_for_qknn")
        is False,
        "checkpoint_load_strict": manifest.get("checkpoint_load_strict") is True,
        "checkpoint_load_audit": all(
            int(load_audit.get(key, -1)) == 0
            for key in ("missing_keys", "unexpected_keys", "skipped_mismatch")
        ),
        "no_prior_post_adapter": "qknn_post_feature_adapter" not in manifest
        and manifest.get("post_feature_adapter_applied") is not True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"frozen cache failed strict validation: {failed}")


def _mark_post_adapter_manifest(
    manifest: dict[str, Any], adapter_info: dict[str, Any]
) -> dict[str, Any]:
    result = dict(manifest)
    result.update(
        {
            "parent_payload_source": manifest.get("payload_source", ""),
            "parent_feature_name": manifest.get("feature_name", ""),
            "parent_identity_only_forward": manifest.get("identity_only_forward"),
            "payload_source": "qknnv42_post_feature_adapter_v1",
            "feature_name": "qknn_post_adapter_z_id",
            "identity_only_forward": False,
            "post_feature_adapter_applied": True,
            "qknn_post_feature_adapter": adapter_info,
        }
    )
    return result


def _sample_key(data: Any, index: int) -> tuple[str, ...]:
    return tuple(str(data[field][index]) for field in KEY_FIELDS)


def _physical_key(data: Any, index: int) -> str:
    return "|".join(str(data[field][index]) for field in KEY_FIELDS[:-1])


def _aligned_source_rows(
    frozen: Any, teacher: Any
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frozen_indices = np.flatnonzero(frozen["dataset_role"].astype(str) == "source")
    teacher_indices = np.flatnonzero(teacher["dataset_role"].astype(str) == "source")
    teacher_by_key = {_sample_key(teacher, int(index)): int(index) for index in teacher_indices}
    if len(teacher_by_key) != len(teacher_indices):
        raise ValueError("teacher source alignment keys are not unique")
    pairs: list[tuple[int, int]] = []
    physical_keys: list[str] = []
    for raw_index in frozen_indices:
        index = int(raw_index)
        key = _sample_key(frozen, index)
        if key not in teacher_by_key:
            raise ValueError(f"frozen source row is missing from teacher cache: {key}")
        pairs.append((index, teacher_by_key[key]))
        physical_keys.append(_physical_key(frozen, index))
    if len(pairs) != len(teacher_indices):
        raise ValueError(
            f"source alignment count mismatch: frozen={len(pairs)} teacher={len(teacher_indices)}"
        )
    x = np.asarray(frozen["features"][[item[0] for item in pairs]], dtype=np.float64)
    y = np.asarray(teacher["features"][[item[1] for item in pairs]], dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"source feature shape mismatch: frozen={x.shape} teacher={y.shape}")
    return x, y, physical_keys


def _group_holdout_mask(keys: list[str]) -> np.ndarray:
    values = []
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:4], "big") % 5 == 0)
    mask = np.asarray(values, dtype=bool)
    if not np.any(mask) or np.all(mask):
        raise ValueError("source-only group holdout split is degenerate")
    return mask


def _cosine_mean(predicted: np.ndarray, target: np.ndarray) -> float:
    pred = predicted / np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1.0e-12)
    truth = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1.0e-12)
    return float(np.mean(np.sum(pred * truth, axis=1)))


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, *, ridge: float
) -> dict[str, np.ndarray | float]:
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0), 1.0e-5)
    y_mean = y.mean(axis=0)
    xn = (x - x_mean) / x_scale
    yn = y - y_mean
    gram = (xn.T @ xn) / max(1, len(xn))
    cross = (xn.T @ yn) / max(1, len(xn))
    weight = np.linalg.solve(gram + float(ridge) * np.eye(gram.shape[0]), cross)
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "weight": weight,
        "ridge": float(ridge),
    }


def _apply(rows: np.ndarray, state: dict[str, np.ndarray | float]) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    return (
        ((x - state["x_mean"]) / state["x_scale"]) @ state["weight"]
        + state["y_mean"]
    ).astype(np.float32)


def _fit_select(
    x: np.ndarray, y: np.ndarray, physical_keys: list[str], ridge_grid: list[float]
) -> tuple[dict[str, np.ndarray | float], list[dict[str, float]]]:
    holdout = _group_holdout_mask(physical_keys)
    rows: list[dict[str, float]] = []
    best_ridge = None
    best_cosine = -np.inf
    for ridge in ridge_grid:
        state = _fit_ridge(x[~holdout], y[~holdout], ridge=float(ridge))
        predicted = _apply(x[holdout], state)
        cosine = _cosine_mean(predicted, y[holdout])
        mse = float(np.mean((predicted - y[holdout]) ** 2))
        rows.append({"ridge": float(ridge), "holdout_cosine": cosine, "holdout_mse": mse})
        if cosine > best_cosine:
            best_cosine = cosine
            best_ridge = float(ridge)
    if best_ridge is None:
        raise RuntimeError("ridge selection produced no candidate")
    return _fit_ridge(x, y, ridge=best_ridge), rows


def _cache_path(root: Path, receiver: str, subdir_base: str, policy: str, name: str) -> Path:
    return root / f"FULL_RX_{receiver}" / f"{subdir_base}_{policy}" / name


def fit_apply(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    teacher = np.load(args.teacher_cache, allow_pickle=False)
    teacher_manifest = _manifest(teacher)
    teacher_evidence = _validate_teacher(
        args.teacher_cache,
        teacher_manifest,
        expected_sha256=str(args.expected_teacher_sha256),
    )
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must not contain duplicates")
    if len(set(args.receivers)) != len(args.receivers):
        raise ValueError("receivers must not contain duplicates")
    if not args.ridge_grid or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in args.ridge_grid
    ):
        raise ValueError("ridge_grid values must be finite and positive")
    if len(set(float(value) for value in args.ridge_grid)) != len(args.ridge_grid):
        raise ValueError("ridge_grid must not contain duplicates")
    policy_summaries: dict[str, Any] = {}
    for policy in args.policies:
        source_path = _cache_path(
            args.frozen_source_root,
            str(args.source_receiver),
            str(args.frozen_subdir_base),
            str(policy),
            str(args.feature_name),
        )
        frozen_source = np.load(source_path, allow_pickle=False)
        _validate_frozen_source(
            _manifest(frozen_source),
            expected_checkpoint_sha256=str(args.expected_checkpoint_sha256),
            expected_policy=str(policy),
            expected_tta_view_count=POLICY_VIEW_COUNTS[str(policy)],
        )
        x, y, physical_keys = _aligned_source_rows(frozen_source, teacher)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("source/teacher features must be finite")
        state, selection = _fit_select(x, y, physical_keys, list(args.ridge_grid))
        adapter_dir = args.out_root / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = adapter_dir / f"source_teacher_ridge_{policy}.npz"
        np.savez(
            adapter_path,
            x_mean=np.asarray(state["x_mean"], dtype=np.float32),
            x_scale=np.asarray(state["x_scale"], dtype=np.float32),
            y_mean=np.asarray(state["y_mean"], dtype=np.float32),
            weight=np.asarray(state["weight"], dtype=np.float32),
            ridge=np.asarray(float(state["ridge"]), dtype=np.float64),
        )
        outputs: list[str] = []
        for receiver in args.receivers:
            input_path = _cache_path(
                args.frozen_target_root,
                str(receiver),
                str(args.frozen_subdir_base),
                str(policy),
                str(args.feature_name),
            )
            with np.load(input_path, allow_pickle=False) as source:
                payload = {key: np.asarray(source[key]) for key in source.files}
                manifest = _manifest(source)
            _validate_frozen_source(
                manifest,
                expected_checkpoint_sha256=str(args.expected_checkpoint_sha256),
                expected_policy=str(policy),
                expected_tta_view_count=POLICY_VIEW_COUNTS[str(policy)],
            )
            if set(np.unique(payload["dataset_role"].astype(str)).tolist()) != {
                "target_old",
                "target_new",
            }:
                raise ValueError(f"target cache contains roles unused by Stage2-C qKNN: {input_path}")
            payload["features"] = _apply(payload["features"], state)
            if not np.all(np.isfinite(payload["features"])):
                raise ValueError(f"adapted target features are non-finite: {input_path}")
            adapter_info = {
                "mode": "source_teacher_ridge",
                "policy": str(policy),
                "ridge": float(state["ridge"]),
                "training_role": "source",
                "training_row_count": int(len(x)),
                "uses_target_rows_for_fit": False,
                "uses_target_labels_for_fit": False,
                "uses_target_query_for_fit": False,
                "updates_adv3b02": False,
                "gradient_updates_adv3b02": 0,
                "feature_dim": int(x.shape[1]),
                "parameter_count": int(x.shape[1] * y.shape[1] + 3 * x.shape[1]),
                "estimated_macs_per_sample": int(x.shape[1] * y.shape[1]),
                "parameter_bytes_fp32": int(
                    4 * (x.shape[1] * y.shape[1] + 3 * x.shape[1])
                ),
                "adapter_path": str(adapter_path),
            }
            manifest = _mark_post_adapter_manifest(manifest, adapter_info)
            payload["manifest_json"] = np.asarray(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            output_path = _cache_path(
                args.out_root,
                str(receiver),
                str(args.output_subdir_base),
                str(policy),
                str(args.feature_name),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **payload)
            outputs.append(str(output_path))
        policy_summaries[str(policy)] = {
            "source_cache": str(source_path),
            "teacher_cache": str(args.teacher_cache),
            "aligned_source_rows": int(len(x)),
            "selected_ridge": float(state["ridge"]),
            "selection": selection,
            "adapter_path": str(adapter_path),
            "adapter_sha256": _sha256(adapter_path),
            "parameter_count": int(x.shape[1] * y.shape[1] + 3 * x.shape[1]),
            "estimated_macs_per_sample": int(x.shape[1] * y.shape[1]),
            "outputs": outputs,
        }
    summary = {
        "schema": "qknnv42_source_teacher_ridge_adapter_v1",
        "training_scope": "source_only",
        "adv3b02_gradient_updates": 0,
        "uses_target_rows_for_fit": False,
        "teacher_manifest_payload_source": teacher_manifest.get("payload_source", ""),
        "teacher_evidence": teacher_evidence,
        "policies": policy_summaries,
    }
    (args.out_root / "adapter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--frozen-target-root", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-receiver", default="20-1")
    parser.add_argument("--receivers", nargs="+", default=["20-1", "3-19", "7-14", "7-7", "8-8"])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--frozen-subdir-base", default="ADV3B02_FROZEN_QKNN_FFT96")
    parser.add_argument("--output-subdir-base", default="ADV3B02_FROZEN_QKNN_RIDGE_FFT96")
    parser.add_argument("--feature-name", default="features_frozen_adv3b02_fft96.npz")
    parser.add_argument(
        "--ridge-grid",
        nargs="+",
        type=float,
        default=[1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0],
    )
    return parser.parse_args()


def main() -> int:
    print(
        json.dumps(
            fit_apply(parse_args()), ensure_ascii=False, indent=2, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
