#!/usr/bin/env python
"""Final-only source-paired C/G diagnostics for frozen P1-PAMR checkpoints.

This evaluator never trains, fits, calibrates, thresholds, sweeps, or selects.
It binds each exported ``z_id`` NPZ to its exact final checkpoint, extracts the
existing GeoSat-C classifier head strictly, and reports frozen source-clean and
source-LEO classifier floors plus raw-cosine angular-margin diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


EXPECTED_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_SOURCE_DAYS = ("2021_03_01", "2021_03_08")
EXPECTED_SOURCE_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
METADATA_FIELDS = (
    "dataset_role",
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "channel_views",
    "sat_scenarios",
)
_PAMR_HEAD_KEY = "id_backbone.cls_head.head.weight"
_PAMR_BINDING_FIELDS = (
    "class_order_contract",
    "dataset_tx_class_order",
    "local_tx_class_order",
    "checkpoint_train_tx_class_order",
    "local_to_dataset_class_ids",
    "local_to_head_class_ids",
    "expected_tx_class_ids",
    "dataset_class_count",
    "local_data_class_count",
    "checkpoint_head_class_count",
    "live_head_class_count",
    "class_count",
)

# File-scoped traceability is retained here because this frozen handoff owns
# precisely this evaluator, its launcher and its focused test file.
FROZEN_POSTFREEZE_CONTRACT = {
    "PAMR-POSTFREEZE-01": "final-only source diagnostics; no fit/calibration/selection",
    "PAMR-POSTFREEZE-02": "C/G clean/LEO physical and ordered metadata binding",
    "PAMR-POSTFREEZE-03": "strict final-checkpoint SHA, local4 class order, and head extraction",
    "PAMR-POSTFREEZE-04": "three disjoint LEO scenarios with TX/RX coverage",
    "PAMR-POSTFREEZE-05": "four classifier floors and raw-cosine angular-margin diagnostics",
}


class PAMRPostfreezePairError(RuntimeError):
    """Raised when a frozen P1-PAMR postfreeze pair fails to close."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_items(value: str | Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        items = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        items = tuple(str(item).strip() for item in value if str(item).strip())
    if not items or len(set(items)) != len(items):
        raise PAMRPostfreezePairError(f"{field} must be a non-empty duplicate-free ordered list")
    return items


def _as_str_array(value: Any, n: int, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        array = np.repeat(array.reshape(1), int(n))
    array = array.reshape(-1)
    if int(array.size) != int(n):
        raise PAMRPostfreezePairError(f"{field} row count mismatch: expected={n} observed={array.size}")
    return np.asarray([str(item).strip() for item in array.tolist()], dtype=object)


def _manifest_from_npz(data: Any, path: Path) -> dict[str, Any]:
    if "manifest_json" not in data.files:
        raise PAMRPostfreezePairError(f"{path} is missing manifest_json")
    try:
        raw = np.asarray(data["manifest_json"])
        item = raw.item() if raw.shape == () else raw.reshape(-1)[0]
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        manifest = json.loads(str(item))
    except Exception as exc:  # pragma: no cover - exact decoder error is environment-dependent
        raise PAMRPostfreezePairError(f"{path} has invalid manifest_json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PAMRPostfreezePairError(f"{path} manifest_json must encode an object")
    return manifest


def _load_npz(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PAMRPostfreezePairError(f"missing NPZ: {source}")
    required = ("features", "tx_logits", *METADATA_FIELDS)
    with np.load(source, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise PAMRPostfreezePairError(f"{source} is missing required arrays: {','.join(missing)}")
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        if features.ndim != 2 or int(features.shape[0]) <= 0:
            raise PAMRPostfreezePairError(f"{source} features must be non-empty rank-2")
        if logits.ndim != 2 or int(logits.shape[0]) != int(features.shape[0]):
            raise PAMRPostfreezePairError(f"{source} tx_logits/features row mismatch")
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise PAMRPostfreezePairError(f"{source} contains non-finite features or logits")
        n = int(features.shape[0])
        payload: dict[str, Any] = {
            "path": str(source),
            "features": features,
            "tx_logits": logits,
            "manifest": _manifest_from_npz(data, source),
        }
        for field in METADATA_FIELDS:
            payload[field] = _as_str_array(data[field], n, field=f"{source}:{field}")
    return payload


def _physical_keys(payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            "\x1f".join(
                str(payload[field][index])
                for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
            )
            for index in range(int(payload["features"].shape[0]))
        ],
        dtype=object,
    )


def _require_unique_physical_keys(payload: Mapping[str, Any], *, label: str) -> np.ndarray:
    keys = _physical_keys(payload)
    if len(set(keys.tolist())) != int(keys.size):
        raise PAMRPostfreezePairError(f"{label} contains duplicate physical keys")
    return keys


def _checkpoint_sha256_from_manifest(payload: Mapping[str, Any], *, label: str) -> str:
    manifest = payload["manifest"]
    digest = str(manifest.get("source_checkpoint_sha256", "")).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PAMRPostfreezePairError(f"{label} lacks a valid source checkpoint SHA256")
    if manifest.get("checkpoint_load_strict") is not True:
        raise PAMRPostfreezePairError(f"{label} checkpoint export was not strict-loaded")
    audit = manifest.get("checkpoint_load_audit")
    if not isinstance(audit, Mapping) or audit.get("checkpoint_load_strict") is not True:
        raise PAMRPostfreezePairError(f"{label} lacks a strict checkpoint-load audit")
    for field in ("missing_keys", "unexpected_keys", "skipped_mismatch"):
        try:
            manifest_count = int(manifest.get(field, -1))
            audit_count = int(audit.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise PAMRPostfreezePairError(f"{label} strict checkpoint-load audit has invalid {field}") from exc
        if manifest_count != 0 or audit_count != 0:
            raise PAMRPostfreezePairError(f"{label} strict checkpoint-load audit has nonzero {field}")
    return digest


def _validate_logit_contract(payload: Mapping[str, Any], source_tx_ids: Sequence[str], *, label: str) -> None:
    manifest = payload["manifest"]
    expected = tuple(source_tx_ids)
    class_order = tuple(str(item).strip() for item in manifest.get("class_id_to_tx", []))
    if class_order != expected:
        raise PAMRPostfreezePairError(
            f"{label} class label/order mismatch: expected={list(expected)} observed={list(class_order)}"
        )
    if list(manifest.get("logit_class_order", [])) != list(range(len(expected))):
        raise PAMRPostfreezePairError(f"{label} logit class order is not the frozen contiguous source order")
    if int(payload["tx_logits"].shape[1]) != len(expected):
        raise PAMRPostfreezePairError(
            f"{label} tx_logits width mismatch: expected={len(expected)} observed={payload['tx_logits'].shape[1]}"
        )
    feature_name = str(manifest.get("feature_name", manifest.get("feature_key", "")))
    if feature_name != "z_id":
        raise PAMRPostfreezePairError(f"{label} must export z_id, got {feature_name!r}")
    _checkpoint_sha256_from_manifest(payload, label=label)


def _source_mask(payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(payload["dataset_role"] == "source", dtype=bool)


def _role_count(payload: Mapping[str, Any], role: str) -> int:
    return int(np.sum(payload["dataset_role"] == str(role)))


def _validate_clean_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_source_count: int,
    expected_target_old_count: int,
    expected_proxy_count: int,
    expected_days: Sequence[str],
    expected_rxs: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    _validate_logit_contract(payload, source_tx_ids, label=label)
    manifest = payload["manifest"]
    if manifest.get("source_only_export") is not False:
        raise PAMRPostfreezePairError(f"{label} clean manifest must not be source-only")
    if str(manifest.get("satellite_tta_policy", "")) != "none":
        raise PAMRPostfreezePairError(f"{label} clean manifest must use satellite_tta_policy=none")
    expected_roles = {"source", "target_old", "proxy_unknown"}
    roles = set(payload["dataset_role"].tolist())
    if roles != expected_roles:
        raise PAMRPostfreezePairError(f"{label} clean roles mismatch: expected={sorted(expected_roles)} observed={sorted(roles)}")
    for role, expected in {
        "source": int(expected_source_count),
        "target_old": int(expected_target_old_count),
        "proxy_unknown": int(expected_proxy_count),
    }.items():
        observed = _role_count(payload, role)
        if observed != expected:
            raise PAMRPostfreezePairError(
                f"{label} clean role count mismatch for {role}: expected={expected} observed={observed}"
            )
    if set(payload["channel_views"].tolist()) != {"clean"}:
        raise PAMRPostfreezePairError(f"{label} clean payload has a non-clean channel view")
    if any(str(item) for item in payload["sat_scenarios"].tolist()):
        raise PAMRPostfreezePairError(f"{label} clean payload must not assign satellite scenarios")
    source = _source_mask(payload)
    if set(payload["tx_ids"][source].tolist()) != set(source_tx_ids):
        raise PAMRPostfreezePairError(f"{label} clean source TX set mismatch")
    if set(payload["day_ids"][source].tolist()) != set(expected_days):
        raise PAMRPostfreezePairError(f"{label} clean source day set mismatch")
    if set(payload["rx_ids"][source].tolist()) != set(expected_rxs):
        raise PAMRPostfreezePairError(f"{label} clean source RX set mismatch")
    return _require_unique_physical_keys(payload, label=label)[source]


def _validate_leo_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_source_count: int,
    expected_scenarios: Sequence[str],
    expected_days: Sequence[str],
    expected_rxs: Sequence[str],
    source_sat_seed: int,
    *,
    label: str,
) -> np.ndarray:
    _validate_logit_contract(payload, source_tx_ids, label=label)
    if set(payload["dataset_role"].tolist()) != {"source"}:
        raise PAMRPostfreezePairError(f"{label} LEO payload must contain source rows only")
    if int(payload["features"].shape[0]) != int(expected_source_count):
        raise PAMRPostfreezePairError(
            f"{label} LEO source row count mismatch: expected={expected_source_count} observed={payload['features'].shape[0]}"
        )
    if set(payload["channel_views"].tolist()) != {"satellite"}:
        raise PAMRPostfreezePairError(f"{label} LEO payload must use exactly channel_view=satellite")
    scenarios = payload["sat_scenarios"]
    if set(scenarios.tolist()) != set(expected_scenarios):
        raise PAMRPostfreezePairError(f"{label} LEO scenario set mismatch")
    if set(payload["tx_ids"].tolist()) != set(source_tx_ids):
        raise PAMRPostfreezePairError(f"{label} LEO source TX set mismatch")
    if set(payload["day_ids"].tolist()) != set(expected_days):
        raise PAMRPostfreezePairError(f"{label} LEO source day set mismatch")
    if set(payload["rx_ids"].tolist()) != set(expected_rxs):
        raise PAMRPostfreezePairError(f"{label} LEO source RX set mismatch")
    manifest = payload["manifest"]
    if manifest.get("source_only_export") is not True:
        raise PAMRPostfreezePairError(f"{label} LEO manifest is not source-only")
    if str(manifest.get("star_ground_channel_impl", "")) != "simplified_leo_residual":
        raise PAMRPostfreezePairError(f"{label} LEO manifest does not use simplified_leo_residual")
    if str(manifest.get("satellite_tta_policy", "")) != "none":
        raise PAMRPostfreezePairError(f"{label} LEO manifest must use satellite_tta_policy=none")
    profile = manifest.get("channel_profile", {}).get("source", {})
    if not isinstance(profile, Mapping) or str(profile.get("view", "")) != "satellite":
        raise PAMRPostfreezePairError(f"{label} LEO source channel profile is not satellite")
    if tuple(str(item) for item in profile.get("scenarios", [])) != tuple(expected_scenarios):
        raise PAMRPostfreezePairError(f"{label} LEO source channel scenarios are not frozen")
    if int(profile.get("sat_seed", -1)) != int(source_sat_seed):
        raise PAMRPostfreezePairError(f"{label} LEO source satellite seed mismatch")
    keys = _require_unique_physical_keys(payload, label=label)
    physical_by_scenario: dict[str, set[str]] = {}
    for scenario in expected_scenarios:
        mask = np.asarray(scenarios == scenario, dtype=bool)
        if set(payload["tx_ids"][mask].tolist()) != set(source_tx_ids):
            raise PAMRPostfreezePairError(f"{label} scenario {scenario} lacks full source TX coverage")
        if set(payload["rx_ids"][mask].tolist()) != set(expected_rxs):
            raise PAMRPostfreezePairError(f"{label} scenario {scenario} lacks full source RX coverage")
        physical_by_scenario[scenario] = set(keys[mask].tolist())
    for index, left in enumerate(expected_scenarios):
        for right in expected_scenarios[index + 1 :]:
            if physical_by_scenario[left] & physical_by_scenario[right]:
                raise PAMRPostfreezePairError(f"{label} scenarios {left}/{right} reuse physical keys")
    return keys


def _assert_pair_metadata(c_payload: Mapping[str, Any], g_payload: Mapping[str, Any], *, label: str) -> None:
    if int(c_payload["features"].shape[0]) != int(g_payload["features"].shape[0]):
        raise PAMRPostfreezePairError(f"C/G {label} metadata row count mismatch")
    if int(c_payload["features"].shape[1]) != int(g_payload["features"].shape[1]):
        raise PAMRPostfreezePairError(f"C/G {label} z_id dimension mismatch")
    for field in METADATA_FIELDS:
        if not np.array_equal(c_payload[field], g_payload[field]):
            raise PAMRPostfreezePairError(f"C/G {label} metadata/scenario mismatch in {field}")


def _tx_order_from_checkpoint(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return _parse_items(value, field=label)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _parse_items(tuple(str(item) for item in value), field=label)
    raise PAMRPostfreezePairError(f"{label} must be an ordered TX list")


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PAMRPostfreezePairError(f"{label} must be a mapping")
    return value


def _load_torch_checkpoint(path: str | Path, *, label: str) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PAMRPostfreezePairError(f"{label} final checkpoint is missing")
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch builds
        checkpoint = torch.load(source, map_location="cpu")
    return _as_mapping(checkpoint, label=f"{label} final checkpoint")


def _expected_binding_sha256(receipt: Mapping[str, Any]) -> str:
    binding = {field: receipt[field] for field in _PAMR_BINDING_FIELDS}
    return hashlib.sha256(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _strict_extract_pamr_head(
    checkpoint_path: str | Path,
    *,
    arm: str,
    source_tx_ids: Sequence[str],
    feature_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract the one frozen local4 head key and validate its full binding."""

    label = f"{arm} final checkpoint"
    checkpoint = _load_torch_checkpoint(checkpoint_path, label=label)
    if str(checkpoint.get("checkpoint_role", "")) != "training_final_only":
        raise PAMRPostfreezePairError(f"{label} is not a training_final_only checkpoint")
    if str(checkpoint.get("checkpoint_selection", "")) != "final_only":
        raise PAMRPostfreezePairError(f"{label} does not retain checkpoint_selection=final_only")
    args = _as_mapping(checkpoint.get("args"), label=f"{label} args")
    if args.get("phase1_pamr_frozen_mode") is not True:
        raise PAMRPostfreezePairError(f"{label} does not retain phase1_pamr_frozen_mode=true")
    if args.get("phase1_pamr_audit_only") is not False:
        raise PAMRPostfreezePairError(f"{label} must not be a technical-audit checkpoint")
    if str(args.get("id_feature_key", "")) != "feat_joint":
        raise PAMRPostfreezePairError(f"{label} z_id is not bound to feat_joint")
    expected_arm_enabled = arm == "G"
    if bool(args.get("phase1_pamr_enabled", False)) is not expected_arm_enabled:
        raise PAMRPostfreezePairError(f"{label} PAMR arm flag does not match C/G identity")
    expected_lambda = 0.05 if expected_arm_enabled else 0.0
    try:
        observed_lambda = float(args.get("lambda_pamr"))
    except (TypeError, ValueError) as exc:
        raise PAMRPostfreezePairError(f"{label} lacks a numeric lambda_pamr") from exc
    if not np.isclose(observed_lambda, expected_lambda, atol=0.0, rtol=0.0):
        raise PAMRPostfreezePairError(f"{label} lambda_pamr does not match frozen C/G contract")
    if int(args.get("num_classes", -1)) != 4 or len(source_tx_ids) != 4:
        raise PAMRPostfreezePairError(f"{label} must bind exactly local4 classifier classes")
    checkpoint_train_tx = _tx_order_from_checkpoint(
        args.get("phase1_source_train_tx_ids"), label=f"{label} phase1_source_train_tx_ids"
    )
    if checkpoint_train_tx != tuple(source_tx_ids):
        raise PAMRPostfreezePairError(f"{label} checkpoint train TX class order mismatch")
    model = _as_mapping(checkpoint.get("model"), label=f"{label} model state")
    if _PAMR_HEAD_KEY not in model:
        raise PAMRPostfreezePairError(f"{label} lacks exact {_PAMR_HEAD_KEY}")
    raw_weight = model[_PAMR_HEAD_KEY]
    if not torch.is_tensor(raw_weight):
        raise PAMRPostfreezePairError(f"{label} {_PAMR_HEAD_KEY} is not a tensor")
    weight = raw_weight.detach().cpu().float().numpy()
    if weight.ndim != 2 or weight.shape != (4, int(feature_dim)):
        raise PAMRPostfreezePairError(
            f"{label} head shape mismatch: expected={(4, int(feature_dim))} observed={tuple(weight.shape)}"
        )
    if not np.isfinite(weight).all() or np.any(np.linalg.norm(weight, axis=1) <= 1.0e-12):
        raise PAMRPostfreezePairError(f"{label} head has non-finite/zero-norm rows")

    split_info = _as_mapping(checkpoint.get("split_info"), label=f"{label} split_info")
    partition = _as_mapping(split_info.get("tx_partition_receipt"), label=f"{label} tx_partition_receipt")
    if partition.get("enabled") is not True or partition.get("held_tx_loaded_by_training") is not False:
        raise PAMRPostfreezePairError(f"{label} does not retain a source-only training partition")
    if _tx_order_from_checkpoint(partition.get("source_known_train_tx"), label=f"{label} source train receipt") != tuple(source_tx_ids):
        raise PAMRPostfreezePairError(f"{label} source train receipt order mismatch")
    if int(partition.get("training_tx_count", -1)) != 4:
        raise PAMRPostfreezePairError(f"{label} source train receipt is not local4")
    expected_reindex = {str(index): tx for index, tx in enumerate(source_tx_ids)}
    if dict(partition.get("training_view_contiguous_reindex", {})) != expected_reindex:
        raise PAMRPostfreezePairError(f"{label} training-view contiguous reindex mismatch")

    receipt = _as_mapping(checkpoint.get("pamr_receipt"), label=f"{label} pamr_receipt")
    missing_binding = [field for field in _PAMR_BINDING_FIELDS if field not in receipt]
    if missing_binding:
        raise PAMRPostfreezePairError(f"{label} PAMR local4 receipt lacks {','.join(missing_binding)}")
    expected_order = list(source_tx_ids)
    if (
        list(receipt["local_tx_class_order"]) != expected_order
        or list(receipt["checkpoint_train_tx_class_order"]) != expected_order
        or list(receipt["local_to_head_class_ids"]) != [0, 1, 2, 3]
        or list(receipt["expected_tx_class_ids"]) != [0, 1, 2, 3]
    ):
        raise PAMRPostfreezePairError(f"{label} PAMR local4 class-order binding mismatch")
    for field in ("local_data_class_count", "checkpoint_head_class_count", "live_head_class_count", "class_count"):
        if int(receipt[field]) != 4:
            raise PAMRPostfreezePairError(f"{label} PAMR {field} is not 4")
    if str(receipt.get("class_order_contract", "")) != "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER":
        raise PAMRPostfreezePairError(f"{label} PAMR class-order contract mismatch")
    dataset_order = [str(item) for item in receipt["dataset_tx_class_order"]]
    local_to_dataset = [int(item) for item in receipt["local_to_dataset_class_ids"]]
    if (
        len(dataset_order) != int(receipt["dataset_class_count"])
        or len(local_to_dataset) != 4
        or any(index < 0 or index >= len(dataset_order) for index in local_to_dataset)
        or [dataset_order[index] for index in local_to_dataset] != expected_order
    ):
        raise PAMRPostfreezePairError(f"{label} PAMR local-to-dataset class binding mismatch")
    observed_binding_sha = str(receipt.get("class_order_binding_sha256", "")).lower()
    if observed_binding_sha != _expected_binding_sha256(receipt):
        raise PAMRPostfreezePairError(f"{label} PAMR class-order binding SHA256 mismatch")
    return weight, {
        "final_checkpoint_sha256": _sha256_file(checkpoint_path),
        "head_state_key": _PAMR_HEAD_KEY,
        "head_rows": 4,
        "head_feature_dim": int(feature_dim),
        "strict_head_extract": True,
        "checkpoint_train_tx_class_order": expected_order,
        "class_order_binding_sha256": observed_binding_sha,
    }


def _rate(values: np.ndarray) -> float:
    if int(values.size) <= 0:
        raise PAMRPostfreezePairError("cannot compute a rate on zero rows")
    return float(np.mean(values.astype(np.float64)))


def _min_group_accuracy(groups: np.ndarray, correct: np.ndarray) -> float:
    rates = [_rate(correct[groups == group]) for group in sorted(set(groups.tolist()))]
    if not rates:
        raise PAMRPostfreezePairError("cannot compute a group floor on zero groups")
    return float(min(rates))


def _predicted_tx(payload: Mapping[str, Any], source_tx_ids: Sequence[str]) -> np.ndarray:
    indices = np.asarray(payload["tx_logits"].argmax(axis=1), dtype=np.int64)
    return np.asarray([source_tx_ids[int(index)] for index in indices.tolist()], dtype=object)


def _classification_summary(payload: Mapping[str, Any], mask: np.ndarray, source_tx_ids: Sequence[str]) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) <= 0:
        raise PAMRPostfreezePairError("classification summary selected zero rows")
    predicted = _predicted_tx(payload, source_tx_ids)
    correct = np.asarray(predicted == payload["tx_ids"], dtype=bool)
    return {
        "count": int(selected.sum()),
        "overall_accuracy": _rate(correct[selected]),
        "min_class_accuracy": _min_group_accuracy(payload["tx_ids"][selected], correct[selected]),
        "min_rx_accuracy": _min_group_accuracy(payload["rx_ids"][selected], correct[selected]),
        "min_day_accuracy": _min_group_accuracy(payload["day_ids"][selected], correct[selected]),
    }


def _delta_pp(c_metrics: Mapping[str, Any], g_metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        metric: 100.0 * (float(g_metrics[metric]) - float(c_metrics[metric]))
        for metric in ("overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy")
    }


def _normalize_rows(values: np.ndarray, *, label: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise PAMRPostfreezePairError(f"{label} contains zero/non-finite z_id rows")
    return rows / norms


def _raw_cosine_margin_summary(
    payload: Mapping[str, Any],
    mask: np.ndarray,
    *,
    head_weight: np.ndarray,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) <= 0:
        raise PAMRPostfreezePairError("raw-cosine angular margin selected zero rows")
    z = _normalize_rows(payload["features"][selected], label="selected source")
    weight = _normalize_rows(head_weight, label="classifier head")
    cosine = np.clip(z @ weight.T, -1.0, 1.0)
    tx_to_index = {tx: index for index, tx in enumerate(source_tx_ids)}
    labels = np.asarray([tx_to_index.get(str(tx), -1) for tx in payload["tx_ids"][selected]], dtype=np.int64)
    if np.any(labels < 0):
        raise PAMRPostfreezePairError("raw-cosine labels are outside the local4 class order")
    predicted = cosine.argmax(axis=1)
    correct = np.asarray(predicted == labels, dtype=bool)
    true_cosine = cosine[np.arange(cosine.shape[0]), labels]
    alternatives = cosine.copy()
    alternatives[np.arange(cosine.shape[0]), labels] = -np.inf
    hard_other = alternatives.max(axis=1)
    # The quantity is positive exactly when the correct class has a smaller
    # angular distance than its nearest competing head row.  It is reported
    # only on raw-cosine-correct rows, never used as a threshold or gate.
    angular_margin_deg = np.degrees(np.arccos(np.clip(hard_other, -1.0, 1.0))) - np.degrees(
        np.arccos(np.clip(true_cosine, -1.0, 1.0))
    )
    correct_margins = angular_margin_deg[correct]
    return {
        "count": int(selected.sum()),
        "raw_cosine_correct_count": int(correct.sum()),
        "raw_cosine_correct_rate": _rate(correct),
        "correct_vs_hardest_other_angular_margin_mean_deg": (
            None if int(correct_margins.size) == 0 else float(np.mean(correct_margins))
        ),
    }


def _angular_delta(c_summary: Mapping[str, Any], g_summary: Mapping[str, Any]) -> dict[str, float | None]:
    c_margin = c_summary["correct_vs_hardest_other_angular_margin_mean_deg"]
    g_margin = g_summary["correct_vs_hardest_other_angular_margin_mean_deg"]
    return {
        "raw_cosine_correct_rate_pp": 100.0
        * (float(g_summary["raw_cosine_correct_rate"]) - float(c_summary["raw_cosine_correct_rate"])),
        "correct_vs_hardest_other_angular_margin_mean_deg": (
            None if c_margin is None or g_margin is None else float(g_margin) - float(c_margin)
        ),
    }


def _paired_cosine_distance_summary(
    clean: Mapping[str, Any], leo: Mapping[str, Any], leo_mask: np.ndarray
) -> dict[str, float]:
    clean_mask = _source_mask(clean)
    clean_keys = _physical_keys(clean)[clean_mask]
    clean_z = _normalize_rows(clean["features"][clean_mask], label="clean source")
    bank = {key: clean_z[index] for index, key in enumerate(clean_keys.tolist())}
    if len(bank) != int(clean_keys.size):
        raise PAMRPostfreezePairError("clean source paired-cosine bank has duplicate physical keys")
    selected = np.asarray(leo_mask, dtype=bool)
    leo_z = _normalize_rows(leo["features"][selected], label="LEO source")
    leo_keys = _physical_keys(leo)[selected]
    distances: list[float] = []
    for z, key in zip(leo_z, leo_keys.tolist()):
        if key not in bank:
            raise PAMRPostfreezePairError("LEO paired-cosine row lacks clean physical binding")
        distances.append(float(1.0 - np.clip(np.dot(z, bank[key]), -1.0, 1.0)))
    if not distances:
        raise PAMRPostfreezePairError("paired-cosine diagnostic selected zero rows")
    return {"paired_clean_leo_cosine_distance_mean": float(np.mean(distances))}


def _paired_cosine_delta(c_summary: Mapping[str, Any], g_summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "paired_clean_leo_cosine_distance_mean": float(g_summary["paired_clean_leo_cosine_distance_mean"])
        - float(c_summary["paired_clean_leo_cosine_distance_mean"])
    }


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise PAMRPostfreezePairError(f"refusing to overwrite final-only pair output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise PAMRPostfreezePairError(f"refusing to overwrite temporary pair output: {temporary}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_tx_ids = _parse_items(args.source_tx_ids, field="source_tx_ids")
    if len(source_tx_ids) != 4:
        raise PAMRPostfreezePairError("P1-PAMR postfreeze is frozen to local4 source TX classes")
    expected_scenarios = _parse_items(args.expected_scenarios, field="expected_scenarios")
    if expected_scenarios != EXPECTED_SCENARIOS:
        raise PAMRPostfreezePairError("expected_scenarios are frozen to the three leo_*_weak scenarios")
    expected_days = _parse_items(args.expected_source_days, field="expected_source_days")
    expected_rxs = _parse_items(args.expected_source_rxs, field="expected_source_rxs")
    if expected_days != EXPECTED_SOURCE_DAYS:
        raise PAMRPostfreezePairError("expected_source_days do not match the frozen WRC LEO v2 slice")
    if expected_rxs != EXPECTED_SOURCE_RXS:
        raise PAMRPostfreezePairError("expected_source_rxs do not match the frozen WRC LEO v2 slice")
    expected_source_count = int(args.expected_source_count)
    expected_target_old_count = int(args.expected_target_old_count)
    expected_proxy_count = int(args.expected_proxy_count)
    if min(expected_source_count, expected_target_old_count, expected_proxy_count) <= 0:
        raise PAMRPostfreezePairError("expected role counts must be positive")

    c_clean = _load_npz(args.c_clean_npz)
    g_clean = _load_npz(args.g_clean_npz)
    c_leo = _load_npz(args.c_leo_npz)
    g_leo = _load_npz(args.g_leo_npz)
    _assert_pair_metadata(c_clean, g_clean, label="clean")
    _assert_pair_metadata(c_leo, g_leo, label="LEO")
    if int(c_clean["features"].shape[1]) != int(c_leo["features"].shape[1]):
        raise PAMRPostfreezePairError("C clean/LEO z_id dimension mismatch")
    if int(g_clean["features"].shape[1]) != int(g_leo["features"].shape[1]):
        raise PAMRPostfreezePairError("G clean/LEO z_id dimension mismatch")

    c_clean_keys = _validate_clean_payload(
        c_clean, source_tx_ids, expected_source_count, expected_target_old_count, expected_proxy_count,
        expected_days, expected_rxs, label="C clean"
    )
    g_clean_keys = _validate_clean_payload(
        g_clean, source_tx_ids, expected_source_count, expected_target_old_count, expected_proxy_count,
        expected_days, expected_rxs, label="G clean"
    )
    c_leo_keys = _validate_leo_payload(
        c_leo, source_tx_ids, expected_source_count, expected_scenarios, expected_days, expected_rxs,
        int(args.source_sat_seed), label="C LEO"
    )
    g_leo_keys = _validate_leo_payload(
        g_leo, source_tx_ids, expected_source_count, expected_scenarios, expected_days, expected_rxs,
        int(args.source_sat_seed), label="G LEO"
    )
    if set(c_clean_keys.tolist()) != set(c_leo_keys.tolist()):
        raise PAMRPostfreezePairError("C clean/LEO source physical key sets differ")
    if set(g_clean_keys.tolist()) != set(g_leo_keys.tolist()):
        raise PAMRPostfreezePairError("G clean/LEO source physical key sets differ")
    if set(c_clean_keys.tolist()) != set(g_clean_keys.tolist()):
        raise PAMRPostfreezePairError("C/G clean source physical key sets differ")
    if set(c_leo_keys.tolist()) != set(g_leo_keys.tolist()):
        raise PAMRPostfreezePairError("C/G LEO source physical key sets differ")

    c_manifest_sha = _checkpoint_sha256_from_manifest(c_clean, label="C clean")
    g_manifest_sha = _checkpoint_sha256_from_manifest(g_clean, label="G clean")
    if c_manifest_sha != _checkpoint_sha256_from_manifest(c_leo, label="C LEO"):
        raise PAMRPostfreezePairError("C clean/LEO source checkpoint SHA256 differs")
    if g_manifest_sha != _checkpoint_sha256_from_manifest(g_leo, label="G LEO"):
        raise PAMRPostfreezePairError("G clean/LEO source checkpoint SHA256 differs")
    c_head, c_head_binding = _strict_extract_pamr_head(
        args.c_final_checkpoint, arm="C", source_tx_ids=source_tx_ids, feature_dim=int(c_clean["features"].shape[1])
    )
    g_head, g_head_binding = _strict_extract_pamr_head(
        args.g_final_checkpoint, arm="G", source_tx_ids=source_tx_ids, feature_dim=int(g_clean["features"].shape[1])
    )
    if c_head_binding["final_checkpoint_sha256"] != c_manifest_sha:
        raise PAMRPostfreezePairError("C final checkpoint SHA256 does not bind the C NPZ exports")
    if g_head_binding["final_checkpoint_sha256"] != g_manifest_sha:
        raise PAMRPostfreezePairError("G final checkpoint SHA256 does not bind the G NPZ exports")

    c_clean_source = _source_mask(c_clean)
    g_clean_source = _source_mask(g_clean)
    c_clean_summary = _classification_summary(c_clean, c_clean_source, source_tx_ids)
    g_clean_summary = _classification_summary(g_clean, g_clean_source, source_tx_ids)
    c_clean_margin = _raw_cosine_margin_summary(c_clean, c_clean_source, head_weight=c_head, source_tx_ids=source_tx_ids)
    g_clean_margin = _raw_cosine_margin_summary(g_clean, g_clean_source, head_weight=g_head, source_tx_ids=source_tx_ids)
    scenario_metrics: dict[str, Any] = {}
    for scenario in expected_scenarios:
        c_mask = np.asarray(c_leo["sat_scenarios"] == scenario, dtype=bool)
        g_mask = np.asarray(g_leo["sat_scenarios"] == scenario, dtype=bool)
        c_summary = _classification_summary(c_leo, c_mask, source_tx_ids)
        g_summary = _classification_summary(g_leo, g_mask, source_tx_ids)
        c_margin = _raw_cosine_margin_summary(c_leo, c_mask, head_weight=c_head, source_tx_ids=source_tx_ids)
        g_margin = _raw_cosine_margin_summary(g_leo, g_mask, head_weight=g_head, source_tx_ids=source_tx_ids)
        c_paired = _paired_cosine_distance_summary(c_clean, c_leo, c_mask)
        g_paired = _paired_cosine_distance_summary(g_clean, g_leo, g_mask)
        scenario_metrics[scenario] = {
            "C": c_summary,
            "G": g_summary,
            "G_minus_C_pp": _delta_pp(c_summary, g_summary),
            "raw_cosine_angular_margin_diagnostic": {
                "C": c_margin,
                "G": g_margin,
                "G_minus_C": _angular_delta(c_margin, g_margin),
            },
            "paired_cosine_diagnostic_source_clean_only": {
                "C": c_paired,
                "G": g_paired,
                "G_minus_C": _paired_cosine_delta(c_paired, g_paired),
            },
        }
    metrics = {
        "schema": "cvs.phase1.pamr_postfreeze_pair.v1",
        "candidate_pair": str(args.candidate_pair),
        "evidence_boundary": "PHASE1_SOURCE_ONLY_FINAL_ONLY_DIAGNOSTIC",
        "frozen_contract": dict(FROZEN_POSTFREEZE_CONTRACT),
        "policy": {
            "fit_performed": False,
            "calibration_performed": False,
            "threshold_used": False,
            "model_selection_performed": False,
            "proxy_rows_used_for_pair_metrics": 0,
            "target_old_rows_used_for_pair_metrics": 0,
            "proxy_rows_used_for_head_or_margin": 0,
            "target_old_rows_used_for_head_or_margin": 0,
            "paired_cosine_is_diagnostic_only": True,
            "raw_cosine_head_source": "strict_final_checkpoint:id_backbone.cls_head.head.weight",
        },
        "source_tx_ids": list(source_tx_ids),
        "expected_source_days": list(expected_days),
        "expected_source_rxs": list(expected_rxs),
        "expected_role_counts": {
            "source": expected_source_count,
            "target_old": expected_target_old_count,
            "proxy_unknown": expected_proxy_count,
        },
        "expected_scenarios": list(expected_scenarios),
        "source_sat_seed": int(args.source_sat_seed),
        "bindings": {
            "c_clean_npz_sha256": _sha256_file(args.c_clean_npz),
            "g_clean_npz_sha256": _sha256_file(args.g_clean_npz),
            "c_leo_npz_sha256": _sha256_file(args.c_leo_npz),
            "g_leo_npz_sha256": _sha256_file(args.g_leo_npz),
            "c_source_checkpoint_sha256": c_manifest_sha,
            "g_source_checkpoint_sha256": g_manifest_sha,
            "C": c_head_binding,
            "G": g_head_binding,
        },
        "clean_source": {
            "C": c_clean_summary,
            "G": g_clean_summary,
            "G_minus_C_pp": _delta_pp(c_clean_summary, g_clean_summary),
            "raw_cosine_angular_margin_diagnostic": {
                "C": c_clean_margin,
                "G": g_clean_margin,
                "G_minus_C": _angular_delta(c_clean_margin, g_clean_margin),
            },
        },
        "leo_scenarios": scenario_metrics,
    }
    _atomic_write_json(args.output_metrics_json, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--c-leo-npz", required=True)
    parser.add_argument("--g-leo-npz", required=True)
    parser.add_argument("--c-final-checkpoint", required=True)
    parser.add_argument("--g-final-checkpoint", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--candidate-pair", required=True)
    parser.add_argument("--expected-scenarios", default=",".join(EXPECTED_SCENARIOS))
    parser.add_argument("--expected-source-days", default=",".join(EXPECTED_SOURCE_DAYS))
    parser.add_argument("--expected-source-rxs", default=",".join(EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-seed", type=int, default=7281718)
    parser.add_argument("--expected-source-count", type=int, default=1600)
    parser.add_argument("--expected-target-old-count", type=int, default=400)
    parser.add_argument("--expected-proxy-count", type=int, default=400)
    parser.add_argument("--output-metrics-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
