#!/usr/bin/env python3
"""Support-selected ablation on sealed single-observation Phase2 packages.

The ``predict`` subcommand has no truth-sidecar argument. It verifies sealed
support/query packages, chooses one unified variant using support leave-one-
physical-sample-out diagnostics only, and writes immutable predictions for all
pre-registered variants. The ``score`` subcommand is a separate post-prediction
process that joins query truth only after the prediction COMMIT exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_diagnostic_bundle_loader import (  # noqa: E402
    load_verified_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    FEATURE_DIM,
    TEMPERATURE,
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair  # noqa: E402
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


EPS = 1.0e-8
DIAG_SHRINKAGE = 0.25
DIAG_MAX_ABS_LOG_SCALE = 0.35
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
PREDICTION_MEMBERS = (
    "query_tokens",
    "scenarios",
    "predicted_class_handles",
)


class SingleObservationAblationError(ValueError):
    """Raised when the independent ablation fails closed."""


@dataclass(frozen=True)
class Variant:
    name: str
    equalizer: bool
    prototype_rule: str
    trim_fraction: float = 0.0
    prototypes_per_class: int = 1
    support_view_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "equalizer": self.equalizer,
            "prototype_rule": self.prototype_rule,
            "trim_fraction": self.trim_fraction,
            "prototypes_per_class": self.prototypes_per_class,
            "support_view_count": self.support_view_count,
            "query_view_count": 1,
            "views_count_as_additional_physical_samples": False,
            "additional_leo_channel_state_generation": False,
        }


VARIANTS = (
    Variant("base_mean", False, "mean"),
    Variant(
        "diag_mean_current3view",
        True,
        "mean",
        support_view_count=3,
    ),
    Variant("base_trimmed10", False, "trimmed_mean", trim_fraction=0.10),
    Variant("base_trimmed20", False, "trimmed_mean", trim_fraction=0.20),
    Variant("base_median", False, "median"),
    Variant(
        "diag_trimmed10",
        True,
        "trimmed_mean",
        trim_fraction=0.10,
        support_view_count=3,
    ),
    Variant(
        "diag_trimmed20",
        True,
        "trimmed_mean",
        trim_fraction=0.20,
        support_view_count=3,
    ),
    Variant(
        "diag_median",
        True,
        "median",
        support_view_count=3,
    ),
    Variant(
        "base_2proto",
        False,
        "deterministic_two_prototype",
        prototypes_per_class=2,
    ),
    Variant(
        "diag_2proto",
        True,
        "deterministic_two_prototype",
        prototypes_per_class=2,
        support_view_count=3,
    ),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _write_json_new(path: Path, payload: Any, *, readonly: bool = True) -> str:
    raw = _canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if readonly:
        _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_text_new(path: Path, text: str, *, readonly: bool = True) -> str:
    raw = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if readonly:
        _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_prediction_new(
    path: Path,
    *,
    query_tokens: np.ndarray,
    scenarios: np.ndarray,
    predicted_class_handles: np.ndarray,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(
            handle,
            query_tokens=np.asarray(query_tokens).astype(str),
            scenarios=np.asarray(scenarios).astype(str),
            predicted_class_handles=np.asarray(predicted_class_handles).astype(str),
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return _sha256_file(path)


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True),
        EPS,
    )


def _diag_log_scale(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = sorted(set(labels.astype(str).tolist()))
    class_energies = np.stack(
        [
            np.mean(
                np.square(features[labels == label], dtype=np.float32),
                axis=0,
            )
            for label in classes
        ]
    )
    energy = np.mean(class_energies, axis=0)
    log_energy = np.log(np.maximum(energy, EPS))
    raw = -0.5 * (log_energy - np.mean(log_energy))
    return np.clip(
        DIAG_SHRINKAGE * raw,
        -DIAG_MAX_ABS_LOG_SCALE,
        DIAG_MAX_ABS_LOG_SCALE,
    ).astype(np.float32)


def _transform_with_labels(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    equalizer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = _normalize(train_x)
    test = _normalize(test_x)
    if not equalizer:
        return train, test, np.zeros(train.shape[1], dtype=np.float32)
    log_scale = _diag_log_scale(train, train_y.astype(str))
    scale = np.exp(log_scale)[None, :]
    return _normalize(train * scale), _normalize(test * scale), log_scale


def _robust_center(rows: np.ndarray, variant: Variant) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if variant.prototype_rule == "mean":
        center = np.mean(values, axis=0)
    elif variant.prototype_rule == "median":
        center = np.median(values, axis=0)
    elif variant.prototype_rule == "trimmed_mean":
        count = int(math.floor(len(values) * float(variant.trim_fraction)))
        ordered = np.sort(values, axis=0)
        selected = (
            ordered[count : len(values) - count]
            if count > 0 and 2 * count < len(values)
            else ordered
        )
        center = np.mean(selected, axis=0)
    else:
        raise SingleObservationAblationError(
            f"unsupported robust center: {variant.prototype_rule}"
        )
    return _normalize(np.asarray(center, dtype=np.float32)[None, :])[0]


def _two_prototypes(rows: np.ndarray) -> np.ndarray:
    values = _normalize(np.asarray(rows, dtype=np.float32))
    if len(values) < 2:
        return values[:1]
    mean = _normalize(np.mean(values, axis=0, keepdims=True))[0]
    first = int(np.argmin(values @ mean))
    second = int(np.argmin(values @ values[first]))
    if second == first:
        second = (first + 1) % len(values)
    centers = np.stack([values[first], values[second]])
    for _ in range(8):
        assignment = np.argmax(values @ centers.T, axis=1)
        next_centers: list[np.ndarray] = []
        for index in range(2):
            selected = values[assignment == index]
            next_centers.append(
                centers[index]
                if len(selected) == 0
                else _normalize(np.mean(selected, axis=0, keepdims=True))[0]
            )
        updated = np.stack(next_centers)
        if np.allclose(updated, centers, rtol=0.0, atol=1.0e-7):
            centers = updated
            break
        centers = updated
    return _normalize(centers)


def _prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    variant: Variant,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    classes = np.asarray(sorted(set(labels.astype(str).tolist())))
    result: list[np.ndarray] = []
    for label in classes.tolist():
        rows = features[labels == label]
        if variant.prototype_rule == "deterministic_two_prototype":
            result.append(_two_prototypes(rows))
        else:
            result.append(_robust_center(rows, variant)[None, :])
    return classes, tuple(result)


def _scores(
    features: np.ndarray,
    prototypes: Sequence[np.ndarray],
) -> np.ndarray:
    query = _normalize(features)
    columns = [
        np.max(query @ _normalize(value).T, axis=1)
        for value in prototypes
    ]
    return TEMPERATURE * np.stack(columns, axis=1)


def _predict(
    features: np.ndarray,
    classes: np.ndarray,
    prototypes: Sequence[np.ndarray],
) -> np.ndarray:
    return classes[np.argmax(_scores(features, prototypes), axis=1)]


def _loo_slice(
    support_x: np.ndarray,
    support_y: np.ndarray,
    variants: Sequence[Variant],
) -> dict[str, dict[str, Any]]:
    x = np.asarray(support_x, dtype=np.float32)
    y = np.asarray(support_y).astype(str)
    classes = np.asarray(sorted(set(y.tolist())))
    records: dict[str, dict[str, list[Any]]] = {
        variant.name: {
            "correct": [],
            "margin": [],
            "label": [],
        }
        for variant in variants
    }
    for heldout in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[heldout] = False
        train_x = x[mask]
        train_y = y[mask]
        test_x = x[heldout : heldout + 1]
        transformed: dict[bool, tuple[np.ndarray, np.ndarray]] = {}
        for equalizer in (False, True):
            train, test, _ = _transform_with_labels(
                train_x,
                train_y,
                test_x,
                equalizer=equalizer,
            )
            transformed[equalizer] = (train, test)
        for variant in variants:
            train, test = transformed[variant.equalizer]
            proto_classes, prototypes = _prototypes(train, train_y, variant)
            if not np.array_equal(proto_classes, classes):
                raise SingleObservationAblationError("LOO class registry drift")
            score = _scores(test, prototypes)[0]
            true_index = int(np.flatnonzero(classes == y[heldout])[0])
            other = np.delete(score, true_index)
            records[variant.name]["correct"].append(
                int(int(np.argmax(score)) == true_index)
            )
            records[variant.name]["margin"].append(
                float(score[true_index] - np.max(other))
            )
            records[variant.name]["label"].append(y[heldout])
    output: dict[str, dict[str, Any]] = {}
    for variant in variants:
        record = records[variant.name]
        per_class: dict[str, Any] = {}
        for label in classes.tolist():
            positions = [
                index
                for index, value in enumerate(record["label"])
                if value == label
            ]
            accuracy = float(
                np.mean([record["correct"][index] for index in positions])
            )
            margins = np.asarray(
                [record["margin"][index] for index in positions],
                dtype=np.float64,
            )
            per_class[label] = {
                "physical_support_count": len(positions),
                "accuracy": accuracy,
                "mean_margin": float(np.mean(margins)),
                "worst_margin": float(np.min(margins)),
            }
        output[variant.name] = {
            "overall_loo_accuracy": float(np.mean(record["correct"])),
            "min_class_loo_accuracy": min(
                float(value["accuracy"]) for value in per_class.values()
            ),
            "worst_margin": float(np.min(record["margin"])),
            "per_class": per_class,
        }
    return output


def _aggregate_support_selection(
    slice_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    variants: Sequence[Variant],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        slices = [
            payload[variant.name]
            for payload in slice_results.values()
        ]
        min_class_values = [
            float(item["min_class_loo_accuracy"]) for item in slices
        ]
        row = {
            **variant.as_dict(),
            "worst_state_scenario_class_loo_accuracy": min(min_class_values),
            "mean_state_scenario_min_class_loo_accuracy": float(
                np.mean(min_class_values)
            ),
            "overall_support_loo_accuracy": float(
                np.mean(
                    [float(item["overall_loo_accuracy"]) for item in slices]
                )
            ),
            "worst_support_loo_margin": min(
                float(item["worst_margin"]) for item in slices
            ),
        }
        row["support_selection_key"] = [
            row["worst_state_scenario_class_loo_accuracy"],
            row["mean_state_scenario_min_class_loo_accuracy"],
            row["overall_support_loo_accuracy"],
            row["worst_support_loo_margin"],
            -float(variant.prototypes_per_class),
            -float(int(variant.equalizer)),
        ]
        rows.append(row)
    ranked = sorted(
        rows,
        key=lambda row: tuple(float(value) for value in row["support_selection_key"]),
        reverse=True,
    )
    selected = ranked[0]
    return {
        "schema": "cvs.phase2.singleobs_view_ablation_support_selection.v1",
        "selection_data": "registered_support_only",
        "query_rows_used_for_selection": 0,
        "query_labels_used_for_selection": False,
        "selection_policy": (
            "lexicographic_max(worst_state_scenario_class_loo_accuracy,"
            "mean_state_scenario_min_class_loo_accuracy,"
            "overall_support_loo_accuracy,worst_support_loo_margin,"
            "lower_prototype_count,lower_equalizer_state)"
        ),
        "before_after_equal_priority": True,
        "selected_variant": selected["name"],
        "ranking": ranked,
        "slice_results": dict(slice_results),
    }


def _member_descriptor(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in manifest.get("members", [])
        if item.get("kind") == kind
    ]
    if len(matches) != 1:
        raise SingleObservationAblationError(
            f"package member descriptor drift: {kind}"
        )
    return matches[0]


def _load_bundle(
    root: str | Path,
    seal: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    return load_verified_somph_predictor_bundle(
        root,
        detached_seal_path=seal,
        expected_seal_sha256=str(expected_seal_sha256).lower(),
    )


def _validate_pair(
    enrollment_manifest: Mapping[str, Any],
    apply_manifest: Mapping[str, Any],
    *,
    state: str,
) -> None:
    expected_stage = "stage2b" if state == "before" else "stage2c"
    if (
        enrollment_manifest.get("profile") != "enrollment_only"
        or apply_manifest.get("profile") != "apply_only"
        or enrollment_manifest.get("stage") != expected_stage
        or apply_manifest.get("stage") != expected_stage
        or enrollment_manifest.get("registration_state") != state
        or apply_manifest.get("registration_state") != state
    ):
        raise SingleObservationAblationError(
            f"{state} enrollment/apply profile drift"
        )
    for field in (
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "registered_classes",
    ):
        if enrollment_manifest.get(field) != apply_manifest.get(field):
            raise SingleObservationAblationError(
                f"{state} enrollment/apply mismatch: {field}"
            )
    if int(enrollment_manifest["k_shot"]) != 10:
        raise SingleObservationAblationError("this development ablation is K10 only")


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if (
        not value.startswith("cuda:")
        or not torch.cuda.is_available()
        or int(value.split(":", 1)[1]) >= torch.cuda.device_count()
    ):
        raise SingleObservationAblationError(
            f"requested device unavailable: {value}"
        )
    return torch.device(value)


def _extract_state_features(
    *,
    model: torch.nn.Module,
    device: torch.device,
    enrollment_payloads: Mapping[str, Mapping[str, np.ndarray]],
    apply_payloads: Mapping[str, Mapping[str, np.ndarray]],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    class_handles = np.asarray(
        [row["class_handle"] for row in manifest["registered_classes"]]
    )
    k_shot = int(manifest["k_shot"])
    result: dict[str, dict[str, np.ndarray]] = {}
    started = time.perf_counter()
    support_forward_count = 0
    query_forward_count = 0
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        support = enrollment_payloads[scenario]
        ranks = np.asarray(support["support_rank_within_class"], dtype=np.int64)
        class_indices = np.asarray(
            support["support_class_indices"], dtype=np.int64
        )
        mask = ranks < k_shot
        support_iq = np.asarray(
            support["support_leo_weak_iq"], dtype=np.float32
        )[mask]
        support_zid = forward_zid160(
            model,
            support_iq,
            device=device,
            batch_size=64,
        )
        support_features = registered_feature(support_iq, support_zid)
        support_labels = class_handles[class_indices[mask]]
        support_tokens = np.asarray(support["support_tokens"]).astype(str)[mask]
        support_hashes = np.asarray(
            support["support_post_channel_iq_sha256"]
        ).astype(str)[mask]
        if (
            len(set(support_tokens.tolist())) != len(support_tokens)
            or len(set(support_hashes.tolist())) != len(support_hashes)
        ):
            raise SingleObservationAblationError(
                f"{scenario} support physical lineage drift"
            )

        query = apply_payloads[scenario]
        query_iq = np.asarray(query["query_leo_weak_iq"], dtype=np.float32)
        query_zid = forward_zid160(
            model,
            query_iq,
            device=device,
            batch_size=1,
        )
        query_features = registered_feature(query_iq, query_zid)
        query_tokens = np.asarray(query["query_tokens"]).astype(str)
        query_hashes = np.asarray(
            query["query_post_channel_iq_sha256"]
        ).astype(str)
        if (
            len(set(query_tokens.tolist())) != len(query_tokens)
            or len(set(query_hashes.tolist())) != len(query_hashes)
            or set(support_tokens.tolist()) & set(query_tokens.tolist())
            or set(support_hashes.tolist()) & set(query_hashes.tolist())
        ):
            raise SingleObservationAblationError(
                f"{scenario} support/query physical disjointness drift"
            )
        result[scenario] = {
            "support_features": support_features,
            "support_labels": support_labels,
            "support_tokens": support_tokens,
            "support_hashes": support_hashes,
            "query_features": query_features,
            "query_tokens": query_tokens,
            "query_hashes": query_hashes,
        }
        support_forward_count += len(support_iq)
        query_forward_count += len(query_iq)
    token_sets = [
        set(result[scenario]["support_tokens"].tolist())
        | set(result[scenario]["query_tokens"].tolist())
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ]
    hash_sets = [
        set(result[scenario]["support_hashes"].tolist())
        | set(result[scenario]["query_hashes"].tolist())
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ]
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            if token_sets[left] & token_sets[right] or hash_sets[left] & hash_sets[right]:
                raise SingleObservationAblationError(
                    "cross-scenario physical observation reuse is forbidden"
                )
    return result, {
        "feature_extraction_elapsed_sec": float(time.perf_counter() - started),
        "support_backbone_forward_count": support_forward_count,
        "query_backbone_forward_count": query_forward_count,
        "query_backbone_forwards_per_sample": 1,
        "query_fft_extractions_per_sample": 1,
        "sealed_runtime_feature_dim": FEATURE_DIM,
        "registered_feature_dim": int(
            result[FORMAL_LEO_WEAK_SCENARIOS[0]]["support_features"].shape[1]
        ),
    }


def _fit_full_state_and_predict(
    *,
    state_features: Mapping[str, Mapping[str, np.ndarray]],
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    all_tokens: list[np.ndarray] = []
    all_scenarios: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    state_rows: list[dict[str, Any]] = []
    total_head_elapsed = 0.0
    total_queries = 0
    total_state_bytes = 0
    total_trainable = 0
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload = state_features[scenario]
        train, query, log_scale = _transform_with_labels(
            payload["support_features"],
            payload["support_labels"],
            payload["query_features"],
            equalizer=variant.equalizer,
        )
        classes, prototypes = _prototypes(
            train,
            payload["support_labels"],
            variant,
        )
        started = time.perf_counter()
        predicted = _predict(query, classes, prototypes)
        elapsed = time.perf_counter() - started
        prototype_rows = int(sum(len(value) for value in prototypes))
        feature_dim = int(train.shape[1])
        state_bytes = int(
            sum(value.nbytes for value in prototypes)
            + (log_scale.nbytes if variant.equalizer else 0)
            + len(_canonical_json_bytes(classes.tolist()))
        )
        trainable = feature_dim if variant.equalizer else 0
        head_macs = int(
            prototype_rows * feature_dim
            + (feature_dim if variant.equalizer else 0)
        )
        if state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise SingleObservationAblationError(
                f"{variant.name} state exceeds 256KB"
            )
        state_rows.append(
            {
                "scenario": scenario,
                "class_count": len(classes),
                "prototype_rows": prototype_rows,
                "feature_dim": feature_dim,
                "trainable_parameters": trainable,
                "persistent_state_bytes": state_bytes,
                "head_macs_per_query": head_macs,
                "head_latency_sec": float(elapsed),
                "head_latency_per_query_ms": float(
                    elapsed * 1000.0 / max(1, len(query))
                ),
            }
        )
        total_head_elapsed += elapsed
        total_queries += len(query)
        total_state_bytes += state_bytes
        total_trainable += trainable
        all_tokens.append(payload["query_tokens"])
        all_scenarios.append(np.asarray([scenario] * len(query)))
        all_predictions.append(predicted)
    return (
        np.concatenate(all_tokens).astype(str),
        np.concatenate(all_scenarios).astype(str),
        np.concatenate(all_predictions).astype(str),
        {
            "schema": "cvs.phase2.singleobs_view_ablation_resource.v1",
            "variant": variant.as_dict(),
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "source_sample_access": False,
            "clean_sample_access": False,
            "additional_leo_channel_state_generation": False,
            "views_count_as_additional_physical_samples": False,
            "trainable_parameters": total_trainable,
            "persistent_state_bytes": total_state_bytes,
            "query_backbone_forwards_per_sample": 1,
            "query_fft_extractions_per_sample": 1,
            "head_latency_sec": float(total_head_elapsed),
            "head_latency_per_query_ms": float(
                total_head_elapsed * 1000.0 / max(1, total_queries)
            ),
            "by_scenario": state_rows,
        },
    )


def predict_command(args: argparse.Namespace) -> int:
    output = Path(args.output_root)
    if output.exists():
        raise SingleObservationAblationError("output root must not exist")
    output.mkdir(parents=True)
    inputs = {
        "before": {
            "enrollment_root": args.before_enrollment_root,
            "enrollment_seal": args.before_enrollment_seal,
            "enrollment_seal_sha256": args.before_enrollment_seal_sha256,
            "apply_root": args.before_apply_root,
            "apply_seal": args.before_apply_seal,
            "apply_seal_sha256": args.before_apply_seal_sha256,
        },
        "after": {
            "enrollment_root": args.after_enrollment_root,
            "enrollment_seal": args.after_enrollment_seal,
            "enrollment_seal_sha256": args.after_enrollment_seal_sha256,
            "apply_root": args.after_apply_root,
            "apply_seal": args.after_apply_seal,
            "apply_seal_sha256": args.after_apply_seal_sha256,
        },
    }
    loaded: dict[str, Any] = {}
    for state, values in inputs.items():
        enrollment_payloads, enrollment_manifest, enrollment_audit = _load_bundle(
            values["enrollment_root"],
            values["enrollment_seal"],
            values["enrollment_seal_sha256"],
        )
        apply_payloads, apply_manifest, apply_audit = _load_bundle(
            values["apply_root"],
            values["apply_seal"],
            values["apply_seal_sha256"],
        )
        _validate_pair(enrollment_manifest, apply_manifest, state=state)
        loaded[state] = {
            "enrollment_payloads": enrollment_payloads,
            "enrollment_manifest": enrollment_manifest,
            "enrollment_audit": enrollment_audit,
            "apply_payloads": apply_payloads,
            "apply_manifest": apply_manifest,
            "apply_audit": apply_audit,
        }
    for field in (
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
    ):
        if (
            loaded["before"]["enrollment_manifest"].get(field)
            != loaded["after"]["enrollment_manifest"].get(field)
        ):
            raise SingleObservationAblationError(
                f"before/after matched-row drift: {field}"
            )
    runtime_device = _device(args.device)
    runtime_manifest = loaded["before"]["enrollment_manifest"]
    model = load_torchscript_backbone_same_fd(
        args.before_enrollment_root,
        _member_descriptor(runtime_manifest, "feature_runtime"),
        device=runtime_device,
    )
    state_features: dict[str, Any] = {}
    extraction_resources: dict[str, Any] = {}
    for state in ("before", "after"):
        features, resource = _extract_state_features(
            model=model,
            device=runtime_device,
            enrollment_payloads=loaded[state]["enrollment_payloads"],
            apply_payloads=loaded[state]["apply_payloads"],
            manifest=loaded[state]["enrollment_manifest"],
        )
        state_features[state] = features
        extraction_resources[state] = resource

    slice_results: dict[str, Any] = {}
    for state in ("before", "after"):
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            key = f"{state}:{scenario}"
            slice_results[key] = _loo_slice(
                state_features[state][scenario]["support_features"],
                state_features[state][scenario]["support_labels"],
                VARIANTS,
            )
    support_selection = _aggregate_support_selection(slice_results, VARIANTS)
    selection_sha256 = _write_json_new(
        output / "support_selection.json",
        support_selection,
    )

    prediction_members: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_root = output / "predictions" / variant.name
        for state in ("before", "after"):
            tokens, scenarios, predicted, resource = _fit_full_state_and_predict(
                state_features=state_features[state],
                variant=variant,
            )
            prediction_path = variant_root / f"{state}_prediction.npz"
            prediction_sha256 = _write_prediction_new(
                prediction_path,
                query_tokens=tokens,
                scenarios=scenarios,
                predicted_class_handles=predicted,
            )
            resource_path = variant_root / f"{state}_resource.json"
            resource_sha256 = _write_json_new(resource_path, resource)
            prediction_members.append(
                {
                    "variant": variant.name,
                    "state": state,
                    "prediction_relative_path": str(
                        prediction_path.relative_to(output)
                    ).replace("\\", "/"),
                    "prediction_sha256": prediction_sha256,
                    "prediction_size_bytes": prediction_path.stat().st_size,
                    "resource_relative_path": str(
                        resource_path.relative_to(output)
                    ).replace("\\", "/"),
                    "resource_sha256": resource_sha256,
                    "resource_size_bytes": resource_path.stat().st_size,
                }
            )
    manifest = {
        "schema": "cvs.phase2.singleobs_view_ablation_prediction_manifest.v1",
        "status": "ALL_VARIANT_IMMUTABLE_PREDICTIONS_COMPLETE",
        "formal_launch_authority": False,
        "claim_scope": "development_only_not_formal_confirmation",
        "truth_sidecar_argument_present_in_predictor": False,
        "query_truth_opened_by_predictor": False,
        "query_rows_used_for_selection": 0,
        "query_labels_used_for_selection": False,
        "selected_variant": support_selection["selected_variant"],
        "support_selection_sha256": selection_sha256,
        "receiver": runtime_manifest["receiver"],
        "seed": runtime_manifest["seed"],
        "k_shot": runtime_manifest["k_shot"],
        "new_class_count": (
            len(loaded["after"]["enrollment_manifest"]["registered_classes"])
            - len(loaded["before"]["enrollment_manifest"]["registered_classes"])
        ),
        "phase1_checkpoint_sha256": runtime_manifest[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": runtime_manifest["feature_runtime_sha256"],
        "method_lock_sha256": runtime_manifest["method_lock_sha256"],
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_source_sample_access": False,
        "phase2_source_derived_signal_access": False,
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_query_post_reception_view_fit_access": False,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "extraction_resources": extraction_resources,
        "variants": [variant.as_dict() for variant in VARIANTS],
        "members": prediction_members,
        "preopen_audit": {
            state: {
                "enrollment": loaded[state]["enrollment_audit"],
                "apply": loaded[state]["apply_audit"],
            }
            for state in ("before", "after")
        },
    }
    manifest_sha256 = _write_json_new(
        output / "prediction_manifest.json",
        manifest,
    )
    commit = {
        "schema": "cvs.phase2.singleobs_view_ablation_prediction_commit.v1",
        "diagnostic_only": True,
        "status": "PREDICTIONS_COMMITTED_BEFORE_TRUTH_JOIN",
        "prediction_manifest_sha256": manifest_sha256,
        "support_selection_sha256": selection_sha256,
        "selected_variant": support_selection["selected_variant"],
        "member_count": len(prediction_members),
        "members_sha256": hashlib.sha256(
            _canonical_json_bytes(prediction_members)
        ).hexdigest(),
    }
    commit_sha256 = _write_json_new(output / "COMMIT.json", commit)
    print(
        json.dumps(
            {
                "status": commit["status"],
                "output_root": str(output.resolve()),
                "selected_variant": support_selection["selected_variant"],
                "commit_sha256": commit_sha256,
                "variant_count": len(VARIANTS),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _verify_prediction_commit(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commit_path = root / "COMMIT.json"
    manifest_path = root / "prediction_manifest.json"
    selection_path = root / "support_selection.json"
    if not commit_path.is_file() or not manifest_path.is_file() or not selection_path.is_file():
        raise SingleObservationAblationError(
            "prediction COMMIT/manifest/support selection is incomplete"
        )
    commit = json.loads(commit_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        commit.get("status") != "PREDICTIONS_COMMITTED_BEFORE_TRUTH_JOIN"
        or manifest.get("status") != "ALL_VARIANT_IMMUTABLE_PREDICTIONS_COMPLETE"
        or commit.get("prediction_manifest_sha256") != _sha256_file(manifest_path)
        or commit.get("support_selection_sha256") != _sha256_file(selection_path)
    ):
        raise SingleObservationAblationError("prediction commit binding drift")
    for member in manifest["members"]:
        prediction = root / member["prediction_relative_path"]
        resource = root / member["resource_relative_path"]
        if (
            _sha256_file(prediction) != member["prediction_sha256"]
            or _sha256_file(resource) != member["resource_sha256"]
            or prediction.stat().st_mode
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            or resource.stat().st_mode
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise SingleObservationAblationError(
                "immutable prediction/resource member drift"
            )
        with np.load(prediction, allow_pickle=False) as archive:
            if tuple(archive.files) != PREDICTION_MEMBERS:
                raise SingleObservationAblationError(
                    "prediction exact schema drift"
                )
    return manifest, json.loads(selection_path.read_text(encoding="utf-8-sig"))


def _report_markdown(
    *,
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    selected = str(selection["selected_variant"])
    selected_score = summary["scores"][selected]
    selected_resource = summary["resources"][selected]
    lines = [
        "# 单观测LEO_weak固定接收IQ后处理消融",
        "",
        "结论：统一support LOO预登记选择的路线为"
        f"`{selected}`。该选择在读取任何query标签之前完成；"
        "全部variant的注册前后prediction均只读提交后，隔离scorer才读取truth sidecar。",
        "",
        "## 协议边界",
        "",
        "- 仅使用sealed LEO_weak support/query与sealed Phase1 runtime；无clean/source输入。",
        "- 每个物理样本只有一份既定LEO接收IQ；接收后view不增加K，也不生成额外LEO状态。",
        "- variant与超参数只由before/after×三场景support LOO选择。",
        "- query逐样本在全部已注册类上决策；无角色Oracle、类别配额、全局重排或query拟合。",
        "- 本结果是单receiver×单development seed的开发消融，不是正式确认矩阵。",
        "",
        "## Support选择排序",
        "",
        "| rank | variant | worst class LOO | mean slice floor | overall LOO | worst margin |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(selection["ranking"], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | "
            f"{row['worst_state_scenario_class_loo_accuracy']:.4f} | "
            f"{row['mean_state_scenario_min_class_loo_accuracy']:.4f} | "
            f"{row['overall_support_loo_accuracy']:.4f} | "
            f"{row['worst_support_loo_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 冻结query结果",
            "",
            "| variant | old before | old after | old floor after | seen-new | H | forgetting(pp) | after MAC/query | after state(B) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in [row["name"] for row in selection["ranking"]]:
        score = summary["scores"][variant]
        resource = summary["resources"][variant]["after"]
        mac = int(
            np.mean(
                [
                    int(row["head_macs_per_query"])
                    for row in resource["by_scenario"]
                ]
            )
        )
        lines.append(
            f"| `{variant}` | {score['before']['old_acc']:.4f} | "
            f"{score['after']['old_acc']:.4f} | "
            f"{score['per_old_class_floor_after']:.4f} | "
            f"{score['after']['seen_new_acc']:.4f} | "
            f"{score['after']['h_old_new']:.4f} | "
            f"{score['old_forgetting_pp']:.2f} | {mac} | "
            f"{resource['persistent_state_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Support-selected路线实际new5结果",
            "",
            f"- `old_acc_before_increment={selected_score['before']['old_acc']:.6f}`",
            f"- `old_acc_after_increment={selected_score['after']['old_acc']:.6f}`",
            f"- `seen_new_acc={selected_score['after']['seen_new_acc']:.6f}`",
            f"- `H_old_new={selected_score['after']['h_old_new']:.6f}`",
            f"- `per_old_class_floor_after={selected_score['per_old_class_floor_after']:.6f}`",
            f"- `old_forgetting_pp={selected_score['old_forgetting_pp']:.6f}`",
            f"- `trainable_parameters_after={selected_resource['after']['trainable_parameters']}`",
            f"- `persistent_state_bytes_after={selected_resource['after']['persistent_state_bytes']}`",
            f"- `query_backbone_forwards_per_sample={selected_resource['after']['query_backbone_forwards_per_sample']}`",
            f"- `query_fft_extractions_per_sample={selected_resource['after']['query_fft_extractions_per_sample']}`",
            "",
            "### 逐TX",
            "",
            "| TX | role | accuracy | count |",
            "|---|---|---:|---:|",
        ]
    )
    for tx, row in selected_score["after"]["by_tx"].items():
        lines.append(
            f"| `{tx}` | {row['role']} | {row['accuracy']:.4f} | {row['count']} |"
        )
    lines.extend(
        [
            "",
            "### 逐场景",
            "",
            "| scenario | old | seen-new | H | old->new | new->old |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, row in selected_score["after"]["by_scenario"].items():
        lines.append(
            f"| `{scenario}` | {row['old_acc']:.4f} | "
            f"{row['seen_new_acc']:.4f} | {row['h_old_new']:.4f} | "
            f"{row['old_to_new_rate']:.4f} | {row['new_to_old_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "该开发row仍需以项目门槛解释：K10 new5要求old总体≥0.92、"
            "每个旧类≥0.88、seen-new≥0.92。未同时达到时只能作为下一轮"
            "floor/新旧类去混淆机制设计证据，不能晋升为125确认候选。",
            "",
        ]
    )
    return "\n".join(lines)


def score_command(args: argparse.Namespace) -> int:
    root = Path(args.prediction_root).resolve(strict=True)
    manifest, selection = _verify_prediction_commit(root)
    score_root = root / "scores"
    score_root.mkdir()
    scores: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    for variant in [row["name"] for row in selection["ranking"]]:
        before_prediction = root / "predictions" / variant / "before_prediction.npz"
        after_prediction = root / "predictions" / variant / "after_prediction.npz"
        result = score_diag_cosine_pair(
            before_prediction_path=before_prediction,
            after_prediction_path=after_prediction,
            truth_sidecar_path=args.truth_sidecar,
            output_path=score_root / f"{variant}.json",
            candidate=variant,
        )
        result.pop("score_artifact_sha256", None)
        scores[variant] = result
        resources[variant] = {
            state: json.loads(
                (
                    root
                    / "predictions"
                    / variant
                    / f"{state}_resource.json"
                ).read_text(encoding="utf-8-sig")
            )
            for state in ("before", "after")
        }
    summary = {
        "schema": "cvs.phase2.singleobs_view_ablation_score_summary.v1",
        "status": "POST_PREDICTION_ISOLATED_SCORING_COMPLETE",
        "claim_scope": "development_only_not_formal_confirmation",
        "query_truth_joined_only_after_all_immutable_predictions": True,
        "query_truth_fed_back_to_predictor": False,
        "support_selected_variant": selection["selected_variant"],
        "prediction_commit_sha256": _sha256_file(root / "COMMIT.json"),
        "truth_sidecar_sha256": _sha256_file(args.truth_sidecar),
        "scores": scores,
        "resources": resources,
    }
    summary_sha256 = _write_json_new(root / "score_summary.json", summary)
    report = _report_markdown(
        manifest=manifest,
        selection=selection,
        summary=summary,
    )
    report_sha256 = _write_text_new(root / "report.md", report)
    score_members = [
        {
            "relative_path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(score_root.glob("*.json"))
    ]
    score_commit = {
        "schema": "cvs.phase2.singleobs_view_ablation_score_commit.v1",
        "diagnostic_only": True,
        "status": "POST_PREDICTION_ISOLATED_SCORING_COMMITTED",
        "support_selected_variant": selection["selected_variant"],
        "score_summary_sha256": summary_sha256,
        "report_sha256": report_sha256,
        "members": score_members,
    }
    score_commit_sha256 = _write_json_new(
        root / "SCORE_COMMIT.json",
        score_commit,
    )
    selected = scores[selection["selected_variant"]]
    print(
        json.dumps(
            {
                "status": score_commit["status"],
                "support_selected_variant": selection["selected_variant"],
                "old_acc_before": selected["before"]["old_acc"],
                "old_acc_after": selected["after"]["old_acc"],
                "seen_new_acc": selected["after"]["seen_new_acc"],
                "h_old_new": selected["after"]["h_old_new"],
                "per_old_class_floor_after": selected[
                    "per_old_class_floor_after"
                ],
                "old_forgetting_pp": selected["old_forgetting_pp"],
                "score_commit_sha256": score_commit_sha256,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser(
        "predict",
        help="support-select variants and commit immutable predictions without truth",
    )
    for state in ("before", "after"):
        predict.add_argument(f"--{state}-enrollment-root", required=True)
        predict.add_argument(f"--{state}-enrollment-seal", required=True)
        predict.add_argument(
            f"--{state}-enrollment-seal-sha256",
            required=True,
        )
        predict.add_argument(f"--{state}-apply-root", required=True)
        predict.add_argument(f"--{state}-apply-seal", required=True)
        predict.add_argument(f"--{state}-apply-seal-sha256", required=True)
    predict.add_argument("--output-root", required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.set_defaults(func=predict_command)

    score = subparsers.add_parser(
        "score",
        help="join truth only after prediction COMMIT exists",
    )
    score.add_argument("--prediction-root", required=True)
    score.add_argument("--truth-sidecar", required=True)
    score.set_defaults(func=score_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
