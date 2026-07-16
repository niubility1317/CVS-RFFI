"""Truth-free SOMP-H enrollment and per-sample apply runtime.

Enrollment receives only a unified registered-support pool. Apply receives
only a sealed SOMP-H head and unlabeled LEO-weak query IQ. Neither API accepts
old/new roles, query truth, query class counts, quotas, ordering hints, or a
query graph.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

import numpy as np
import torch

from cvsrffi.stage2_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
SOMPH_METHOD_LOCK_SCHEMA = "cvs.phase2.somph_method_lock.v1"
SOMPH_HEAD_CAPSULE_SCHEMA = "cvs.phase2.somph_runtime_head_capsule.v1"
SOMPH_ENROLLMENT_BINDING_SCHEMA = "cvs.phase2.somph_enrollment_binding.v1"
ADV3B02_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
FEATURE_DIM = 160
EPS = 1.0e-8

_HEAD_TENSOR_NAMES = (
    "prototypes_fp16",
    "prototype_class_ids_uint16",
    "centroids_fp16",
    "residual_scale_fp16",
    "class_hubness_penalty_fp16",
    "scalars_fp16",
)
_SUPPORT_PAYLOAD_KEYS = {
    "support_leo_weak_iq",
    "support_class_indices",
    "support_rank_within_class",
    "support_tokens",
    "support_overlay_tokens",
    "support_satellite_seeds",
    "support_post_channel_iq_sha256",
}
_QUERY_PAYLOAD_KEYS = {
    "query_leo_weak_iq",
    "query_tokens",
    "query_overlay_tokens",
    "query_satellite_seeds",
    "query_post_channel_iq_sha256",
}
_OPAQUE_CLASS_PREFIX = "cls_"


class SomphPredictorRuntimeError(ValueError):
    """Raised when SOMP-H runtime input or state fails closed."""


def _phase2_contract() -> dict[str, Any]:
    return {
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "query_labels_used_for_fit": False,
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "dense_query_graph_used": False,
    }


def expected_somph_method_lock() -> dict[str, Any]:
    return {
        "schema": SOMPH_METHOD_LOCK_SCHEMA,
        "method_id": "SOMPH_V1",
        "base_model_id": "ADV3B02_CORE90_SOFT_E200",
        "checkpoint_sha256": ADV3B02_CHECKPOINT_SHA256,
        "feature_schema": "adv3b02_z_id160_fp32",
        "feature_dim": FEATURE_DIM,
        "feature_runtime_policy": "sealed_adv3b02_identity_runtime_only",
        "adapter_type": "none",
        "trainable_parameters": 0,
        "updated_original_parameters": 0,
        "adaptation_epochs": 0,
        "optimizer_steps": 0,
        "max_prototypes_per_class": 2,
        "residual_shrinkage": 0.5,
        "residual_scale_min": 0.5,
        "residual_scale_max": 2.0,
        "max_mix": 0.75,
        "hubness_weight": 0.25,
        "trainable_parameter_cap": 50_000,
        "persistent_state_cap_bytes": 256 * 1024,
        "support_view_policy": "one_preoverlaid_leo_weak_view_per_scenario",
        "selection_policy": "fixed_support_geometry_no_query_label_selection",
        "development_receiver": "20-1",
        "development_seed": 713101,
        "development_k_shot": 10,
        "phase2_contract": _phase2_contract(),
    }


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    rows = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(rows.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(rows.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(rows.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_sha256(values: np.ndarray) -> str:
    rows = np.asarray(values).astype(str)
    return hashlib.sha256(
        json.dumps(rows.tolist(), ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_runtime_sample_metadata(
    payload: Mapping[str, Any],
    *,
    token_key: str,
    token_prefix: str,
    overlay_key: str,
    seed_key: str,
    iq_sha_key: str,
    expected_count: int,
    context: str,
) -> np.ndarray:
    raw_tokens = np.asarray(payload[token_key])
    if raw_tokens.ndim != 1 or raw_tokens.dtype.kind not in {"S", "U"}:
        raise SomphPredictorRuntimeError(f"SOMP-H {context} token dtype drift")
    tokens = raw_tokens.astype(str)
    if (
        len(tokens) != expected_count
        or len(tokens) != len(set(tokens.tolist()))
        or any(
            not token.startswith(token_prefix)
            or len(token) != len(token_prefix) + 64
            or not _is_sha256(token[len(token_prefix) :])
            for token in tokens.tolist()
        )
    ):
        raise SomphPredictorRuntimeError(f"SOMP-H {context} token schema drift")
    raw_overlays = np.asarray(payload[overlay_key])
    overlays = raw_overlays.astype(str)
    if (
        raw_overlays.ndim != 1
        or raw_overlays.dtype.kind not in {"S", "U"}
        or overlays.shape != (expected_count,)
        or len(overlays) != len(set(overlays.tolist()))
        or any(
            not token.startswith("oid_")
            or len(token) != 68
            or not _is_sha256(token[4:])
            for token in overlays.tolist()
        )
    ):
        raise SomphPredictorRuntimeError(
            f"SOMP-H {context} overlay token schema drift"
        )
    seeds = np.asarray(payload[seed_key])
    if seeds.dtype != np.int64 or seeds.shape != (expected_count,):
        raise SomphPredictorRuntimeError(
            f"SOMP-H {context} satellite seed schema drift"
        )
    raw_hashes = np.asarray(payload[iq_sha_key])
    hashes = raw_hashes.astype(str)
    if (
        raw_hashes.ndim != 1
        or raw_hashes.dtype.kind not in {"S", "U"}
        or hashes.shape != (expected_count,)
        or any(not _is_sha256(value) for value in hashes.tolist())
    ):
        raise SomphPredictorRuntimeError(
            f"SOMP-H {context} post-channel IQ digest schema drift"
        )
    return tokens


def validate_enrollment_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema",
        "stage",
        "registration_state",
        "receiver",
        "seed",
        "k_shot",
        "registered_class_handles",
        "enrollment_package_root_sha256",
        "enrollment_package_seal_sha256",
        "checkpoint_sha256",
        "method_lock_sha256",
        "support_token_sha256_by_scenario",
        "support_feature_sha256_by_scenario",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment binding exact schema drift"
        )
    if value["schema"] != SOMPH_ENROLLMENT_BINDING_SCHEMA:
        raise SomphPredictorRuntimeError("SOMP-H enrollment binding schema drift")
    if value["stage"] not in {"stage2b", "stage2c"}:
        raise SomphPredictorRuntimeError("SOMP-H enrollment stage drift")
    if value["registration_state"] not in {"before", "after"}:
        raise SomphPredictorRuntimeError("SOMP-H enrollment registration state drift")
    if value["stage"] == "stage2b" and value["registration_state"] != "before":
        raise SomphPredictorRuntimeError("Stage2-B permits only the before registry")
    if not isinstance(value["receiver"], str) or not value["receiver"]:
        raise SomphPredictorRuntimeError("SOMP-H enrollment receiver drift")
    for field in ("seed", "k_shot"):
        if (
            not isinstance(value[field], int)
            or isinstance(value[field], bool)
            or value[field] < 1
        ):
            raise SomphPredictorRuntimeError(
                f"SOMP-H enrollment binding integer drift: {field}"
            )
    handles = value["registered_class_handles"]
    if (
        not isinstance(handles, list)
        or len(handles) < 2
        or len(handles) != len(set(handles))
        or any(
            not isinstance(handle, str)
            or not handle.startswith(_OPAQUE_CLASS_PREFIX)
            or len(handle) != 68
            or not _is_sha256(handle[4:])
            for handle in handles
        )
    ):
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment class registry is not unique opaque"
        )
    if value["checkpoint_sha256"] != ADV3B02_CHECKPOINT_SHA256:
        raise SomphPredictorRuntimeError("SOMP-H enrollment checkpoint drift")
    for field in (
        "enrollment_package_root_sha256",
        "enrollment_package_seal_sha256",
        "method_lock_sha256",
    ):
        if not _is_sha256(value[field]):
            raise SomphPredictorRuntimeError(
                f"SOMP-H enrollment binding SHA256 drift: {field}"
            )
    for field in (
        "support_token_sha256_by_scenario",
        "support_feature_sha256_by_scenario",
    ):
        mapping = value[field]
        if (
            not isinstance(mapping, dict)
            or tuple(mapping) != FORMAL_LEO_WEAK_SCENARIOS
            or any(not _is_sha256(mapping[scenario]) for scenario in mapping)
        ):
            raise SomphPredictorRuntimeError(
                f"SOMP-H enrollment binding scenario digest drift: {field}"
            )
    return json.loads(json.dumps(dict(value), sort_keys=True))


def validate_somph_method_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_somph_method_lock()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SomphPredictorRuntimeError("SOMP-H method lock exact schema drift")
    failed = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if failed:
        raise SomphPredictorRuntimeError(f"SOMP-H method lock drift: {failed}")
    return json.loads(json.dumps(dict(value), sort_keys=True))


def _normalize(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), EPS)


def _residual_scale(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    shrinkage: float,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    residuals = []
    for class_index in range(class_count):
        selected = rows[labels == class_index]
        residuals.append(selected - selected.mean(axis=0, keepdims=True))
    residual = np.concatenate(residuals, axis=0)
    within_var = np.mean(np.square(residual), axis=0)
    global_var = float(np.mean(within_var))
    shrunk = (1.0 - shrinkage) * within_var + shrinkage * global_var
    inverse_std = 1.0 / np.sqrt(np.maximum(shrunk, EPS))
    inverse_std /= max(float(np.median(inverse_std)), EPS)
    return np.clip(inverse_std, scale_min, scale_max).astype(np.float32)


def _farthest_first(rows: np.ndarray, count: int) -> np.ndarray:
    value = _normalize(rows)
    target = min(max(1, int(count)), len(value))
    centroid = _normalize(value.mean(axis=0, keepdims=True))[0]
    chosen = [int(np.argmax(value @ centroid))]
    while len(chosen) < target:
        similarity = value @ value[np.asarray(chosen)].T
        nearest = np.max(similarity, axis=1)
        nearest[np.asarray(chosen)] = np.inf
        chosen.append(int(np.argmin(nearest)))
    return np.asarray(chosen, dtype=np.int64)


def _fit_head(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    class_count: int,
    method_lock: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    rows = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64).reshape(-1)
    if (
        rows.ndim != 2
        or rows.shape != (len(labels), FEATURE_DIM)
        or not np.isfinite(rows).all()
    ):
        raise SomphPredictorRuntimeError("SOMP-H support feature schema drift")
    if class_count < 2 or labels.size < class_count:
        raise SomphPredictorRuntimeError("SOMP-H requires at least two registered classes")
    if labels.min(initial=0) < 0 or labels.max(initial=-1) >= class_count:
        raise SomphPredictorRuntimeError("SOMP-H support label is outside the registry")
    counts = np.bincount(labels, minlength=class_count)
    if np.any(counts == 0) or len(set(counts.tolist())) != 1:
        raise SomphPredictorRuntimeError("SOMP-H requires equal nonzero K per registered class")

    scale = _residual_scale(
        rows,
        labels,
        class_count=class_count,
        shrinkage=float(method_lock["residual_shrinkage"]),
        scale_min=float(method_lock["residual_scale_min"]),
        scale_max=float(method_lock["residual_scale_max"]),
    )
    transformed = _normalize(rows * scale[None, :])
    prototypes: list[np.ndarray] = []
    prototype_ids: list[int] = []
    centroids: list[np.ndarray] = []
    per_class = min(int(method_lock["max_prototypes_per_class"]), int(counts.min()))
    for class_index in range(class_count):
        selected = transformed[labels == class_index]
        centroids.append(_normalize(selected.mean(axis=0, keepdims=True))[0])
        chosen = selected[_farthest_first(selected, per_class)]
        prototypes.extend(chosen)
        prototype_ids.extend([class_index] * len(chosen))
    centroid_bank = np.stack(centroids).astype(np.float32)
    prototype_bank = np.stack(prototypes).astype(np.float32)
    gram = centroid_bank @ centroid_bank.T
    np.fill_diagonal(gram, -np.inf)
    hubness = np.max(gram, axis=1)
    hubness -= float(np.mean(hubness))
    class_penalty = float(method_lock["hubness_weight"]) * hubness
    return {
        "prototypes_fp16": prototype_bank.astype(np.float16),
        "prototype_class_ids_uint16": np.asarray(prototype_ids, dtype=np.uint16),
        "centroids_fp16": centroid_bank.astype(np.float16),
        "residual_scale_fp16": scale.astype(np.float16),
        "class_hubness_penalty_fp16": class_penalty.astype(np.float16),
        "scalars_fp16": np.asarray(
            [method_lock["max_mix"], method_lock["hubness_weight"]],
            dtype=np.float16,
        ),
    }


def _head_prefix(scenario: str, name: str) -> str:
    return f"{scenario}__{name}"


def somph_head_capsule_members() -> tuple[str, ...]:
    members = [
        "schema_utf8",
        "method_lock_sha256_utf8",
        "enrollment_binding_json_utf8",
        "class_count_uint16",
        "feature_dim_uint16",
        "k_shot_uint16",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        members.extend(_head_prefix(scenario, name) for name in _HEAD_TENSOR_NAMES)
    return tuple(members)


def _utf8(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(value: np.ndarray, *, context: str) -> str:
    raw = np.asarray(value)
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise SomphPredictorRuntimeError(f"{context} must be a uint8 vector")
    try:
        return raw.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SomphPredictorRuntimeError(f"{context} is not UTF-8") from exc


def _head_from_capsule(
    capsule: Mapping[str, np.ndarray],
    *,
    scenario: str,
    class_count: int,
) -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(capsule[_head_prefix(scenario, name)])
        for name in _HEAD_TENSOR_NAMES
    }
    prototypes = result["prototypes_fp16"]
    prototype_ids = result["prototype_class_ids_uint16"]
    centroids = result["centroids_fp16"]
    scale = result["residual_scale_fp16"]
    penalty = result["class_hubness_penalty_fp16"]
    scalars = result["scalars_fp16"]
    if prototypes.dtype != np.float16 or prototypes.ndim != 2:
        raise SomphPredictorRuntimeError("SOMP-H prototype capsule drift")
    if prototype_ids.dtype != np.uint16 or prototype_ids.shape != (len(prototypes),):
        raise SomphPredictorRuntimeError("SOMP-H prototype class map drift")
    if centroids.dtype != np.float16 or centroids.shape != (class_count, FEATURE_DIM):
        raise SomphPredictorRuntimeError("SOMP-H centroid capsule drift")
    if prototypes.shape[1] != FEATURE_DIM:
        raise SomphPredictorRuntimeError("SOMP-H prototype feature dimension drift")
    if scale.dtype != np.float16 or scale.shape != (FEATURE_DIM,):
        raise SomphPredictorRuntimeError("SOMP-H residual scale capsule drift")
    if penalty.dtype != np.float16 or penalty.shape != (class_count,):
        raise SomphPredictorRuntimeError("SOMP-H hubness capsule drift")
    if scalars.dtype != np.float16 or scalars.shape != (2,):
        raise SomphPredictorRuntimeError("SOMP-H scalar capsule drift")
    if any(not np.isfinite(value).all() for value in result.values()):
        raise SomphPredictorRuntimeError("SOMP-H head capsule contains non-finite values")
    ids = prototype_ids.astype(np.int64)
    if ids.size == 0 or ids.max(initial=-1) >= class_count:
        raise SomphPredictorRuntimeError("SOMP-H prototype class map is outside registry")
    if np.any(np.bincount(ids, minlength=class_count) == 0):
        raise SomphPredictorRuntimeError("SOMP-H capsule drops a registered class")
    return result


def validate_somph_head_capsule(
    capsule: Mapping[str, np.ndarray],
    *,
    method_lock: Mapping[str, Any],
    expected_enrollment_binding_sha256: str | None = None,
) -> dict[str, Any]:
    lock = validate_somph_method_lock(method_lock)
    if tuple(capsule) != somph_head_capsule_members():
        raise SomphPredictorRuntimeError("SOMP-H head capsule exact member/order drift")
    if _decode_utf8(capsule["schema_utf8"], context="SOMP-H capsule schema") != SOMPH_HEAD_CAPSULE_SCHEMA:
        raise SomphPredictorRuntimeError("SOMP-H head capsule schema drift")
    if _decode_utf8(
        capsule["method_lock_sha256_utf8"], context="SOMP-H method lock digest"
    ) != canonical_sha256(lock):
        raise SomphPredictorRuntimeError("SOMP-H head capsule/method lock mismatch")
    try:
        binding = json.loads(
            _decode_utf8(
                capsule["enrollment_binding_json_utf8"],
                context="SOMP-H enrollment binding",
            )
        )
    except json.JSONDecodeError as exc:
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment binding is not valid JSON"
        ) from exc
    binding = validate_enrollment_binding(binding)
    binding_sha256 = canonical_sha256(binding)
    if (
        expected_enrollment_binding_sha256 is not None
        and binding_sha256 != expected_enrollment_binding_sha256
    ):
        raise SomphPredictorRuntimeError(
            "SOMP-H capsule/enrollment binding digest mismatch"
        )
    if binding["method_lock_sha256"] != canonical_sha256(lock):
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment binding/method lock mismatch"
        )
    class_count = len(binding["registered_class_handles"])
    count = np.asarray(capsule["class_count_uint16"])
    dimension = np.asarray(capsule["feature_dim_uint16"])
    k_value = np.asarray(capsule["k_shot_uint16"])
    if count.dtype != np.uint16 or count.shape != (1,) or int(count[0]) != class_count:
        raise SomphPredictorRuntimeError("SOMP-H capsule class count drift")
    if (
        dimension.dtype != np.uint16
        or dimension.shape != (1,)
        or int(dimension[0]) != FEATURE_DIM
    ):
        raise SomphPredictorRuntimeError("SOMP-H capsule feature dimension drift")
    if (
        k_value.dtype != np.uint16
        or k_value.shape != (1,)
        or int(k_value[0]) != int(binding["k_shot"])
    ):
        raise SomphPredictorRuntimeError("SOMP-H capsule K drift")
    k_shot = int(k_value[0])
    expected_per_class = min(int(lock["max_prototypes_per_class"]), k_shot)
    state_bytes = 0
    per_query_macs: list[int] = []
    prototype_counts: list[int] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        head = _head_from_capsule(capsule, scenario=scenario, class_count=class_count)
        ids = np.asarray(head["prototype_class_ids_uint16"], dtype=np.int64)
        if not np.all(
            np.bincount(ids, minlength=class_count) == expected_per_class
        ):
            raise SomphPredictorRuntimeError(
                "SOMP-H capsule prototype count/method lock drift"
            )
        scale = np.asarray(head["residual_scale_fp16"], dtype=np.float32)
        tolerance = 2.0e-3
        if (
            float(scale.min()) < float(lock["residual_scale_min"]) - tolerance
            or float(scale.max()) > float(lock["residual_scale_max"]) + tolerance
        ):
            raise SomphPredictorRuntimeError(
                "SOMP-H capsule residual scale/method lock drift"
            )
        scalars = np.asarray(head["scalars_fp16"], dtype=np.float32)
        if (
            abs(float(scalars[0]) - float(lock["max_mix"])) > tolerance
            or abs(float(scalars[1]) - float(lock["hubness_weight"])) > tolerance
        ):
            raise SomphPredictorRuntimeError(
                "SOMP-H capsule scoring scalar/method lock drift"
            )
        centroids = _normalize(
            np.asarray(head["centroids_fp16"], dtype=np.float32)
        )
        gram = centroids @ centroids.T
        np.fill_diagonal(gram, -np.inf)
        hubness = np.max(gram, axis=1)
        hubness -= float(np.mean(hubness))
        expected_penalty = float(lock["hubness_weight"]) * hubness
        penalty = np.asarray(
            head["class_hubness_penalty_fp16"], dtype=np.float32
        )
        if not np.allclose(penalty, expected_penalty, atol=3.0e-3, rtol=0.0):
            raise SomphPredictorRuntimeError(
                "SOMP-H capsule hubness penalty/centroid drift"
            )
        state_bytes += sum(int(head[name].nbytes) for name in _HEAD_TENSOR_NAMES)
        prototype_count = int(len(head["prototypes_fp16"]))
        prototype_counts.append(prototype_count)
        per_query_macs.append(
            3 * FEATURE_DIM
            + prototype_count * FEATURE_DIM
            + class_count * FEATURE_DIM
            + 2 * prototype_count
            + 3 * class_count
        )
    if state_bytes > int(lock["persistent_state_cap_bytes"]):
        raise SomphPredictorRuntimeError("SOMP-H deployment state exceeds method cap")
    return {
        "class_count": class_count,
        "registered_class_handles": list(binding["registered_class_handles"]),
        "enrollment_binding": binding,
        "enrollment_binding_sha256": binding_sha256,
        "feature_dim": FEATURE_DIM,
        "prototype_count_by_scenario": prototype_counts,
        "candidate_state_bytes_fp16": state_bytes,
        "active_scenario_state_bytes_fp16": state_bytes // len(FORMAL_LEO_WEAK_SCENARIOS),
        "candidate_extra_macs_per_query": max(per_query_macs),
        "capsule_array_bytes": int(sum(np.asarray(value).nbytes for value in capsule.values())),
    }


def _strict_zid160_forward(
    model: torch.nn.Module,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    source = np.asarray(rows, dtype=np.float32)
    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source), int(batch_size)):
            batch = torch.from_numpy(
                np.asarray(source[start : start + int(batch_size)], dtype=np.float32)
            ).to(device)
            output = model(batch)
            if isinstance(output, dict):
                feature_value = output.get("features")
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                feature_value = output[0]
            else:
                raise SomphPredictorRuntimeError(
                    "sealed ADV3B02 runtime must return features and logits"
                )
            if (
                not torch.is_tensor(feature_value)
                or feature_value.dtype != torch.float32
                or feature_value.ndim != 2
                or int(feature_value.shape[1]) != FEATURE_DIM
            ):
                raise SomphPredictorRuntimeError(
                    "sealed ADV3B02 runtime must return adv3b02_z_id160_fp32"
                )
            values = feature_value.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise SomphPredictorRuntimeError(
                    "sealed ADV3B02 runtime returned non-finite z_id160"
                )
            features.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(features, axis=0)


def enroll_somph_heads(
    model: torch.nn.Module,
    support_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    *,
    enrollment_binding: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit one role-symmetric head per LEO scenario from registered support only."""

    lock = validate_somph_method_lock(method_lock)
    input_binding = dict(enrollment_binding)
    for field in (
        "support_token_sha256_by_scenario",
        "support_feature_sha256_by_scenario",
    ):
        if field in input_binding:
            raise SomphPredictorRuntimeError(
                "SOMP-H caller must not predeclare runtime-derived support digests"
            )
    input_keys = {
        "schema",
        "stage",
        "registration_state",
        "receiver",
        "seed",
        "k_shot",
        "registered_class_handles",
        "enrollment_package_root_sha256",
        "enrollment_package_seal_sha256",
        "checkpoint_sha256",
        "method_lock_sha256",
    }
    if set(input_binding) != input_keys:
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment input binding exact schema drift"
        )
    k_shot = input_binding["k_shot"]
    registered_class_count = len(input_binding["registered_class_handles"])
    if input_binding["method_lock_sha256"] != canonical_sha256(lock):
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment input/method lock mismatch"
        )
    if input_binding["checkpoint_sha256"] != ADV3B02_CHECKPOINT_SHA256:
        raise SomphPredictorRuntimeError(
            "SOMP-H enrollment input/checkpoint mismatch"
        )
    validate_enrollment_binding(
        {
            **input_binding,
            "support_token_sha256_by_scenario": {
                scenario: "0" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
            "support_feature_sha256_by_scenario": {
                scenario: "0" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
        }
    )
    if tuple(support_by_scenario) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphPredictorRuntimeError("SOMP-H enrollment scenario registry drift")
    if (
        not isinstance(k_shot, int)
        or isinstance(k_shot, bool)
        or k_shot < 1
        or k_shot > 20
    ):
        raise SomphPredictorRuntimeError("SOMP-H K must be in the sealed K20 pool")
    if registered_class_count < 2:
        raise SomphPredictorRuntimeError("SOMP-H registry must contain at least two classes")
    support_token_sha256_by_scenario: dict[str, str] = {}
    support_feature_sha256_by_scenario: dict[str, str] = {}
    packed_by_scenario: dict[str, dict[str, np.ndarray]] = {}
    elapsed: dict[str, float] = {}
    forward_counts: dict[str, int] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        started = time.perf_counter()
        support = support_by_scenario[scenario]
        if set(support) != _SUPPORT_PAYLOAD_KEYS:
            raise SomphPredictorRuntimeError(
                "SOMP-H support payload exact schema drift"
            )
        labels = np.asarray(support["support_class_indices"])
        ranks = np.asarray(support["support_rank_within_class"])
        iq = np.asarray(support["support_leo_weak_iq"])
        if (
            iq.dtype != np.float32
            or iq.ndim != 3
            or iq.shape[1] != 2
            or not np.isfinite(iq).all()
        ):
            raise SomphPredictorRuntimeError(
                "SOMP-H support IQ shape/dtype drift"
            )
        if labels.dtype != np.int64 or ranks.dtype != np.int64:
            raise SomphPredictorRuntimeError(
                "SOMP-H support class/rank dtype drift"
            )
        tokens = _validate_runtime_sample_metadata(
            support,
            token_key="support_tokens",
            token_prefix="sid_",
            overlay_key="support_overlay_tokens",
            seed_key="support_satellite_seeds",
            iq_sha_key="support_post_channel_iq_sha256",
            expected_count=len(iq),
            context="support",
        )
        mask = ranks < int(k_shot)
        expected = [
            (class_index, rank)
            for class_index in range(registered_class_count)
            for rank in range(int(k_shot))
        ]
        if (
            labels.shape != ranks.shape
            or labels.shape != (len(iq),)
            or tokens.shape != (len(iq),)
            or list(zip(labels[mask].tolist(), ranks[mask].tolist())) != expected
        ):
            raise SomphPredictorRuntimeError(
                "SOMP-H support payload is not the exact nested K prefix"
            )
        support_iq = iq[mask]
        support_labels = labels[mask]
        support_tokens = tokens[mask]
        features = _strict_zid160_forward(
            model, support_iq, device=device, batch_size=batch_size
        )
        packed = _fit_head(
            features,
            support_labels,
            class_count=registered_class_count,
            method_lock=lock,
        )
        packed_by_scenario[scenario] = packed
        support_token_sha256_by_scenario[scenario] = _ordered_sha256(
            support_tokens
        )
        support_feature_sha256_by_scenario[scenario] = _array_sha256(features)
        elapsed[scenario] = float(time.perf_counter() - started)
        forward_counts[scenario] = int(len(support_iq))
    bound = validate_enrollment_binding(
        {
            **input_binding,
            "support_token_sha256_by_scenario": support_token_sha256_by_scenario,
            "support_feature_sha256_by_scenario": support_feature_sha256_by_scenario,
        }
    )
    capsule: dict[str, np.ndarray] = {
        "schema_utf8": _utf8(SOMPH_HEAD_CAPSULE_SCHEMA),
        "method_lock_sha256_utf8": _utf8(canonical_sha256(lock)),
        "enrollment_binding_json_utf8": _utf8(
            json.dumps(
                bound,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "class_count_uint16": np.asarray([registered_class_count], dtype=np.uint16),
        "feature_dim_uint16": np.asarray([FEATURE_DIM], dtype=np.uint16),
        "k_shot_uint16": np.asarray([k_shot], dtype=np.uint16),
    }
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for name in _HEAD_TENSOR_NAMES:
            capsule[_head_prefix(scenario, name)] = packed_by_scenario[scenario][name]
    resource = validate_somph_head_capsule(
        capsule,
        method_lock=lock,
        expected_enrollment_binding_sha256=canonical_sha256(bound),
    )
    return capsule, {
        "schema": "cvs.phase2.somph_enrollment_resource_receipt.v1",
        "trainable_parameters": 0,
        "updated_original_parameters": 0,
        "adaptation_epochs": 0,
        "optimizer_steps": 0,
        "query_rows_used_for_fit": 0,
        "enrollment_binding_sha256": canonical_sha256(bound),
        "support_enrollment_backbone_forwards_by_scenario": forward_counts,
        "support_enrollment_latency_ms_by_scenario": {
            key: value * 1000.0 for key, value in elapsed.items()
        },
        **resource,
    }


def _score_head(features: np.ndarray, head: Mapping[str, np.ndarray]) -> np.ndarray:
    prototypes = _normalize(np.asarray(head["prototypes_fp16"], dtype=np.float32))
    prototype_ids = np.asarray(
        head["prototype_class_ids_uint16"], dtype=np.int64
    )
    centroids = _normalize(np.asarray(head["centroids_fp16"], dtype=np.float32))
    scale = np.asarray(head["residual_scale_fp16"], dtype=np.float32)
    penalty = np.asarray(head["class_hubness_penalty_fp16"], dtype=np.float32)
    max_mix = float(np.asarray(head["scalars_fp16"], dtype=np.float32)[0])
    rows = _normalize(np.asarray(features, dtype=np.float32) * scale[None, :])
    prototype_scores = rows @ prototypes.T
    centroid_scores = rows @ centroids.T
    scores = np.empty((len(rows), len(centroids)), dtype=np.float32)
    for class_index in range(len(centroids)):
        selected = prototype_scores[:, prototype_ids == class_index]
        scores[:, class_index] = (
            max_mix * np.max(selected, axis=1)
            + (1.0 - max_mix) * centroid_scores[:, class_index]
            - penalty[class_index]
        )
    return scores


def apply_somph_heads(
    model: torch.nn.Module,
    query_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    capsule: Mapping[str, np.ndarray],
    *,
    registered_class_handles: list[str],
    expected_enrollment_binding_sha256: str,
    method_lock: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the sealed head independently to every query over one registry."""

    if batch_size != 1:
        raise SomphPredictorRuntimeError(
            "SOMP-H formal apply requires singleton backbone forwards"
        )
    lock = validate_somph_method_lock(method_lock)
    resource = validate_somph_head_capsule(
        capsule,
        method_lock=lock,
        expected_enrollment_binding_sha256=expected_enrollment_binding_sha256,
    )
    if list(registered_class_handles) != resource["registered_class_handles"]:
        raise SomphPredictorRuntimeError(
            "SOMP-H apply registry does not match enrollment capsule"
        )
    registered_class_count = len(registered_class_handles)
    if tuple(query_by_scenario) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphPredictorRuntimeError("SOMP-H apply scenario registry drift")
    tokens_out: list[np.ndarray] = []
    scenarios_out: list[np.ndarray] = []
    predictions_out: list[np.ndarray] = []
    counts_out: list[np.ndarray] = []
    latency_mean: dict[str, float] = {}
    latency_p95: dict[str, float] = {}
    latency_max: dict[str, float] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        query = query_by_scenario[scenario]
        if set(query) != _QUERY_PAYLOAD_KEYS:
            raise SomphPredictorRuntimeError(
                "SOMP-H query payload exact schema drift"
            )
        rows = np.asarray(query["query_leo_weak_iq"])
        if (
            rows.dtype != np.float32
            or rows.ndim != 3
            or rows.shape[1] != 2
            or not np.isfinite(rows).all()
        ):
            raise SomphPredictorRuntimeError("SOMP-H query IQ shape/dtype drift")
        tokens = _validate_runtime_sample_metadata(
            query,
            token_key="query_tokens",
            token_prefix="qid_",
            overlay_key="query_overlay_tokens",
            seed_key="query_satellite_seeds",
            iq_sha_key="query_post_channel_iq_sha256",
            expected_count=len(rows),
            context="query",
        )
        head = _head_from_capsule(
            capsule, scenario=scenario, class_count=registered_class_count
        )
        predictions = np.empty(len(rows), dtype=np.int64)
        singleton_latency: list[float] = []
        for index in range(len(rows)):
            started = time.perf_counter()
            feature = _strict_zid160_forward(
                model,
                rows[index : index + 1],
                device=device,
                batch_size=1,
            )
            predictions[index] = int(
                np.argmax(_score_head(feature, head), axis=1)[0]
            )
            singleton_latency.append(
                float((time.perf_counter() - started) * 1000.0)
            )
        latency_values = np.asarray(singleton_latency, dtype=np.float64)
        latency_mean[scenario] = float(np.mean(latency_values))
        latency_p95[scenario] = float(np.percentile(latency_values, 95))
        latency_max[scenario] = float(np.max(latency_values))
        tokens_out.append(tokens)
        scenarios_out.append(np.asarray([scenario] * len(tokens)))
        predictions_out.append(predictions)
        counts_out.append(np.ones(len(tokens), dtype=np.uint8))
    return {
        "query_tokens": np.concatenate(tokens_out),
        "scenarios": np.concatenate(scenarios_out),
        "predicted_class_indices": np.concatenate(predictions_out),
        "backbone_forward_counts": np.concatenate(counts_out),
    }, {
        "schema": "cvs.phase2.somph_apply_resource_receipt.v1",
        "latency_measurement_policy": (
            "singleton_backbone_plus_singleton_head_end_to_end"
        ),
        "query_latency_ms_by_scenario": latency_mean,
        "query_p95_latency_ms_by_scenario": latency_p95,
        "query_max_latency_ms_by_scenario": latency_max,
        "mean_query_latency_ms": float(np.mean(list(latency_mean.values()))),
        "mean_backbone_forward_count": 1.0,
        "p95_backbone_forward_count": 1,
        "max_backbone_forward_count": 1,
        "single_view_only": True,
        **resource,
    }


def assert_role_oracle_free_public_api() -> None:
    forbidden = {
        "old_class_count",
        "new_class_count",
        "query_labels",
        "query_roles",
        "query_per_tx",
        "class_quota",
        "batch_assignment",
    }
    for function in (enroll_somph_heads, apply_somph_heads):
        code = function.__code__
        parameter_count = code.co_argcount + code.co_kwonlyargcount
        overlap = forbidden & set(code.co_varnames[:parameter_count])
        if overlap:
            raise SomphPredictorRuntimeError(
                f"SOMP-H public API exposes forbidden role/query controls: {sorted(overlap)}"
            )


__all__ = [
    "ADV3B02_CHECKPOINT_SHA256",
    "FEATURE_DIM",
    "SOMPH_HEAD_CAPSULE_SCHEMA",
    "SOMPH_ENROLLMENT_BINDING_SCHEMA",
    "SOMPH_METHOD_LOCK_SCHEMA",
    "SomphPredictorRuntimeError",
    "apply_somph_heads",
    "assert_role_oracle_free_public_api",
    "canonical_sha256",
    "enroll_somph_heads",
    "expected_somph_method_lock",
    "somph_head_capsule_members",
    "validate_somph_head_capsule",
    "validate_enrollment_binding",
    "validate_somph_method_lock",
]
