"""Strict single-stream Stage2-B/C SOMP-H deployment capsules.

The global method lock fixes a pure, zero-gradient SOMP-H head over the sealed
ADV3B02 z_id160 runtime.  Receiver/seed/K/new-count live in a separate row
manifest.  Pre-registration and post-registration enrollment are independent:
the former accepts only old-class support, while the latter accepts all
registered classes.  Baselines are separate prediction artifacts; this module
never hides an adapter or baseline stream inside the SOMP-H resource claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np

from paper_reproduction.cvs_aligned.support_only_multiprototype_head import (
    PACKED_HEAD_KEYS,
    fit_support_only_multiprototype_head,
    pack_support_only_multiprototype_head,
    predict_support_only_multiprototype_head,
    unpack_support_only_multiprototype_head,
)


METHOD_LOCK_SCHEMA = "cvs.phase2.somph_method_lock.v1"
ROW_MANIFEST_SCHEMA = "cvs.phase2.somph_row_manifest.v1"
STAGE_INPUT_SCHEMA = "cvs.phase2.somph_stage_input_binding.v1"
STAGE_HEAD_SCHEMA = "cvs.phase2.somph_stage_head_capsule.v1"
REGISTRATION_PAIR_SCHEMA = "cvs.phase2.somph_registration_pair.v1"
K_FAMILY_SCHEMA = "cvs.phase2.somph_k_family.v1"
FORMAL_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
FORMAL_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
FORMAL_OLD_CLASSES = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
FORMAL_NEW20 = (
    "1-16", "1-18", "18-10", "14-11", "8-3",
    "18-8", "10-10", "16-19", "20-12", "4-10",
    "13-14", "2-5", "1-8", "19-13", "19-9",
    "3-8", "19-8", "11-19", "2-16", "19-6",
)
ADV3B02_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
STAGES = ("before_registration", "after_registration")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_SUPPORT_TOKEN = re.compile(r"^sid_[0-9a-f]{64}$")
_OPAQUE_QUERY_TOKEN = re.compile(r"^qid_[0-9a-f]{64}$")


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


def _utf8(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(value: np.ndarray, *, field: str) -> str:
    rows = np.asarray(value)
    if rows.dtype != np.uint8 or rows.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional uint8 array")
    try:
        return rows.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} is not valid UTF-8") from exc


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_values_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in values], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def opaque_class_handle(value: str) -> str:
    payload = f"SOMPH_V1_CLASS::{value}".encode("utf-8")
    return f"cls_{hashlib.sha256(payload).hexdigest()}"


def array_sha256(value: np.ndarray) -> str:
    rows = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(rows.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(rows.shape), separators=(",", ":")).encode("ascii"))
    digest.update(rows.tobytes(order="C"))
    return digest.hexdigest()


def _validate_opaque_tokens(values: np.ndarray, *, kind: str) -> np.ndarray:
    tokens = np.asarray(values).astype(str)
    pattern = _OPAQUE_SUPPORT_TOKEN if kind == "support" else _OPAQUE_QUERY_TOKEN
    if tokens.ndim != 1 or not len(tokens) or len(set(tokens.tolist())) != len(tokens):
        raise ValueError(f"SOMP-H opaque {kind} token layout drift")
    if any(pattern.fullmatch(value) is None for value in tokens.tolist()):
        raise ValueError(f"SOMP-H {kind} tokens are not opaque")
    return tokens


def _head_payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in PACKED_HEAD_KEYS:
        value = np.ascontiguousarray(np.asarray(payload[key]))
        digest.update(key.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def expected_method_lock() -> dict[str, Any]:
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "method_id": "SOMPH_V1",
        "base_model_id": "ADV3B02_CORE90_SOFT_E200",
        "checkpoint_sha256": ADV3B02_SHA256,
        "feature_schema": "adv3b02_z_id160_fp32",
        "feature_dim": 160,
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


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_method_lock()
    if not isinstance(lock, dict) or set(lock) != set(expected):
        raise ValueError("SOMP-H method lock exact schema drift")
    failed = [key for key, value in expected.items() if lock.get(key) != value]
    if failed:
        raise ValueError(f"SOMP-H method lock drift: {failed}")
    return json.loads(json.dumps(dict(lock), sort_keys=True))


def validate_row_manifest(
    row: Mapping[str, Any], *, method_lock_sha256: str
) -> dict[str, Any]:
    expected_keys = {
        "schema", "method_lock_sha256", "split_role", "receiver", "seed",
        "k_shot", "new_class_count", "support_pool_max_k", "scenarios",
    }
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ValueError("SOMP-H row manifest exact schema drift")
    if row.get("schema") != ROW_MANIFEST_SCHEMA:
        raise ValueError("SOMP-H row manifest schema drift")
    if row.get("method_lock_sha256") != method_lock_sha256:
        raise ValueError("SOMP-H row/method lock hash drift")
    receiver = str(row.get("receiver"))
    seed = int(row.get("seed", -1))
    k_shot = int(row.get("k_shot", -1))
    split_role = str(row.get("split_role"))
    if receiver not in FORMAL_RECEIVERS:
        raise ValueError("SOMP-H row receiver is outside the formal target set")
    if split_role == "development":
        if (receiver, seed, k_shot) != ("20-1", 713101, 10):
            raise ValueError("SOMP-H development row must use the locked K10 unit")
    elif split_role == "confirmation":
        if seed not in {713102, 713103, 713104, 713105, 713106} or k_shot not in {1, 5, 10, 20}:
            raise ValueError("SOMP-H confirmation row seed/K is outside the independent lock")
    else:
        raise ValueError("SOMP-H split_role must be development or confirmation")
    new_count = int(row.get("new_class_count", -1))
    if new_count not in {0, 5, 10, 20}:
        raise ValueError("SOMP-H new class count must be 0, 5, 10, or 20")
    if int(row.get("support_pool_max_k", -1)) != 20:
        raise ValueError("SOMP-H row must use the formal K20 support pool")
    if tuple(str(value) for value in row.get("scenarios", [])) != FORMAL_SCENARIOS:
        raise ValueError("SOMP-H row scenario registry drift")
    return json.loads(json.dumps(dict(row), sort_keys=True))


def validate_stage_input_binding(
    binding: Mapping[str, Any], *, row_manifest_sha256: str
) -> dict[str, Any]:
    expected_keys = {
        "schema", "stage", "row_manifest_sha256", "sealed_package_sha256",
        "preopen_audit_sha256", "runtime_access_audit_policy_sha256",
        "feature_runtime_sha256", "registered_class_order_sha256",
        "support_class_handles", "support_pool_ids_sha256_by_scenario",
        "support_ids_sha256_by_scenario", "support_label_sequence_sha256_by_scenario",
        "support_feature_sha256_by_scenario", "satellite_seed_by_scenario",
        "support_prefix_policy", "support_query_overlap_count",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ValueError("SOMP-H stage input binding exact schema drift")
    if binding.get("schema") != STAGE_INPUT_SCHEMA:
        raise ValueError("SOMP-H stage input binding schema drift")
    stage = str(binding.get("stage"))
    if stage not in STAGES:
        raise ValueError("SOMP-H stage input binding stage drift")
    if binding.get("row_manifest_sha256") != row_manifest_sha256:
        raise ValueError("SOMP-H stage/row manifest hash drift")
    for field in (
        "sealed_package_sha256", "preopen_audit_sha256",
        "runtime_access_audit_policy_sha256", "feature_runtime_sha256",
        "registered_class_order_sha256",
    ):
        if _SHA256.fullmatch(str(binding.get(field))) is None:
            raise ValueError(f"SOMP-H invalid stage input digest: {field}")
    digest_maps = (
        "support_pool_ids_sha256_by_scenario", "support_ids_sha256_by_scenario",
        "support_label_sequence_sha256_by_scenario", "support_feature_sha256_by_scenario",
    )
    for field in digest_maps:
        values = binding.get(field)
        if not isinstance(values, dict) or tuple(values) != FORMAL_SCENARIOS:
            raise ValueError(f"SOMP-H {field} scenario registry drift")
        if any(_SHA256.fullmatch(str(values[scenario])) is None for scenario in FORMAL_SCENARIOS):
            raise ValueError(f"SOMP-H {field} digest drift")
    seeds = binding.get("satellite_seed_by_scenario")
    if not isinstance(seeds, dict) or tuple(seeds) != FORMAL_SCENARIOS:
        raise ValueError("SOMP-H satellite seed scenario registry drift")
    if any(int(seeds[scenario]) < 0 for scenario in FORMAL_SCENARIOS):
        raise ValueError("SOMP-H satellite seed must be nonnegative")
    for field in (
        "support_pool_ids_sha256_by_scenario", "support_ids_sha256_by_scenario",
        "support_label_sequence_sha256_by_scenario",
    ):
        if len(set(binding[field].values())) != 1:
            raise ValueError(f"SOMP-H {field} must bind the same physical rows across scenarios")
    handles = [str(value) for value in binding.get("support_class_handles", [])]
    if not handles or len(set(handles)) != len(handles):
        raise ValueError("SOMP-H stage support class registry drift")
    if binding.get("registered_class_order_sha256") != ordered_values_sha256(handles):
        raise ValueError("SOMP-H registered class order digest drift")
    if binding.get("support_prefix_policy") != "rank_lt_k_from_locked_k20_pool":
        raise ValueError("SOMP-H K-prefix policy drift")
    if int(binding.get("support_query_overlap_count", -1)) != 0:
        raise ValueError("SOMP-H support/query overlap must be zero")
    return json.loads(json.dumps(dict(binding), sort_keys=True))


def _prefixed(prefix: str, payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}__{key}": np.asarray(payload[key]) for key in PACKED_HEAD_KEYS}


def _unprefix(capsule: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {key: np.asarray(capsule[f"{prefix}__{key}"]) for key in PACKED_HEAD_KEYS}


def stage_head_members() -> list[str]:
    output = [
        "schema_utf8", "stage_utf8", "class_handles_json_utf8",
        "method_lock_sha256_utf8", "row_manifest_sha256_utf8",
        "stage_input_binding_sha256_utf8",
    ]
    for scenario in FORMAL_SCENARIOS:
        output.append(f"head_payload_sha256__{scenario}_utf8")
        output.extend(f"head__{scenario}__{key}" for key in PACKED_HEAD_KEYS)
    return output


def build_stage_head_capsule(
    *,
    features_by_scenario: Mapping[str, np.ndarray],
    support_tokens_by_scenario: Mapping[str, np.ndarray],
    support_pool_tokens_by_scenario: Mapping[str, np.ndarray],
    support_pool_labels: np.ndarray,
    support_pool_ranks: np.ndarray,
    method_lock: Mapping[str, Any],
    row_manifest: Mapping[str, Any],
    stage_input_binding: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    lock = validate_method_lock(method_lock)
    method_hash = canonical_sha256(lock)
    row = validate_row_manifest(row_manifest, method_lock_sha256=method_hash)
    row_hash = canonical_sha256(row)
    binding = validate_stage_input_binding(stage_input_binding, row_manifest_sha256=row_hash)
    stage = str(binding["stage"])
    if int(row["new_class_count"]) == 0 and stage != "before_registration":
        raise ValueError("SOMP-H Stage2-B row only supports before_registration")
    expected_handles = list(FORMAL_OLD_CLASSES)
    if stage == "after_registration":
        expected_handles.extend(FORMAL_NEW20[: int(row["new_class_count"])])
    handles = [str(value) for value in binding["support_class_handles"]]
    if handles != expected_handles:
        raise ValueError("SOMP-H stage cannot access classes outside its registration state")
    pool_labels = np.asarray(support_pool_labels, dtype=np.int64).reshape(-1)
    pool_ranks = np.asarray(support_pool_ranks, dtype=np.int64).reshape(-1)
    if pool_labels.shape != pool_ranks.shape or pool_labels.size == 0:
        raise ValueError("SOMP-H K20 support pool label/rank layout drift")
    if pool_labels.min(initial=0) < 0 or pool_labels.max(initial=-1) >= len(handles):
        raise ValueError("SOMP-H support pool labels are outside the stage registry")
    for class_id in range(len(handles)):
        selected_ranks = np.sort(pool_ranks[pool_labels == class_id])
        if not np.array_equal(selected_ranks, np.arange(20, dtype=np.int64)):
            raise ValueError("SOMP-H support pool must contain ranks 0..19 per class")
    selected_mask = pool_ranks < int(row["k_shot"])
    labels = pool_labels[selected_mask]
    if not np.all(np.bincount(labels, minlength=len(handles)) == int(row["k_shot"])):
        raise ValueError("SOMP-H stage support must contain exactly K physical rows per class")
    label_sequence = [handles[int(index)] for index in labels.tolist()]
    label_digest = ordered_values_sha256(label_sequence)
    if any(
        binding["support_label_sequence_sha256_by_scenario"][scenario] != label_digest
        for scenario in FORMAL_SCENARIOS
    ):
        raise ValueError("SOMP-H support token/label sequence digest drift")
    for name, mapping in (
        ("support feature", features_by_scenario),
        ("support token", support_tokens_by_scenario),
        ("support pool token", support_pool_tokens_by_scenario),
    ):
        if tuple(mapping) != FORMAL_SCENARIOS:
            raise ValueError(f"SOMP-H {name} scenario registry drift")
    capsule: dict[str, np.ndarray] = {
        "schema_utf8": _utf8(STAGE_HEAD_SCHEMA),
        "stage_utf8": _utf8(stage),
        "class_handles_json_utf8": _utf8(
            json.dumps(handles, ensure_ascii=True, separators=(",", ":"))
        ),
        "method_lock_sha256_utf8": _utf8(method_hash),
        "row_manifest_sha256_utf8": _utf8(row_hash),
        "stage_input_binding_sha256_utf8": _utf8(canonical_sha256(binding)),
    }
    fit_kwargs = {
        "max_prototypes_per_class": int(lock["max_prototypes_per_class"]),
        "residual_shrinkage": float(lock["residual_shrinkage"]),
        "residual_scale_min": float(lock["residual_scale_min"]),
        "residual_scale_max": float(lock["residual_scale_max"]),
        "max_mix": float(lock["max_mix"]),
        "hubness_weight": float(lock["hubness_weight"]),
    }
    for scenario in FORMAL_SCENARIOS:
        pool_tokens = _validate_opaque_tokens(
            support_pool_tokens_by_scenario[scenario], kind="support"
        )
        support_tokens = _validate_opaque_tokens(
            support_tokens_by_scenario[scenario], kind="support"
        )
        if len(pool_tokens) != len(pool_labels):
            raise ValueError("SOMP-H support pool token layout drift")
        if len(support_tokens) != len(labels):
            raise ValueError("SOMP-H selected support token layout drift")
        expected_tokens = pool_tokens[selected_mask]
        if not np.array_equal(support_tokens, expected_tokens):
            raise ValueError("SOMP-H selected support is not the locked rank<K prefix")
        if ordered_values_sha256(pool_tokens.tolist()) != binding[
            "support_pool_ids_sha256_by_scenario"
        ][scenario]:
            raise ValueError("SOMP-H support pool token digest/order drift")
        if ordered_values_sha256(support_tokens.tolist()) != binding[
            "support_ids_sha256_by_scenario"
        ][scenario]:
            raise ValueError("SOMP-H selected support token digest/order drift")
        rows = np.asarray(features_by_scenario[scenario], dtype=np.float32)
        if rows.shape != (len(labels), int(lock["feature_dim"])):
            raise ValueError("SOMP-H support feature layout/schema drift")
        if array_sha256(rows) != binding["support_feature_sha256_by_scenario"][scenario]:
            raise ValueError("SOMP-H support feature tensor digest drift")
        head = fit_support_only_multiprototype_head(
            rows, labels, class_count=len(handles), **fit_kwargs
        )
        packed = pack_support_only_multiprototype_head(head)
        capsule[f"head_payload_sha256__{scenario}_utf8"] = _utf8(
            _head_payload_sha256(packed)
        )
        capsule.update(_prefixed(f"head__{scenario}", packed))
    return capsule


def _validate_head_against_lock(
    head_payload: Mapping[str, np.ndarray],
    *,
    lock: Mapping[str, Any],
    class_count: int,
    k_shot: int,
) -> None:
    head = unpack_support_only_multiprototype_head(dict(head_payload))
    expected_per_class = min(int(lock["max_prototypes_per_class"]), int(k_shot))
    counts = np.bincount(head.prototype_class_ids, minlength=class_count)
    if head.feature_dim != int(lock["feature_dim"]) or head.class_count != class_count:
        raise ValueError("SOMP-H head feature/class schema drift")
    if not np.all(counts == expected_per_class):
        raise ValueError("SOMP-H head prototype count drift")
    tolerance = 2.0e-3
    if np.min(head.residual_scale) < float(lock["residual_scale_min"]) - tolerance:
        raise ValueError("SOMP-H residual scale is below the method lock")
    if np.max(head.residual_scale) > float(lock["residual_scale_max"]) + tolerance:
        raise ValueError("SOMP-H residual scale exceeds the method lock")
    if abs(head.max_mix - float(lock["max_mix"])) > tolerance:
        raise ValueError("SOMP-H max_mix drift")
    if abs(head.hubness_weight - float(lock["hubness_weight"])) > tolerance:
        raise ValueError("SOMP-H hubness weight drift")
    expected_audit = {
        "fit_scope": "registered_support_only",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_true_batch_class_count_used": False,
        "query_class_quota_used": False,
        "query_global_assignment_used": False,
        "dense_query_graph_used": False,
        "role_symmetric_rule": True,
        "class_count": class_count,
        "min_physical_support_per_class": int(k_shot),
        "prototypes_per_class": expected_per_class,
        "residual_shrinkage": float(lock["residual_shrinkage"]),
        "residual_scale_min": float(lock["residual_scale_min"]),
        "residual_scale_max": float(lock["residual_scale_max"]),
        "max_mix": float(lock["max_mix"]),
        "hubness_weight": float(lock["hubness_weight"]),
    }
    if head.support_audit != expected_audit:
        raise ValueError("SOMP-H support audit/method lock drift")


def validate_stage_head_capsule(
    capsule: Mapping[str, np.ndarray],
    *,
    method_lock: Mapping[str, Any],
    row_manifest: Mapping[str, Any],
    stage_input_binding: Mapping[str, Any],
) -> tuple[str, list[str]]:
    lock = validate_method_lock(method_lock)
    method_hash = canonical_sha256(lock)
    row = validate_row_manifest(row_manifest, method_lock_sha256=method_hash)
    row_hash = canonical_sha256(row)
    binding = validate_stage_input_binding(stage_input_binding, row_manifest_sha256=row_hash)
    if list(capsule) != stage_head_members():
        raise ValueError("SOMP-H stage capsule exact member/order drift")
    if _decode_utf8(capsule["schema_utf8"], field="schema_utf8") != STAGE_HEAD_SCHEMA:
        raise ValueError("SOMP-H stage capsule schema drift")
    stage = _decode_utf8(capsule["stage_utf8"], field="stage_utf8")
    if stage != binding["stage"]:
        raise ValueError("SOMP-H stage capsule/input stage drift")
    expected_hashes = {
        "method_lock_sha256_utf8": method_hash,
        "row_manifest_sha256_utf8": row_hash,
        "stage_input_binding_sha256_utf8": canonical_sha256(binding),
    }
    for field, expected in expected_hashes.items():
        if _decode_utf8(capsule[field], field=field) != expected:
            raise ValueError(f"SOMP-H stage capsule hash drift: {field}")
    handles = json.loads(
        _decode_utf8(capsule["class_handles_json_utf8"], field="class_handles_json_utf8")
    )
    if handles != binding["support_class_handles"]:
        raise ValueError("SOMP-H stage capsule class registry drift")
    for scenario in FORMAL_SCENARIOS:
        payload = _unprefix(capsule, f"head__{scenario}")
        expected_payload_hash = _decode_utf8(
            capsule[f"head_payload_sha256__{scenario}_utf8"],
            field=f"head_payload_sha256__{scenario}_utf8",
        )
        if _head_payload_sha256(payload) != expected_payload_hash:
            raise ValueError("SOMP-H head payload digest drift")
        _validate_head_against_lock(
            payload,
            lock=lock,
            class_count=len(handles),
            k_shot=int(row["k_shot"]),
        )
    return stage, [str(value) for value in handles]


def stage_head_resource_audit(
    capsule: Mapping[str, np.ndarray],
    *,
    method_lock: Mapping[str, Any],
    row_manifest: Mapping[str, Any],
    stage_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    stage, _handles = validate_stage_head_capsule(
        capsule,
        method_lock=method_lock,
        row_manifest=row_manifest,
        stage_input_binding=stage_input_binding,
    )
    state_bytes = 0
    macs: list[int] = []
    for scenario in FORMAL_SCENARIOS:
        head = unpack_support_only_multiprototype_head(
            _unprefix(capsule, f"head__{scenario}")
        )
        state_bytes += head.persistent_state_bytes_fp16
        macs.append(head.extra_macs_per_query)
    serialized_array_bytes = int(sum(np.asarray(value).nbytes for value in capsule.values()))
    active_scenario_state_bytes = int(max(
        unpack_support_only_multiprototype_head(
            _unprefix(capsule, f"head__{scenario}")
        ).persistent_state_bytes_fp16
        for scenario in FORMAL_SCENARIOS
    ))
    cap = int(method_lock["persistent_state_cap_bytes"])
    if state_bytes > cap:
        raise ValueError("SOMP-H candidate deployment state exceeds the method cap")
    return {
        "stage": stage,
        "adapter_type": "none",
        "trainable_parameters": 0,
        "updated_original_parameters": 0,
        "adaptation_epochs": 0,
        "optimizer_steps": 0,
        "optimizer_state_bytes": 0,
        "optimizer_state_deployment_required": False,
        "query_rows_used_for_fit": 0,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "candidate_state_bytes_fp16": int(state_bytes),
        "active_scenario_state_bytes_fp16": active_scenario_state_bytes,
        "candidate_state_cap_bytes": cap,
        "candidate_state_within_cap": True,
        "candidate_extra_macs_per_query": int(max(macs)),
        "capsule_array_bytes_including_registry_and_audit": serialized_array_bytes,
    }


def apply_stage_head_capsule(
    *,
    scenario: str,
    query_features: np.ndarray,
    query_tokens: np.ndarray,
    capsule: Mapping[str, np.ndarray],
    method_lock: Mapping[str, Any],
    row_manifest: Mapping[str, Any],
    stage_input_binding: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if scenario not in FORMAL_SCENARIOS:
        raise ValueError("SOMP-H apply scenario drift")
    stage, handles = validate_stage_head_capsule(
        capsule,
        method_lock=method_lock,
        row_manifest=row_manifest,
        stage_input_binding=stage_input_binding,
    )
    rows = np.asarray(query_features, dtype=np.float32)
    tokens = _validate_opaque_tokens(query_tokens, kind="query")
    if rows.ndim != 2 or rows.shape[1] != int(method_lock["feature_dim"]):
        raise ValueError("SOMP-H query feature schema drift")
    if not np.all(np.isfinite(rows)):
        raise ValueError("SOMP-H query features must be finite")
    if len(tokens) != len(rows):
        raise ValueError("SOMP-H opaque query token layout drift")
    head = unpack_support_only_multiprototype_head(
        _unprefix(capsule, f"head__{scenario}")
    )
    prediction_ids = predict_support_only_multiprototype_head(rows, head)
    return {
        "stage": np.asarray([stage] * len(rows)),
        "query_tokens": tokens,
        "prediction": np.asarray(
            [opaque_class_handle(handles[int(index)]) for index in prediction_ids],
            dtype=str,
        ),
    }


def validate_registration_pair(
    pair: Mapping[str, Any],
    *,
    method_lock: Mapping[str, Any],
    row_manifest: Mapping[str, Any],
    before_binding: Mapping[str, Any],
    after_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema", "row_manifest_sha256", "before_binding_sha256",
        "after_binding_sha256", "old_support_physical_ids_sha256_before",
        "old_support_physical_ids_sha256_after", "old_query_physical_ids_sha256_before",
        "old_query_physical_ids_sha256_after",
    }
    if not isinstance(pair, dict) or set(pair) != expected_keys:
        raise ValueError("SOMP-H registration pair exact schema drift")
    lock = validate_method_lock(method_lock)
    row = validate_row_manifest(
        row_manifest, method_lock_sha256=canonical_sha256(lock)
    )
    row_hash = canonical_sha256(row)
    before = validate_stage_input_binding(before_binding, row_manifest_sha256=row_hash)
    after = validate_stage_input_binding(after_binding, row_manifest_sha256=row_hash)
    if pair.get("schema") != REGISTRATION_PAIR_SCHEMA:
        raise ValueError("SOMP-H registration pair schema drift")
    if pair.get("row_manifest_sha256") != row_hash:
        raise ValueError("SOMP-H registration pair row hash drift")
    if pair.get("before_binding_sha256") != canonical_sha256(before):
        raise ValueError("SOMP-H registration pair before hash drift")
    if pair.get("after_binding_sha256") != canonical_sha256(after):
        raise ValueError("SOMP-H registration pair after hash drift")
    if before["stage"] != "before_registration" or after["stage"] != "after_registration":
        raise ValueError("SOMP-H registration pair stage order drift")
    if before["feature_runtime_sha256"] != after["feature_runtime_sha256"]:
        raise ValueError("SOMP-H registration pair feature runtime drift")
    if before["satellite_seed_by_scenario"] != after["satellite_seed_by_scenario"]:
        raise ValueError("SOMP-H registration pair satellite seed drift")
    for suffix in ("support", "query"):
        if pair[f"old_{suffix}_physical_ids_sha256_before"] != pair[
            f"old_{suffix}_physical_ids_sha256_after"
        ]:
            raise ValueError(f"SOMP-H registration pair old {suffix} mismatch")
    return json.loads(json.dumps(dict(pair), sort_keys=True))


def validate_k_family(
    family: Mapping[str, Any],
    *,
    method_lock: Mapping[str, Any],
    row_manifests_by_k: Mapping[int, Mapping[str, Any]],
    bindings_by_k: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {"schema", "stage", "row_manifest_sha256_by_k", "binding_sha256_by_k"}
    if not isinstance(family, dict) or set(family) != expected_keys:
        raise ValueError("SOMP-H K-family exact schema drift")
    if family.get("schema") != K_FAMILY_SCHEMA or family.get("stage") not in STAGES:
        raise ValueError("SOMP-H K-family schema/stage drift")
    ks = (1, 5, 10, 20)
    if tuple(sorted(row_manifests_by_k)) != ks or tuple(sorted(bindings_by_k)) != ks:
        raise ValueError("SOMP-H K-family must contain K1/K5/K10/K20")
    lock = validate_method_lock(method_lock)
    reference_row: dict[str, Any] | None = None
    reference_binding: dict[str, Any] | None = None
    for k_shot in ks:
        row = validate_row_manifest(
            row_manifests_by_k[k_shot], method_lock_sha256=canonical_sha256(lock)
        )
        binding = validate_stage_input_binding(
            bindings_by_k[k_shot], row_manifest_sha256=canonical_sha256(row)
        )
        if int(row["k_shot"]) != k_shot or binding["stage"] != family["stage"]:
            raise ValueError("SOMP-H K-family row K/stage drift")
        if str(k_shot) not in family["row_manifest_sha256_by_k"] or str(k_shot) not in family[
            "binding_sha256_by_k"
        ]:
            raise ValueError("SOMP-H K-family digest registry drift")
        if family["row_manifest_sha256_by_k"][str(k_shot)] != canonical_sha256(row):
            raise ValueError("SOMP-H K-family row digest drift")
        if family["binding_sha256_by_k"][str(k_shot)] != canonical_sha256(binding):
            raise ValueError("SOMP-H K-family binding digest drift")
        if reference_row is None:
            reference_row, reference_binding = row, binding
            continue
        for field in (
            "split_role", "receiver", "seed", "new_class_count",
            "support_pool_max_k", "scenarios",
        ):
            if row[field] != reference_row[field]:
                raise ValueError(f"SOMP-H K-family row mismatch: {field}")
        for field in (
            "feature_runtime_sha256", "registered_class_order_sha256",
            "support_pool_ids_sha256_by_scenario", "satellite_seed_by_scenario",
        ):
            if binding[field] != reference_binding[field]:
                raise ValueError(f"SOMP-H K-family binding mismatch: {field}")
    return json.loads(json.dumps(dict(family), sort_keys=True))
