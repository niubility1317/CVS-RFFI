"""Artifact-closed Phase1-held54 falsifier for GRB-JP4-CFM-qKNN-D92.

This module is deliberately non-promotable.  It builds one coverage-selected
held-receiver matrix, seals query features separately from truth, publishes
immutable predictions, and only then joins opaque query IDs in the scorer.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.grb_jp4_cfm_held_heads import (
    HeldD92State,
    d92_resource_receipt,
    fit_held_d92_head,
    score_held_d92_head,
)
from cvsrffi.grb_jp4_cfm_phase1_held_builder import (
    build_phase1_method_lock,
    build_phase1_qknn_locks,
    build_source_aggregate,
)
from cvsrffi.phase1_grb_jp4_cfm_bundle import (
    GRBJP4CFMPhase1Component,
    build_grb_jp4_cfm_component,
    canonical_array_sha256,
    class_handle_binding_sha256,
)
from cvsrffi import stage2_grb_jp4_cfm_d92 as cfm
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    deserialize_typed_zid_runtime_state,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
)


SCOPE = "PHASE1_HELD_PROXY_NON_PROMOTABLE"
SCHEMA = "cvs.phase1.grb_jp4_cfm.held54_falsifier.v1"
CANDIDATE = "GRB-JP4-CFM-qKNN-D92/r2-sharedK1"
K_VALUES = (1, 5, 10)
SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
ARMS = ("M0", "M92", "M_DA", "M_DA92")
COUNTERFACTUALS = (
    "ground_off",
    "tx_permuted",
    "equal_energy_random_q4",
)
GROUND_VARIANTS = ("real_q4", "tx_permuted", "equal_energy_random_q4")
ROW_COUNT = 6 * len(SCENES) * len(K_VALUES)
COUNTERFACTUAL_SEED = 60720260724
STATE_LIMIT_BYTES = 262_144
POST_BACKBONE_MAC_LIMIT = 262_144
PACKET_NAME = "held54.packet.json"
QUERY_NAME = "held54.query.npz"
TRUTH_NAME = "held54.truth.json"
BUILD_RECEIPT_NAME = "held54.build-receipt.json"
ROW_SHARD_RECEIPT_NAME = "held54.row-shard-receipt.json"
PREDICTION_NAME = "held54.prediction.json"
SCORE_NAME = "held54.score.json"
BUILD_RECEIPT_SCHEMA = SCHEMA + ".build-receipt.v1"
ROW_SHARD_SCHEMA = SCHEMA + ".row-shard.v1"
ROW_SHARD_RECEIPT_SCHEMA = ROW_SHARD_SCHEMA + ".receipt.v1"
Z_DIM = 160
HIDDEN_DIM = 320


class GRBJP4HeldError(ValueError):
    """Raised when the held54 artifact or scientific closure drifts."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canon(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canon(value)
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GRBJP4HeldError(f"{name} must be a lowercase SHA256")
    return value


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.ndim > 0:
        array = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "sha256": canonical_array_sha256(array),
    }


def _decode_array(wire: Mapping[str, Any]) -> np.ndarray:
    if set(wire) != {"dtype", "shape", "data_b64", "sha256"}:
        raise GRBJP4HeldError("array wire allowlist drift")
    try:
        dtype = np.dtype(str(wire["dtype"]))
        shape = tuple(int(value) for value in wire["shape"])
        raw = base64.b64decode(str(wire["data_b64"]), validate=True)
        result = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
    except (ValueError, TypeError, KeyError) as exc:
        raise GRBJP4HeldError("array wire codec drift") from exc
    if canonical_array_sha256(result) != wire["sha256"]:
        raise GRBJP4HeldError("array wire SHA256 drift")
    return result if result.ndim == 0 else np.ascontiguousarray(result)


def _validate_archive(archive: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "z_id",
        "hidden",
        "pre_relu",
        "joint_weight",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "class_ids",
        "observation_ids",
    }
    if set(archive) != required:
        raise GRBJP4HeldError("tap archive member allowlist drift")
    result = {name: np.asarray(value) for name, value in archive.items()}
    count = len(result["labels"])
    for name, width in (
        ("z_id", Z_DIM),
        ("hidden", HIDDEN_DIM),
        ("pre_relu", Z_DIM),
    ):
        value = result[name]
        if (
            value.dtype != np.float32
            or value.shape != (count, width)
            or not np.isfinite(value).all()
        ):
            raise GRBJP4HeldError(f"tap {name} contract drift")
    recomputed = np.maximum(result["pre_relu"], np.float32(0.0))
    if not np.array_equal(result["z_id"], recomputed):
        raise GRBJP4HeldError("tap z_id must be byte-exact ReLU(pre_relu)")
    weight = result["joint_weight"]
    if (
        weight.dtype != np.float32
        or weight.shape != (Z_DIM, HIDDEN_DIM)
        or not np.isfinite(weight).all()
    ):
        raise GRBJP4HeldError("tap joint weight contract drift")
    for name in (
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
    ):
        value = result[name]
        if (
            value.ndim != 1
            or len(value) != count
            or any(not item for item in value.astype(str).tolist())
        ):
            raise GRBJP4HeldError(f"tap metadata contract drift: {name}")
    classes = tuple(result["class_ids"].astype(str).tolist())
    if (
        len(classes) != 6
        or len(set(classes)) != 6
        or set(classes) != set(result["labels"].astype(str).tolist())
        or len(set(result["physical_ids"].astype(str).tolist())) != count
        or len(set(result["observation_ids"].astype(str).tolist())) != count
        or set(result["scenario_names"].astype(str).tolist()) != set(SCENES)
    ):
        raise GRBJP4HeldError("tap class/physical/scene closure drift")
    return result


def _artifact_binding(
    value: Mapping[str, Any], coverage_sha256: str
) -> dict[str, str]:
    expected = {
        "archive_schema",
        "archive_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "coverage_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise GRBJP4HeldError("artifact binding allowlist drift")
    result = {key: str(item) for key, item in value.items()}
    if result["archive_schema"] != "cvs.phase1.jp4_tap_archive.v1":
        raise GRBJP4HeldError("tap archive schema binding drift")
    for name in (
        "archive_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "coverage_sha256",
    ):
        _require_sha(result[name], name)
    if result["coverage_sha256"] != coverage_sha256:
        raise GRBJP4HeldError("coverage SHA256 binding drift")
    return result


def _held_receiver(receivers: Sequence[str], coverage_sha256: str) -> str:
    values = tuple(sorted(str(value) for value in receivers))
    if len(values) < 4 or len(set(values)) != len(values):
        raise GRBJP4HeldError("held matrix requires at least four receivers")
    index = int.from_bytes(bytes.fromhex(coverage_sha256)[:8], "big") % len(values)
    return values[index]


def _source_archive(
    archive: Mapping[str, np.ndarray], held_receiver: str
) -> dict[str, np.ndarray]:
    mask = archive["receiver_ids"].astype(str) != held_receiver
    result: dict[str, np.ndarray] = {}
    row_fields = {
        "z_id",
        "hidden",
        "pre_relu",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
    }
    for name, value in archive.items():
        result[name] = np.asarray(value[mask] if name in row_fields else value)
    if held_receiver in set(result["receiver_ids"].astype(str).tolist()):
        raise GRBJP4HeldError("held receiver leaked into Phase1 source aggregate")
    return result


def _typed_component(
    source: Mapping[str, np.ndarray],
    *,
    locks: Mapping[int, Phase1ZIDStudentTLock],
    checkpoint_sha256: str,
) -> tuple[GRBJP4CFMPhase1Component, dict[str, Any]]:
    classes = tuple(source["class_ids"].astype(str).tolist())
    binding = class_handle_binding_sha256(classes)
    method_input = build_phase1_method_lock(
        checkpoint_sha256=checkpoint_sha256,
        class_handle_binding_sha256=binding,
        qknn_locks=locks,
    )
    aggregate = build_source_aggregate(source, qknn_locks=locks)
    payload, manifest = build_grb_jp4_cfm_component(
        aggregate,
        class_registry=classes,
        checkpoint_joint_proj_weight=source["joint_weight"],
        checkpoint_sha256=checkpoint_sha256,
        class_handle_binding_sha256=binding,
        generation_code_sha256=_sha(
            {"module": SCHEMA, "source_builder": "held_source_aggregate.v1"}
        ),
        generation_config_sha256=_sha(
            {"K_values": K_VALUES, "counterfactual_seed": COUNTERFACTUAL_SEED}
        ),
        method_lock=method_input,
        provenance_status="PHASE1_HELD_SOURCE_ONLY_DEVELOPMENT",
    )
    deterministic_root = _sha(
        {
            "manifest": manifest,
            "arrays": {
                name: canonical_array_sha256(value)
                for name, value in sorted(payload.items())
            },
        }
    )
    typed_manifest = {
        **manifest,
        "pre_sign_content_root_sha256": deterministic_root,
    }
    component = GRBJP4CFMPhase1Component(
        p_g_q=np.asarray(payload["p_g_q"]),
        p_g_scale=np.asarray(payload["p_g_scale"]),
        p_g_weight=np.asarray(payload["p_g_weight"]),
        p_g_radius=np.asarray(payload["p_g_radius"]),
        p_g_mask=np.asarray(payload["p_g_mask"]),
        p_g_physical_counts=np.asarray(payload["p_g_physical_counts"]),
        p_g_receipt_sha256=np.asarray(payload["p_g_receipt_sha256"]),
        p_g_source_prototype_sha256=np.asarray(
            payload["p_g_source_prototype_sha256"]
        ),
        p_g_quantization_max_abs_error=np.asarray(
            payload["p_g_quantization_max_abs_error"]
        ),
        p_g_quantization_certificate_sha256=np.asarray(
            payload["p_g_quantization_certificate_sha256"]
        ),
        l_g_q=np.asarray(payload["l_g_q"]),
        l_g_scale=np.asarray(payload["l_g_scale"]),
        r_q=np.asarray(payload["r_q"]),
        r_scale=np.asarray(payload["r_scale"]),
        direction_energy_a=np.asarray(payload["direction_energy_a"]),
        delta_q=float(payload["delta_q"]),
        tau_q=float(payload["tau_q"]),
        class_registry=classes,
        method_lock=typed_manifest["method_lock"],
        manifest=typed_manifest,
    )
    return component, typed_manifest


def _quantize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(rows, dtype=np.float64)
    maximum = np.max(np.abs(value), axis=1)
    if np.any(maximum <= 0.0) or not np.isfinite(maximum).all():
        raise GRBJP4HeldError("random q4 row degeneracy")
    scales = np.asarray(maximum / 127.0, dtype="<f2")
    if np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise GRBJP4HeldError("random q4 FP16 scale drift")
    codes = np.rint(value / scales.astype(np.float64)[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def _ground_variants(ground: cfm.GroundCFMInput) -> dict[str, cfm.GroundCFMInput]:
    permutation = np.roll(np.arange(6), -1)
    permuted = replace(
        ground,
        prototype_codes=np.asarray(ground.prototype_codes)[permutation],
        prototype_scales=np.asarray(ground.prototype_scales)[permutation],
        prototype_mask=np.asarray(ground.prototype_mask)[permutation],
        prototype_weights=np.asarray(ground.prototype_weights)[permutation],
        prototype_radii=np.asarray(ground.prototype_radii)[permutation],
        component_digest=_sha(
            {
                "falsifier": "tx_permuted",
                "base": ground.digest,
                "permutation": permutation.tolist(),
            }
        ),
    )
    generator = np.random.Generator(np.random.PCG64(COUNTERFACTUAL_SEED))
    gaussian = generator.standard_normal((Z_DIM, 4))
    q, _ = np.linalg.qr(gaussian, mode="reduced")
    random_rows = q.T
    for row in random_rows:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    left_codes, left_scales = _quantize_rows(random_rows)
    random_ground = replace(
        ground,
        left_codes=left_codes,
        left_scales=left_scales,
        component_digest=_sha(
            {
                "falsifier": "equal_energy_random_q4",
                "base": ground.digest,
                "seed": COUNTERFACTUAL_SEED,
                "left_codes_sha256": canonical_array_sha256(left_codes),
                "left_scales_sha256": canonical_array_sha256(left_scales),
            }
        ),
    )
    return {
        "real_q4": ground,
        "tx_permuted": permuted,
        "equal_energy_random_q4": random_ground,
    }


def _ground_wire(ground: cfm.GroundCFMInput) -> dict[str, Any]:
    arrays = {
        name: _encode_array(np.asarray(getattr(ground, name)))
        for name in (
            "prototype_codes",
            "prototype_scales",
            "prototype_mask",
            "prototype_weights",
            "prototype_radii",
            "left_codes",
            "left_scales",
            "right_codes",
            "right_scales",
            "direction_energy",
            "delta_q_fp16",
            "tau_q_fp16",
        )
    }
    wire = {
        "schema": ground.schema,
        "arrays": arrays,
        "old_class_order": list(ground.old_class_order),
        "checkpoint_sha256": ground.checkpoint_sha256,
        "joint_weight_sha256": ground.joint_weight_sha256,
        "phase1_method_lock_sha256": ground.phase1_method_lock_sha256,
        "component_digest": ground.component_digest,
        "phase1_resource_receipt": dict(ground.phase1_resource_receipt),
        "phase1_resource_receipt_sha256": ground.phase1_resource_receipt_sha256,
        "digest": ground.digest,
    }
    wire["wire_sha256"] = _sha(wire)
    return wire


def _ground_unwire(wire: Mapping[str, Any]) -> cfm.GroundCFMInput:
    signed = dict(wire)
    actual_wire = signed.pop("wire_sha256", None)
    if _sha(signed) != actual_wire:
        raise GRBJP4HeldError("ground wire SHA256 drift")
    arrays = wire.get("arrays")
    if not isinstance(arrays, Mapping):
        raise GRBJP4HeldError("ground wire arrays drift")
    ground = cfm.GroundCFMInput(
        *[
            _decode_array(arrays[name])
            for name in (
                "prototype_codes",
                "prototype_scales",
                "prototype_mask",
                "prototype_weights",
                "prototype_radii",
                "left_codes",
                "left_scales",
                "right_codes",
                "right_scales",
                "direction_energy",
                "delta_q_fp16",
                "tau_q_fp16",
            )
        ],
        tuple(str(value) for value in wire["old_class_order"]),
        str(wire["checkpoint_sha256"]),
        str(wire["joint_weight_sha256"]),
        str(wire["phase1_method_lock_sha256"]),
        str(wire["component_digest"]),
        dict(wire["phase1_resource_receipt"]),
        str(wire["phase1_resource_receipt_sha256"]),
        schema=str(wire["schema"]),
    )
    if ground.digest != wire.get("digest"):
        raise GRBJP4HeldError("ground semantic digest drift")
    return ground


def _support_and_query_indices(
    archive: Mapping[str, np.ndarray],
    *,
    held_receiver: str,
    scene: str,
    classes: Sequence[str],
    k_shot: int,
    coverage_sha256: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    support: list[int] = []
    query_ids: list[str] = []
    labels = archive["labels"].astype(str)
    receivers = archive["receiver_ids"].astype(str)
    scenes = archive["scenario_names"].astype(str)
    physical = archive["physical_ids"].astype(str)
    for class_id in classes:
        indices = np.flatnonzero(
            (labels == class_id)
            & (receivers == held_receiver)
            & (scenes == scene)
        )
        ordered = sorted(
            indices.tolist(),
            key=lambda index: _sha(
                {
                    "coverage_sha256": coverage_sha256,
                    "scene": scene,
                    "physical_sample_id": physical[index],
                }
            ),
        )
        if len(ordered) <= k_shot:
            raise GRBJP4HeldError("held cell lacks K support plus query")
        support.extend(ordered[:k_shot])
        query_ids.extend(physical[index] for index in ordered[k_shot:])
    support = sorted(support, key=lambda index: physical[index].encode("utf-8"))
    if len(set(query_ids)) != len(query_ids):
        raise GRBJP4HeldError("held query physical IDs are not unique")
    return np.asarray(support, dtype=np.int64), tuple(sorted(query_ids))


def _runtime_wire(
    rows: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    lock: Phase1ZIDStudentTLock,
) -> dict[str, Any]:
    bank = build_typed_zid_support_bank(
        np.asarray(rows, dtype=np.float32),
        list(labels),
        tuple(classes),
        config=lock,
    )
    metric = identity_shared_psd_metric(config=lock)
    wire = serialize_typed_zid_runtime_state(bank, metric)
    return {
        "wire_b64": base64.b64encode(wire).decode("ascii"),
        "wire_sha256": _sha(wire),
        "wire_bytes": len(wire),
    }


def _runtime_unwire(wire: Mapping[str, Any]):
    if set(wire) != {"wire_b64", "wire_sha256", "wire_bytes"}:
        raise GRBJP4HeldError("qKNN runtime wire allowlist drift")
    try:
        raw = base64.b64decode(str(wire["wire_b64"]), validate=True)
    except ValueError as exc:
        raise GRBJP4HeldError("qKNN runtime base64 drift") from exc
    if _sha(raw) != wire["wire_sha256"] or len(raw) != wire["wire_bytes"]:
        raise GRBJP4HeldError("qKNN runtime SHA/length drift")
    return deserialize_typed_zid_runtime_state(raw)


def _d92_wire(state: HeldD92State) -> dict[str, Any]:
    wire = {
        "schema": state.schema,
        "classes": list(state.classes),
        "old_classes": list(state.old_classes),
        "new_classes": list(state.new_classes),
        "K": state.k_shot,
        "coefficient_fp32": _encode_array(state.coefficient_fp32),
        "intercept_fp32": _encode_array(state.intercept_fp32),
        "old_covariance_fp32": _encode_array(state.old_covariance_fp32),
        "new_covariance_fp32": _encode_array(state.new_covariance_fp32),
        "active": state.active,
        "audit": dict(state.audit),
        "state_sha256": state.state_sha256,
    }
    wire["wire_sha256"] = _sha(wire)
    return wire


def _d92_unwire(wire: Mapping[str, Any]) -> HeldD92State:
    signed = dict(wire)
    actual = signed.pop("wire_sha256", None)
    if _sha(signed) != actual:
        raise GRBJP4HeldError("D92 wire SHA256 drift")
    return HeldD92State(
        classes=tuple(str(value) for value in wire["classes"]),
        old_classes=tuple(str(value) for value in wire["old_classes"]),
        new_classes=tuple(str(value) for value in wire["new_classes"]),
        k_shot=int(wire["K"]),
        coefficient_fp32=_decode_array(wire["coefficient_fp32"]),
        intercept_fp32=_decode_array(wire["intercept_fp32"]),
        old_covariance_fp32=_decode_array(wire["old_covariance_fp32"]),
        new_covariance_fp32=_decode_array(wire["new_covariance_fp32"]),
        active=bool(wire["active"]),
        audit=dict(wire["audit"]),
        state_sha256=str(wire["state_sha256"]),
        schema=str(wire["schema"]),
    )


def _adapt_features(
    z_id: np.ndarray,
    hidden: np.ndarray,
    pre_relu: np.ndarray,
    state: cfm.CFMFitState,
    ground: cfm.GroundCFMInput,
) -> np.ndarray:
    z = np.asarray(z_id)
    h = np.asarray(hidden)
    pre = np.asarray(pre_relu)
    if (
        z.dtype != np.float32
        or h.dtype != np.float32
        or pre.dtype != np.float32
        or z.shape != pre.shape
        or z.shape != (len(h), Z_DIM)
        or h.shape[1:] != (HIDDEN_DIM,)
        or not np.array_equal(z, np.maximum(pre, np.float32(0.0)))
    ):
        raise GRBJP4HeldError("tap adaptation input closure drift")
    delta = np.einsum(
        "r,ri,rj->ij",
        state.theta().astype(np.float64),
        ground.weighted_left().astype(np.float64),
        ground.right().astype(np.float64),
    ).astype(np.float32)
    adapted_pre = pre + h @ delta.T
    adapted = np.maximum(adapted_pre, np.float32(0.0))
    return np.asarray(normalize_zid_rows(adapted.astype(np.float32)), dtype=np.float32)


def _fit_state(
    archive: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    old: Sequence[str],
    new: Sequence[str],
    ground: cfm.GroundCFMInput,
    lock: cfm.CFMMethodLock,
    checkpoint_sha256: str,
    ground_off: bool,
) -> cfm.CFMFitState:
    function = (
        cfm.fit_cfm_ground_off_falsifier_from_taps
        if ground_off
        else cfm.fit_cfm_from_taps
    )
    return function(
        base_z_id=np.ascontiguousarray(archive["z_id"][indices], dtype=np.float32),
        hidden=np.ascontiguousarray(archive["hidden"][indices], dtype=np.float32),
        pre_relu=np.ascontiguousarray(
            archive["pre_relu"][indices], dtype=np.float32
        ),
        support_labels=archive["labels"][indices].astype(str).tolist(),
        support_physical_tokens=archive["physical_ids"][indices]
        .astype(str)
        .tolist(),
        registered_old_classes=tuple(old),
        registered_new_classes=tuple(new),
        ground=ground,
        lock=lock,
        checkpoint_weight=archive["joint_weight"],
        checkpoint_sha256=checkpoint_sha256,
    )


def _query_binding(query: Mapping[str, Any]) -> str:
    ids = np.asarray(query["query_ids"])
    arrays = {
        name: np.asarray(query[name])
        for name in ("z_id", "hidden", "pre_relu")
    }
    if (
        ids.ndim != 1
        or len(set(ids.astype(str).tolist())) != len(ids)
        or any(
            arrays[name].dtype != np.float32
            or arrays[name].shape
            != (
                len(ids),
                Z_DIM if name != "hidden" else HIDDEN_DIM,
            )
            or not np.isfinite(arrays[name]).all()
            for name in arrays
        )
    ):
        raise GRBJP4HeldError("query feature artifact contract drift")
    return _sha(
        {
            "query_ids": ids.astype(str).tolist(),
            "arrays": {
                name: canonical_array_sha256(value)
                for name, value in arrays.items()
            },
        }
    )


def _class_tie_tokens(
    archive: Mapping[str, np.ndarray],
    support_indices: np.ndarray,
    classes: Sequence[str],
    coverage_sha256: str,
) -> dict[str, str]:
    labels = archive["labels"].astype(str)
    physical = archive["physical_ids"].astype(str)
    result = {}
    for class_id in classes:
        result[class_id] = _sha(
            {
                "coverage_sha256": coverage_sha256,
                "support_physical_ids": sorted(
                    physical[
                        support_indices[labels[support_indices] == class_id]
                    ].tolist()
                ),
            }
        )
    if len(set(result.values())) != len(result):
        raise GRBJP4HeldError("class tie token collision")
    return result


def _resource_receipt(
    *,
    ground: cfm.GroundCFMInput,
    runtimes: Mapping[str, Mapping[str, Any]],
    d92_base: HeldD92State,
    d92_adapted: HeldD92State,
    fit_state: cfm.CFMFitState,
    fit_state_wire_bytes: int,
    class_count: int,
    support_count: int,
) -> dict[str, Any]:
    phase1 = dict(ground.phase1_resource_receipt)
    qknn_before = int(runtimes["M0_before"]["wire_bytes"])
    qknn_after = int(runtimes["M0_after"]["wire_bytes"])
    qknn_adapted = int(runtimes["M_DA_after"]["wire_bytes"])
    d92_base_resource = d92_resource_receipt(d92_base)
    d92_adapted_resource = d92_resource_receipt(d92_adapted)
    qknn_state = max(qknn_before, qknn_after)
    component = int(phase1["total_component_bytes"])
    support_resource = dict(fit_state.fit_receipt["resource_receipt"])
    support_fit_mac = int(support_resource["support_fit_mac_upper_bound"])
    support_fit_mac_limit = int(support_resource["support_fit_mac_limit"])
    full = {
        "M0": qknn_state,
        "M92": qknn_state
        + int(d92_base_resource["full_head_state_bytes"]),
        "M_DA": qknn_adapted + component + int(fit_state_wire_bytes),
        "M_DA92": qknn_adapted
        + int(d92_adapted_resource["full_head_state_bytes"])
        + component
        + int(fit_state_wire_bytes),
    }
    qknn_mac = support_count * Z_DIM + class_count * (Z_DIM + 1)
    post_mac = {
        "M0": qknn_mac,
        "M92": qknn_mac
        + int(d92_base_resource["post_backbone_mac_per_query"]),
        "M_DA": qknn_mac,
        "M_DA92": qknn_mac
        + int(d92_adapted_resource["post_backbone_mac_per_query"]),
    }
    if (
        phase1["jp4_update_factor_wire_bytes"] > 4096
        or support_fit_mac >= support_fit_mac_limit
        or any(value > STATE_LIMIT_BYTES for value in full.values())
        or any(value > POST_BACKBONE_MAC_LIMIT for value in post_mac.values())
    ):
        raise GRBJP4HeldError("held row resource hard gate failed")
    return {
        "update_factor_wire_bytes": int(
            phase1["jp4_update_factor_wire_bytes"]
        ),
        "ground_wire_bytes": int(phase1["ground_wire_bytes"]),
        "total_component_bytes": component,
        "fit_state_wire_bytes": int(fit_state_wire_bytes),
        "full_arm_state_bytes": full,
        "arm_state_limit_bytes": STATE_LIMIT_BYTES,
        "full_arm_post_backbone_mac_per_query": post_mac,
        "post_backbone_mac_limit_per_query": POST_BACKBONE_MAC_LIMIT,
        "support_fit_mac_upper_bound": support_fit_mac,
        "support_fit_mac_limit": support_fit_mac_limit,
        "M92_qknn_diagnostic_state_and_mac_fully_accounted": True,
        "M_DA92_qknn_diagnostic_state_and_mac_fully_accounted": True,
        "backbone_common_mac_per_query": 0,
        "cross_arm_shared_state_credit_bytes": 0,
        "query_rows_used_for_fit": 0,
    }


def _build_packet_impl(
    tap_archive: Mapping[str, Any],
    *,
    coverage_sha256: str,
    artifact_binding: Mapping[str, Any],
    selected_cell: tuple[str, str, int] | None,
    complete_matrix: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build either the complete matrix or one independently executable row."""

    archive = _validate_archive(tap_archive)
    coverage = _require_sha(coverage_sha256, "coverage_sha256")
    binding = _artifact_binding(dict(artifact_binding), coverage)
    classes = tuple(archive["class_ids"].astype(str).tolist())
    receivers = tuple(sorted(set(archive["receiver_ids"].astype(str).tolist())))
    held_receiver = _held_receiver(receivers, coverage)
    source = _source_archive(archive, held_receiver)
    qknn_locks = build_phase1_qknn_locks()
    component, component_manifest = _typed_component(
        source,
        locks=qknn_locks,
        checkpoint_sha256=binding["checkpoint_sha256"],
    )
    ground = cfm.GroundCFMInput.from_phase1_component(
        component, checkpoint_weight=archive["joint_weight"]
    )
    variants = _ground_variants(ground)
    variant_wires = {
        name: _ground_wire(variants[name]) for name in GROUND_VARIANTS
    }
    rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    all_query_ids: set[str] = set()
    labels = archive["labels"].astype(str)
    physical = archive["physical_ids"].astype(str)
    for pseudo_new in classes:
        old = tuple(value for value in classes if value != pseudo_new)
        new = (pseudo_new,)
        for scene in SCENES:
            for k_shot in K_VALUES:
                if (
                    selected_cell is not None
                    and (pseudo_new, scene, k_shot) != selected_cell
                ):
                    continue
                support, query_ids = _support_and_query_indices(
                    archive,
                    held_receiver=held_receiver,
                    scene=scene,
                    classes=classes,
                    k_shot=k_shot,
                    coverage_sha256=coverage,
                )
                all_query_ids.update(query_ids)
                qknn_lock = qknn_locks[k_shot]
                method_lock = cfm.CFMMethodLock.from_mapping(
                    component.method_lock, qknn_lock=qknn_lock
                )
                fit_states = {
                    "real_q4": _fit_state(
                        archive,
                        support,
                        old=old,
                        new=new,
                        ground=variants["real_q4"],
                        lock=method_lock,
                        checkpoint_sha256=binding["checkpoint_sha256"],
                        ground_off=False,
                    ),
                    "ground_off": _fit_state(
                        archive,
                        support,
                        old=old,
                        new=new,
                        ground=variants["real_q4"],
                        lock=method_lock,
                        checkpoint_sha256=binding["checkpoint_sha256"],
                        ground_off=True,
                    ),
                    "tx_permuted": _fit_state(
                        archive,
                        support,
                        old=old,
                        new=new,
                        ground=variants["tx_permuted"],
                        lock=method_lock,
                        checkpoint_sha256=binding["checkpoint_sha256"],
                        ground_off=False,
                    ),
                    "equal_energy_random_q4": _fit_state(
                        archive,
                        support,
                        old=old,
                        new=new,
                        ground=variants["equal_energy_random_q4"],
                        lock=method_lock,
                        checkpoint_sha256=binding["checkpoint_sha256"],
                        ground_off=False,
                    ),
                }
                adapted_support = {
                    name: _adapt_features(
                        archive["z_id"][support],
                        archive["hidden"][support],
                        archive["pre_relu"][support],
                        state,
                        variants["real_q4" if name == "ground_off" else name],
                    )
                    for name, state in fit_states.items()
                }
                support_labels = labels[support].tolist()
                old_mask = np.asarray(
                    [value in set(old) for value in support_labels], dtype=np.bool_
                )
                base_support = np.asarray(
                    normalize_zid_rows(archive["z_id"][support]), dtype=np.float32
                )
                runtimes = {
                    "M0_before": _runtime_wire(
                        base_support[old_mask],
                        np.asarray(support_labels)[old_mask].tolist(),
                        old,
                        qknn_lock,
                    ),
                    "M0_after": _runtime_wire(
                        base_support, support_labels, classes, qknn_lock
                    ),
                    "M_DA_before": _runtime_wire(
                        adapted_support["real_q4"][old_mask],
                        np.asarray(support_labels)[old_mask].tolist(),
                        old,
                        qknn_lock,
                    ),
                    "M_DA_after": _runtime_wire(
                        adapted_support["real_q4"],
                        support_labels,
                        classes,
                        qknn_lock,
                    ),
                }
                counterfactual_runtime = {
                    name: _runtime_wire(
                        adapted_support[name],
                        support_labels,
                        classes,
                        qknn_lock,
                    )
                    for name in COUNTERFACTUALS
                }
                d92_base = fit_held_d92_head(
                    base_support,
                    support_labels,
                    old_classes=old,
                    new_classes=new,
                    k_shot=k_shot,
                )
                d92_adapted = {
                    name: fit_held_d92_head(
                        adapted_support[name],
                        support_labels,
                        old_classes=old,
                        new_classes=new,
                        k_shot=k_shot,
                    )
                    for name in ("real_q4", *COUNTERFACTUALS)
                }
                fit_wires = {
                    name: cfm.serialize_cfm_fit_state(state)
                    for name, state in fit_states.items()
                }
                resource = _resource_receipt(
                    ground=ground,
                    runtimes=runtimes,
                    d92_base=d92_base,
                    d92_adapted=d92_adapted["real_q4"],
                    fit_state=fit_states["real_q4"],
                    fit_state_wire_bytes=len(fit_wires["real_q4"]),
                    class_count=len(classes),
                    support_count=len(support),
                )
                row_id = _sha(
                    {
                        "coverage_sha256": coverage,
                        "held_receiver": held_receiver,
                        "pseudo_new": pseudo_new,
                        "scene": scene,
                        "K": k_shot,
                    }
                )
                row = {
                    "row_id": row_id,
                    "pseudo_new": pseudo_new,
                    "scene": scene,
                    "K": k_shot,
                    "old_classes": list(old),
                    "new_classes": list(new),
                    "query_ids": list(query_ids),
                    "support_physical_ids": physical[support].tolist(),
                    "support_labels": support_labels,
                    "class_tie_tokens": _class_tie_tokens(
                        archive, support, classes, coverage
                    ),
                    "qknn_lock_digest": qknn_lock.lock_digest,
                    "runtimes": runtimes,
                    "counterfactual_runtimes": counterfactual_runtime,
                    "fit_states": {
                        name: {
                            "wire_b64": base64.b64encode(wire).decode("ascii"),
                            "wire_sha256": _sha(wire),
                            "wire_bytes": len(wire),
                        }
                        for name, wire in fit_wires.items()
                    },
                    "fit_state_receipts": {
                        name: {
                            "receipt_sha256": state.receipt_sha256,
                            "query_rows_used_for_fit": int(
                                state.fit_receipt["query_rows_used_for_fit"]
                            ),
                            "ground_equation_enabled": bool(
                                state.fit_receipt["ground_equation_enabled"]
                            ),
                            "loco_stability_gate_pass": bool(
                                (
                                    state.fit_receipt.get("loco_receipt") or {}
                                ).get("stability_gate_pass", k_shot > 1)
                            ),
                            "theta_code_sha256": canonical_array_sha256(
                                state.theta_codes
                            ),
                        }
                        for name, state in fit_states.items()
                    },
                    "d92_states": {
                        "base": _d92_wire(d92_base),
                        **{
                            name: _d92_wire(state)
                            for name, state in d92_adapted.items()
                        },
                    },
                    "d92_k1_exact_qknn_fallback": k_shot == 1,
                    "resource": resource,
                }
                rows.append(row)
                query_set = set(query_ids)
                truth_rows.append(
                    {
                        "row_id": row_id,
                        "query_labels": {
                            physical[index]: labels[index]
                            for index in np.flatnonzero(
                                (archive["receiver_ids"].astype(str) == held_receiver)
                                & (
                                    archive["scenario_names"].astype(str)
                                    == scene
                                )
                            )
                            if physical[index] in query_set
                        },
                    }
                )
    query_ids = sorted(all_query_ids)
    index = {value: i for i, value in enumerate(physical.tolist())}
    output_schema = SCHEMA if complete_matrix else SCHEMA + ".row-shard.v1"
    query: dict[str, Any] = {
        "schema": output_schema + ".query.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "packet_core_sha256": "",
        "query_ids": np.asarray(query_ids),
        "z_id": np.asarray(
            [archive["z_id"][index[value]] for value in query_ids],
            dtype=np.float32,
        ),
        "hidden": np.asarray(
            [archive["hidden"][index[value]] for value in query_ids],
            dtype=np.float32,
        ),
        "pre_relu": np.asarray(
            [archive["pre_relu"][index[value]] for value in query_ids],
            dtype=np.float32,
        ),
        "query_binding_sha256": "",
    }
    query["query_binding_sha256"] = _query_binding(query)
    packet_core = {
        "schema": output_schema,
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "formal_phase2_eligible": False,
        "target25_authorized": False,
        "coverage_sha256": coverage,
        "input_artifact_binding": binding,
        "held_receiver": held_receiver,
        "receivers": list(receivers),
        "classes": list(classes),
        "K_values": list(K_VALUES),
        "scenes": list(SCENES),
        "arms": list(ARMS),
        "counterfactuals": list(COUNTERFACTUALS),
        "counterfactual_seed": COUNTERFACTUAL_SEED,
        "phase1_component_manifest_sha256": _sha(component_manifest),
        "ground_variants": variant_wires,
        "query_binding_sha256": query["query_binding_sha256"],
        "rows": rows,
    }
    packet_core_sha256 = _sha(packet_core)
    query["packet_core_sha256"] = packet_core_sha256
    truth = {
        "schema": output_schema + ".truth.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "packet_core_sha256": packet_core_sha256,
        "rows": truth_rows,
    }
    truth["truth_sha256"] = _sha(truth)
    packet = {
        **packet_core,
        "packet_core_sha256": packet_core_sha256,
        "truth_commitment_sha256": truth["truth_sha256"],
    }
    packet["packet_sha256"] = _sha(packet)
    if complete_matrix:
        _verify_packet(packet)
    else:
        _verify_row_shard(packet, query, truth)
    return packet, query, truth


def build_packet(
    tap_archive: Mapping[str, Any],
    *,
    coverage_sha256: str,
    artifact_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the complete sealed 54-row packet, query artifact, and truth."""

    return _build_packet_impl(
        tap_archive,
        coverage_sha256=coverage_sha256,
        artifact_binding=artifact_binding,
        selected_cell=None,
        complete_matrix=True,
    )


def build_row_shard(
    tap_archive: Mapping[str, Any],
    *,
    coverage_sha256: str,
    artifact_binding: Mapping[str, Any],
    pseudo_new: str,
    scene: str,
    k_shot: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit exactly one held row in an independent process-ready shard."""

    cell = (str(pseudo_new), str(scene), int(k_shot))
    return _build_packet_impl(
        tap_archive,
        coverage_sha256=coverage_sha256,
        artifact_binding=artifact_binding,
        selected_cell=cell,
        complete_matrix=False,
    )


def _verify_packet(packet: Mapping[str, Any]) -> None:
    signed = dict(packet)
    packet_digest = signed.pop("packet_sha256", None)
    core = dict(signed)
    core_digest = core.pop("packet_core_sha256", None)
    core.pop("truth_commitment_sha256", None)
    if (
        packet.get("schema") != SCHEMA
        or packet.get("candidate") != CANDIDATE
        or packet.get("evaluation_scope") != SCOPE
        or packet.get("formal_phase2_eligible") is not False
        or packet.get("target25_authorized") is not False
        or _sha(signed) != packet_digest
        or _sha(core) != core_digest
        or type(packet.get("rows")) is not list
        or len(packet["rows"]) != ROW_COUNT
        or packet.get("K_values") != list(K_VALUES)
        or packet.get("scenes") != list(SCENES)
        or packet.get("arms") != list(ARMS)
        or packet.get("counterfactuals") != list(COUNTERFACTUALS)
        or packet.get("counterfactual_seed") != COUNTERFACTUAL_SEED
        or set(packet.get("ground_variants", {})) != set(GROUND_VARIANTS)
    ):
        raise GRBJP4HeldError("packet schema/hash/row-count drift")
    for name in (
        "packet_sha256",
        "packet_core_sha256",
        "truth_commitment_sha256",
        "coverage_sha256",
        "query_binding_sha256",
        "phase1_component_manifest_sha256",
    ):
        _require_sha(packet.get(name), f"packet {name}")
    cells = {
        (row.get("pseudo_new"), row.get("scene"), row.get("K"))
        for row in packet["rows"]
    }
    expected = {
        (pseudo_new, scene, k_shot)
        for pseudo_new in packet["classes"]
        for scene in SCENES
        for k_shot in K_VALUES
    }
    row_ids = [row.get("row_id") for row in packet["rows"]]
    if cells != expected or len(set(row_ids)) != ROW_COUNT:
        raise GRBJP4HeldError("packet held54 row bijection drift")
    for wire in packet["ground_variants"].values():
        _ground_unwire(wire)
    for row in packet["rows"]:
        if (
            row.get("old_classes")
            != [
                value
                for value in packet["classes"]
                if value != row.get("pseudo_new")
            ]
            or row.get("new_classes") != [row.get("pseudo_new")]
            or set(row.get("fit_states", {}))
            != {"real_q4", "ground_off", "tx_permuted", "equal_energy_random_q4"}
            or set(row.get("class_tie_tokens", {})) != set(packet["classes"])
            or len(set(row["class_tie_tokens"].values())) != len(packet["classes"])
            or len(row.get("query_ids", [])) != len(set(row.get("query_ids", [])))
            or any(
                int(value) > STATE_LIMIT_BYTES
                for value in row["resource"]["full_arm_state_bytes"].values()
            )
        ):
            raise GRBJP4HeldError("packet row registry/state/resource drift")


def _verify_row_shard(
    packet: Mapping[str, Any],
    query: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> None:
    signed = dict(packet)
    packet_digest = signed.pop("packet_sha256", None)
    core = dict(signed)
    core_digest = core.pop("packet_core_sha256", None)
    core.pop("truth_commitment_sha256", None)
    if (
        packet.get("schema") != ROW_SHARD_SCHEMA
        or packet.get("candidate") != CANDIDATE
        or packet.get("evaluation_scope") != SCOPE
        or packet.get("formal_phase2_eligible") is not False
        or packet.get("target25_authorized") is not False
        or _sha(signed) != packet_digest
        or _sha(core) != core_digest
        or len(packet.get("rows", [])) != 1
        or packet.get("K_values") != list(K_VALUES)
        or packet.get("scenes") != list(SCENES)
        or packet.get("arms") != list(ARMS)
        or set(packet.get("ground_variants", {})) != set(GROUND_VARIANTS)
    ):
        raise GRBJP4HeldError("row shard packet contract drift")
    row = packet["rows"][0]
    if (
        row.get("pseudo_new") not in packet.get("classes", [])
        or row.get("scene") not in SCENES
        or row.get("K") not in K_VALUES
        or row.get("new_classes") != [row.get("pseudo_new")]
    ):
        raise GRBJP4HeldError("row shard cell contract drift")
    if (
        query.get("schema") != ROW_SHARD_SCHEMA + ".query.v1"
        or query.get("candidate") != CANDIDATE
        or query.get("evaluation_scope") != SCOPE
        or query.get("packet_core_sha256") != packet["packet_core_sha256"]
        or _query_binding(query) != packet["query_binding_sha256"]
        or set(np.asarray(query["query_ids"]).astype(str).tolist())
        != set(row["query_ids"])
        or truth.get("schema") != ROW_SHARD_SCHEMA + ".truth.v1"
        or truth.get("candidate") != CANDIDATE
        or truth.get("evaluation_scope") != SCOPE
        or truth.get("packet_core_sha256") != packet["packet_core_sha256"]
        or len(truth.get("rows", [])) != 1
        or truth["rows"][0].get("row_id") != row["row_id"]
        or truth.get("truth_sha256") != packet["truth_commitment_sha256"]
        or _sha(
            {
                key: value
                for key, value in truth.items()
                if key != "truth_sha256"
            }
        )
        != truth.get("truth_sha256")
    ):
        raise GRBJP4HeldError("row shard query/truth binding drift")


def assemble_row_shards(
    shards: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    ],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Assemble exactly one immutable shard for every frozen held54 cell."""

    if type(shards) not in (list, tuple) or len(shards) != ROW_COUNT:
        raise GRBJP4HeldError("row shard assembly requires exactly 54 shards")
    common: dict[str, Any] | None = None
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    truths: dict[tuple[str, str, int], dict[str, Any]] = {}
    query_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for packet_value, query_value, truth_value in shards:
        packet = dict(packet_value)
        query = dict(query_value)
        truth = dict(truth_value)
        _verify_row_shard(packet, query, truth)
        packet_common = {
            key: value
            for key, value in packet.items()
            if key
            not in {
                "schema",
                "rows",
                "query_binding_sha256",
                "packet_core_sha256",
                "truth_commitment_sha256",
                "packet_sha256",
            }
        }
        if common is None:
            common = packet_common
        elif _canon(common) != _canon(packet_common):
            raise GRBJP4HeldError("row shard common input/ground drift")
        row = dict(packet["rows"][0])
        cell = (row["pseudo_new"], row["scene"], int(row["K"]))
        if cell in rows:
            raise GRBJP4HeldError("duplicate row shard cell")
        rows[cell] = row
        truths[cell] = dict(truth["rows"][0])
        ids = np.asarray(query["query_ids"]).astype(str).tolist()
        for index, query_id in enumerate(ids):
            values = tuple(
                np.ascontiguousarray(query[name][index], dtype=np.float32)
                for name in ("z_id", "hidden", "pre_relu")
            )
            if query_id in query_values and any(
                not np.array_equal(left, right)
                for left, right in zip(query_values[query_id], values)
            ):
                raise GRBJP4HeldError("row shard query feature disagreement")
            query_values[query_id] = values
    assert common is not None
    classes = tuple(str(value) for value in common["classes"])
    expected = [
        (pseudo_new, scene, k_shot)
        for pseudo_new in classes
        for scene in SCENES
        for k_shot in K_VALUES
    ]
    if set(rows) != set(expected):
        raise GRBJP4HeldError("row shard cell bijection drift")
    ordered_rows = [rows[cell] for cell in expected]
    ordered_truth = [truths[cell] for cell in expected]
    query_ids = sorted(query_values)
    query: dict[str, Any] = {
        "schema": SCHEMA + ".query.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "packet_core_sha256": "",
        "query_ids": np.asarray(query_ids),
        "z_id": np.asarray(
            [query_values[value][0] for value in query_ids], dtype=np.float32
        ),
        "hidden": np.asarray(
            [query_values[value][1] for value in query_ids], dtype=np.float32
        ),
        "pre_relu": np.asarray(
            [query_values[value][2] for value in query_ids], dtype=np.float32
        ),
        "query_binding_sha256": "",
    }
    query["query_binding_sha256"] = _query_binding(query)
    packet_core = {
        "schema": SCHEMA,
        **common,
        "query_binding_sha256": query["query_binding_sha256"],
        "rows": ordered_rows,
    }
    packet_core_sha256 = _sha(packet_core)
    query["packet_core_sha256"] = packet_core_sha256
    truth = {
        "schema": SCHEMA + ".truth.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "packet_core_sha256": packet_core_sha256,
        "rows": ordered_truth,
    }
    truth["truth_sha256"] = _sha(truth)
    packet = {
        **packet_core,
        "packet_core_sha256": packet_core_sha256,
        "truth_commitment_sha256": truth["truth_sha256"],
    }
    packet["packet_sha256"] = _sha(packet)
    _verify_packet(packet)
    return packet, query, truth


def _fit_unwire(
    wire: Mapping[str, Any],
    *,
    ground: cfm.GroundCFMInput,
    lock: cfm.CFMMethodLock,
    checkpoint_sha256: str,
) -> cfm.CFMFitState:
    if set(wire) != {"wire_b64", "wire_sha256", "wire_bytes"}:
        raise GRBJP4HeldError("fit-state wire allowlist drift")
    try:
        raw = base64.b64decode(str(wire["wire_b64"]), validate=True)
    except ValueError as exc:
        raise GRBJP4HeldError("fit-state base64 drift") from exc
    if _sha(raw) != wire["wire_sha256"] or len(raw) != wire["wire_bytes"]:
        raise GRBJP4HeldError("fit-state wire SHA/length drift")
    return cfm.deserialize_cfm_fit_state(
        raw,
        expected_ground_digest=ground.digest,
        expected_lock_digest=lock.digest,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_joint_weight_sha256_before=ground.joint_weight_sha256,
    )


def _stable_predictions(
    logits: np.ndarray,
    classes: Sequence[str],
    tie_tokens: Mapping[str, str],
) -> tuple[list[str], int]:
    values = np.asarray(logits)
    registry = tuple(str(value) for value in classes)
    if (
        values.ndim != 2
        or values.shape[1] != len(registry)
        or not np.isfinite(values).all()
        or not set(registry).issubset(set(tie_tokens))
    ):
        raise GRBJP4HeldError("prediction logits/registry contract drift")
    predictions = []
    ties = 0
    for row in values:
        maximum = float(np.max(row))
        candidates = np.flatnonzero(row == maximum).tolist()
        if len(candidates) > 1:
            ties += 1
        winner = min(candidates, key=lambda index: tie_tokens[registry[index]])
        predictions.append(registry[winner])
    return predictions, ties


def _qknn_payload(
    runtime: Mapping[str, Any],
    query_zid: np.ndarray,
    tie_tokens: Mapping[str, str],
    support_physical_ids: Sequence[str],
) -> dict[str, Any]:
    bank, metric = _runtime_unwire(runtime)
    logits = score_zid_student_t_logits(bank, query_zid, metric=metric)
    classes = list(bank.classes)
    prediction, ties = _stable_predictions(logits, classes, tie_tokens)
    decoded = (
        bank.codes_qint8.astype(np.float32)
        * bank.scales_fp16.astype(np.float32)[:, None]
    )
    query_unit = np.asarray(normalize_zid_rows(query_zid), dtype=np.float32)
    membership_index = np.argmax(query_unit @ decoded.T, axis=1)
    support_ids = tuple(str(value) for value in support_physical_ids)
    if len(support_ids) != len(decoded):
        raise GRBJP4HeldError("neighbor membership/support runtime drift")
    return {
        "classes": classes,
        "prediction": prediction,
        "logits": _encode_array(logits),
        "neighbor_membership": [
            support_ids[int(index)] for index in membership_index
        ],
        "top_score_tie_rows": ties,
    }


def _d92_payload(
    state_wire: Mapping[str, Any],
    query_zid: np.ndarray,
    *,
    qknn_neighbor_payload: Mapping[str, Any],
    tie_tokens: Mapping[str, str],
) -> dict[str, Any]:
    state = _d92_unwire(state_wire)
    logits = score_held_d92_head(state, query_zid)
    prediction, ties = _stable_predictions(logits, state.classes, tie_tokens)
    return {
        "classes": list(state.classes),
        "prediction": prediction,
        "logits": _encode_array(logits),
        "neighbor_membership": list(
            qknn_neighbor_payload["neighbor_membership"]
        ),
        "top_score_tie_rows": ties,
    }


def predict_packet(
    packet: Mapping[str, Any], query: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish immutable all-class predictions without reading truth."""

    _verify_packet(packet)
    expected_query_keys = {
        "schema",
        "candidate",
        "evaluation_scope",
        "packet_core_sha256",
        "query_ids",
        "z_id",
        "hidden",
        "pre_relu",
        "query_binding_sha256",
    }
    if (
        set(query) != expected_query_keys
        or query.get("schema") != SCHEMA + ".query.v1"
        or query.get("candidate") != CANDIDATE
        or query.get("evaluation_scope") != SCOPE
        or query.get("packet_core_sha256") != packet["packet_core_sha256"]
        or _query_binding(query) != query.get("query_binding_sha256")
        or query.get("query_binding_sha256") != packet["query_binding_sha256"]
    ):
        raise GRBJP4HeldError("query artifact binding drift")
    query_ids = np.asarray(query["query_ids"]).astype(str).tolist()
    lookup = {value: index for index, value in enumerate(query_ids)}
    grounds = {
        name: _ground_unwire(packet["ground_variants"][name])
        for name in GROUND_VARIANTS
    }
    checkpoint_sha = packet["input_artifact_binding"]["checkpoint_sha256"]
    qknn_locks = build_phase1_qknn_locks()
    output_rows = []
    for row in packet["rows"]:
        if any(value not in lookup for value in row["query_ids"]):
            raise GRBJP4HeldError("query artifact misses packet row IDs")
        indices = np.asarray([lookup[value] for value in row["query_ids"]])
        qz = np.ascontiguousarray(query["z_id"][indices], dtype=np.float32)
        qh = np.ascontiguousarray(query["hidden"][indices], dtype=np.float32)
        qp = np.ascontiguousarray(query["pre_relu"][indices], dtype=np.float32)
        # Reconstruct the lock from the frozen Phase1 method-lock binding held
        # by the same K-specific qKNN lock.  The numeric constants are repeated
        # in the typed ground input and cannot be changed by query data.
        qlock = qknn_locks[row["K"]]
        method_lock = cfm.CFMMethodLock(
            qknn_neighbor_count=qlock.active_k,
            student_nu=qlock.student_nu,
            kernel_effective_dim=float(qlock.kernel_effective_dim),
            kernel_volume_gamma=qlock.kernel_volume_gamma,
            kernel_scale=qlock.shared_h0,
            qknn_lock_digest=qlock.lock_digest,
            phase1_method_lock_sha256=grounds[
                "real_q4"
            ].phase1_method_lock_sha256,
            delta_q=float(grounds["real_q4"].delta_q_fp16),
            tau_q=float(grounds["real_q4"].tau_q_fp16),
            scale_prior_strength=qlock.scale_prior_strength,
            scale_min_ratio=qlock.scale_min_ratio,
            scale_max_ratio=qlock.scale_max_ratio,
            temperature=qlock.temperature,
        )
        states = {
            name: _fit_unwire(
                row["fit_states"][name],
                ground=grounds["real_q4" if name == "ground_off" else name],
                lock=method_lock,
                checkpoint_sha256=checkpoint_sha,
            )
            for name in ("real_q4", *COUNTERFACTUALS)
        }
        adapted = {
            name: _adapt_features(
                qz,
                qh,
                qp,
                state,
                grounds["real_q4" if name == "ground_off" else name],
            )
            for name, state in states.items()
        }
        tie = row["class_tie_tokens"]
        old_support_ids = [
            token
            for token, label in zip(
                row["support_physical_ids"], row["support_labels"]
            )
            if label in set(row["old_classes"])
        ]
        base_before = _qknn_payload(
            row["runtimes"]["M0_before"], qz, tie, old_support_ids
        )
        base_after = _qknn_payload(
            row["runtimes"]["M0_after"],
            qz,
            tie,
            row["support_physical_ids"],
        )
        da_before = _qknn_payload(
            row["runtimes"]["M_DA_before"],
            adapted["real_q4"],
            tie,
            old_support_ids,
        )
        da_after = _qknn_payload(
            row["runtimes"]["M_DA_after"],
            adapted["real_q4"],
            tie,
            row["support_physical_ids"],
        )
        if row["K"] == 1:
            m92_after = copy.deepcopy(base_after)
            mda92_after = copy.deepcopy(da_after)
        else:
            m92_after = _d92_payload(
                row["d92_states"]["base"],
                qz,
                qknn_neighbor_payload=base_after,
                tie_tokens=tie,
            )
            mda92_after = _d92_payload(
                row["d92_states"]["real_q4"],
                adapted["real_q4"],
                qknn_neighbor_payload=da_after,
                tie_tokens=tie,
            )
        counterfactuals = {}
        for name in COUNTERFACTUALS:
            qknn_payload = _qknn_payload(
                row["counterfactual_runtimes"][name],
                adapted[name],
                tie,
                row["support_physical_ids"],
            )
            d92_payload = (
                copy.deepcopy(qknn_payload)
                if row["K"] == 1
                else _d92_payload(
                    row["d92_states"][name],
                    adapted[name],
                    qknn_neighbor_payload=qknn_payload,
                    tie_tokens=tie,
                )
            )
            counterfactuals[name] = {
                "M_DA": qknn_payload,
                "M_DA92": d92_payload,
            }
        output_rows.append(
            {
                "row_id": row["row_id"],
                "query_ids": list(row["query_ids"]),
                "before": {
                    "M0": base_before,
                    "M92": copy.deepcopy(base_before),
                    "M_DA": da_before,
                    "M_DA92": copy.deepcopy(da_before),
                },
                "after": {
                    "M0": base_after,
                    "M92": m92_after,
                    "M_DA": da_after,
                    "M_DA92": mda92_after,
                },
                "counterfactuals": counterfactuals,
            }
        )
    result = {
        "schema": SCHEMA + ".prediction.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "target25_authorized": False,
        "packet_sha256": packet["packet_sha256"],
        "query_binding_sha256": packet["query_binding_sha256"],
        "rows": output_rows,
    }
    result["COMMIT"] = _sha(result)
    return result


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    query_count: int,
    expected_classes: Sequence[str],
    tie_tokens: Mapping[str, str],
) -> tuple[list[str], np.ndarray, list[str]]:
    if (
        set(payload)
        != {
            "classes",
            "prediction",
            "logits",
            "neighbor_membership",
            "top_score_tie_rows",
        }
        or payload.get("classes") != list(expected_classes)
        or type(payload.get("prediction")) is not list
        or len(payload["prediction"]) != query_count
        or type(payload.get("neighbor_membership")) is not list
        or len(payload["neighbor_membership"]) != query_count
    ):
        raise GRBJP4HeldError("prediction arm payload drift")
    logits = _decode_array(payload["logits"])
    prediction, ties = _stable_predictions(logits, expected_classes, tie_tokens)
    if prediction != payload["prediction"] or ties != payload["top_score_tie_rows"]:
        raise GRBJP4HeldError("prediction argmax/logit binding drift")
    return prediction, logits, list(payload["neighbor_membership"])


def _arm_metrics(
    before: Sequence[str],
    after: Sequence[str],
    truth: Sequence[str],
    *,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    all_classes: Sequence[str],
) -> dict[str, Any]:
    old_set = set(old_classes)
    new_set = set(new_classes)
    old_positions = [index for index, value in enumerate(truth) if value in old_set]
    new_positions = [index for index, value in enumerate(truth) if value in new_set]
    old_before = float(
        np.mean([before[index] == truth[index] for index in old_positions])
    )
    old_after = float(
        np.mean([after[index] == truth[index] for index in old_positions])
    )
    seen_new = float(
        np.mean([after[index] == truth[index] for index in new_positions])
    )
    per_class = {
        class_id: float(
            np.mean(
                [
                    after[index] == truth[index]
                    for index in range(len(truth))
                    if truth[index] == class_id
                ]
            )
        )
        for class_id in all_classes
    }
    h_score = (
        0.0
        if old_after + seen_new == 0.0
        else 2.0 * old_after * seen_new / (old_after + seen_new)
    )
    return {
        "old_before": old_before,
        "old_after": old_after,
        "seen_new": seen_new,
        "H_old_new": h_score,
        "floor": min(per_class.values()),
        "min_new": min(per_class[value] for value in new_classes),
        "per_class": per_class,
        "per_old_class": {value: per_class[value] for value in old_classes},
        "forgetting": old_before - old_after,
    }


def _comparison_record(
    *,
    packet_row: Mapping[str, Any],
    name: str,
    baseline_arm: str,
    candidate_arm: str,
    arms: Mapping[str, Any],
    predictions: Mapping[str, Sequence[str]],
    memberships: Mapping[str, Sequence[str]],
    truth: Sequence[str],
    counterfactual_arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = arms[baseline_arm]
    candidate = arms[candidate_arm]
    base_prediction = predictions[baseline_arm]
    candidate_prediction = predictions[candidate_arm]
    wrong_to_correct = sum(
        left != target and right == target
        for left, right, target in zip(
            base_prediction, candidate_prediction, truth
        )
    )
    correct_to_wrong = sum(
        left == target and right != target
        for left, right, target in zip(
            base_prediction, candidate_prediction, truth
        )
    )
    control_net = {}
    control_h = {}
    for control, metrics in counterfactual_arms.items():
        control_prediction = metrics[candidate_arm]["prediction"]
        c_w2c = sum(
            left != target and right == target
            for left, right, target in zip(
                base_prediction, control_prediction, truth
            )
        )
        c_c2w = sum(
            left == target and right != target
            for left, right, target in zip(
                base_prediction, control_prediction, truth
            )
        )
        control_net[control] = c_w2c - c_c2w
        control_h[control] = (
            metrics[candidate_arm]["metrics"]["H_old_new"]
            - baseline["H_old_new"]
        )
    return {
        "row_id": packet_row["row_id"],
        "K": packet_row["K"],
        "scene": packet_row["scene"],
        "pseudo_new": packet_row["pseudo_new"],
        "comparison": name,
        "neighbor_membership_changes": sum(
            left != right
            for left, right in zip(
                memberships[baseline_arm], memberships[candidate_arm]
            )
        ),
        "argmax_changes": sum(
            left != right
            for left, right in zip(base_prediction, candidate_prediction)
        ),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "old_after_delta": candidate["old_after"] - baseline["old_after"],
        "seen_new_delta": candidate["seen_new"] - baseline["seen_new"],
        "H_delta": candidate["H_old_new"] - baseline["H_old_new"],
        "floor_delta": candidate["floor"] - baseline["floor"],
        "min_new_delta": candidate["min_new"] - baseline["min_new"],
        "per_old_class_deltas": {
            class_id: candidate["per_old_class"][class_id]
            - baseline["per_old_class"][class_id]
            for class_id in packet_row["old_classes"]
        },
        "forgetting_delta": candidate["forgetting"] - baseline["forgetting"],
        "loco_stable": bool(
            packet_row["fit_state_receipts"]["real_q4"][
                "loco_stability_gate_pass"
            ]
            if packet_row["K"] == 1
            else True
        ),
        "counterfactual_net_corrections": control_net,
        "counterfactual_H_deltas": control_h,
    }


def _matched_causal_summary(
    records: Sequence[Mapping[str, Any]],
    fields: tuple[str, ...],
    comparison: str,
) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records if record["comparison"] == comparison]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in selected:
        grouped.setdefault(tuple(record[field] for field in fields), []).append(record)
    output = []
    tolerance = 1.0e-12
    for key in sorted(grouped):
        group = grouped[key]
        item: dict[str, Any] = {
            field: value for field, value in zip(fields, key)
        }
        item.update(
            {
                "comparison": comparison,
                "rows": len(group),
                "neighbor_membership_changes": sum(
                    int(record["neighbor_membership_changes"]) for record in group
                ),
                "argmax_changes": sum(
                    int(record["argmax_changes"]) for record in group
                ),
                "wrong_to_correct": sum(
                    int(record["wrong_to_correct"]) for record in group
                ),
                "correct_to_wrong": sum(
                    int(record["correct_to_wrong"]) for record in group
                ),
            }
        )
        for metric in (
            "old_after_delta",
            "seen_new_delta",
            "H_delta",
            "floor_delta",
            "min_new_delta",
            "forgetting_delta",
        ):
            item[metric] = float(np.mean([record[metric] for record in group]))
        per_old: dict[str, list[float]] = {}
        for record in group:
            for class_id, value in record["per_old_class_deltas"].items():
                per_old.setdefault(class_id, []).append(float(value))
        item["per_old_class_deltas"] = {
            class_id: float(np.mean(values))
            for class_id, values in sorted(per_old.items())
        }
        control_net = {
            name: sum(
                int(record["counterfactual_net_corrections"][name])
                for record in group
            )
            for name in COUNTERFACTUALS
        }
        control_h = {
            name: float(
                np.mean(
                    [
                        record["counterfactual_H_deltas"][name]
                        for record in group
                    ]
                )
            )
            for name in COUNTERFACTUALS
        }
        item["counterfactual_net_corrections"] = control_net
        item["counterfactual_H_deltas"] = control_h
        net = item["wrong_to_correct"] - item["correct_to_wrong"]
        item["net_corrections"] = net
        item["gate_pass"] = bool(
            item["neighbor_membership_changes"] > 0
            and item["argmax_changes"] > 0
            and item["wrong_to_correct"] > item["correct_to_wrong"]
            and item["old_after_delta"] >= -tolerance
            and item["seen_new_delta"] >= -tolerance
            and item["H_delta"] >= -tolerance
            and item["floor_delta"] >= -tolerance
            and item["min_new_delta"] >= -tolerance
            and all(
                value >= -tolerance
                for value in item["per_old_class_deltas"].values()
            )
            and item["forgetting_delta"] <= tolerance
            and (
                item["H_delta"] > tolerance
                or item["floor_delta"] > tolerance
            )
            and net > max(control_net.values())
            and item["H_delta"] > max(control_h.values())
            and all(bool(record["loco_stable"]) for record in group)
        )
        output.append(item)
    return output


def score_packet(
    packet: Mapping[str, Any],
    prediction: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    commit: str,
    truth_sha256: str,
) -> dict[str, Any]:
    """Join truth only after the immutable prediction commitment."""

    _verify_packet(packet)
    pred_signed = dict(prediction)
    actual_commit = pred_signed.pop("COMMIT", None)
    truth_signed = dict(truth)
    actual_truth = truth_signed.pop("truth_sha256", None)
    if (
        prediction.get("schema") != SCHEMA + ".prediction.v1"
        or prediction.get("candidate") != CANDIDATE
        or prediction.get("evaluation_scope") != SCOPE
        or prediction.get("target25_authorized") is not False
        or prediction.get("packet_sha256") != packet["packet_sha256"]
        or actual_commit != commit
        or _sha(pred_signed) != commit
        or truth.get("schema") != SCHEMA + ".truth.v1"
        or truth.get("candidate") != CANDIDATE
        or truth.get("evaluation_scope") != SCOPE
        or truth.get("packet_core_sha256") != packet["packet_core_sha256"]
        or actual_truth != truth_sha256
        or _sha(truth_signed) != truth_sha256
        or truth_sha256 != packet["truth_commitment_sha256"]
        or len(prediction.get("rows", [])) != ROW_COUNT
        or len(truth.get("rows", [])) != ROW_COUNT
    ):
        raise GRBJP4HeldError("prediction/truth seal or row-count drift")
    metrics = []
    comparisons = []
    for packet_row, pred_row, truth_row in zip(
        packet["rows"], prediction["rows"], truth["rows"]
    ):
        if (
            pred_row.get("row_id") != packet_row["row_id"]
            or truth_row.get("row_id") != packet_row["row_id"]
            or pred_row.get("query_ids") != packet_row["query_ids"]
            or set(truth_row.get("query_labels", {}))
            != set(packet_row["query_ids"])
        ):
            raise GRBJP4HeldError("prediction/truth row identity drift")
        y = [
            truth_row["query_labels"][query_id]
            for query_id in packet_row["query_ids"]
        ]
        if set(y) != set(packet["classes"]):
            raise GRBJP4HeldError("truth row must cover every registered class")
        arm_metrics = {}
        arm_predictions = {}
        arm_memberships = {}
        for arm in ARMS:
            expected_after = (
                packet["classes"]
                if packet_row["K"] == 1 or arm in ("M0", "M_DA")
                else packet_row["old_classes"] + packet_row["new_classes"]
            )
            before_prediction, _, _ = _validate_payload(
                pred_row["before"][arm],
                query_count=len(y),
                expected_classes=packet_row["old_classes"],
                tie_tokens=packet_row["class_tie_tokens"],
            )
            after_prediction, _, membership = _validate_payload(
                pred_row["after"][arm],
                query_count=len(y),
                expected_classes=expected_after,
                tie_tokens=packet_row["class_tie_tokens"],
            )
            arm_predictions[arm] = after_prediction
            arm_memberships[arm] = membership
            arm_metrics[arm] = _arm_metrics(
                before_prediction,
                after_prediction,
                y,
                old_classes=packet_row["old_classes"],
                new_classes=packet_row["new_classes"],
                all_classes=packet["classes"],
            )
        control_metrics: dict[str, dict[str, Any]] = {}
        for name in COUNTERFACTUALS:
            control_metrics[name] = {}
            for arm in ("M_DA", "M_DA92"):
                expected_classes = (
                    packet["classes"]
                    if packet_row["K"] == 1 or arm == "M_DA"
                    else packet_row["old_classes"] + packet_row["new_classes"]
                )
                control_prediction, _, membership = _validate_payload(
                    pred_row["counterfactuals"][name][arm],
                    query_count=len(y),
                    expected_classes=expected_classes,
                    tie_tokens=packet_row["class_tie_tokens"],
                )
                before_arm = "M0" if arm == "M_DA" else "M92"
                before_prediction = pred_row["before"][before_arm]["prediction"]
                control_metrics[name][arm] = {
                    "prediction": control_prediction,
                    "neighbor_membership": membership,
                    "metrics": _arm_metrics(
                        before_prediction,
                        control_prediction,
                        y,
                        old_classes=packet_row["old_classes"],
                        new_classes=packet_row["new_classes"],
                        all_classes=packet["classes"],
                    ),
                }
        row_metric = {
            "row_id": packet_row["row_id"],
            "pseudo_new": packet_row["pseudo_new"],
            "scene": packet_row["scene"],
            "K": packet_row["K"],
            "arms": arm_metrics,
            "counterfactuals": {
                name: {
                    arm: value["metrics"]
                    for arm, value in control_metrics[name].items()
                }
                for name in COUNTERFACTUALS
            },
            "resource": packet_row["resource"],
        }
        metrics.append(row_metric)
        comparisons.append(
            _comparison_record(
                packet_row=packet_row,
                name="G_DA",
                baseline_arm="M0",
                candidate_arm="M_DA",
                arms=arm_metrics,
                predictions=arm_predictions,
                memberships=arm_memberships,
                truth=y,
                counterfactual_arms=control_metrics,
            )
        )
        comparisons.append(
            _comparison_record(
                packet_row=packet_row,
                name="G_DA92",
                baseline_arm="M92",
                candidate_arm="M_DA92",
                arms=arm_metrics,
                predictions=arm_predictions,
                memberships=arm_memberships,
                truth=y,
                counterfactual_arms=control_metrics,
            )
        )
    summary_by_k = []
    summary_by_k_scene = []
    summary_by_k_pseudo = []
    for comparison in ("G_DA", "G_DA92"):
        summary_by_k.extend(
            _matched_causal_summary(comparisons, ("K",), comparison)
        )
        summary_by_k_scene.extend(
            _matched_causal_summary(comparisons, ("K", "scene"), comparison)
        )
        summary_by_k_pseudo.extend(
            _matched_causal_summary(
                comparisons, ("K", "pseudo_new"), comparison
            )
        )
    resource_gate = all(
        row["resource"]["update_factor_wire_bytes"] <= 4096
        and row["resource"]["support_fit_mac_upper_bound"]
        < row["resource"]["support_fit_mac_limit"]
        and all(
            value <= STATE_LIMIT_BYTES
            for value in row["resource"]["full_arm_state_bytes"].values()
        )
        and all(
            value <= POST_BACKBONE_MAC_LIMIT
            for value in row["resource"][
                "full_arm_post_backbone_mac_per_query"
            ].values()
        )
        for row in packet["rows"]
    )
    all_summaries = summary_by_k + summary_by_k_scene + summary_by_k_pseudo
    comparison_pass = {
        name: all(
            item["gate_pass"]
            for item in all_summaries
            if item["comparison"] == name
        )
        for name in ("G_DA", "G_DA92")
    }
    held_gate = bool(
        resource_gate
        and comparison_pass["G_DA"]
        and comparison_pass["G_DA92"]
    )
    return {
        "schema": SCHEMA + ".score.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "promotion_scope": SCOPE,
        "formal_phase2_eligible": False,
        "target25_authorized": False,
        "target25_blocked_reason": (
            "HELD54_REQUIRES_PASS_PLUS_INDEPENDENT_REVIEW_AND_SEPARATE_"
            "TARGET25_PREREGISTRATION"
        ),
        "packet_sha256": packet["packet_sha256"],
        "COMMIT": commit,
        "truth_sha256": truth_sha256,
        "metrics": metrics,
        "comparisons": comparisons,
        "summary_by_K": summary_by_k,
        "summary_by_K_scene": summary_by_k_scene,
        "summary_by_K_pseudo_new": summary_by_k_pseudo,
        "resource_gate_pass": resource_gate,
        "comparison_gate_pass": comparison_pass,
        "held_proxy_gate_pass": held_gate,
        "verdict": (
            "PHASE1_HELD_PROXY_PASS_REVIEW_REQUIRED"
            if held_gate
            else "PHASE1_HELD_PROXY_NEGATIVE"
        ),
    }


def _method_lock_for_ground(
    ground: cfm.GroundCFMInput, qknn_lock: Phase1ZIDStudentTLock
) -> cfm.CFMMethodLock:
    return cfm.CFMMethodLock(
        qknn_neighbor_count=qknn_lock.active_k,
        student_nu=qknn_lock.student_nu,
        kernel_effective_dim=float(qknn_lock.kernel_effective_dim),
        kernel_volume_gamma=qknn_lock.kernel_volume_gamma,
        kernel_scale=qknn_lock.shared_h0,
        qknn_lock_digest=qknn_lock.lock_digest,
        phase1_method_lock_sha256=ground.phase1_method_lock_sha256,
        delta_q=float(ground.delta_q_fp16),
        tau_q=float(ground.tau_q_fp16),
        scale_prior_strength=qknn_lock.scale_prior_strength,
        scale_min_ratio=qknn_lock.scale_min_ratio,
        scale_max_ratio=qknn_lock.scale_max_ratio,
        temperature=qknn_lock.temperature,
    )


def _relabel_payload(
    payload: Mapping[str, Any], inverse: Mapping[str, str]
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["classes"] = [inverse[value] for value in result["classes"]]
    result["prediction"] = [inverse[value] for value in result["prediction"]]
    return result


def _numeric_runtime_root(row: Mapping[str, Any]) -> str:
    arrays: list[str] = []
    for wire in (
        *row["runtimes"].values(),
        *row["counterfactual_runtimes"].values(),
    ):
        bank, metric = _runtime_unwire(wire)
        for value in (
            bank.codes_qint8,
            bank.scales_fp16,
            bank.class_indices_int16,
            bank.class_scales_fp16,
            metric.basis_codes_qint8,
            metric.basis_scales_fp16,
            metric.attenuation_fp16,
        ):
            arrays.append(canonical_array_sha256(np.asarray(value)))
    for wire in row["d92_states"].values():
        for name in (
            "coefficient_fp32",
            "intercept_fp32",
            "old_covariance_fp32",
            "new_covariance_fp32",
        ):
            arrays.append(canonical_array_sha256(_decode_array(wire[name])))
    return _sha(arrays)


def _inverse_metric_labels(
    value: Any, inverse: Mapping[str, str]
) -> Any:
    if isinstance(value, Mapping):
        return {
            inverse.get(str(key), str(key)): _inverse_metric_labels(item, inverse)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_inverse_metric_labels(item, inverse) for item in value]
    if isinstance(value, str):
        return inverse.get(value, value)
    return value


def audit_label_permutation(
    tap_archive: Mapping[str, Any],
    *,
    coverage_sha256: str,
    artifact_binding: Mapping[str, Any],
    packet: Mapping[str, Any] | None = None,
    query: Mapping[str, Any] | None = None,
    truth: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
    score: Mapping[str, Any] | None = None,
    pseudo_new: str | None = None,
    scene: str = SCENES[0],
    k_shot: int = 5,
) -> dict[str, Any]:
    """Genuinely refit one seeded within-role label permutation.

    Numeric rows and physical IDs remain byte-identical.  Only opaque class
    handles are permuted within the selected row's old/new roles; the complete
    builder→bundle→solver→heads path is rerun before inverse comparison.
    """

    if packet is None or query is None or truth is None:
        packet, query, truth = build_packet(
            tap_archive,
            coverage_sha256=coverage_sha256,
            artifact_binding=artifact_binding,
        )
    _verify_packet(packet)
    if prediction is None:
        prediction = predict_packet(packet, query)
    if score is None:
        score = score_packet(
            packet,
            prediction,
            truth,
            commit=prediction["COMMIT"],
            truth_sha256=truth["truth_sha256"],
        )
    chosen_new = packet["classes"][-1] if pseudo_new is None else str(pseudo_new)
    if (
        chosen_new not in packet["classes"]
        or scene not in SCENES
        or k_shot not in K_VALUES
    ):
        raise GRBJP4HeldError("label audit selected cell drift")
    old = [value for value in packet["classes"] if value != chosen_new]
    new = [chosen_new]
    generator = np.random.Generator(np.random.PCG64(COUNTERFACTUAL_SEED))
    old_permutation = generator.permutation(len(old))
    if np.array_equal(old_permutation, np.arange(len(old))):
        old_permutation = np.roll(old_permutation, -1)
    new_permutation = generator.permutation(len(new))
    permuted_old = [old[int(index)] for index in old_permutation]
    permuted_new = [new[int(index)] for index in new_permutation]
    mapping = dict(zip(old + new, permuted_old + permuted_new))
    inverse = {value: key for key, value in mapping.items()}
    if len(inverse) != len(mapping) or all(mapping[key] == key for key in old):
        raise GRBJP4HeldError("label audit Fisher-Yates permutation is trivial")

    permuted_archive = {
        name: np.array(value, copy=True)
        for name, value in _validate_archive(tap_archive).items()
    }
    permuted_archive["labels"] = np.asarray(
        [mapping[str(value)] for value in permuted_archive["labels"]]
    )
    permuted_archive["class_ids"] = np.asarray(
        [mapping[str(value)] for value in permuted_archive["class_ids"]]
    )
    permuted_binding = dict(artifact_binding)
    permuted_binding["archive_sha256"] = _sha(
        {
            "base": artifact_binding["archive_sha256"],
            "label_permutation": mapping,
        }
    )
    permuted_binding["manifest_sha256"] = _sha(
        {
            "base": artifact_binding["manifest_sha256"],
            "label_permutation": mapping,
        }
    )
    perm_packet, perm_query, perm_truth = build_packet(
        permuted_archive,
        coverage_sha256=coverage_sha256,
        artifact_binding=permuted_binding,
    )
    perm_prediction = predict_packet(perm_packet, perm_query)
    perm_score = score_packet(
        perm_packet,
        perm_prediction,
        perm_truth,
        commit=perm_prediction["COMMIT"],
        truth_sha256=perm_truth["truth_sha256"],
    )

    original_rows = {
        (row["pseudo_new"], row["scene"], row["K"]): row
        for row in packet["rows"]
    }
    permuted_rows = {
        (row["pseudo_new"], row["scene"], row["K"]): row
        for row in perm_packet["rows"]
    }
    original_predictions = {
        row["row_id"]: row for row in prediction["rows"]
    }
    permuted_predictions = {
        row["row_id"]: row for row in perm_prediction["rows"]
    }
    original_metrics = {row["row_id"]: row for row in score["metrics"]}
    permuted_metrics = {
        row["row_id"]: row for row in perm_score["metrics"]
    }
    original_comparisons = {
        (row["row_id"], row["comparison"]): row
        for row in score["comparisons"]
    }
    permuted_comparisons = {
        (row["row_id"], row["comparison"]): row
        for row in perm_score["comparisons"]
    }
    original_grounds = {
        name: _ground_unwire(packet["ground_variants"][name])
        for name in GROUND_VARIANTS
    }
    permuted_grounds = {
        name: _ground_unwire(perm_packet["ground_variants"][name])
        for name in GROUND_VARIANTS
    }
    ground_numeric_equal = all(
        all(
            canonical_array_sha256(np.asarray(getattr(original_grounds[name], field)))
            == canonical_array_sha256(
                np.asarray(getattr(permuted_grounds[name], field))
            )
            for field in (
                "prototype_codes",
                "prototype_scales",
                "prototype_mask",
                "prototype_weights",
                "prototype_radii",
                "left_codes",
                "left_scales",
                "right_codes",
                "right_scales",
                "direction_energy",
                "delta_q_fp16",
                "tau_q_fp16",
            )
        )
        and [inverse[value] for value in permuted_grounds[name].old_class_order]
        == list(original_grounds[name].old_class_order)
        for name in GROUND_VARIANTS
    )
    original_lookup = {
        str(value): index
        for index, value in enumerate(np.asarray(query["query_ids"]).tolist())
    }
    permuted_lookup = {
        str(value): index
        for index, value in enumerate(
            np.asarray(perm_query["query_ids"]).tolist()
        )
    }
    query_receipt_equal = bool(
        packet["held_receiver"] == perm_packet["held_receiver"]
        and packet["coverage_sha256"] == perm_packet["coverage_sha256"]
        and np.array_equal(query["query_ids"], perm_query["query_ids"])
        and all(
            np.array_equal(query[name], perm_query[name])
            for name in ("z_id", "hidden", "pre_relu")
        )
        and query["query_binding_sha256"] == perm_query["query_binding_sha256"]
    )
    theta_equal = True
    resource_equal = True
    numeric_state_equal = ground_numeric_equal
    adapted_equal = True
    prediction_equal = True
    metrics_equal = True
    split_receipts_equal = query_receipt_equal
    theta_mismatches: list[dict[str, Any]] = []
    qknn_locks = build_phase1_qknn_locks()
    for key, original_row in original_rows.items():
        mapped_key = (mapping[key[0]], key[1], key[2])
        if mapped_key not in permuted_rows:
            raise GRBJP4HeldError("label audit row alignment drift")
        permuted_row = permuted_rows[mapped_key]
        split_receipts_equal = bool(
            split_receipts_equal
            and original_row["support_physical_ids"]
            == permuted_row["support_physical_ids"]
            and original_row["query_ids"] == permuted_row["query_ids"]
            and original_row["support_labels"]
            == [inverse[value] for value in permuted_row["support_labels"]]
        )
        for name in ("real_q4", *COUNTERFACTUALS):
            if (
                original_row["fit_state_receipts"][name]["theta_code_sha256"]
                != permuted_row["fit_state_receipts"][name][
                    "theta_code_sha256"
                ]
            ):
                theta_mismatches.append(
                    {
                        "cell": list(key),
                        "mapped_cell": list(mapped_key),
                        "variant": name,
                    }
                )
        theta_equal = not theta_mismatches
        resource_equal = bool(
            resource_equal and original_row["resource"] == permuted_row["resource"]
        )
        numeric_state_equal = bool(
            numeric_state_equal
            and _numeric_runtime_root(original_row)
            == _numeric_runtime_root(permuted_row)
        )
        ids = original_row["query_ids"]
        original_indices = np.asarray([original_lookup[value] for value in ids])
        permuted_indices = np.asarray([permuted_lookup[value] for value in ids])
        for name in ("real_q4", *COUNTERFACTUALS):
            ground_name = "real_q4" if name == "ground_off" else name
            qlock = qknn_locks[original_row["K"]]
            original_ground = original_grounds[ground_name]
            permuted_ground = permuted_grounds[ground_name]
            original_state = _fit_unwire(
                original_row["fit_states"][name],
                ground=original_ground,
                lock=_method_lock_for_ground(original_ground, qlock),
                checkpoint_sha256=packet["input_artifact_binding"][
                    "checkpoint_sha256"
                ],
            )
            permuted_state = _fit_unwire(
                permuted_row["fit_states"][name],
                ground=permuted_ground,
                lock=_method_lock_for_ground(permuted_ground, qlock),
                checkpoint_sha256=perm_packet["input_artifact_binding"][
                    "checkpoint_sha256"
                ],
            )
            original_adapted = _adapt_features(
                query["z_id"][original_indices],
                query["hidden"][original_indices],
                query["pre_relu"][original_indices],
                original_state,
                original_ground,
            )
            permuted_adapted = _adapt_features(
                perm_query["z_id"][permuted_indices],
                perm_query["hidden"][permuted_indices],
                perm_query["pre_relu"][permuted_indices],
                permuted_state,
                permuted_ground,
            )
            adapted_equal = bool(
                adapted_equal
                and np.array_equal(original_adapted, permuted_adapted)
            )
        original_prediction_row = original_predictions[original_row["row_id"]]
        normalized_prediction = copy.deepcopy(
            permuted_predictions[permuted_row["row_id"]]
        )
        normalized_prediction["row_id"] = original_row["row_id"]
        for stage in ("before", "after"):
            normalized_prediction[stage] = {
                arm: _relabel_payload(payload, inverse)
                for arm, payload in normalized_prediction[stage].items()
            }
        normalized_prediction["counterfactuals"] = {
            name: {
                arm: _relabel_payload(payload, inverse)
                for arm, payload in arms.items()
            }
            for name, arms in normalized_prediction["counterfactuals"].items()
        }
        prediction_equal = bool(
            prediction_equal
            and _canon(normalized_prediction) == _canon(original_prediction_row)
        )
        normalized_metric = _inverse_metric_labels(
            permuted_metrics[permuted_row["row_id"]], inverse
        )
        normalized_metric["row_id"] = original_row["row_id"]
        metrics_equal = bool(
            metrics_equal
            and _canon(normalized_metric)
            == _canon(original_metrics[original_row["row_id"]])
        )
        for comparison in ("G_DA", "G_DA92"):
            normalized_comparison = _inverse_metric_labels(
                permuted_comparisons[
                    (permuted_row["row_id"], comparison)
                ],
                inverse,
            )
            normalized_comparison["row_id"] = original_row["row_id"]
            metrics_equal = bool(
                metrics_equal
                and _canon(normalized_comparison)
                == _canon(
                    original_comparisons[
                        (original_row["row_id"], comparison)
                    ]
                )
            )

    def normalized_summary(
        values: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        normalized = [_inverse_metric_labels(value, inverse) for value in values]
        return sorted(normalized, key=_canon)

    gates_equal = all(
        _canon(sorted(score[name], key=_canon))
        == _canon(normalized_summary(perm_score[name]))
        for name in (
            "summary_by_K",
            "summary_by_K_scene",
            "summary_by_K_pseudo_new",
        )
    ) and all(
        score[name] == perm_score[name]
        for name in (
            "resource_gate_pass",
            "comparison_gate_pass",
            "held_proxy_gate_pass",
            "verdict",
        )
    )
    gate = all(
        (
            theta_equal,
            resource_equal,
            numeric_state_equal,
            adapted_equal,
            prediction_equal,
            metrics_equal,
            gates_equal,
            split_receipts_equal,
        )
    )
    return {
        "schema": SCHEMA + ".label-permutation-audit.v2",
        "seed": COUNTERFACTUAL_SEED,
        "selected_cell": {
            "pseudo_new": chosen_new,
            "scene": scene,
            "K": k_shot,
        },
        "old_group_fisher_yates": True,
        "new_group_fisher_yates": True,
        "new_group_singleton": len(new) == 1,
        "mapping": mapping,
        "theta_mismatches": theta_mismatches,
        "theta_bytes_equal": theta_equal,
        "resource_receipts_equal": resource_equal,
        "unlabeled_numeric_state_equal": numeric_state_equal,
        "adapted_numeric_features_equal": adapted_equal,
        "predictions_equal_after_inverse": prediction_equal,
        "metrics_equal_after_inverse": metrics_equal,
        "gates_equal_after_inverse": gates_equal,
        "held_receiver_and_split_receipts_equal": split_receipts_equal,
        "gate_pass": gate,
    }


def _write_new(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return _canon(value) + b"\n"


def _source_commit(value: Any) -> str:
    commit = str(value)
    if (
        len(commit) != 40
        or commit != commit.lower()
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise GRBJP4HeldError("source_git_commit must be a lowercase Git SHA1")
    return commit


def _write_query_npz_new(path: Path, query: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(query["schema"]),
            candidate=np.asarray(query["candidate"]),
            evaluation_scope=np.asarray(query["evaluation_scope"]),
            packet_core_sha256=np.asarray(query["packet_core_sha256"]),
            query_ids=np.asarray(query["query_ids"]),
            z_id=np.asarray(query["z_id"]),
            hidden=np.asarray(query["hidden"]),
            pre_relu=np.asarray(query["pre_relu"]),
            query_binding_sha256=np.asarray(query["query_binding_sha256"]),
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read_query_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "candidate",
            "evaluation_scope",
            "packet_core_sha256",
            "query_ids",
            "z_id",
            "hidden",
            "pre_relu",
            "query_binding_sha256",
        }
        if set(archive.files) != required:
            raise GRBJP4HeldError("external query member allowlist drift")
        return {
            name: (
                str(archive[name].item())
                if name
                in {
                    "schema",
                    "candidate",
                    "evaluation_scope",
                    "packet_core_sha256",
                    "query_binding_sha256",
                }
                else np.asarray(archive[name])
            )
            for name in archive.files
        }


def write_row_shard_artifacts(
    output_dir: str | Path,
    packet: Mapping[str, Any],
    query: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    source_git_commit: str,
) -> dict[str, Any]:
    """Publish one independently fitted row shard without overwriting."""

    _verify_row_shard(packet, query, truth)
    commit = _source_commit(source_git_commit)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    packet_path = root / PACKET_NAME
    query_path = root / QUERY_NAME
    truth_path = root / TRUTH_NAME
    _write_new(packet_path, _json_bytes(packet))
    _write_query_npz_new(query_path, query)
    _write_new(truth_path, _json_bytes(truth))
    row = packet["rows"][0]
    receipt = {
        "schema": ROW_SHARD_RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "target25_authorized": False,
        "source_git_commit": commit,
        "row_id": row["row_id"],
        "pseudo_new": row["pseudo_new"],
        "scene": row["scene"],
        "K": row["K"],
        "packet_file_sha256": _sha_file(packet_path),
        "query_file_sha256": _sha_file(query_path),
        "truth_file_sha256": _sha_file(truth_path),
        "packet_sha256": packet["packet_sha256"],
        "query_binding_sha256": packet["query_binding_sha256"],
        "truth_commitment_sha256": packet["truth_commitment_sha256"],
    }
    receipt["receipt_sha256"] = _sha(receipt)
    _write_new(root / ROW_SHARD_RECEIPT_NAME, _json_bytes(receipt))
    return receipt


def load_row_shard_artifacts(
    output_dir: str | Path,
    *,
    expected_source_git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(output_dir)
    expected = {
        PACKET_NAME,
        QUERY_NAME,
        TRUTH_NAME,
        ROW_SHARD_RECEIPT_NAME,
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise GRBJP4HeldError("row shard artifact member allowlist drift")
    receipt = json.loads(
        (root / ROW_SHARD_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    signed = dict(receipt)
    actual_receipt = signed.pop("receipt_sha256", None)
    if (
        receipt.get("schema") != ROW_SHARD_RECEIPT_SCHEMA
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("evaluation_scope") != SCOPE
        or receipt.get("target25_authorized") is not False
        or receipt.get("source_git_commit")
        != _source_commit(expected_source_git_commit)
        or _sha(signed) != actual_receipt
        or _sha_file(root / PACKET_NAME) != receipt.get("packet_file_sha256")
        or _sha_file(root / QUERY_NAME) != receipt.get("query_file_sha256")
        or _sha_file(root / TRUTH_NAME) != receipt.get("truth_file_sha256")
    ):
        raise GRBJP4HeldError("row shard receipt/SHA256 drift")
    packet = json.loads((root / PACKET_NAME).read_text(encoding="utf-8"))
    query = _read_query_npz(root / QUERY_NAME)
    truth = json.loads((root / TRUTH_NAME).read_text(encoding="utf-8"))
    _verify_row_shard(packet, query, truth)
    row = packet["rows"][0]
    if (
        receipt["row_id"] != row["row_id"]
        or receipt["pseudo_new"] != row["pseudo_new"]
        or receipt["scene"] != row["scene"]
        or receipt["K"] != row["K"]
        or receipt["packet_sha256"] != packet["packet_sha256"]
        or receipt["query_binding_sha256"] != packet["query_binding_sha256"]
        or receipt["truth_commitment_sha256"]
        != packet["truth_commitment_sha256"]
    ):
        raise GRBJP4HeldError("row shard semantic receipt drift")
    return packet, query, truth, receipt


def write_build_artifacts(
    output_dir: str | Path,
    packet: Mapping[str, Any],
    query: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Write four non-overwriting externally hash-bound build artifacts."""

    _verify_packet(packet)
    if (
        _query_binding(query) != packet["query_binding_sha256"]
        or truth.get("truth_sha256") != packet["truth_commitment_sha256"]
    ):
        raise GRBJP4HeldError("build artifact semantic binding drift")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    packet_path = root / PACKET_NAME
    truth_path = root / TRUTH_NAME
    query_path = root / QUERY_NAME
    _write_new(packet_path, _json_bytes(packet))
    _write_new(truth_path, _json_bytes(truth))
    _write_query_npz_new(query_path, query)
    receipt = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "target25_authorized": False,
        "packet_file_sha256": _sha_file(packet_path),
        "query_file_sha256": _sha_file(query_path),
        "truth_file_sha256": _sha_file(truth_path),
        "packet_sha256": packet["packet_sha256"],
        "packet_core_sha256": packet["packet_core_sha256"],
        "query_binding_sha256": packet["query_binding_sha256"],
        "truth_commitment_sha256": packet["truth_commitment_sha256"],
    }
    receipt["receipt_sha256"] = _sha(receipt)
    _write_new(root / BUILD_RECEIPT_NAME, _json_bytes(receipt))
    return receipt


def load_prediction_inputs(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load packet/query for prediction without parsing the truth artifact."""

    root = Path(output_dir)
    expected = {PACKET_NAME, QUERY_NAME, TRUTH_NAME, BUILD_RECEIPT_NAME}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise GRBJP4HeldError("external build artifact member allowlist drift")
    receipt = json.loads(
        (root / BUILD_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    signed = dict(receipt)
    actual_receipt = signed.pop("receipt_sha256", None)
    if (
        receipt.get("schema") != BUILD_RECEIPT_SCHEMA
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("evaluation_scope") != SCOPE
        or receipt.get("target25_authorized") is not False
        or _sha(signed) != actual_receipt
        or _sha_file(root / PACKET_NAME) != receipt.get("packet_file_sha256")
        or _sha_file(root / QUERY_NAME) != receipt.get("query_file_sha256")
        or _sha_file(root / TRUTH_NAME) != receipt.get("truth_file_sha256")
    ):
        raise GRBJP4HeldError("external build receipt/SHA256 drift")
    packet = json.loads((root / PACKET_NAME).read_text(encoding="utf-8"))
    query = _read_query_npz(root / QUERY_NAME)
    _verify_packet(packet)
    if (
        packet["packet_sha256"] != receipt["packet_sha256"]
        or packet["packet_core_sha256"] != receipt["packet_core_sha256"]
        or packet["query_binding_sha256"] != receipt["query_binding_sha256"]
        or packet["truth_commitment_sha256"]
        != receipt["truth_commitment_sha256"]
        or _query_binding(query) != packet["query_binding_sha256"]
    ):
        raise GRBJP4HeldError("external prediction-input semantic receipt drift")
    return packet, query


def load_build_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Scorer-side full loader; truth is parsed only after prediction inputs."""

    root = Path(output_dir)
    packet, query = load_prediction_inputs(root)
    truth = json.loads((root / TRUTH_NAME).read_text(encoding="utf-8"))
    if (
        truth.get("schema") != SCHEMA + ".truth.v1"
        or truth.get("candidate") != CANDIDATE
        or truth.get("evaluation_scope") != SCOPE
        or truth.get("packet_core_sha256") != packet["packet_core_sha256"]
        or truth.get("truth_sha256") != packet["truth_commitment_sha256"]
        or _sha(
            {
                key: value
                for key, value in truth.items()
                if key != "truth_sha256"
            }
        )
        != truth.get("truth_sha256")
    ):
        raise GRBJP4HeldError("external scorer truth semantic receipt drift")
    return packet, query, truth


def write_prediction_artifact(
    path: str | Path, prediction: Mapping[str, Any]
) -> str:
    data = _json_bytes(prediction)
    _write_new(Path(path), data)
    return _sha(data)


def write_score_artifact(path: str | Path, score: Mapping[str, Any]) -> str:
    data = _json_bytes(score)
    _write_new(Path(path), data)
    return _sha(data)


__all__ = [
    "ARMS",
    "BUILD_RECEIPT_NAME",
    "CANDIDATE",
    "COUNTERFACTUALS",
    "COUNTERFACTUAL_SEED",
    "GRBJP4HeldError",
    "K_VALUES",
    "PACKET_NAME",
    "PREDICTION_NAME",
    "QUERY_NAME",
    "ROW_COUNT",
    "SCENES",
    "SCHEMA",
    "SCOPE",
    "SCORE_NAME",
    "TRUTH_NAME",
    "_matched_causal_summary",
    "audit_label_permutation",
    "build_packet",
    "load_build_artifacts",
    "load_prediction_inputs",
    "predict_packet",
    "score_packet",
    "write_build_artifacts",
    "write_prediction_artifact",
    "write_score_artifact",
]
