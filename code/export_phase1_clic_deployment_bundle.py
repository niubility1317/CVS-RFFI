"""Immutable source-only CLIC deployment bundle container and verifier.

The archive contains only model/CLIC state plus aggregate source geometry and
source-frozen decision state.  It explicitly excludes raw IQ, row features,
row logits, identities, proxy rows, target rows, and replaceable sidecars.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import torch

import export_phase1_clic_features as _clean
import evaluate_phase1_clic_postfreeze_pair as _pair
from cvsrffi import phase1_clic as _clic


BUNDLE_SCHEMA = "cvs.phase1.clic_deployment_bundle.v1"
STATE_FORMAT = "cvs.phase1.clic_state_bytes.v1"
MEMBER_NAMES = (
    "model_state.bin",
    "clic_state.bin",
    "source_geometry.json",
    "source_frozen_unknown_rule.json",
    "config.json",
    "manifest.json",
)
STATE_MEMBER_NAMES = ("model_state.bin", "clic_state.bin")
EXPECTED_OPERATOR_MODE = "complex_local_invariant_curvature"
EXPECTED_SCENARIOS = tuple(_clic.FORMAL_LEO_WEAK_SCENARIOS)
LOCAL_CLASS_COUNT = 4
REAL_RULE_SCHEMA = "cvs.phase1.clic_source_frozen_unknown_rules.v1"
CLIC_STATE_PREFIX = "id_backbone.clic."
RUNTIME_REBUILD_SCHEMA = "cvs.phase1.clic_runtime_rebuild.v1"

# Exactly the model construction surface used by post_stage_common's
# build_baseline_model.  No data path, sample identity, source split, or
# receiver/target state is serialized in this deployment recipe.
RUNTIME_MODEL_DEFAULTS: dict[str, Any] = {
    "num_classes": LOCAL_CLASS_COUNT,
    "num_domains": 1,
    "model_size": "M",
    "dataset": "wisig",
    "input_len": _clic.CLIC_INPUT_LENGTH,
    "sample_rate_hz": 25e6,
    "id_feature_key": "feat_joint",
    "dom_feature_key": "feat_imp",
    # lite_d is the smallest frozen CLIC-compatible 160-dimensional identity
    # head; lite_c is deliberately rejected by the model's own CLIC contract.
    "model_variant": "lite_d",
    "branch_ablation": "none",
    "mixstyle_on": False,
    "mixstyle_p": 0.3,
    "mixstyle_alpha": 0.1,
    "mixstyle_eps": 1e-6,
    "mixstyle_layers": "time_down,t1",
    "mixstyle_use_domain_label": True,
    "mixstyle_mix": "crossdomain",
    "mixstyle_strength": 1.0,
    "mixstyle_fallback": "random",
    "domain_branch_ablation": "same",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    "use_circularity": True,
    "use_freq_stats": True,
    "use_pa_stats": True,
    "use_freq_band_gate": True,
    "freq_feature_source": "raw_fft",
    "pa_feature_source": "raw_iq",
    "pa_orders": None,
    "use_aux_spectral_stats": True,
    "channel_trim_scale": 1.0,
    "id_time_stability_mode": "off",
    "id_freq_stability_mode": "off",
    "domain_time_stability_mode": "off",
    "domain_freq_stability_mode": "off",
    "time_stability_channels": 8,
    "freq_stability_channels": 4,
    "fast_infer_when_no_aux": True,
    "use_tx_adv_on_zdom": False,
    "arch_family": "cvsincnet",
    "representation_mode": "dual",
    "phase1_clic_frozen_mode": True,
    "phase1_clic_operator_mode": EXPECTED_OPERATOR_MODE,
}
RUNTIME_MODEL_KEYS = frozenset(RUNTIME_MODEL_DEFAULTS)


class CLICBundleError(RuntimeError):
    """Raised when an immutable CLIC bundle cannot be verified fail-closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, Mapping):
            return {str(key): normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CLICBundleError("non-finite value cannot enter CLIC bundle")
            return item
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        raise CLICBundleError(f"unsupported CLIC bundle JSON type: {type(item).__name__}")
    try:
        return json.dumps(normalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CLICBundleError("cannot canonicalize CLIC bundle state") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise CLICBundleError(f"{label} hash is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CLICBundleError(f"{label} hash is invalid") from exc
    return value


def _reject_forbidden(value: Any, *, label: str) -> None:
    forbidden = (
        "raw_iq", "clean_iq", "received_iq", "sample_feature", "sample_logit", "target_row", "proxy_row",
        "receiver_id", "sample_id", "physical_id", "target_id", "proxy_id", "query_row",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            lower = str(key).lower()
            provenance_hash = lower in {
                "received_iq_sha256", "existing_received_iq_sha256",
                "physical_order_sha256",
            }
            if not provenance_hash and any(token in lower for token in forbidden):
                raise CLICBundleError(f"forbidden CLIC bundle state field: {key}")
            _reject_forbidden(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden(item, label=label)


def _state_value_bytes(value: Any) -> tuple[dict[str, Any], bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"kind": "bytes", "size_bytes": len(raw), "sha256": _sha256_bytes(raw)}, raw
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
    else:
        raise CLICBundleError("model/CLIC state values must be bytes, numpy arrays, or tensors")
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise CLICBundleError("model/CLIC state dtype is unsafe")
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise CLICBundleError("model/CLIC state contains non-finite floating values")
    raw = array.tobytes(order="C")
    return {
        "kind": "array", "dtype": array.dtype.str, "shape": list(array.shape),
        "size_bytes": len(raw), "sha256": _sha256_bytes(raw),
    }, raw


def _pack_state(state: Mapping[str, Any], *, label: str) -> bytes:
    if not isinstance(state, Mapping) or not state:
        raise CLICBundleError(f"{label} must be a nonempty mapping")
    descriptors: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    for key in sorted(str(item) for item in state):
        if not key or key not in state:
            raise CLICBundleError(f"{label} key is invalid")
        _reject_forbidden({key: None}, label=label)
        descriptor, raw = _state_value_bytes(state[key])
        descriptors.append({"key": key, **descriptor})
        chunks.append(raw)
    header = {"schema": STATE_FORMAT, "entries": descriptors}
    return b"CLICSTATE1\n" + _canonical_json_bytes(header) + b"\n" + b"".join(chunks)


def _unpack_state(payload: bytes, *, label: str) -> dict[str, Any]:
    marker = b"CLICSTATE1\n"
    if not payload.startswith(marker):
        raise CLICBundleError(f"{label} state header is invalid")
    rest = payload[len(marker):]
    try:
        header_raw, blob = rest.split(b"\n", 1)
        header = json.loads(header_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLICBundleError(f"{label} state descriptor is invalid") from exc
    if not isinstance(header, dict) or header.get("schema") != STATE_FORMAT or not isinstance(header.get("entries"), list):
        raise CLICBundleError(f"{label} state schema is invalid")
    offset = 0
    result: dict[str, Any] = {}
    for descriptor in header["entries"]:
        if not isinstance(descriptor, Mapping) or set(descriptor).difference({"key", "kind", "dtype", "shape", "size_bytes", "sha256"}):
            raise CLICBundleError(f"{label} state descriptor fields are invalid")
        key = descriptor.get("key")
        kind = descriptor.get("kind")
        size = descriptor.get("size_bytes")
        if not isinstance(key, str) or not key or key in result or type(size) is not int or size < 0:
            raise CLICBundleError(f"{label} state descriptor is invalid")
        _reject_forbidden({key: None}, label=label)
        raw = blob[offset:offset + size]
        offset += size
        if len(raw) != size or _sha256_bytes(raw) != _require_sha256(descriptor.get("sha256"), label=f"{label} member"):
            raise CLICBundleError(f"{label} state byte hash drifted")
        if kind == "bytes":
            if set(descriptor) != {"key", "kind", "size_bytes", "sha256"}:
                raise CLICBundleError(f"{label} bytes descriptor fields drifted")
            result[key] = raw
        elif kind == "array":
            if set(descriptor) != {"key", "kind", "dtype", "shape", "size_bytes", "sha256"}:
                raise CLICBundleError(f"{label} array descriptor fields drifted")
            try:
                dtype = np.dtype(str(descriptor["dtype"]))
                shape = tuple(int(item) for item in descriptor["shape"])
            except (TypeError, ValueError) as exc:
                raise CLICBundleError(f"{label} array descriptor shape/dtype is invalid") from exc
            if (
                dtype.hasobject
                or dtype.fields is not None
                or dtype.kind not in {"b", "i", "u", "f", "c"}
                or any(item < 0 for item in shape)
                or int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != size
            ):
                raise CLICBundleError(f"{label} array state byte/shape/dtype drifted")
            array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            if dtype.kind in {"f", "c"} and not np.isfinite(array).all():
                raise CLICBundleError(f"{label} state contains non-finite floating values")
            result[key] = array
        else:
            raise CLICBundleError(f"{label} state descriptor kind is invalid")
    if offset != len(blob) or not result:
        raise CLICBundleError(f"{label} state byte length drifted")
    return result


def _member_descriptor(name: str, payload: bytes) -> dict[str, Any]:
    return {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}


def _rule_with_scene_hashes(rule: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(rule, Mapping):
        raise CLICBundleError("source-frozen unknown rule must be a mapping")
    _reject_forbidden(rule, label="source-frozen unknown rule")
    normalized = dict(rule)
    supplied = normalized.get("rule_sha256")
    core = {key: value for key, value in normalized.items() if key != "rule_sha256"}
    rule_sha = _canonical_sha256(core)
    if supplied is not None and supplied != rule_sha:
        raise CLICBundleError("source-frozen unknown rule hash drifted")
    normalized["rule_sha256"] = rule_sha
    policies = normalized.get("per_scene_policies")
    scene_hashes: dict[str, str] = {}
    if policies is not None:
        if not isinstance(policies, Mapping) or set(str(key) for key in policies) != set(EXPECTED_SCENARIOS):
            raise CLICBundleError("per-scene CLIC policy coverage is incomplete")
        for scene in EXPECTED_SCENARIOS:
            policy = policies[scene]
            if not isinstance(policy, Mapping) or policy.get("scene") != scene:
                raise CLICBundleError("per-scene CLIC policy scene binding drifted")
            policy_sha = policy.get("policy_rule_sha256", policy.get("rule_sha256"))
            scene_hashes[scene] = _require_sha256(policy_sha, label=f"{scene} policy/rule")
    else:
        # Compatibility path for minimal mechanical fixtures.  It still produces a
        # distinct scene-bound rule digest and never substitutes geometry SHA.
        for scene in EXPECTED_SCENARIOS:
            scene_hashes[scene] = _canonical_sha256({"scene": scene, "source_rule_sha256": rule_sha})
    return normalized, scene_hashes


def _validated_real_source_rule(
    rule: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the aggregate-only three-scene rule with PAIR's frozen API."""

    if not isinstance(rule, Mapping):
        raise CLICBundleError("real source-frozen unknown rule must be a mapping")
    _reject_forbidden(rule, label="real source-frozen unknown rule")
    expected_fields = {
        "schema", "geometry_state_sha256", "received_iq_sha256",
        "physical_order_sha256", "per_scene_policies", "state_sha256",
    }
    if set(rule) != expected_fields or rule.get("schema") != REAL_RULE_SCHEMA:
        raise CLICBundleError("real source-frozen unknown rule fields/schema drifted")
    try:
        _, _, _, geometry_sha = _pair._validated_geometry(geometry)
    except _pair.CLICPostfreezePairError as exc:
        raise CLICBundleError("real source geometry failed strict PAIR validation") from exc
    if rule.get("geometry_state_sha256") != geometry_sha:
        raise CLICBundleError("real source-frozen rule geometry SHA binding drifted")
    received_sha = _require_sha256(rule.get("received_iq_sha256"), label="real source received-IQ")
    physical_sha = _require_sha256(rule.get("physical_order_sha256"), label="real source physical order")
    policies = rule.get("per_scene_policies")
    if not isinstance(policies, Mapping) or set(str(key) for key in policies) != set(EXPECTED_SCENARIOS):
        raise CLICBundleError("real source-frozen rule lacks exact three-scene policies")
    scene_hashes: dict[str, str] = {}
    for scene in EXPECTED_SCENARIOS:
        policy = policies.get(scene)
        try:
            _, _, policy_sha = _pair._validated_policy(policy, geometry=geometry, scene=scene)
        except _pair.CLICPostfreezePairError as exc:
            raise CLICBundleError(f"real {scene} source tail policy failed strict PAIR validation") from exc
        if (
            policy.get("geometry_state_sha256") != geometry_sha
            or policy.get("received_iq_sha256") != received_sha
            or policy.get("physical_order_sha256") != physical_sha
        ):
            raise CLICBundleError(f"real {scene} policy geometry/received/physical binding drifted")
        scene_hashes[scene] = policy_sha
    payload = dict(rule)
    state_sha = payload.pop("state_sha256", None)
    _require_sha256(state_sha, label="real source-frozen rule state")
    if _canonical_sha256(payload) != state_sha:
        raise CLICBundleError("real source-frozen rule state hash drifted")
    return dict(rule), scene_hashes


def _require_real_tensor_state(state: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise CLICBundleError(f"real {label} state must be nonempty")
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise CLICBundleError(f"real {label} state key is invalid")
        if not (torch.is_tensor(value) or isinstance(value, np.ndarray)):
            raise CLICBundleError(f"real {label} state must retain tensor shape/dtype/bytes")


def _validated_real_bundle_state(
    *,
    model_state: Mapping[str, Any],
    clic_state: Mapping[str, Any],
    source_geometry: Mapping[str, Any],
    source_frozen_unknown_rule: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Cross-check all state that makes a bundle eligible for a real reload."""

    _require_real_tensor_state(model_state, label="model")
    _require_real_tensor_state(clic_state, label="CLIC")
    checkpoint_clic = {
        key: value for key, value in model_state.items()
        if key.startswith(CLIC_STATE_PREFIX)
    }
    if not checkpoint_clic:
        raise CLICBundleError("real model state lacks id_backbone.clic.* checkpoint subset")
    if set(clic_state) != set(checkpoint_clic):
        raise CLICBundleError("CLIC state keys do not equal checkpoint id_backbone.clic.* subset")
    if _pack_state(clic_state, label="CLIC") != _pack_state(checkpoint_clic, label="checkpoint CLIC subset"):
        raise CLICBundleError("CLIC state bytes/shape/dtype do not equal checkpoint CLIC subset")
    try:
        _pair._validated_geometry(source_geometry)
    except _pair.CLICPostfreezePairError as exc:
        raise CLICBundleError("real source geometry failed strict PAIR validation") from exc
    return _validated_real_source_rule(
        source_frozen_unknown_rule, geometry=source_geometry
    )


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CLICBundleError(f"{label} must not be boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CLICBundleError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise CLICBundleError(f"{label} must be finite")
    return number


def _validate_runtime_rebuild(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CLICBundleError("real CLIC runtime rebuild config must be a mapping")
    expected_fields = {"schema", "input_len", "model_kwargs"}
    if set(value) != expected_fields or value.get("schema") != RUNTIME_REBUILD_SCHEMA:
        raise CLICBundleError("real CLIC runtime rebuild config fields/schema drifted")
    input_len = value.get("input_len")
    if type(input_len) is not int or input_len != int(_clic.CLIC_INPUT_LENGTH):
        raise CLICBundleError("real CLIC runtime input length drifted")
    kwargs = value.get("model_kwargs")
    if not isinstance(kwargs, Mapping) or set(kwargs) != RUNTIME_MODEL_KEYS:
        raise CLICBundleError("real CLIC runtime model reconstruction keys drifted")
    normalized = dict(kwargs)
    for field in ("num_classes", "num_domains", "input_len", "time_stability_channels", "freq_stability_channels"):
        if type(normalized.get(field)) is not int or int(normalized[field]) <= 0:
            raise CLICBundleError(f"real CLIC runtime {field} is invalid")
    if normalized["num_classes"] != LOCAL_CLASS_COUNT or normalized["input_len"] != input_len:
        raise CLICBundleError("real CLIC runtime local4/input-length binding drifted")
    for field in (
        "sample_rate_hz", "mixstyle_p", "mixstyle_alpha", "mixstyle_eps", "mixstyle_strength",
        "domain_enhancer_strength", "channel_trim_scale",
    ):
        normalized[field] = _finite_number(normalized.get(field), label=f"real CLIC runtime {field}")
    for field in (
        "mixstyle_on", "mixstyle_use_domain_label", "fast_infer_when_no_aux",
        "phase1_clic_frozen_mode", "use_circularity", "use_freq_stats", "use_pa_stats",
        "use_freq_band_gate", "use_aux_spectral_stats", "use_tx_adv_on_zdom",
    ):
        if type(normalized.get(field)) is not bool:
            raise CLICBundleError(f"real CLIC runtime {field} must be boolean")
    for field in (
        "model_size", "dataset", "id_feature_key", "dom_feature_key", "model_variant",
        "branch_ablation", "mixstyle_layers", "mixstyle_mix", "mixstyle_fallback",
        "domain_branch_ablation", "domain_enhancer", "id_time_stability_mode",
        "id_freq_stability_mode", "domain_time_stability_mode", "domain_freq_stability_mode",
        "arch_family", "representation_mode", "phase1_clic_operator_mode",
        "freq_feature_source", "pa_feature_source",
    ):
        if not isinstance(normalized.get(field), str) or not normalized[field]:
            raise CLICBundleError(f"real CLIC runtime {field} is invalid")
    if (
        normalized["phase1_clic_frozen_mode"] is not True
        or normalized["phase1_clic_operator_mode"] != EXPECTED_OPERATOR_MODE
        or normalized["representation_mode"] != "dual"
        or normalized["arch_family"] != "cvsincnet"
    ):
        raise CLICBundleError("real CLIC runtime frozen invariant-curvature model binding drifted")
    pa_orders = normalized.get("pa_orders")
    if pa_orders is not None:
        if not isinstance(pa_orders, (list, tuple)) or not pa_orders:
            raise CLICBundleError("real CLIC runtime pa_orders must be nonempty list or null")
        if any(type(order) is not int or order <= 0 or order % 2 == 0 for order in pa_orders):
            raise CLICBundleError("real CLIC runtime pa_orders must be positive odd integers")
        normalized["pa_orders"] = [int(order) for order in pa_orders]
    _reject_forbidden(normalized, label="real CLIC runtime model config")
    return {
        "schema": RUNTIME_REBUILD_SCHEMA,
        "input_len": int(input_len),
        "model_kwargs": normalized,
    }


def _validate_config(config: Mapping[str, Any], *, require_runtime_rebuild: bool = False) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise CLICBundleError("CLIC bundle config must be a mapping")
    normalized = dict(config)
    expected = {"z_id_dim", "z_dom_dim", "q_clic_dim"}
    if require_runtime_rebuild:
        expected.add("runtime_rebuild")
    if set(normalized) != expected:
        raise CLICBundleError("CLIC bundle config fields drifted")
    for field in ("z_id_dim", "z_dom_dim", "q_clic_dim"):
        value = normalized.get(field)
        if type(value) is not int or value <= 0:
            raise CLICBundleError(f"CLIC bundle config {field} is invalid")
    if require_runtime_rebuild:
        normalized["runtime_rebuild"] = _validate_runtime_rebuild(normalized.get("runtime_rebuild"))
    _reject_forbidden(normalized, label="CLIC bundle config")
    return normalized


def _runtime_rebuild_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only architecture values required for an offline strict rebuild."""

    args = checkpoint.get("args")
    state = checkpoint.get("model")
    if not isinstance(args, Mapping) or not isinstance(state, Mapping) or not state:
        raise CLICBundleError("real CLIC checkpoint lacks args/model for runtime reconstruction")
    try:
        from cvsrffi.checkpoint_loading import infer_num_domains_from_state
        num_domains = int(infer_num_domains_from_state(state))
    except Exception as exc:
        raise CLICBundleError("cannot infer real CLIC runtime domain width from checkpoint state") from exc
    input_len = args.get("wisig_out_len")
    if type(input_len) is not int or input_len != int(_clic.CLIC_INPUT_LENGTH):
        raise CLICBundleError("real CLIC checkpoint input length drifted")
    model_kwargs = dict(RUNTIME_MODEL_DEFAULTS)
    model_kwargs["num_domains"] = num_domains
    model_kwargs["input_len"] = int(input_len)
    for key in (
        "num_classes", "model_size", "dataset", "sample_rate_hz", "id_feature_key",
        "dom_feature_key", "model_variant", "branch_ablation", "domain_branch_ablation",
        "domain_enhancer", "domain_enhancer_strength", "id_time_stability_mode",
        "id_freq_stability_mode", "domain_time_stability_mode", "domain_freq_stability_mode",
        "time_stability_channels", "freq_stability_channels", "fast_infer_when_no_aux",
        "arch_family", "representation_mode", "phase1_clic_frozen_mode", "phase1_clic_operator_mode",
        "use_circularity", "use_freq_stats", "use_pa_stats", "use_freq_band_gate",
        "freq_feature_source", "pa_feature_source", "pa_orders", "use_aux_spectral_stats",
        "channel_trim_scale", "use_tx_adv_on_zdom",
    ):
        if key in args:
            model_kwargs[key] = args[key]
    for bundle_key, checkpoint_key in (
        ("mixstyle_on", "use_mixstyle"),
        ("mixstyle_p", "mixstyle_p"),
        ("mixstyle_alpha", "mixstyle_alpha"),
        ("mixstyle_eps", "mixstyle_eps"),
        ("mixstyle_layers", "mixstyle_layers"),
        ("mixstyle_use_domain_label", "mixstyle_use_domain_label"),
        ("mixstyle_mix", "mixstyle_mix"),
        ("mixstyle_strength", "mixstyle_strength"),
        ("mixstyle_fallback", "mixstyle_fallback"),
    ):
        if checkpoint_key in args:
            model_kwargs[bundle_key] = args[checkpoint_key]
    return _validate_runtime_rebuild(
        {
            "schema": RUNTIME_REBUILD_SCHEMA,
            "input_len": int(input_len),
            "model_kwargs": model_kwargs,
        }
    )


def _rebuild_real_model(
    model_state: Mapping[str, Any],
    *,
    runtime_rebuild: Mapping[str, Any],
) -> torch.nn.Module:
    """Materialize the sealed state once, with no checkpoint or source access."""

    runtime = _validate_runtime_rebuild(runtime_rebuild)
    _require_real_tensor_state(model_state, label="model")
    tensor_state: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
        else:
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject or array.dtype.fields is not None or not array.dtype.isnative:
                raise CLICBundleError("real model state dtype cannot be reconstructed")
            try:
                tensor = torch.from_numpy(array)
            except (TypeError, ValueError) as exc:
                raise CLICBundleError("real model state tensor conversion failed") from exc
        if tensor.dtype == torch.bfloat16:
            raise CLICBundleError("real model bfloat16 state is not an allowed sealed format")
        tensor_state[key] = tensor
    try:
        from model_dual_cvsincnet import build_dual_model
        model = build_dual_model(**runtime["model_kwargs"]).to(torch.device("cpu"))
        model.load_state_dict(tensor_state, strict=True)
        model.eval()
    except (RuntimeError, TypeError, ValueError, ImportError) as exc:
        raise CLICBundleError("strict real CLIC model reconstruction/state load failed") from exc
    return model


def _checkpoint_terminal_binding(checkpoint_path: Path, terminal_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not checkpoint_path.is_file() or not terminal_path.is_file():
        raise CLICBundleError("CLIC bundle checkpoint or terminal receipt is missing")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICBundleError("cannot load CLIC final checkpoint for bundle") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise CLICBundleError("CLIC final checkpoint is malformed")
    args = checkpoint["args"]
    try:
        source = _clean._parse_csv(args.get("phase1_source_train_tx_ids", ""), label="checkpoint source TX IDs")
        known = _clean._parse_csv(args.get("phase1_source_known_validation_tx_ids", ""), label="checkpoint held validation TX IDs")
        proxy = _clean._parse_csv(args.get("phase1_source_proxy_unknown_tx_ids", ""), label="checkpoint proxy TX IDs")
        _, receipt, arm = _clean.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_path,
            terminal_receipt_path=terminal_path,
            source_tx_ids=source,
            known_validation_tx_ids=known,
            proxy_unknown_tx_ids=proxy,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICBundleError(f"CLIC bundle terminal/checkpoint reopening failed: {exc}") from exc
    if arm != "G" or args.get("phase1_clic_operator_mode") != EXPECTED_OPERATOR_MODE:
        raise CLICBundleError("only frozen G invariant-curvature CLIC checkpoints may produce deployment bundles")
    return checkpoint, receipt


def _bundle_clean_masks(
    clean: Mapping[str, Any],
    *,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
    checkpoint_sha256: str,
    terminal_receipt_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bind a clean exporter artifact to the current G checkpoint before fitting."""

    manifest = clean.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CLICBundleError("bundle clean feature manifest is absent")
    if (
        manifest.get("source_checkpoint_sha256") != checkpoint_sha256
        or manifest.get("terminal_receipt_sha256") != terminal_receipt_sha256
        or tuple(str(item) for item in manifest.get("source_tx_ids", ())) != tuple(source_tx_ids)
        or tuple(str(item) for item in manifest.get("known_validation_tx_ids", ())) != tuple(known_validation_tx_ids)
        or tuple(str(item) for item in manifest.get("proxy_unknown_tx_ids", ())) != tuple(proxy_unknown_tx_ids)
    ):
        raise CLICBundleError("bundle clean NPZ checkpoint/terminal/TX binding drifted")
    roles = np.asarray(clean.get("roles"), dtype=str).reshape(-1)
    tx_ids = np.asarray(clean.get("tx_ids"), dtype=str).reshape(-1)
    row_count = int(clean.get("row_count", -1))
    if roles.size != row_count or tx_ids.size != row_count:
        raise CLICBundleError("bundle clean NPZ role/TX row alignment drifted")
    labeled = roles == "labeled_fit"
    validation = roles == "source_validation_known"
    proxy = roles == "proxy_unknown"
    if not (np.any(labeled) and np.any(validation) and int(np.sum(proxy)) == 400):
        raise CLICBundleError("bundle clean NPZ L/V/fixed400 role rows do not close")
    if set(tx_ids[labeled]) != set(source_tx_ids) or set(tx_ids[validation]) != set(known_validation_tx_ids):
        raise CLICBundleError("bundle clean NPZ source-L/source-V TX roles drifted")
    if set(tx_ids[proxy]) != set(proxy_unknown_tx_ids):
        raise CLICBundleError("bundle clean NPZ fixed400 proxy TX role drifted")
    if any(int(np.sum(tx_ids[labeled] == name)) <= 1 for name in source_tx_ids):
        raise CLICBundleError("bundle clean NPZ every source-L local4 class needs more than one row")
    return labeled, validation, proxy


def _strict_leo_artifact_for_bundle(
    *,
    leo_npz_path: str | Path,
    leo_binding_path: str | Path,
    source_tx_ids: Sequence[str],
    checkpoint_sha256: str,
    terminal_receipt_sha256: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Reopen the one-existing-observation LEO NPZ and its byte binding."""

    loaded = _pair._load_binding_json(leo_binding_path, "G")
    binding = loaded["binding"]
    feature = loaded["feature"]
    passed_path = Path(leo_npz_path).resolve()
    if passed_path != Path(feature["path"]).resolve():
        raise CLICBundleError("bundle LEO NPZ path does not equal the sealed binding NPZ")
    if binding.get("checkpoint_sha256") != checkpoint_sha256 or binding.get("terminal_receipt_sha256") != terminal_receipt_sha256:
        raise CLICBundleError("bundle LEO binding checkpoint/terminal SHA drifted")
    if tuple(str(item) for item in binding.get("source_tx_ids", ())) != tuple(source_tx_ids):
        raise CLICBundleError("bundle LEO binding source TX ordering drifted")
    arrays = feature.get("arrays")
    if not isinstance(arrays, Mapping):
        raise CLICBundleError("bundle LEO NPZ arrays are absent")
    tx_ids = np.asarray(feature.get("tx_ids"), dtype=str).reshape(-1)
    if set(tx_ids).difference(source_tx_ids):
        raise CLICBundleError("bundle LEO NPZ contains non-source or target TX rows")
    scenes = np.asarray(arrays.get("sat_scenarios"), dtype=str).reshape(-1)
    slots = np.asarray(arrays.get("source_rx_slot"))
    physical = np.asarray(arrays.get("physical_sample_id"), dtype=str).reshape(-1)
    rx_ids = np.asarray(arrays.get("rx_ids"), dtype=str).reshape(-1)
    day_ids = np.asarray(arrays.get("day_ids"), dtype=str).reshape(-1)
    if not all(values.size == feature["row_count"] for values in (scenes, slots.reshape(-1), physical, rx_ids, day_ids)):
        raise CLICBundleError("bundle LEO NPZ metadata row alignment drifted")
    if slots.dtype.kind not in {"i", "u"} or np.any(slots < 0) or np.any(slots >= 7):
        raise CLICBundleError("bundle LEO NPZ source RX slots drifted")
    if set(scenes) != set(EXPECTED_SCENARIOS):
        raise CLICBundleError("bundle LEO NPZ formal three-scene coverage drifted")
    if not all(str(value) for value in physical):
        raise CLICBundleError("bundle LEO NPZ physical_sample_id is empty")
    # Stable physical identity is global, including across the three sealed
    # scenes.  Scene metadata cannot make one raw emission eligible for more
    # than one tail-calibration cell or bundle policy.
    if len(set(physical)) != int(feature["row_count"]):
        raise CLICBundleError("bundle LEO NPZ physical_sample_id must be globally unique across scenes")
    physical_keys = [
        "|".join((tx_ids[index], rx_ids[index], day_ids[index], physical[index]))
        for index in range(int(feature["row_count"]))
    ]
    if len(physical_keys) != len(set(physical_keys)):
        raise CLICBundleError("bundle LEO NPZ physical row order has duplicates")
    if binding.get("physical_keys") != physical_keys or _canonical_sha256(physical_keys) != binding.get("physical_order_sha256"):
        raise CLICBundleError("bundle LEO binding physical order does not equal current NPZ rows")
    return feature, binding


def _source_rule_from_clean_and_leo(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_file: Path,
    terminal_file: Path,
    clean_npz_path: str | Path,
    leo_npz_path: str | Path,
    leo_binding_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive all real bundle state from sealed G artifacts, never CLI values."""

    args = checkpoint.get("args")
    model_state = checkpoint.get("model")
    if not isinstance(args, Mapping) or not isinstance(model_state, Mapping) or not model_state:
        raise CLICBundleError("real G checkpoint lacks args/model state for bundle derivation")
    try:
        source_tx_ids = _clean._parse_csv(args.get("phase1_source_train_tx_ids", ""), label="checkpoint source TX IDs")
        known_validation_tx_ids = _clean._parse_csv(args.get("phase1_source_known_validation_tx_ids", ""), label="checkpoint held validation TX IDs")
        proxy_unknown_tx_ids = _clean._parse_csv(args.get("phase1_source_proxy_unknown_tx_ids", ""), label="checkpoint proxy TX IDs")
    except _clean.CLICSplitExportError as exc:
        raise CLICBundleError("real G checkpoint source/validation/proxy TX args are invalid") from exc
    if len(source_tx_ids) != LOCAL_CLASS_COUNT or len(known_validation_tx_ids) != 1 or len(proxy_unknown_tx_ids) != 1:
        raise CLICBundleError("real G checkpoint local4/held/proxy TX cardinality drifted")
    checkpoint_sha = _sha256_file(checkpoint_file)
    terminal_sha = _sha256_file(terminal_file)
    clean = _pair._load_feature_npz(clean_npz_path, _clean.EXPECTED_LV_EXPORT_SCHEMA, "G")
    labeled, validation, proxy = _bundle_clean_masks(
        clean,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
    )
    diagnostic = _pair.compute_clic_proxy_diagnostic(
        clean["z_id"][labeled], clean["tx_ids"][labeled], clean["z_id"][validation],
        clean["z_id"][proxy], clean["tx_ids"][proxy], source_tx_ids,
    )
    geometry = diagnostic["geometry"]
    leo, binding = _strict_leo_artifact_for_bundle(
        leo_npz_path=leo_npz_path,
        leo_binding_path=leo_binding_path,
        source_tx_ids=source_tx_ids,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
    )
    arrays = leo["arrays"]
    scenes = np.asarray(arrays["sat_scenarios"], dtype=str).reshape(-1)
    slots = np.asarray(arrays["source_rx_slot"], dtype=np.int64).reshape(-1)
    policies: dict[str, Any] = {}
    physical_binding = {
        "received_iq_sha256": binding["received_iq_sha256"],
        "physical_order_sha256": binding["physical_order_sha256"],
        "source_only": True,
        "single_leo_observation": True,
    }
    for scene in EXPECTED_SCENARIOS:
        scene_mask = scenes == scene
        policies[scene] = _pair.freeze_clic_tail_policy(
            geometry, leo["z_id"][scene_mask], scenes[scene_mask], slots[scene_mask], leo["tx_ids"][scene_mask], physical_binding
        )
    source_rule: dict[str, Any] = {
        "schema": REAL_RULE_SCHEMA,
        "geometry_state_sha256": geometry["state_sha256"],
        "received_iq_sha256": binding["received_iq_sha256"],
        "physical_order_sha256": binding["physical_order_sha256"],
        "per_scene_policies": policies,
    }
    source_rule["state_sha256"] = _canonical_sha256(source_rule)
    candidate_match = _clean.EXPECTED_CANDIDATE_PATTERN.fullmatch(str(args.get("candidate_id", "")))
    if candidate_match is None or candidate_match.group(2) != "G":
        raise CLICBundleError("real G checkpoint candidate/fold binding drifted during source policy sealing")
    source_policy_state = _pair.build_clic_source_policy_state(
        fold_index=int(candidate_match.group(1)),
        arm="G",
        operator_mode=EXPECTED_OPERATOR_MODE,
        geometry=geometry,
        policies=policies,
        checkpoint_sha256=checkpoint_sha,
        terminal_receipt_sha256=terminal_sha,
    )
    clic_state = {
        key: value for key, value in model_state.items()
        if str(key).startswith(CLIC_STATE_PREFIX)
    }
    if not clic_state:
        raise CLICBundleError("real G checkpoint lacks id_backbone.clic.* state for bundle")
    runtime_rebuild = _runtime_rebuild_from_checkpoint(checkpoint)
    model = _rebuild_real_model(model_state, runtime_rebuild=runtime_rebuild)
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise CLICBundleError("real G checkpoint model has no parameters") from exc
    synthetic_input = torch.zeros(
        (1, 2, int(runtime_rebuild["input_len"])), dtype=parameter.dtype, device=torch.device("cpu")
    )
    try:
        with torch.no_grad():
            outputs = model(synthetic_input, y_tx=None, grl_lambda=1.0, return_aux=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CLICBundleError("real G checkpoint cannot produce sealed runtime output dimensions") from exc
    if not isinstance(outputs, Mapping):
        raise CLICBundleError("real G checkpoint sealed runtime output is not a mapping")
    output_dims: dict[str, int] = {}
    for field in ("z_id", "z_dom", "q_clic"):
        value = outputs.get(field)
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != 1 or value.shape[1] <= 0:
            raise CLICBundleError(f"real G checkpoint sealed runtime output {field} shape drifted")
        output_dims[field] = int(value.shape[1])
    if output_dims["z_id"] != int(geometry["feature_dim"]):
        raise CLICBundleError("real G checkpoint z_id dimension does not equal source geometry")
    config = {
        "z_id_dim": output_dims["z_id"],
        "z_dom_dim": output_dims["z_dom"],
        "q_clic_dim": output_dims["q_clic"],
    }
    return dict(model_state), clic_state, geometry, source_rule, source_policy_state, config


def _zip_write(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100600 << 16
    archive.writestr(info, payload)


def export_bundle(
    *,
    checkpoint_path: str | Path,
    terminal_receipt_path: str | Path,
    output_path: str | Path,
    model_state: Mapping[str, Any] | None = None,
    clic_state: Mapping[str, Any] | None = None,
    source_geometry: Mapping[str, Any] | None = None,
    source_frozen_unknown_rule: Mapping[str, Any] | None = None,
    source_policy_state: Mapping[str, Any] | None = None,
    operator_mode: str | None = None,
    config: Mapping[str, Any] | None = None,
    clean_npz_path: str | Path | None = None,
    leo_npz_path: str | Path | None = None,
    leo_binding_path: str | Path | None = None,
) -> str:
    """Create one immutable G-fold archive; it never copies the training checkpoint.

    The public CLI passes only the sealed G checkpoint/terminal plus clean and
    existing-LEO raw artifact paths.  This function then reopens and derives
    model state, CLIC subset, geometry, and all three source tail policies.
    The explicit-state branch remains only for the pre-existing synthetic
    archive mechanics fixture; callers cannot reach it through ``main``.
    """

    checkpoint_file = Path(checkpoint_path).resolve()
    terminal_file = Path(terminal_receipt_path).resolve()
    checkpoint, _ = _checkpoint_terminal_binding(checkpoint_file, terminal_file)
    raw_paths = (clean_npz_path, leo_npz_path, leo_binding_path)
    explicit_state = (model_state, clic_state, source_geometry, source_frozen_unknown_rule, operator_mode, config)
    if any(value is not None for value in raw_paths):
        if not all(value is not None for value in raw_paths):
            raise CLICBundleError("bundle raw artifact derivation requires clean, LEO NPZ, and LEO binding together")
        if any(value is not None for value in explicit_state) or source_policy_state is not None:
            raise CLICBundleError("bundle cannot mix raw-derived and caller-injected state")
        model_state, clic_state, source_geometry, source_frozen_unknown_rule, source_policy_state, config = _source_rule_from_clean_and_leo(
            checkpoint=checkpoint,
            checkpoint_file=checkpoint_file,
            terminal_file=terminal_file,
            clean_npz_path=clean_npz_path,
            leo_npz_path=leo_npz_path,
            leo_binding_path=leo_binding_path,
        )
        operator_mode = EXPECTED_OPERATOR_MODE
    elif any(value is None for value in explicit_state):
        raise CLICBundleError("bundle explicit-state branch is incomplete")
    assert model_state is not None
    assert clic_state is not None
    assert source_geometry is not None
    assert source_frozen_unknown_rule is not None
    assert operator_mode is not None
    assert config is not None
    if operator_mode != EXPECTED_OPERATOR_MODE:
        raise CLICBundleError("CLIC deployment bundle operator mode drifted")
    output = Path(output_path).resolve()
    if output.exists():
        raise CLICBundleError(f"refusing to overwrite immutable CLIC bundle: {output}")
    normalized_config = _validate_config(config)
    _reject_forbidden(source_geometry, label="source geometry")
    if not isinstance(source_geometry, Mapping):
        raise CLICBundleError("source geometry must be a mapping")
    model_payload = _pack_state(model_state, label="model")
    clic_payload = _pack_state(clic_state, label="CLIC")
    checkpoint_model = checkpoint.get("model")
    state_origin = "synthetic_fixture"
    if isinstance(checkpoint_model, Mapping) and checkpoint_model:
        if model_payload != _pack_state(checkpoint_model, label="checkpoint model"):
            raise CLICBundleError("model_state does not equal exact current checkpoint model state")
        rule, scene_rule_sha = _validated_real_bundle_state(
            model_state=model_state,
            clic_state=clic_state,
            source_geometry=source_geometry,
            source_frozen_unknown_rule=source_frozen_unknown_rule,
        )
        if source_policy_state is None:
            raise CLICBundleError("real CLIC bundle lacks pair-sealed G source policy state")
        try:
            normalized_policy_state = _pair._validated_clic_source_policy_state(
                source_policy_state,
                arm="G",
                checkpoint_sha256=_sha256_file(checkpoint_file),
                terminal_receipt_sha256=_sha256_file(terminal_file),
            )
        except _pair.CLICPostfreezePairError as exc:
            raise CLICBundleError("real CLIC bundle G source policy state failed strict PAIR validation") from exc
        if (
            normalized_policy_state.get("geometry") != source_geometry
            or normalized_policy_state.get("policies") != rule.get("per_scene_policies")
        ):
            raise CLICBundleError("real CLIC bundle G source policy state/rule drifted")
        normalized_config = _validate_config(
            {
                **normalized_config,
                "runtime_rebuild": _runtime_rebuild_from_checkpoint(checkpoint),
            },
            require_runtime_rebuild=True,
        )
        state_origin = "checkpoint_model_exact"
    else:
        if source_policy_state is not None:
            raise CLICBundleError("synthetic fixture bundle must not carry a real source policy state")
        rule, scene_rule_sha = _rule_with_scene_hashes(source_frozen_unknown_rule)
        normalized_policy_state = None
    geometry_payload = _canonical_json_bytes(dict(source_geometry)) + b"\n"
    rule_payload = _canonical_json_bytes(rule) + b"\n"
    config_payload = _canonical_json_bytes(normalized_config) + b"\n"
    members = {
        "model_state.bin": model_payload,
        "clic_state.bin": clic_payload,
        "source_geometry.json": geometry_payload,
        "source_frozen_unknown_rule.json": rule_payload,
        "config.json": config_payload,
    }
    descriptors = {name: _member_descriptor(name, payload) for name, payload in members.items()}
    content_root = _canonical_sha256(descriptors)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "member_allowlist": list(MEMBER_NAMES),
        "members": descriptors,
        "content_root_sha256": content_root,
        "state_sha256": content_root,
        "checkpoint_sha256": _sha256_file(checkpoint_file),
        "terminal_receipt_sha256": _sha256_file(terminal_file),
        "operator_mode": EXPECTED_OPERATOR_MODE,
        "state_origin": state_origin,
        "config": normalized_config,
        "z_id_shape": [1, int(normalized_config["z_id_dim"])],
        "z_id_dtype": "float64",
        "z_dom_shape": [1, int(normalized_config["z_dom_dim"])],
        "q_clic_shape": [1, int(normalized_config["q_clic_dim"])],
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "single_leo_observation_required": True,
        "source_geometry_sha256": _sha256_bytes(geometry_payload),
        "source_frozen_unknown_rule": rule,
        "clic_source_policy_state": normalized_policy_state,
        "source_frozen_unknown_rule_sha256": _sha256_bytes(rule_payload),
        "per_scene_policy_rule_sha256": scene_rule_sha,
        "bundle_has_raw_checkpoint": False,
        "bundle_has_sample_rows": False,
    }
    manifest_payload = _canonical_json_bytes(manifest) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise CLICBundleError(f"refusing to overwrite temporary CLIC bundle: {temporary}")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_STORED) as archive:
            for name in MEMBER_NAMES:
                _zip_write(archive, name, manifest_payload if name == "manifest.json" else members[name])
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return str(output)


def _read_zip_members(path: Path) -> dict[str, bytes]:
    if not path.is_file():
        raise CLICBundleError("CLIC bundle archive is missing")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != set(MEMBER_NAMES):
                forbidden = sorted(set(names).difference(MEMBER_NAMES))
                hint = forbidden[0] if forbidden else "member allowlist"
                raise CLICBundleError(f"CLIC bundle member allowlist/forbidden member drifted: {hint}")
            if any(info.is_dir() or info.compress_type != zipfile.ZIP_STORED for info in infos):
                raise CLICBundleError("CLIC bundle member format drifted")
            return {name: archive.read(name) for name in MEMBER_NAMES}
    except zipfile.BadZipFile as exc:
        raise CLICBundleError("CLIC bundle archive is invalid") from exc


def _json_member(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLICBundleError(f"CLIC bundle {label} JSON is invalid") from exc
    if not isinstance(value, dict):
        raise CLICBundleError(f"CLIC bundle {label} must be an object")
    return value


def verify_clic_bundle(path: str | Path) -> dict[str, Any]:
    """Verify every archive member and cross-binding before materializing state."""

    members = _read_zip_members(Path(path).resolve())
    manifest = _json_member(members["manifest.json"], label="manifest")
    expected_fields = {
        "schema", "member_allowlist", "members", "content_root_sha256", "state_sha256", "checkpoint_sha256",
        "terminal_receipt_sha256", "operator_mode", "state_origin", "config", "z_id_shape", "z_id_dtype",
        "z_dom_shape", "q_clic_shape", "clean_source_runtime_access", "query_fit_access",
        "single_leo_observation_required", "source_geometry_sha256", "source_frozen_unknown_rule", "clic_source_policy_state",
        "source_frozen_unknown_rule_sha256", "per_scene_policy_rule_sha256", "bundle_has_raw_checkpoint",
        "bundle_has_sample_rows",
    }
    if set(manifest) != expected_fields or manifest.get("schema") != BUNDLE_SCHEMA:
        raise CLICBundleError("CLIC bundle manifest state/schema field drifted")
    if manifest.get("member_allowlist") != list(MEMBER_NAMES):
        raise CLICBundleError("CLIC bundle member allowlist drifted")
    descriptors = manifest.get("members")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(STATE_MEMBER_NAMES + ("source_geometry.json", "source_frozen_unknown_rule.json", "config.json")):
        raise CLICBundleError("CLIC bundle member descriptor drifted")
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"sha256", "size_bytes"}:
            raise CLICBundleError("CLIC bundle member descriptor fields drifted")
        if _sha256_bytes(members[name]) != _require_sha256(descriptor.get("sha256"), label=f"{name} member") or len(members[name]) != descriptor.get("size_bytes"):
            raise CLICBundleError(f"CLIC bundle member byte hash drifted: {name}")
    content_root = _canonical_sha256(dict(descriptors))
    if manifest.get("content_root_sha256") != content_root or manifest.get("state_sha256") != content_root:
        raise CLICBundleError("CLIC bundle state/hash root drifted")
    _require_sha256(manifest.get("checkpoint_sha256"), label="checkpoint")
    _require_sha256(manifest.get("terminal_receipt_sha256"), label="terminal receipt")
    if manifest.get("operator_mode") != EXPECTED_OPERATOR_MODE:
        raise CLICBundleError("CLIC bundle operator mode drifted")
    if manifest.get("state_origin") not in {"checkpoint_model_exact", "synthetic_fixture"}:
        raise CLICBundleError("CLIC bundle state origin drifted")
    real_state = manifest.get("state_origin") == "checkpoint_model_exact"
    config = _validate_config(
        manifest.get("config"), require_runtime_rebuild=real_state
    )
    if _json_member(members["config.json"], label="config") != config:
        raise CLICBundleError("CLIC bundle config member drifted")
    if manifest.get("z_id_shape") != [1, config["z_id_dim"]] or manifest.get("z_id_dtype") != "float64":
        raise CLICBundleError("CLIC bundle z_id shape/dtype drifted")
    if manifest.get("z_dom_shape") != [1, config["z_dom_dim"]] or manifest.get("q_clic_shape") != [1, config["q_clic_dim"]]:
        raise CLICBundleError("CLIC bundle output shape drifted")
    if manifest.get("clean_source_runtime_access") is not False or manifest.get("query_fit_access") is not False or manifest.get("single_leo_observation_required") is not True:
        raise CLICBundleError("CLIC bundle source/query/single-LEO contract drifted")
    if manifest.get("bundle_has_raw_checkpoint") is not False or manifest.get("bundle_has_sample_rows") is not False:
        raise CLICBundleError("CLIC bundle forbidden raw/sample declaration drifted")
    geometry = _json_member(members["source_geometry.json"], label="source geometry")
    _reject_forbidden(geometry, label="source geometry")
    if manifest.get("source_geometry_sha256") != _sha256_bytes(members["source_geometry.json"]):
        raise CLICBundleError("CLIC bundle source geometry hash drifted")
    rule = _json_member(members["source_frozen_unknown_rule.json"], label="source-frozen unknown rule")
    model_state = _unpack_state(members["model_state.bin"], label="model")
    clic_state = _unpack_state(members["clic_state.bin"], label="CLIC")
    if manifest.get("state_origin") == "checkpoint_model_exact":
        normalized_rule, scene_hashes = _validated_real_bundle_state(
            model_state=model_state,
            clic_state=clic_state,
            source_geometry=geometry,
            source_frozen_unknown_rule=rule,
        )
    else:
        normalized_rule, scene_hashes = _rule_with_scene_hashes(rule)
    if normalized_rule != manifest.get("source_frozen_unknown_rule"):
        raise CLICBundleError("CLIC bundle source-frozen unknown rule member drifted")
    if manifest.get("source_frozen_unknown_rule_sha256") != _sha256_bytes(members["source_frozen_unknown_rule.json"]):
        raise CLICBundleError("CLIC bundle source-frozen unknown rule hash drifted")
    if manifest.get("per_scene_policy_rule_sha256") != scene_hashes:
        raise CLICBundleError("CLIC bundle per-scene policy/rule hash drifted")
    policy_state = manifest.get("clic_source_policy_state")
    if real_state:
        try:
            normalized_policy_state = _pair._validated_clic_source_policy_state(
                policy_state,
                arm="G",
                checkpoint_sha256=manifest["checkpoint_sha256"],
                terminal_receipt_sha256=manifest["terminal_receipt_sha256"],
            )
        except _pair.CLICPostfreezePairError as exc:
            raise CLICBundleError("real CLIC bundle source policy state failed strict PAIR validation") from exc
        if (
            normalized_policy_state.get("geometry") != geometry
            or normalized_policy_state.get("policies") != normalized_rule.get("per_scene_policies")
        ):
            raise CLICBundleError("real CLIC bundle source policy state/rule drifted")
    elif policy_state is not None:
        raise CLICBundleError("synthetic fixture bundle must not claim a source policy state")
    # This is deliberately derived from sealed, verified bundle state rather
    # than accepted from a caller-controlled manifest assertion.  A compact
    # synthetic fixture is useful for archive-mechanics tests only and can
    # never advertise a checkpoint-model reload.
    real_rebuild_verified = False
    if real_state:
        # A real flag is derived only after the sealed architecture recipe and
        # exact sealed tensors can independently recreate the model.
        _rebuild_real_model(
            model_state, runtime_rebuild=config["runtime_rebuild"]
        )
        real_rebuild_verified = True
    verified = dict(manifest)
    verified["real_checkpoint_state_rebuild_verified"] = real_rebuild_verified
    # Verification proves that the sealed state reconstructs; it has not yet
    # received IQ or run a model forward.  Only reload_forward can claim that.
    verified["real_checkpoint_reload_verified"] = False
    return verified


def bundle_member_names(path: str | Path) -> set[str]:
    """Return the verified archive member names, never a filesystem directory walk."""

    _read_zip_members(Path(path).resolve())
    return set(MEMBER_NAMES)


def _synthetic_fixture_reload(
    verified: Mapping[str, Any],
    received_i: Any,
) -> dict[str, Any]:
    """Deterministic archive test adapter, never a model or deployment path.

    Older mechanical tests intentionally construct an empty checkpoint plus a
    bytes-only placeholder state.  It cannot be rebuilt as a CVSincNet model;
    keep that compatibility strictly isolated and fail closed to ``defer``.
    In particular it does not score source geometry, derive a threshold, or
    claim real checkpoint-forward evidence.
    """

    try:
        values = np.asarray(received_i, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CLICBundleError("synthetic fixture received-IQ is not numeric") from exc
    if values.size <= 0 or not np.isfinite(values).all():
        raise CLICBundleError("synthetic fixture received-IQ is non-finite or empty")
    config = _validate_config(verified.get("config"))
    flat = np.ascontiguousarray(values.reshape(-1), dtype=np.float64)
    digest = hashlib.sha256(
        str(verified["state_sha256"]).encode("ascii") + flat.tobytes(order="C")
    ).digest()
    perturb = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) / 255.0

    def vector(dim: int, *, offset: int) -> np.ndarray:
        base = np.resize(flat, dim)
        noise = np.resize(perturb[offset:], dim) - 0.5
        return (base + noise * 1.0e-12).reshape(1, dim).astype(np.float64, copy=False)

    z_id = vector(int(config["z_id_dim"]), offset=0)
    z_dom = vector(int(config["z_dom_dim"]), offset=7)
    q_clic = vector(int(config["q_clic_dim"]), offset=13)
    base = float(np.sum(flat, dtype=np.float64))
    tx_logits = np.asarray(
        [[base + float(index) * 1.0e-12 for index in range(LOCAL_CLASS_COUNT)]],
        dtype=np.float64,
    )
    return {
        "z_id": z_id,
        "z_dom": z_dom,
        "q_clic": q_clic,
        "tx_logits": tx_logits,
        "e_unknown": float(np.linalg.norm(z_id, ord=2)),
        "decision": "defer",
        "state_sha256": str(verified["state_sha256"]),
        "real_checkpoint_state_rebuild_verified": False,
        "real_checkpoint_reload_verified": False,
        "synthetic_fixture": True,
    }


def _reload_members_after_verify(path: str | Path, verified: Mapping[str, Any]) -> dict[str, bytes]:
    """Close the verify-to-forward byte window without trusting a sidecar."""

    members = _read_zip_members(Path(path).resolve())
    manifest = _json_member(members["manifest.json"], label="manifest")
    expected_manifest = {
        key: value for key, value in verified.items()
        if key not in {
            "real_checkpoint_state_rebuild_verified",
            "real_checkpoint_reload_verified",
        }
    }
    if manifest != expected_manifest:
        raise CLICBundleError("CLIC bundle manifest changed after strict verification")
    descriptors = verified.get("members")
    if not isinstance(descriptors, Mapping):
        raise CLICBundleError("CLIC bundle verified member descriptors are absent")
    for name, descriptor in descriptors.items():
        if (
            name not in members
            or _sha256_bytes(members[name]) != descriptor.get("sha256")
            or len(members[name]) != descriptor.get("size_bytes")
        ):
            raise CLICBundleError("CLIC bundle member bytes changed after strict verification")
    return members


def _strict_received_iq_for_reload(received_i: Any, *, input_len: int) -> torch.Tensor:
    if torch.is_tensor(received_i):
        values = received_i.detach().cpu().numpy()
    else:
        try:
            values = np.asarray(received_i)
        except (TypeError, ValueError) as exc:
            raise CLICBundleError("CLIC reload received_i cannot be materialized") from exc
    if values.dtype.kind != "f":
        raise CLICBundleError("CLIC reload received_i must be float32 or float64")
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape != (1, 2, int(input_len)):
        raise CLICBundleError("CLIC reload received_i must have strict [2,T] or [1,2,T] shape")
    if not np.isfinite(values).all():
        raise CLICBundleError("CLIC reload received_i is non-finite")
    return torch.from_numpy(np.ascontiguousarray(values))


def _tensor_output(
    output: Mapping[str, Any],
    field: str,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    value = output.get(field)
    if not torch.is_tensor(value):
        raise CLICBundleError(f"strict CLIC reload model output lacks tensor {field}")
    array = value.detach().cpu().numpy()
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise CLICBundleError(f"strict CLIC reload {field} output shape/nonfinite drifted")
    return np.ascontiguousarray(array, dtype=np.float64)


def _reload_real_forward(
    path: str | Path,
    verified: Mapping[str, Any],
    received_i: Any,
    *,
    scene: str | None,
) -> dict[str, Any]:
    if not isinstance(scene, str) or scene not in EXPECTED_SCENARIOS:
        raise CLICBundleError("strict real CLIC reload requires one formal LEO scene")
    members = _reload_members_after_verify(path, verified)
    config = _validate_config(verified.get("config"), require_runtime_rebuild=True)
    geometry = _json_member(members["source_geometry.json"], label="source geometry")
    rule = _json_member(members["source_frozen_unknown_rule.json"], label="source-frozen unknown rule")
    model_state = _unpack_state(members["model_state.bin"], label="model")
    clic_state = _unpack_state(members["clic_state.bin"], label="CLIC")
    normalized_rule, _ = _validated_real_bundle_state(
        model_state=model_state,
        clic_state=clic_state,
        source_geometry=geometry,
        source_frozen_unknown_rule=rule,
    )
    model = _rebuild_real_model(
        model_state, runtime_rebuild=config["runtime_rebuild"]
    )
    input_tensor = _strict_received_iq_for_reload(
        received_i, input_len=int(config["runtime_rebuild"]["input_len"])
    )
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise CLICBundleError("strict real CLIC reload model has no parameters") from exc
    input_tensor = input_tensor.to(dtype=parameter.dtype, device=torch.device("cpu"))
    try:
        with torch.no_grad():
            # Exactly one received-IQ forward.  No channel synthesis, TTA,
            # source/proxy access, fit, threshold update, or feedback occurs.
            output = model(input_tensor, y_tx=None, grl_lambda=1.0, return_aux=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CLICBundleError("strict real CLIC reload forward failed") from exc
    if not isinstance(output, Mapping):
        raise CLICBundleError("strict real CLIC reload model output is not a mapping")
    z_id = _tensor_output(output, "z_id", expected_shape=(1, int(config["z_id_dim"])))
    z_dom = _tensor_output(output, "z_dom", expected_shape=(1, int(config["z_dom_dim"])))
    q_clic = _tensor_output(output, "q_clic", expected_shape=(1, int(config["q_clic_dim"])))
    tx_logits = _tensor_output(output, "tx_logits", expected_shape=(1, LOCAL_CLASS_COUNT))
    try:
        scored = _pair.score_clic_open_set(
            geometry,
            normalized_rule["per_scene_policies"][scene],
            z_id,
            tx_logits,
            scene,
        )
    except _pair.CLICPostfreezePairError as exc:
        raise CLICBundleError("strict real CLIC reload PAIR scoring failed") from exc
    return {
        "z_id": z_id,
        "z_dom": z_dom,
        "q_clic": q_clic,
        "tx_logits": tx_logits,
        "e_unknown": scored["e_unknown"],
        "decision": scored["decision"],
        "predicted_index": scored["predicted_index"],
        "predicted_class": scored["predicted_class"],
        "zero_flag": scored["zero_flag"],
        "scene": scene,
        "state_sha256": str(verified["state_sha256"]),
        "geometry_state_sha256": scored["geometry_state_sha256"],
        "policy_rule_sha256": scored["policy_rule_sha256"],
        "real_checkpoint_state_rebuild_verified": True,
        "real_checkpoint_reload_verified": True,
        "synthetic_fixture": False,
    }


def reload_forward(path: str | Path, received_i: Any, *, scene: str | None = None) -> dict[str, Any]:
    """Verify then execute exactly one sealed bundle forward.

    The synthetic branch exists solely to retain the RED archive fixture.  A
    checkpoint-backed bundle takes the independent strict reconstruction path
    below; it is intentionally not allowed to fall back to this adapter.
    """

    verified = verify_clic_bundle(path)
    if verified["state_origin"] == "synthetic_fixture":
        return _synthetic_fixture_reload(verified, received_i)
    return _reload_real_forward(path, verified, received_i, scene=scene)


def build_parser() -> argparse.ArgumentParser:
    """Expose only raw G artifacts; sealed state is always derived internally."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--terminal-receipt-json", required=True)
    parser.add_argument("--clean-npz", required=True)
    parser.add_argument("--leo-npz", required=True)
    parser.add_argument("--leo-binding-json", required=True)
    parser.add_argument("--output-bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = export_bundle(
        checkpoint_path=args.checkpoint,
        terminal_receipt_path=args.terminal_receipt_json,
        clean_npz_path=args.clean_npz,
        leo_npz_path=args.leo_npz,
        leo_binding_path=args.leo_binding_json,
        output_path=args.output_bundle,
    )
    print(json.dumps({"output_bundle": output}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
