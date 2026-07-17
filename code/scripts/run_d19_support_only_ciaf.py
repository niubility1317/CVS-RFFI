#!/usr/bin/env python3
"""Run D20 max-old/FFT-RF development selection on sealed K10 support only.

The executable deliberately exposes no query, scorer, truth, role, quota, or
batch-assignment input.  It opens only the already sealed LEO_weak enrollment
packages and one immutable int8 Phase1 aggregate-prototype component.  The
historical component is explicitly development-only; therefore every output
disables formal metric and performance claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (CODE, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cvsrffi.phase1_int8_prototype_bundle import (  # noqa: E402
    ALLOWED_NPZ_MEMBERS,
    NPZ_NAME,
    sha256_file,
    validate_int8_component,
)
from cvsrffi.somph_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
    finalize_somph_enrollment_authority_after_materialization,
    materialize_somph_enrollment_with_signed_authority,
)
from cvsrffi.stage2_ciaf import FEATURE_DIM, Int8DomainClassComponent  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    CANDIDATE_D1_B0_CAP,
    TEMPERATURE as DIAG_TEMPERATURE,
    fit_diag_cosine_state,
    log_scale_bounds,
    registered_feature,
)
from cvsrffi.stage2_dali import (  # noqa: E402
    DaliConfig,
    fit_old_dali,
    predict_one_dali,
    register_new_dali,
    rerank_old_scores_dali,
    score_one_dali,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)
MODE = "development_select_unverified_component"
SUPPORT_QUERY_DISJOINTNESS_STATUS = "SUPPORT_ONLY_NO_QUERY_CLAIM"
HELD_RANKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
IDENTITY_CANDIDATE = "Z0_SUPPORT_ONLY"
GROUND_CANDIDATE = "B1_MAXOLD_G005"
GROUND_DIRECT_CANDIDATE = "B2_MAXOLD_G005_L025"
DIAG_CANDIDATE = "B3_SINGLE_IQ_DIAG_FFTRF"
DIAG_MAXOLD_CANDIDATE = "B4_SINGLE_IQ_DIAG_FFTRF_MAXOLD"


class D19RunnerError(ValueError):
    """Raised when the historical-component D20 screen fails closed."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _tensor_from_numpy_dlpack(
    value: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Bridge NumPy to Torch without the NumPy C-API used by from_numpy."""

    rows = np.ascontiguousarray(value)
    tensor = torch.utils.dlpack.from_dlpack(rows)
    return tensor.to(dtype=dtype, device=device)


@contextmanager
def _numpy2_torch21_as_tensor_compatibility():
    """Scope the old Torch/new NumPy workaround to the imported D1 fit."""

    original = torch.as_tensor

    def compatible(value: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            if args:
                raise D19RunnerError(
                    "NumPy compatibility bridge forbids positional options"
                )
            dtype = kwargs.pop("dtype", None)
            device = kwargs.pop("device", None)
            if kwargs:
                raise D19RunnerError("NumPy compatibility bridge option drift")
            tensor = torch.utils.dlpack.from_dlpack(np.ascontiguousarray(value))
            return tensor.to(dtype=dtype, device=device)
        return original(value, *args, **kwargs)

    torch.as_tensor = compatible
    try:
        yield
    finally:
        torch.as_tensor = original


def _member(manifest: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    matches = [row for row in manifest.get("members", []) if row.get("kind") == kind]
    if len(matches) != 1:
        raise D19RunnerError(f"sealed member missing or duplicated: {kind}")
    return matches[0]


def _verified_json_member(
    root: Path, manifest: Mapping[str, Any], *, kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    member = _member(manifest, kind)
    relative = str(member.get("relative_path", ""))
    path = (root / relative).resolve(strict=True)
    if (
        path.parent != root.resolve(strict=True)
        or path.is_symlink()
        or path.name != relative
    ):
        raise D19RunnerError(f"unsafe sealed member path: {kind}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != member.get("sha256") or len(raw) != int(member.get("size_bytes", -1)):
        raise D19RunnerError(f"sealed member hash/size drift: {kind}")
    return json.loads(raw.decode("utf-8")), {
        "kind": kind,
        "relative_path": relative,
        "sha256": digest,
        "size_bytes": len(raw),
    }


def _overlay_index(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    value, audit = _verified_json_member(root, manifest, kind="overlay_provenance")
    if (
        value.get("schema") != "cvs.phase2.somph_overlay_provenance.v1"
        or value.get("receiver") != manifest.get("receiver")
        or int(value.get("seed", -1)) != int(manifest.get("seed", -2))
        or not isinstance(value.get("samples"), list)
    ):
        raise D19RunnerError("overlay provenance envelope drift")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    overlays: set[str] = set()
    for row in value["samples"]:
        key = (
            str(row.get("sample_token", "")),
            str(row.get("post_channel_iq_sha256", "")),
            str(row.get("scenario", "")),
        )
        overlay = str(row.get("overlay_token", ""))
        if (
            key in result
            or key[2] not in FORMAL_LEO_WEAK_SCENARIOS
            or len(key[1]) != 64
            or not overlay.startswith("oid_")
            or len(overlay) != 68
            or overlay in overlays
            or not isinstance(row.get("satellite_seed"), int)
        ):
            raise D19RunnerError("overlay provenance row drift")
        result[key] = dict(row)
        overlays.add(overlay)
    audit.update({"sample_count": len(result), "unique_overlay_token_count": len(overlays)})
    return result, audit


def _payload_rows(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    handles = np.asarray(
        [str(row["class_handle"]) for row in manifest["registered_classes"]]
    )
    indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    labels = handles[indices].astype(str)
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    order = np.asarray(
        sorted(range(len(labels)), key=lambda index: (labels[index], int(ranks[index]))),
        dtype=np.int64,
    )
    rows = {
        "iq": np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[order],
        "labels": labels[order],
        "ranks": ranks[order],
        "tokens": np.asarray(payload["support_tokens"]).astype(str)[order],
        "hashes": np.asarray(payload["support_post_channel_iq_sha256"]).astype(str)[order],
    }
    classes, counts = np.unique(rows["labels"], return_counts=True)
    computed = np.asarray(
        [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows["iq"]]
    )
    if (
        rows["iq"].ndim != 3
        or rows["iq"].shape[1] != 2
        or not np.isfinite(rows["iq"]).all()
        or set(counts.tolist()) != {10}
        or any(
            set(rows["ranks"][rows["labels"] == label].tolist()) != set(range(10))
            for label in classes
        )
        or not np.array_equal(computed, rows["hashes"])
        or len(set(rows["tokens"].tolist())) != len(rows["tokens"])
        or len(set(rows["hashes"].tolist())) != len(rows["hashes"])
    ):
        raise D19RunnerError(f"strict K10 payload drift: {scenario}")
    return rows


def _rows_with_overlay(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    overlay: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    rows = _payload_rows(payload, manifest, scenario=scenario)
    raw_handles = np.asarray(
        [str(row["class_handle"]) for row in manifest["registered_classes"]]
    )
    raw_labels = raw_handles[np.asarray(payload["support_class_indices"], dtype=np.int64)]
    raw_ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    order = np.asarray(
        sorted(range(len(raw_labels)), key=lambda i: (str(raw_labels[i]), int(raw_ranks[i]))),
        dtype=np.int64,
    )
    payload_overlays = np.asarray(payload["support_overlay_tokens"]).astype(str)[order]
    payload_seeds = np.asarray(payload["support_satellite_seeds"], dtype=np.int64)[order]
    bound_seeds: list[int] = []
    for index, (token, parent) in enumerate(zip(rows["tokens"], rows["hashes"])):
        item = overlay.get((str(token), str(parent), scenario))
        if item is None:
            raise D19RunnerError("support row absent from overlay provenance")
        if (
            str(item.get("overlay_token")) != str(payload_overlays[index])
            or int(item.get("satellite_seed", -1)) != int(payload_seeds[index])
        ):
            raise D19RunnerError("support NPZ/overlay binding drift")
        bound_seeds.append(int(item["satellite_seed"]))
    rows["overlay_tokens"] = payload_overlays
    rows["satellite_seeds"] = np.asarray(bound_seeds, dtype=np.int64)
    return rows


def _manifest_binding(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if (
        before.get("receiver") != after.get("receiver")
        or int(before.get("seed", -1)) != int(after.get("seed", -2))
        or int(before.get("k_shot", -1)) != 10
        or int(after.get("k_shot", -1)) != 10
        or before.get("feature_runtime_sha256") != after.get("feature_runtime_sha256")
        or before.get("phase1_checkpoint_sha256") != after.get("phase1_checkpoint_sha256")
    ):
        raise D19RunnerError("before/after package binding drift")
    old = {str(row["class_handle"]) for row in before["registered_classes"]}
    all_classes = {str(row["class_handle"]) for row in after["registered_classes"]}
    if not old < all_classes:
        raise D19RunnerError("real new-class registration set required")


def _require_post_materialization_authority(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    required = "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
    for name, audit in (("before", before), ("after", after)):
        if (
            audit.get("iq_payload_materialized") is not True
            or audit.get("formal_launch_authority") is not True
            or audit.get("formal_metric_claim_allowed") is not False
            or audit.get("support_query_disjointness_status")
            != SUPPORT_QUERY_DISJOINTNESS_STATUS
            or audit.get("runtime_authorization_schema")
            != SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA
            or audit.get("status") != required
        ):
            raise D19RunnerError(f"post-materialization authority drift: {name}")


def _old_reuse(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]) -> None:
    old_classes = set(np.asarray(before["labels"]).astype(str).tolist())
    def keyed(rows: Mapping[str, np.ndarray]) -> dict[tuple[str, int], tuple[str, str, str, int]]:
        return {
            (str(rows["labels"][i]), int(rows["ranks"][i])): (
                str(rows["tokens"][i]), str(rows["hashes"][i]),
                str(rows["overlay_tokens"][i]), int(rows["satellite_seeds"][i]),
            )
            for i in range(len(rows["labels"]))
            if str(rows["labels"][i]) in old_classes
        }
    if keyed(before) != keyed(after):
        raise D19RunnerError("before/after old support exact reuse drift")


def _cross_scene_disjointness(
    rows: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    scenarios = tuple(FORMAL_LEO_WEAK_SCENARIOS)
    for left_index, left in enumerate(scenarios):
        for right in scenarios[left_index + 1 :]:
            overlap = {
                key: len(set(rows[left][field].tolist()) & set(rows[right][field].tolist()))
                for key, field in (
                    ("physical_sample_id", "tokens"),
                    ("parent_received_iq_sha256", "hashes"),
                    ("overlay_token", "overlay_tokens"),
                )
            }
            pairs.append({"left": left, "right": right, "overlap_count": overlap, "pass": not any(overlap.values())})
    if not all(row["pass"] for row in pairs):
        raise D19RunnerError("cross-scene support reuse")
    return {"pairs": pairs, "all_pairwise_disjoint": True}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise D19RunnerError(f"refusing to overwrite output: {path}")
    raw = (
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if path.exists():
        raise D19RunnerError(f"refusing to overwrite output: {path}")
    raw = b"".join(
        (
            json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _registered_handles(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list) or not rows:
        raise D19RunnerError("registered class manifest missing")
    ordered = sorted(rows, key=lambda row: int(row["class_index"]))
    indices = [int(row["class_index"]) for row in ordered]
    handles = tuple(str(row["class_handle"]) for row in ordered)
    if indices != list(range(len(indices))) or len(set(handles)) != len(handles):
        raise D19RunnerError("registered class order drift")
    return handles


def _preopen_manifest(
    root: Path,
    seal: Path,
    *,
    expected_seal_sha256: str,
) -> dict[str, Any]:
    """Verify seal and manifest without opening any IQ archive."""

    if sha256_file(seal) != expected_seal_sha256:
        raise D19RunnerError("detached enrollment seal SHA256 drift")
    seal_value = json.loads(seal.read_text(encoding="utf-8"))
    if (
        seal_value.get("schema") != "cvs.phase2.somph_predictor_bundle_seal.v1"
        or seal_value.get("manifest_relative_path") != "package_manifest.json"
    ):
        raise D19RunnerError("enrollment seal schema drift")
    manifest_path = root / "package_manifest.json"
    raw = manifest_path.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != seal_value.get("manifest_sha256")
        or len(raw) != int(seal_value.get("manifest_size_bytes", -1))
    ):
        raise D19RunnerError("pre-open package manifest hash/size drift")
    return json.loads(raw.decode("utf-8"))


def _load_component(
    component_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    bound_old_handles: tuple[str, ...],
    class_binding_path: Path,
    expected_class_binding_sha256: str,
) -> tuple[Int8DomainClassComponent, dict[str, Any]]:
    manifest_path = component_dir / "manifest.json"
    if len(expected_manifest_sha256) != 64:
        raise D19RunnerError("detached component manifest SHA256 required")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise D19RunnerError("component manifest SHA256 drift")
    manifest = validate_int8_component(component_dir)
    if (
        manifest.get("formal_phase2_eligible") is not False
        or manifest.get("provenance_status") != "UNVERIFIED_UNDER_CURRENT_PROTOCOL"
        or manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
        or int(manifest.get("class_count", -1)) != len(bound_old_handles)
        or manifest.get("phase2_phase1_prototype_component_immutable") is not True
        or manifest.get("phase2_phase1_prototype_update_access") is not False
        or manifest.get("phase2_phase1_prototype_member_or_exemplar_access")
        is not False
    ):
        raise D19RunnerError("development-only int8 component contract drift")
    npz_path = component_dir / NPZ_NAME
    if sha256_file(class_binding_path) != expected_class_binding_sha256:
        raise D19RunnerError("ADV3B02 class binding SHA256 drift")
    binding_value = json.loads(class_binding_path.read_text(encoding="utf-8"))
    entries = binding_value.get("entries")
    binding_evidence = binding_value.get("evidence")
    if (
        binding_value.get("schema") != "cvs.phase2.d20_adv3b02_class_binding.v2"
        or binding_value.get("checkpoint_sha256") != expected_checkpoint_sha256
        or not isinstance(entries, list)
        or [int(row.get("class_index", -1)) for row in entries]
        != list(range(len(bound_old_handles)))
        or [int(row.get("direct_logit_index", -1)) for row in entries]
        != list(range(len(bound_old_handles)))
        or tuple(str(row.get("registered_class_handle", "")) for row in entries)
        != bound_old_handles
        or not isinstance(binding_evidence, dict)
        or len(str(binding_evidence.get("feature_runtime_sha256", ""))) != 64
        or not str(binding_evidence.get("direct_logit_head_state_key", ""))
        or len(str(binding_evidence.get("direct_logit_head_tensor_sha256", "")))
        != 64
        or any(
            len(str(row.get("direct_logit_weight_row_sha256", ""))) != 64
            for row in entries
        )
    ):
        raise D19RunnerError("ADV3B02 class binding contract drift")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != ALLOWED_NPZ_MEMBERS:
            raise D19RunnerError("component NPZ allowlist drift")
        phase1_registry = tuple(str(value) for value in arrays["class_registry"])
        if phase1_registry != tuple(str(row.get("phase1_tx", "")) for row in entries):
            raise D19RunnerError("Phase1 prototype column/TX binding drift")
        component = Int8DomainClassComponent(
            arrays["domain_class_q"],
            arrays["domain_class_scale"],
            arrays["domain_class_mask"],
            bound_old_handles,
            str(arrays["feature_schema"].item()),
        )
    binding = {
        "status": "DEVELOPMENT_EXACT_PHASE1_TX_TO_REGISTERED_HANDLE_BINDING",
        "formal_mapping_claim_allowed": False,
        "phase1_column_registry": list(phase1_registry),
        "phase2_bound_old_handles": list(bound_old_handles),
        "phase1_to_phase2_column_index": list(range(len(bound_old_handles))),
        "direct_logit_indices": [
            int(row["direct_logit_index"]) for row in entries
        ],
        "direct_logit_to_class_handle_order_bound": True,
        "feature_runtime_sha256": str(
            binding_evidence["feature_runtime_sha256"]
        ),
        "direct_logit_head_state_key": str(
            binding_evidence["direct_logit_head_state_key"]
        ),
        "direct_logit_head_tensor_sha256": str(
            binding_evidence["direct_logit_head_tensor_sha256"]
        ),
        "direct_logit_weight_row_sha256": [
            str(row["direct_logit_weight_row_sha256"]) for row in entries
        ],
        "class_binding_sha256": expected_class_binding_sha256,
        "strict_replay_class_mapping_required_before_formal_use": True,
        "component_manifest_sha256": actual_manifest_sha256,
        "component_npz_sha256": str(manifest["component_npz_sha256"]),
        "component_serialized_bytes": int(manifest["serialized_component_bytes"]),
        "component_logical_state_bytes": int(component.state_bytes),
        "component_provenance_status": str(manifest["provenance_status"]),
    }
    return component, {"manifest": manifest, "column_binding": binding}


def _verify_runtime_direct_logit_binding(
    model: torch.nn.Module,
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the sealed runtime head rows before any support IQ is opened."""

    if manifest.get("feature_runtime_sha256") != binding.get(
        "feature_runtime_sha256"
    ):
        raise D19RunnerError("runtime/class-binding SHA256 drift")
    state_key = str(binding.get("direct_logit_head_state_key", ""))
    state = model.state_dict()
    if state_key not in state:
        raise D19RunnerError("bound direct-logit head absent from sealed runtime")
    weight = np.ascontiguousarray(
        np.asarray(state[state_key].detach().float().cpu().tolist(), dtype=np.float32)
    )
    expected_rows = list(binding.get("direct_logit_weight_row_sha256", []))
    if weight.ndim != 2 or weight.shape[0] != len(expected_rows):
        raise D19RunnerError("sealed runtime direct-logit head shape drift")
    tensor_sha256 = hashlib.sha256(weight.tobytes()).hexdigest()
    row_sha256 = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in weight
    ]
    if (
        tensor_sha256 != binding.get("direct_logit_head_tensor_sha256")
        or row_sha256 != expected_rows
    ):
        raise D19RunnerError("sealed runtime direct-logit row binding drift")
    return {
        "verified_before_support_open": True,
        "feature_runtime_sha256": str(manifest["feature_runtime_sha256"]),
        "direct_logit_head_state_key": state_key,
        "direct_logit_head_tensor_sha256": tensor_sha256,
        "direct_logit_weight_row_sha256": row_sha256,
        "direct_logit_to_class_handle_order_bound": True,
    }


def preregistered_candidates() -> dict[str, DaliConfig]:
    """Return the small unified candidate set fixed before support opening."""

    return {
        IDENTITY_CANDIDATE: DaliConfig(ground_weight=0.0, direct_weight=0.0),
        GROUND_CANDIDATE: DaliConfig(ground_weight=0.05, direct_weight=0.0),
        GROUND_DIRECT_CANDIDATE: DaliConfig(
            ground_weight=0.05, direct_weight=0.25
        ),
        DIAG_CANDIDATE: DaliConfig(ground_weight=0.0, direct_weight=0.0),
        DIAG_MAXOLD_CANDIDATE: DaliConfig(
            ground_weight=0.05, direct_weight=0.25
        ),
    }


def _extract_scene_signals(
    model: torch.nn.Module,
    device: torch.device,
    rows: Mapping[str, np.ndarray],
    direct_logit_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    feature_hashes: list[str] = []
    logit_hashes: list[str] = []
    model.eval()
    for iq in np.asarray(rows["iq"], dtype=np.float32):
        batch = _tensor_from_numpy_dlpack(
            np.ascontiguousarray(np.asarray(iq, dtype=np.float32)[None, ...]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            output = model(batch)
        if isinstance(output, dict):
            feature_value = output.get("features")
            logit_value = output.get("logits")
        elif isinstance(output, (tuple, list)) and len(output) == 2:
            feature_value, logit_value = output
        else:
            raise D19RunnerError("sealed runtime must return features and logits")
        if not torch.is_tensor(feature_value) or not torch.is_tensor(logit_value):
            raise D19RunnerError("sealed runtime feature/logit tensor drift")
        feature = np.asarray(
            feature_value.detach().float().cpu().tolist(), dtype=np.float32
        )
        direct = np.asarray(
            logit_value.detach().float().cpu().tolist(), dtype=np.float32
        )
        if feature.shape != (1, 160) or direct.ndim != 2 or direct.shape[0] != 1:
            raise D19RunnerError("sealed runtime z_id shape drift")
        vector = np.ascontiguousarray(feature[0], dtype=np.float32)
        indices = np.asarray(direct_logit_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or len(indices) == 0
            or len(set(indices.tolist())) != len(indices)
            or bool(np.any(indices < 0))
            or bool(np.any(indices >= direct.shape[1]))
        ):
            raise D19RunnerError("sealed runtime direct-logit binding drift")
        logit_vector = np.ascontiguousarray(direct[0, indices], dtype=np.float32)
        if not np.isfinite(vector).all() or not np.isfinite(logit_vector).all():
            raise D19RunnerError("sealed runtime output contains non-finite values")
        values.append(vector)
        logits.append(logit_vector)
        feature_hashes.append(hashlib.sha256(vector.tobytes()).hexdigest())
        logit_hashes.append(hashlib.sha256(logit_vector.tobytes()).hexdigest())
    matrix = np.stack(values).astype(np.float32)
    logit_matrix = np.stack(logits).astype(np.float32)
    return matrix, logit_matrix, {
        "physical_support_rows": int(len(matrix)),
        "backbone_forwards": int(len(matrix)),
        "physical_batch_size": 1,
        "one_forward_per_unique_support": True,
        "zid_and_direct_logits_from_same_forward": True,
        "direct_logit_indices": [int(value) for value in direct_logit_indices],
        "direct_logit_to_class_handle_order_bound": True,
        "derived_views_per_support": 0,
        "feature_sha256": feature_hashes,
        "direct_logit_sha256": logit_hashes,
    }


def _post_reception_view_lineage(
    rows: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    tokens = np.asarray(rows["tokens"]).astype(str)
    hashes = np.asarray(rows["hashes"]).astype(str)
    if (
        len(tokens) != len(hashes)
        or any(len(value) != 64 for value in hashes)
        or len(set(tokens.tolist())) != len(tokens)
    ):
        raise D19RunnerError("post-reception view lineage parent drift")
    operators = (
        "adv3b02_zid160_base_v1",
        "same_received_iq_fft96_v1",
        "same_received_iq_rf32_v1",
    )
    return [
        {
            "physical_sample_id": str(token),
            "parent_received_iq_sha256": str(parent),
            "operators": [
                {"operator_id": operator, "view_seed": 0}
                for operator in operators
            ],
            "post_reception_view_count": len(operators),
            "additional_physical_sample_count": 0,
            "additional_leo_overlay_count": 0,
        }
        for token, parent in zip(tokens, hashes)
    ]


def _metric_block(
    truth: Sequence[str], predictions: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    y = np.asarray(truth).astype(str)
    pred = np.asarray(predictions).astype(str)
    per_class: dict[str, float] = {}
    for label in classes:
        mask = y == str(label)
        if not bool(np.any(mask)):
            raise D19RunnerError("metric class has no held support")
        per_class[str(label)] = float(np.mean(pred[mask] == y[mask]))
    return {
        "overall_accuracy": float(np.mean(pred == y)),
        "per_class_accuracy": per_class,
        "class_floor_accuracy": float(min(per_class.values())),
        "sample_count": int(len(y)),
    }


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right <= 0.0 else 2.0 * left * right / (left + right)


def _normalize_matrix(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise D19RunnerError("diag feature matrix drift")
    return rows / np.maximum(
        np.linalg.norm(rows, axis=1, keepdims=True), np.float32(1.0e-8)
    )


def _target_support_centroids(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    rows = _normalize_matrix(features)
    values: list[np.ndarray] = []
    for label in classes:
        selected = rows[labels == str(label)]
        if not len(selected):
            raise D19RunnerError("target support centroid class missing")
        values.append(np.mean(selected, axis=0, dtype=np.float64).astype(np.float32))
    return _normalize_matrix(np.stack(values))


def _fit_diag_registered_state(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    fit_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Fit Stage2-B, then register/fit new weights while freezing the old state."""

    with _numpy2_torch21_as_tensor_compatibility():
        state = fit_diag_cosine_state(
            features[train & old],
            labels[train & old],
            seed=int(fit_seed),
            device=device,
            candidate=CANDIDATE_D1_B0_CAP,
        )
    old_lookup = {str(label): index for index, label in enumerate(state.classes)}
    if set(old_lookup) != set(old_classes):
        raise D19RunnerError("diag old registry drift")
    old_weights = np.ascontiguousarray(
        np.stack([state.weights[old_lookup[label]] for label in old_classes]),
        dtype=np.float32,
    )
    lower, upper = log_scale_bounds(CANDIDATE_D1_B0_CAP, features.shape[1])
    scale = np.ascontiguousarray(
        np.minimum(np.maximum(state.log_scale, lower), upper), dtype=np.float32
    )
    scaled_support = np.asarray(features, dtype=np.float32) * np.exp(scale)[None, :]
    new_weights: list[np.ndarray] = []
    for label in new_classes:
        class_rows = scaled_support[train & new & (labels == label)]
        if not len(class_rows):
            raise D19RunnerError("diag new registration class has no support")
        prototype = np.mean(class_rows, axis=0, dtype=np.float64).astype(np.float32)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm <= 1.0e-8:
            raise D19RunnerError("diag new registration prototype drift")
        new_weights.append(prototype / norm)
    registry = old_classes + new_classes
    class_to_index = {label: index for index, label in enumerate(registry)}
    train_labels = labels[train]
    targets_np = np.asarray(
        [class_to_index[str(label)] for label in train_labels], dtype=np.int64
    )
    counts = np.bincount(targets_np, minlength=len(registry))
    if len(counts) != len(registry) or bool(np.any(counts <= 0)):
        raise D19RunnerError("diag all-registered class balance drift")
    k_shot = int(np.min(counts))
    registration_epochs = min(20, 5 * max(k_shot - 1, 0))
    stage2c_trainable_parameters = (
        int(len(new_classes) * features.shape[1]) if registration_epochs else 0
    )
    stage2c_estimated_adaptation_macs = int(
        registration_epochs
        * int(np.sum(train))
        * len(registry)
        * features.shape[1]
        * 3
    )
    old_tensor = _tensor_from_numpy_dlpack(
        _normalize_matrix(old_weights), dtype=torch.float32, device=device
    )
    new_initial = np.stack(new_weights).astype(np.float32)
    new_tensor = torch.nn.Parameter(
        _tensor_from_numpy_dlpack(
            new_initial, dtype=torch.float32, device=device
        ).clone()
    )
    train_tensor = _tensor_from_numpy_dlpack(
        _normalize_matrix(scaled_support[train]),
        dtype=torch.float32,
        device=device,
    )
    targets = _tensor_from_numpy_dlpack(
        targets_np, dtype=torch.long, device=device
    )
    new_initial_tensor = _tensor_from_numpy_dlpack(
        new_initial, dtype=torch.float32, device=device
    )
    registration_trace: list[dict[str, Any]] = []
    if registration_epochs:
        optimizer = torch.optim.SGD([new_tensor], lr=0.05, momentum=0.0)
        for epoch in range(1, registration_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            weights_tensor = torch.cat(
                [old_tensor, torch.nn.functional.normalize(new_tensor, dim=1)],
                dim=0,
            )
            logits = DIAG_TEMPERATURE * (train_tensor @ weights_tensor.T)
            sample_loss = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            class_loss = torch.stack(
                [
                    sample_loss[targets == index].mean()
                    for index in range(len(registry))
                ]
            )
            worst_surrogate = 0.25 * (
                torch.logsumexp(class_loss / 0.25, dim=0)
                - math.log(len(registry))
            )
            anchor_loss = torch.mean(
                (
                    torch.nn.functional.normalize(new_tensor, dim=1)
                    - new_initial_tensor
                )
                ** 2
            )
            loss = class_loss.mean() + 0.20 * worst_surrogate + 0.01 * anchor_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([new_tensor], max_norm=5.0)
            optimizer.step()
            row = {
                "phase": "stage2c_all_registered_new_weight_fit_old_state_frozen",
                "epoch": epoch,
                "total_epochs": registration_epochs,
                "loss": float(loss.detach().cpu()),
                "mean_class_loss": float(class_loss.mean().detach().cpu()),
                "worst_class_loss": float(class_loss.max().detach().cpu()),
                "worst_class_surrogate": float(worst_surrogate.detach().cpu()),
                "new_centroid_anchor_loss": float(anchor_loss.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "support_accuracy": float(
                    (logits.argmax(dim=1) == targets).float().mean().detach().cpu()
                ),
            }
            if not all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key != "phase"
            ):
                raise D19RunnerError("diag Stage2-C loss trace is non-finite")
            registration_trace.append(row)
    final_new_weights = np.asarray(
        torch.nn.functional.normalize(new_tensor.detach(), dim=1).cpu().tolist(),
        dtype=np.float32,
    )
    all_weights = np.ascontiguousarray(
        np.concatenate([old_weights, final_new_weights], axis=0), dtype=np.float32
    )
    registry_bytes = len(_canonical({"classes": list(registry)}))
    persistent_bytes = int(scale.nbytes + all_weights.nbytes + registry_bytes)
    return {
        "classes": registry,
        "old_class_count": len(old_classes),
        "scale": scale,
        "old_weights": old_weights,
        "weights": all_weights,
        "trace": [dict(row) for row in state.trace] + registration_trace,
        "resource": {
            **dict(state.resource),
            "support_view_count": 1,
            "query_view_count": 1,
            "physical_leo_observations_per_support": 1,
            "same_received_iq_feature_branches": ["z_id160", "fft96", "rf32"],
            "same_received_iq_additional_backbone_forwards": 0,
            "cross_scenario_support_pooling": False,
            "stage2c_registration_rule": (
                "all_registered_class_balanced_worst_class_new_weight_fit_"
                "old_diag_and_old_head_bitwise_frozen"
            ),
            "stage2c_class_balanced_loss": True,
            "stage2c_worst_class_surrogate_weight": 0.20,
            "stage2c_new_centroid_anchor_weight": 0.01,
            "stage2c_registration_epochs": registration_epochs,
            "stage2c_k1_registration_rule": "centroid_only_zero_optimizer_epoch",
            "max_adaptation_epochs_per_event": 20,
            "stage2c_old_raw_score_columns_bitwise_unchanged": True,
            "stage2b_trainable_parameters": int(
                state.resource["trainable_parameters"]
            ),
            "stage2c_trainable_parameters": stage2c_trainable_parameters,
            "registered_class_count": len(registry),
            "class_count": len(registry),
            "registry_state_bytes": registry_bytes,
            "parameter_state_bytes": int(scale.nbytes + all_weights.nbytes),
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_limit_bytes": 256 * 1024,
            "persistent_state_limit_pass": persistent_bytes <= 256 * 1024,
            "trainable_parameters": int(
                state.resource["trainable_parameters"]
            )
            + stage2c_trainable_parameters,
            "stage2b_optimizer_steps": int(state.resource["optimizer_steps"]),
            "stage2c_optimizer_steps": registration_epochs,
            "optimizer_steps": int(state.resource["optimizer_steps"])
            + registration_epochs,
            "stage2b_estimated_adaptation_macs": int(
                state.resource["estimated_adaptation_macs"]
            ),
            "stage2c_estimated_adaptation_macs": stage2c_estimated_adaptation_macs,
            "estimated_adaptation_macs": int(
                state.resource["estimated_adaptation_macs"]
            )
            + stage2c_estimated_adaptation_macs,
            "estimated_adaptation_macs_scope": (
                "stage2b_diag_fit_plus_stage2c_all_registered_full_batch_"
                "new_weight_forward_backward_excludes_backbone_fft_rf"
            ),
            "stage2b_support_enrollment_rows": int(
                state.resource["support_enrollment_rows"]
            ),
            "support_enrollment_rows": int(np.sum(train)),
            "estimated_macs_per_query": int(
                features.shape[1] + len(registry) * features.shape[1]
            ),
        },
    }


def _diag_scores(
    state: Mapping[str, Any],
    features: np.ndarray,
    *,
    include_new: bool,
) -> np.ndarray:
    rows = np.asarray(features, dtype=np.float32)
    scale = np.asarray(state["scale"], dtype=np.float32)
    weights = np.asarray(
        state["weights"] if include_new else state["old_weights"],
        dtype=np.float32,
    )
    if rows.ndim != 2 or rows.shape[1] != len(scale):
        raise D19RunnerError("diag scoring feature shape drift")
    return np.ascontiguousarray(
        DIAG_TEMPERATURE
        * (_normalize_matrix(rows * np.exp(scale)[None, :]) @ _normalize_matrix(weights).T),
        dtype=np.float32,
    )


def _evaluate_fold(
    component: Int8DomainClassComponent,
    rows: Mapping[str, np.ndarray],
    z_id: np.ndarray,
    direct_logits: np.ndarray,
    diag_features: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: DaliConfig,
    fit_seed: int,
    device: torch.device,
    diag_cache: dict[tuple[str, tuple[int, int]], dict[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D19RunnerError("leave-two-out class symmetry drift")
    use_diag = candidate_id in {DIAG_CANDIDATE, DIAG_MAXOLD_CANDIDATE}
    use_maxold = candidate_id == DIAG_MAXOLD_CANDIDATE
    if use_diag:
        cache_key = (str(held_ranks), held_ranks)
        diag_state = diag_cache.get(cache_key)
        if diag_state is None:
            diag_state = _fit_diag_registered_state(
                diag_features,
                labels,
                train,
                old,
                new,
                old_classes=old_classes,
                new_classes=new_classes,
                fit_seed=fit_seed,
                device=device,
            )
            diag_cache[cache_key] = diag_state
        before_base = _diag_scores(
            diag_state, diag_features[held & old], include_new=False
        )
        after_old_base = _diag_scores(
            diag_state, diag_features[held & old], include_new=True
        )
        after_new_base = _diag_scores(
            diag_state, diag_features[held & new], include_new=True
        )
        if use_maxold:
            before = fit_old_dali(
                component,
                z_id[train & old],
                labels[train & old],
                direct_logits[train & old],
                config=config,
            )
            after = register_new_dali(
                before,
                z_id[train & new],
                labels[train & new],
                registered_classes=new_classes,
            )
            before_score_rows = np.stack(
                [
                    rerank_old_scores_dali(before, base, row, logit)
                    for base, row, logit in zip(
                        before_base,
                        z_id[held & old],
                        direct_logits[held & old],
                    )
                ]
            )
            after_old_score_rows = np.stack(
                [
                    rerank_old_scores_dali(after, base, row, logit)
                    for base, row, logit in zip(
                        after_old_base,
                        z_id[held & old],
                        direct_logits[held & old],
                    )
                ]
            )
            after_new_score_rows = np.stack(
                [
                    rerank_old_scores_dali(after, base, row, logit)
                    for base, row, logit in zip(
                        after_new_base,
                        z_id[held & new],
                        direct_logits[held & new],
                    )
                ]
            )
        else:
            before_score_rows = before_base
            after_old_score_rows = after_old_base
            after_new_score_rows = after_new_base
        before_predictions = [
            old_classes[int(index)] for index in np.argmax(before_score_rows, axis=1)
        ]
        all_classes = old_classes + new_classes
        after_old_predictions = [
            all_classes[int(index)]
            for index in np.argmax(after_old_score_rows, axis=1)
        ]
        after_new_predictions = [
            all_classes[int(index)]
            for index in np.argmax(after_new_score_rows, axis=1)
        ]
        old_scores_unchanged = np.array_equal(
            before_score_rows,
            after_old_score_rows[:, : len(old_classes)],
        )
        resource = dict(diag_state["resource"])
        dali_resource = after.resource_audit() if use_maxold else None
        rerank_macs = (
            int(dali_resource["estimated_head_macs_per_query"])
            - int(dali_resource["prototype_score_macs_per_query"])
            if dali_resource is not None
            else 0
        )
        combined_state_bytes = int(resource["persistent_state_bytes"]) + (
            int(dali_resource["persistent_state_bytes"])
            if dali_resource is not None
            else 0
        )
        resource.update(
            {
                "int8_component_used_for_prediction": use_maxold,
                "int8_component_state_bytes": (
                    component.state_bytes if use_maxold else 0
                ),
                "diag_registered_state_bytes": int(
                    diag_state["resource"]["persistent_state_bytes"]
                ),
                "dali_rerank_state_bytes": (
                    int(dali_resource["persistent_state_bytes"])
                    if dali_resource is not None
                    else 0
                ),
                "persistent_state_bytes": combined_state_bytes,
                "persistent_state_limit_bytes": 256 * 1024,
                "persistent_state_limit_pass": combined_state_bytes <= 256 * 1024,
                "max_old_preserving_identity_rerank": use_maxold,
                "estimated_head_macs_per_query": int(
                    resource["estimated_macs_per_query"]
                    + rerank_macs
                ),
                "complete_loss_trace": diag_state["trace"],
            }
        )
    elif candidate_id == IDENTITY_CANDIDATE:
        old_proto = _target_support_centroids(
            z_id[train & old], labels[train & old], old_classes
        )
        new_proto = _target_support_centroids(
            z_id[train & new], labels[train & new], new_classes
        )
        all_proto = np.concatenate([old_proto, new_proto], axis=0)
        before_score_rows = _normalize_matrix(z_id[held & old]) @ old_proto.T
        after_old_score_rows = _normalize_matrix(z_id[held & old]) @ all_proto.T
        after_new_score_rows = _normalize_matrix(z_id[held & new]) @ all_proto.T
        before_predictions = [
            old_classes[int(index)] for index in np.argmax(before_score_rows, axis=1)
        ]
        all_classes = old_classes + new_classes
        after_old_predictions = [
            all_classes[int(index)]
            for index in np.argmax(after_old_score_rows, axis=1)
        ]
        after_new_predictions = [
            all_classes[int(index)]
            for index in np.argmax(after_new_score_rows, axis=1)
        ]
        old_scores_unchanged = np.array_equal(
            before_score_rows, after_old_score_rows[:, : len(old_classes)]
        )
        registry_bytes = len(_canonical({"classes": list(all_classes)}))
        persistent_bytes = int(all_proto.nbytes + registry_bytes)
        resource = {
            "schema": "cvs.phase2.d20_target_centroid_baseline.resource.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "trainable_parameters": 0,
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_limit_bytes": 256 * 1024,
            "persistent_state_limit_pass": persistent_bytes <= 256 * 1024,
            "int8_component_used_for_prediction": False,
            "int8_component_state_bytes": 0,
            "estimated_head_macs_per_query": int(len(all_classes) * FEATURE_DIM),
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "dense_query_graph_bytes": 0,
            "per_sample_all_registered_classes": True,
            "source_sample_access": False,
            "sample_level_source_feature_access": False,
        }
    else:
        before = fit_old_dali(
            component,
            z_id[train & old],
            labels[train & old],
            direct_logits[train & old],
            config=config,
        )
        after = register_new_dali(
            before,
            z_id[train & new],
            labels[train & new],
            registered_classes=new_classes,
        )
        if after.classes != old_classes + new_classes:
            raise D19RunnerError("registered class order drift after enrollment")
        before_predictions = [
            predict_one_dali(before, row, logit)[0]
            for row, logit in zip(z_id[held & old], direct_logits[held & old])
        ]
        after_old_predictions = [
            predict_one_dali(after, row, logit)[0]
            for row, logit in zip(z_id[held & old], direct_logits[held & old])
        ]
        after_new_predictions = [
            predict_one_dali(after, row, logit)[0]
            for row, logit in zip(z_id[held & new], direct_logits[held & new])
        ]
        old_scores_unchanged = all(
            np.array_equal(
                score_one_dali(before, row, logit),
                score_one_dali(after, row, logit)[: len(old_classes)],
            )
            for row, logit in zip(z_id[held & old], direct_logits[held & old])
        )
        resource = after.resource_audit()
    before_old = _metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = _metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = _metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    h_value = _harmonic(
        float(after_old["overall_accuracy"]),
        float(after_new["overall_accuracy"]),
    )
    forgetting = float(
        before_old["overall_accuracy"] - after_old["overall_accuracy"]
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": h_value,
        "joint_floor": float(
            min(
                after_old["class_floor_accuracy"],
                after_new["class_floor_accuracy"],
            )
        ),
        "forgetting": forgetting,
        "resource": resource,
        "old_score_columns_bitwise_unchanged": bool(
            old_scores_unchanged
        ),
    }


def _aggregate_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(FORMAL_LEO_WEAK_SCENARIOS) * len(HELD_RANKS):
        raise D19RunnerError("candidate fold matrix is incomplete")

    def values(path: tuple[str, ...]) -> np.ndarray:
        result: list[float] = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return np.asarray(result, dtype=np.float64)

    return {
        "candidate_id": str(rows[0]["candidate_id"]),
        "fold_count": len(rows),
        "mean_before_old": float(values(("before_old", "overall_accuracy")).mean()),
        "mean_after_old": float(values(("after_old", "overall_accuracy")).mean()),
        "mean_after_new": float(values(("after_new", "overall_accuracy")).mean()),
        "mean_H_old_new": float(values(("H_old_new",)).mean()),
        "mean_forgetting": float(values(("forgetting",)).mean()),
        "worst_before_old_floor": float(
            values(("before_old", "class_floor_accuracy")).min()
        ),
        "worst_after_old_floor": float(
            values(("after_old", "class_floor_accuracy")).min()
        ),
        "worst_after_new_floor": float(
            values(("after_new", "class_floor_accuracy")).min()
        ),
        "worst_joint_floor": float(values(("joint_floor",)).min()),
        "max_forgetting": float(values(("forgetting",)).max()),
        "all_old_columns_bitwise_unchanged": all(
            bool(row["old_score_columns_bitwise_unchanged"]) for row in rows
        ),
    }


def _select_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    baseline = list(folds_by_candidate[IDENTITY_CANDIDATE])
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float]] = []
    tolerance = 1.0e-12
    for candidate_id, rows in folds_by_candidate.items():
        aggregate = _aggregate_candidate(rows)
        if candidate_id == IDENTITY_CANDIDATE:
            decision = {
                **aggregate,
                "atomic_noninferiority_vs_Z0": True,
                "strict_aggregate_improvement_vs_Z0": False,
                "eligible_positive_route": False,
                "fallback": True,
            }
            decisions.append(decision)
            continue
        fold_guards: list[bool] = []
        for row, zero in zip(rows, baseline):
            old_classwise = all(
                float(row["after_old"]["per_class_accuracy"][label]) + tolerance
                >= float(zero["after_old"]["per_class_accuracy"][label])
                for label in row["after_old"]["per_class_accuracy"]
            )
            new_classwise = all(
                float(row["after_new"]["per_class_accuracy"][label]) + tolerance
                >= float(zero["after_new"]["per_class_accuracy"][label])
                for label in row["after_new"]["per_class_accuracy"]
            )
            fold_guards.append(
                float(row["before_old"]["class_floor_accuracy"])
                + tolerance
                >= float(zero["before_old"]["class_floor_accuracy"])
                and float(row["after_old"]["class_floor_accuracy"])
                + tolerance
                >= float(zero["after_old"]["class_floor_accuracy"])
                and float(row["after_new"]["class_floor_accuracy"])
                + tolerance
                >= float(zero["after_new"]["class_floor_accuracy"])
                and float(row["H_old_new"]) + tolerance
                >= float(zero["H_old_new"])
                and float(row["forgetting"])
                <= float(zero["forgetting"]) + tolerance
                and old_classwise
                and new_classwise
            )
        zero_aggregate = _aggregate_candidate(baseline)
        strict_old_identity = (
            aggregate["worst_after_old_floor"]
            > zero_aggregate["worst_after_old_floor"] + tolerance
        )
        strict = bool(strict_old_identity)
        atomic = all(fold_guards)
        paired_atomic = True
        paired_strict = True
        paired_noninferior_count = len(rows)
        if candidate_id == DIAG_MAXOLD_CANDIDATE:
            paired_rows = list(folds_by_candidate[DIAG_CANDIDATE])
            paired_guards: list[bool] = []
            for row, paired in zip(rows, paired_rows):
                old_classwise = all(
                    float(row["after_old"]["per_class_accuracy"][label])
                    + tolerance
                    >= float(paired["after_old"]["per_class_accuracy"][label])
                    for label in row["after_old"]["per_class_accuracy"]
                )
                new_exact = all(
                    float(row["after_new"]["per_class_accuracy"][label])
                    == float(paired["after_new"]["per_class_accuracy"][label])
                    for label in row["after_new"]["per_class_accuracy"]
                )
                paired_guards.append(
                    old_classwise
                    and new_exact
                    and float(row["H_old_new"]) + tolerance
                    >= float(paired["H_old_new"])
                    and float(row["forgetting"])
                    <= float(paired["forgetting"]) + tolerance
                )
            paired_aggregate = _aggregate_candidate(paired_rows)
            paired_atomic = all(paired_guards)
            paired_noninferior_count = int(sum(paired_guards))
            paired_strict = (
                aggregate["worst_after_old_floor"]
                > paired_aggregate["worst_after_old_floor"] + tolerance
            )
        eligible_positive = bool(
            atomic and strict and paired_atomic and paired_strict
        )
        decision = {
            **aggregate,
            "atomic_noninferiority_vs_Z0": atomic,
            "noninferior_fold_count": int(sum(fold_guards)),
            "strict_aggregate_improvement_vs_Z0": strict,
            "strict_worst_old_floor_improvement_vs_Z0": strict_old_identity,
            "paired_base_candidate_id": (
                DIAG_CANDIDATE
                if candidate_id == DIAG_MAXOLD_CANDIDATE
                else None
            ),
            "paired_atomic_noninferiority": paired_atomic,
            "paired_noninferior_fold_count": paired_noninferior_count,
            "paired_strict_worst_old_floor_improvement": paired_strict,
            "eligible_positive_route": eligible_positive,
            "fallback": False,
        }
        decisions.append(decision)
        if eligible_positive:
            eligible.append(
                (
                    candidate_id,
                    float(aggregate["worst_joint_floor"]),
                    float(aggregate["mean_H_old_new"]),
                )
            )
    selected = (
        max(eligible, key=lambda item: (item[1], item[2], item[0]))[0]
        if eligible
        else IDENTITY_CANDIDATE
    )
    return selected, decisions


def _deployment_state_audit(
    component: Int8DomainClassComponent,
    rows: Mapping[str, np.ndarray],
    z_id: np.ndarray,
    direct_logits: np.ndarray,
    diag_features: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    candidate_id: str,
    config: DaliConfig,
    fit_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    use_diag = candidate_id in {DIAG_CANDIDATE, DIAG_MAXOLD_CANDIDATE}
    use_maxold = candidate_id == DIAG_MAXOLD_CANDIDATE
    if use_diag:
        all_rows = np.ones(len(labels), dtype=bool)
        diag_state = _fit_diag_registered_state(
            diag_features,
            labels,
            all_rows,
            old,
            new,
            old_classes=old_classes,
            new_classes=new_classes,
            fit_seed=fit_seed,
            device=device,
        )
        resource = dict(diag_state["resource"])
        resource["int8_component_used_for_prediction"] = use_maxold
        if use_maxold:
            before = fit_old_dali(
                component, z_id[old], labels[old], direct_logits[old], config=config
            )
            after = register_new_dali(
                before, z_id[new], labels[new], registered_classes=new_classes
            )
            dali_resource = after.resource_audit()
            rerank_macs = int(
                dali_resource["estimated_head_macs_per_query"]
            ) - int(dali_resource["prototype_score_macs_per_query"])
            diag_state_bytes = int(resource["persistent_state_bytes"])
            combined_state_bytes = diag_state_bytes + int(
                dali_resource["persistent_state_bytes"]
            )
            resource.update(
                {
                    "int8_component_state_bytes": component.state_bytes,
                    "diag_registered_state_bytes": diag_state_bytes,
                    "dali_rerank_state_bytes": int(
                        dali_resource["persistent_state_bytes"]
                    ),
                    "persistent_state_bytes": combined_state_bytes,
                    "persistent_state_limit_bytes": 256 * 1024,
                    "persistent_state_limit_pass": (
                        combined_state_bytes <= 256 * 1024
                    ),
                    "max_old_preserving_identity_rerank": True,
                    "estimated_head_macs_per_query": int(
                        resource["estimated_macs_per_query"]
                        + rerank_macs
                    ),
                    "old_score_columns_bitwise_unchanged_after_registration": True,
                }
            )
        else:
            diag_state_bytes = int(resource["persistent_state_bytes"])
            resource.update(
                {
                    "int8_component_state_bytes": 0,
                    "diag_registered_state_bytes": diag_state_bytes,
                    "dali_rerank_state_bytes": 0,
                    "max_old_preserving_identity_rerank": False,
                    "estimated_head_macs_per_query": int(
                        resource["estimated_macs_per_query"]
                    ),
                    "old_score_columns_bitwise_unchanged_after_registration": True,
                }
            )
    elif candidate_id == IDENTITY_CANDIDATE:
        old_proto = _target_support_centroids(z_id[old], labels[old], old_classes)
        new_proto = _target_support_centroids(z_id[new], labels[new], new_classes)
        prototypes = np.concatenate([old_proto, new_proto], axis=0)
        registry_bytes = len(
            _canonical({"classes": list(old_classes + new_classes)})
        )
        persistent_bytes = int(prototypes.nbytes + registry_bytes)
        resource = {
            "schema": "cvs.phase2.d20_target_centroid_baseline.resource.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "trainable_parameters": 0,
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_limit_bytes": 256 * 1024,
            "persistent_state_limit_pass": persistent_bytes <= 256 * 1024,
            "int8_component_used_for_prediction": False,
            "int8_component_state_bytes": 0,
            "estimated_head_macs_per_query": int(len(prototypes) * FEATURE_DIM),
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "dense_query_graph_bytes": 0,
            "per_sample_all_registered_classes": True,
            "source_sample_access": False,
            "sample_level_source_feature_access": False,
        }
    else:
        before = fit_old_dali(
            component, z_id[old], labels[old], direct_logits[old], config=config
        )
        after = register_new_dali(
            before, z_id[new], labels[new], registered_classes=new_classes
        )
        resource = after.resource_audit()
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "estimated_score_mac_reduction_vs_identity_single_qknn": float(
                1.0 - resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "old_score_columns_bitwise_unchanged_after_registration": bool(
                resource.get("old_score_columns_bitwise_unchanged_after_registration", True)
            ),
        }
    )
    return resource


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    before_formal_policy: Path,
    before_formal_policy_authorization: Path,
    before_signed_policy_authorization_envelope: Path,
    expected_before_signed_policy_authorization_envelope_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    after_formal_policy: Path,
    after_formal_policy_authorization: Path,
    after_signed_policy_authorization_envelope: Path,
    expected_after_signed_policy_authorization_envelope_sha256: str,
    component_dir: Path,
    expected_component_manifest_sha256: str,
    class_binding_path: Path,
    expected_class_binding_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = MODE,
) -> dict[str, Any]:
    if mode != MODE:
        raise D19RunnerError("D20 historical component runner is development-only")
    if output.exists():
        raise D19RunnerError("output path already exists")
    candidates = preregistered_candidates()

    # The historical int8 screening exception still requires all component,
    # checkpoint and class-column checks before any support IQ materialization.
    before_preopen_manifest = _preopen_manifest(
        before_root,
        before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
    )
    after_preopen_manifest = _preopen_manifest(
        after_root,
        after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
    )
    _manifest_binding(before_preopen_manifest, after_preopen_manifest)
    preopen_old_classes = _registered_handles(before_preopen_manifest)
    component, component_audit = _load_component(
        component_dir,
        expected_manifest_sha256=expected_component_manifest_sha256,
        expected_checkpoint_sha256=str(
            before_preopen_manifest["phase1_checkpoint_sha256"]
        ),
        bound_old_handles=preopen_old_classes,
        class_binding_path=class_binding_path,
        expected_class_binding_sha256=expected_class_binding_sha256,
    )
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root,
        _member(before_preopen_manifest, "feature_runtime"),
        device=device,
    )
    runtime_direct_logit_binding_audit = _verify_runtime_direct_logit_binding(
        model,
        before_preopen_manifest,
        component_audit["column_binding"],
    )

    before_evidence = materialize_somph_enrollment_with_signed_authority(
        before_root,
        detached_seal_path=before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
        formal_policy_path=before_formal_policy,
        formal_policy_authorization_path=before_formal_policy_authorization,
        signed_policy_authorization_envelope_path=before_signed_policy_authorization_envelope,
        expected_signed_policy_authorization_envelope_sha256=(
            expected_before_signed_policy_authorization_envelope_sha256
        ),
    )
    after_evidence = materialize_somph_enrollment_with_signed_authority(
        after_root,
        detached_seal_path=after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
        formal_policy_path=after_formal_policy,
        formal_policy_authorization_path=after_formal_policy_authorization,
        signed_policy_authorization_envelope_path=after_signed_policy_authorization_envelope,
        expected_signed_policy_authorization_envelope_sha256=(
            expected_after_signed_policy_authorization_envelope_sha256
        ),
    )
    before_authority = finalize_somph_enrollment_authority_after_materialization(
        before_evidence
    )
    after_authority = finalize_somph_enrollment_authority_after_materialization(
        after_evidence
    )
    _require_post_materialization_authority(before_authority, after_authority)
    before_manifest = before_evidence.manifest
    after_manifest = after_evidence.manifest
    _manifest_binding(before_manifest, after_manifest)
    old_classes = _registered_handles(before_manifest)
    all_classes = _registered_handles(after_manifest)
    if all_classes[: len(old_classes)] != old_classes:
        raise D19RunnerError("after registry does not append new classes")
    new_classes = all_classes[len(old_classes) :]
    if old_classes != preopen_old_classes:
        raise D19RunnerError("post-materialization registry differs from pre-open binding")

    before_overlay, before_overlay_audit = _overlay_index(before_root, before_manifest)
    after_overlay, after_overlay_audit = _overlay_index(after_root, after_manifest)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    scene_rows: dict[str, dict[str, np.ndarray]] = {}
    scene_features: dict[str, np.ndarray] = {}
    scene_direct_logits: dict[str, np.ndarray] = {}
    scene_diag_features: dict[str, np.ndarray] = {}
    extraction_audits: dict[str, Any] = {}
    old_reuse_audits: dict[str, Any] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _rows_with_overlay(
            before_evidence.materialized_payloads[scenario],
            before_manifest,
            before_overlay,
            scenario=scenario,
        )
        after_rows = _rows_with_overlay(
            after_evidence.materialized_payloads[scenario],
            after_manifest,
            after_overlay,
            scenario=scenario,
        )
        _old_reuse(before_rows, after_rows)
        features, direct_logits, extraction = _extract_scene_signals(
            model,
            device,
            after_rows,
            component_audit["column_binding"]["direct_logit_indices"],
        )
        scene_rows[scenario] = after_rows
        scene_features[scenario] = features
        scene_direct_logits[scenario] = direct_logits
        scene_diag_features[scenario] = registered_feature(after_rows["iq"], features)
        extraction.update(
            {
                "same_received_iq_fft96_extractions": int(len(after_rows["iq"])),
                "same_received_iq_rf32_extractions": int(len(after_rows["iq"])),
                "registered_feature_dim": int(scene_diag_features[scenario].shape[1]),
                "additional_backbone_forwards_for_fft_rf": 0,
                "post_reception_operator_id": "d20_same_received_iq_fft96_rf32_v1",
                "derived_views_per_support": 2,
                "post_reception_view_used": True,
                "post_reception_view_count": 3,
                "post_reception_view_lineage": _post_reception_view_lineage(
                    after_rows
                ),
            }
        )
        extraction_audits[scenario] = extraction
        old_reuse_audits[scenario] = {
            "old_support_exact_reuse": True,
            "before_old_rows": int(len(before_rows["labels"])),
            "after_total_rows": int(len(after_rows["labels"])),
        }
    cross_scene = _cross_scene_disjointness(scene_rows)

    training_log: list[dict[str, Any]] = []
    folds_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidates
    }
    diag_caches: dict[str, dict[tuple[str, tuple[int, int]], dict[str, Any]]] = {
        scenario: {} for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    for candidate_id, config in candidates.items():
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            for fold_index, held_ranks in enumerate(HELD_RANKS):
                row = _evaluate_fold(
                    component,
                    scene_rows[scenario],
                    scene_features[scenario],
                    scene_direct_logits[scenario],
                    scene_diag_features[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    held_ranks=held_ranks,
                    candidate_id=candidate_id,
                    config=config,
                    fit_seed=int(before_manifest["seed"]) + fold_index,
                    device=device,
                    diag_cache=diag_caches[scenario],
                )
                row.update(
                    {
                        "schema": "cvs.phase2.d20_maxold_fftrf.support_fold.v1",
                        "scenario": scenario,
                        "fold_index": fold_index,
                        "query_opened": False,
                        "formal_metric_claim_allowed": False,
                        "performance_claim_allowed": False,
                    }
                )
                folds_by_candidate[candidate_id].append(row)
                training_log.append(row)
    selected_id, candidate_decisions = _select_candidate(folds_by_candidate)
    deployment_resources = {
        scenario: _deployment_state_audit(
            component,
            scene_rows[scenario],
            scene_features[scenario],
            scene_direct_logits[scenario],
            scene_diag_features[scenario],
            old_classes=old_classes,
            new_classes=new_classes,
            candidate_id=selected_id,
            config=candidates[selected_id],
            fit_seed=int(before_manifest["seed"]),
            device=device,
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    elapsed = time.perf_counter() - start
    support_audit = {
        "schema": "cvs.phase2.d20_maxold_fftrf.support_audit.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_USER_AUTHORIZED_PREBUNDLE_INT8_SCREEN",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "query_rows_opened": 0,
        "query_labels_opened": 0,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "sample_level_source_feature_access": False,
        "authorized_int8_phase1_aggregate_component_access": True,
        "int8_component_update_access": False,
        "one_physical_support_one_leo_channel_observation": True,
        "derived_support_views_per_physical_sample": 2,
        "post_reception_view_count_including_base": 3,
        "derived_views_count_toward_k": False,
        "old_reuse_by_scenario": old_reuse_audits,
        "cross_scene_disjointness": cross_scene,
        "before_overlay_audit": before_overlay_audit,
        "after_overlay_audit": after_overlay_audit,
        "feature_extraction": extraction_audits,
        "runtime_direct_logit_binding": runtime_direct_logit_binding_audit,
        "component": component_audit,
        "before_post_materialization_audit_sha256": before_authority[
            "post_materialization_audit_sha256"
        ],
        "after_post_materialization_audit_sha256": after_authority[
            "post_materialization_audit_sha256"
        ],
    }
    training_log_sha256 = _write_jsonl(output / "training_log.jsonl", training_log)
    support_audit_sha256 = _write_json(output / "support_audit.json", support_audit)
    selection = {
        "schema": "cvs.phase2.d20_maxold_fftrf.selection.v1",
        "selected_candidate_id": selected_id,
        "selected_positive_route": selected_id != IDENTITY_CANDIDATE,
        "fallback_to_identity": selected_id == IDENTITY_CANDIDATE,
        "selection_rule": (
            "all_15_folds_classwise_noninferior_vs_Z0_plus_strict_aggregate_"
            "improvement_and_B4_paired_noninferior_strict_worst_old_floor_"
            "gain_vs_B3"
        ),
        "candidate_decisions": candidate_decisions,
    }
    selection_sha256 = _write_json(output / "selection.json", selection)
    resource_sha256 = _write_json(
        output / "resource_audit.json",
        {
            "schema": "cvs.phase2.d20_maxold_fftrf.resource_matrix.v1",
            "selected_candidate_id": selected_id,
            "by_scenario": deployment_resources,
        },
    )
    receipt = {
        "schema": "cvs.phase2.d20_maxold_fftrf.receipt.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_COMPLETE",
        "mode": mode,
        "selected_candidate_id": selected_id,
        "selected_positive_route": selected_id != IDENTITY_CANDIDATE,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "receiver": str(before_manifest["receiver"]),
        "seed": int(before_manifest["seed"]),
        "k_shot": int(before_manifest["k_shot"]),
        "old_class_count": len(old_classes),
        "new_class_count": len(new_classes),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "candidate_count": len(candidates),
        "folds_per_candidate": len(FORMAL_LEO_WEAK_SCENARIOS) * len(HELD_RANKS),
        "elapsed_seconds": elapsed,
        "training_log_sha256": training_log_sha256,
        "support_audit_sha256": support_audit_sha256,
        "selection_sha256": selection_sha256,
        "resource_audit_sha256": resource_sha256,
        "component_manifest_sha256": expected_component_manifest_sha256,
        "component_npz_sha256": component_audit["manifest"]["component_npz_sha256"],
        "component_provenance_status": component_audit["manifest"][
            "provenance_status"
        ],
    }
    receipt_sha256 = _write_json(output / "RECEIPT.json", receipt)
    return {"receipt_sha256": receipt_sha256, **receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--before-formal-policy", type=Path, required=True)
    parser.add_argument("--before-formal-policy-authorization", type=Path, required=True)
    parser.add_argument(
        "--before-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--before-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--after-formal-policy", type=Path, required=True)
    parser.add_argument("--after-formal-policy-authorization", type=Path, required=True)
    parser.add_argument(
        "--after-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--after-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--component-manifest-sha256", required=True)
    parser.add_argument("--class-binding", type=Path, required=True)
    parser.add_argument("--class-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=(MODE,), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        before_root=args.before_root,
        before_seal=args.before_seal,
        expected_before_seal_sha256=args.before_seal_sha256,
        before_formal_policy=args.before_formal_policy,
        before_formal_policy_authorization=args.before_formal_policy_authorization,
        before_signed_policy_authorization_envelope=(
            args.before_signed_policy_authorization_envelope
        ),
        expected_before_signed_policy_authorization_envelope_sha256=(
            args.before_signed_policy_authorization_envelope_sha256
        ),
        after_root=args.after_root,
        after_seal=args.after_seal,
        expected_after_seal_sha256=args.after_seal_sha256,
        after_formal_policy=args.after_formal_policy,
        after_formal_policy_authorization=args.after_formal_policy_authorization,
        after_signed_policy_authorization_envelope=(
            args.after_signed_policy_authorization_envelope
        ),
        expected_after_signed_policy_authorization_envelope_sha256=(
            args.after_signed_policy_authorization_envelope_sha256
        ),
        component_dir=args.component_dir,
        expected_component_manifest_sha256=args.component_manifest_sha256,
        class_binding_path=args.class_binding,
        expected_class_binding_sha256=args.class_binding_sha256,
        output=args.output,
        device_name=args.device,
        mode=args.mode,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
