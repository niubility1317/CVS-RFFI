#!/usr/bin/env python3
"""Export a development-only Phase1 joint-projection tap archive.

The exporter reuses the already validated source-only weak-IQ selection and
the SHA-bound ADV3B02 checkpoint.  It publishes only frozen features needed
for the Phase1-held GRB-JP4 falsifier: z_id, joint_proj.0 input, pre-ReLU
output, and the bound joint weight.  Received IQ is never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
import uuid

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    while _value in sys.path:
        sys.path.remove(_value)
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, _value)

from baseline_origin_sat_view import SatViewStage  # noqa: E402
from cvsrffi.checkpoint_loading import (  # noqa: E402
    build_exact_ssdg_model_from_checkpoint,
)
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA_V1,
    LEO_WEAK_CACHE_SET_SCHEMA_V1,
)
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256  # noqa: E402
from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import (  # noqa: E402
    strict_zid_with_hook,
)
from scripts.export_phase1_singleobs_dual_feature_archive import (  # noqa: E402
    CACHE_LOADER,
    MEMBERS as REFERENCE_MEMBERS,
    _load_selection_salt,
    _select_verified_observations,
)
from scripts.export_phase1_singleobs_feature_archive import (  # noqa: E402
    KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256,
)


SCHEMA = "cvs.phase1.jp4_tap_archive.v1"
NPZ_NAME = "phase1_jp4_tap_archive.npz"
MANIFEST_NAME = "phase1_jp4_tap_archive.manifest.json"
Z_DIM = 160
HIDDEN_DIM = 320
MEMBERS = (
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
)


class Phase1JP4TapArchiveError(ValueError):
    """Raised when the source-only tap export closure drifts."""


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def _require_sha(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Phase1JP4TapArchiveError(f"{name} must be lowercase SHA256")
    return text


def _regular_bound(path: str | Path, expected: str, name: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise Phase1JP4TapArchiveError(f"{name} must be a regular file")
    value = value.resolve()
    if _sha_file(value) != _require_sha(expected, f"{name} SHA256"):
        raise Phase1JP4TapArchiveError(f"{name} SHA256 drift")
    return value


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _joint_linear(model: Any) -> torch.nn.Linear:
    try:
        linear = model.id_backbone.cls_head.joint_proj[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise Phase1JP4TapArchiveError(
            "checkpoint lacks id_backbone.cls_head.joint_proj.0"
        ) from exc
    if (
        not isinstance(linear, torch.nn.Linear)
        or tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM)
        or linear.weight.dtype != torch.float32
        or not bool(torch.isfinite(linear.weight).all().item())
    ):
        raise Phase1JP4TapArchiveError("checkpoint joint weight contract drift")
    return linear


def _forward_taps(
    model: Any,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= 256:
        raise Phase1JP4TapArchiveError("batch_size must be in [1,256]")
    outputs: list[list[np.ndarray]] = [[], [], []]
    calls = 0
    for start in range(0, len(rows), int(batch_size)):
        chunk = np.ascontiguousarray(
            rows[start : start + int(batch_size)], dtype=np.float32
        )
        tensor = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
        forward = strict_zid_with_hook(model, tensor)
        calls += 1
        for target, value, shape in (
            (outputs[0], forward.z_id, (len(chunk), Z_DIM)),
            (outputs[1], forward.hidden, (len(chunk), HIDDEN_DIM)),
            (outputs[2], forward.pre_relu, (len(chunk), Z_DIM)),
        ):
            array = np.asarray(value)
            if (
                array.dtype != np.float32
                or array.shape != shape
                or not np.isfinite(array).all()
            ):
                raise Phase1JP4TapArchiveError("eager JP4 tap output drift")
            target.append(np.ascontiguousarray(array))
    return (
        np.ascontiguousarray(np.concatenate(outputs[0]), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(outputs[1]), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(outputs[2]), dtype=np.float32),
        calls,
    )


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(arrays) != MEMBERS:
        raise Phase1JP4TapArchiveError("tap archive member order drift")
    count = len(arrays["labels"])
    for name, width in (
        ("z_id", Z_DIM),
        ("hidden", HIDDEN_DIM),
        ("pre_relu", Z_DIM),
    ):
        value = np.asarray(arrays[name])
        if (
            value.dtype != np.float32
            or value.shape != (count, width)
            or not np.isfinite(value).all()
        ):
            raise Phase1JP4TapArchiveError(f"{name} array contract drift")
    weight = np.asarray(arrays["joint_weight"])
    if (
        weight.dtype != np.float32
        or weight.shape != (Z_DIM, HIDDEN_DIM)
        or not np.isfinite(weight).all()
    ):
        raise Phase1JP4TapArchiveError("joint_weight array contract drift")
    for name in (
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
    ):
        value = np.asarray(arrays[name])
        if (
            value.ndim != 1
            or len(value) != count
            or any(not item for item in value.astype(str).tolist())
        ):
            raise Phase1JP4TapArchiveError(f"{name} metadata contract drift")
    classes = np.asarray(arrays["class_ids"])
    if (
        classes.ndim != 1
        or len(classes) != 6
        or len(set(classes.astype(str).tolist())) != 6
        or set(classes.astype(str).tolist())
        != set(np.asarray(arrays["labels"]).astype(str).tolist())
        or len(set(np.asarray(arrays["physical_ids"]).astype(str).tolist()))
        != count
        or len(set(np.asarray(arrays["observation_ids"]).astype(str).tolist()))
        != count
    ):
        raise Phase1JP4TapArchiveError("tap registry/physical-ID closure drift")
    # The joint layer contains a bias.  Exact hidden/linear/pre-ReLU binding is
    # checked live by strict_zid_with_hook; the persisted closure can still
    # verify the byte-exact ReLU relation without storing that unchanged bias.
    if not np.array_equal(arrays["z_id"], np.maximum(arrays["pre_relu"], 0.0)):
        raise Phase1JP4TapArchiveError("z_id/pre_relu ReLU binding drift")


def export_phase1_jp4_tap_archive(
    *,
    cache_set_path: str | Path,
    cache_set_sha256: str,
    selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    reference_archive_path: str | Path,
    reference_archive_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    expected_checkpoint = _require_sha(checkpoint_sha256, "checkpoint SHA256")
    if expected_checkpoint != BASE_CHECKPOINT_SHA256:
        raise Phase1JP4TapArchiveError("checkpoint is not the frozen ADV3B02 base")
    checkpoint_file = _regular_bound(
        checkpoint_path, expected_checkpoint, "checkpoint"
    )
    reference_file = _regular_bound(
        reference_archive_path, reference_archive_sha256, "reference archive"
    )
    cache_file = _regular_bound(cache_set_path, cache_set_sha256, "cache set")
    if cache_set_sha256 not in KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256:
        raise Phase1JP4TapArchiveError("cache set is not a known source-validation lineage")
    try:
        runtime_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise Phase1JP4TapArchiveError("device is invalid") from exc
    if (
        runtime_device.type != "cuda"
        or not torch.cuda.is_available()
        or runtime_device.index is None
        or runtime_device.index >= torch.cuda.device_count()
    ):
        raise Phase1JP4TapArchiveError("tap export requires an explicit available CUDA device")

    with torch.serialization.safe_globals([SatViewStage]):
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, Mapping):
        raise Phase1JP4TapArchiveError("checkpoint safe load did not return a mapping")
    model, rebuild_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=256, device=runtime_device
    )
    model.to(runtime_device).eval()
    linear = _joint_linear(model)
    joint_weight = np.ascontiguousarray(
        linear.weight.detach().cpu().numpy(), dtype=np.float32
    )

    salt = _load_selection_salt(
        selection_salt_receipt_path,
        selection_salt_receipt_sha256,
        checkpoint_sha=expected_checkpoint,
    )
    arrays_by_scenario, cache_payload, cache_audit = CACHE_LOADER(
        cache_file, expected_scope="source_validation", allowed_roles={"source"}
    )
    if (
        cache_audit.get("outer_observed_schema") != LEO_WEAK_CACHE_SET_SCHEMA_V1
        or set(cache_audit.get("inner_observed_schema_by_scenario", {}).values())
        != {LEO_WEAK_CACHE_SCHEMA_V1}
        or cache_audit.get("legacy_schema_compatibility") is not True
        or cache_payload.get("cache_scope") != "source_validation"
    ):
        raise Phase1JP4TapArchiveError("verified source cache schema drift")
    metadata, selected_iq = _select_verified_observations(
        arrays_by_scenario, salt["selection_salt_sha256"]
    )
    if selected_iq.shape[1:] != (2, 256):
        raise Phase1JP4TapArchiveError("selected received-IQ shape drift")
    z_id, hidden, pre_relu, calls = _forward_taps(
        model,
        selected_iq,
        device=runtime_device,
        batch_size=batch_size,
    )
    del selected_iq, model, checkpoint
    torch.cuda.empty_cache()

    with np.load(reference_file, allow_pickle=False) as reference_npz:
        if tuple(reference_npz.files) != REFERENCE_MEMBERS:
            raise Phase1JP4TapArchiveError("reference archive member closure drift")
        reference = {
            name: np.asarray(reference_npz[name]) for name in reference_npz.files
        }
    for name in (
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
    ):
        if not np.array_equal(metadata[name], reference[name]):
            raise Phase1JP4TapArchiveError(f"reference metadata mismatch: {name}")
    maximum = float(
        np.max(np.abs(z_id.astype(np.float64) - reference["z_id"].astype(np.float64)))
    )
    if not np.isfinite(maximum) or maximum > 1.0e-5:
        raise Phase1JP4TapArchiveError(
            f"eager/reference z_id parity failed: max_abs={maximum}"
        )
    class_ids = np.asarray(reference["class_ids"], dtype=np.str_)
    arrays = {
        "z_id": z_id,
        "hidden": hidden,
        "pre_relu": pre_relu,
        "joint_weight": joint_weight,
        "labels": metadata["labels"],
        "receiver_ids": metadata["receiver_ids"],
        "day_ids": metadata["day_ids"],
        "physical_ids": metadata["physical_ids"],
        "scenario_names": metadata["scenario_names"],
        "class_ids": class_ids,
        "observation_ids": metadata["observation_ids"],
    }
    _validate_arrays(arrays)

    root = Path(output_dir).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to overwrite JP4 tap archive output")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        archive_path = staging / NPZ_NAME
        with archive_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema": SCHEMA,
            "status": "DEVELOPMENT_ONLY_NOT_FORMAL",
            "artifact_stage": "phase1_offline_before_target_access",
            "formal_phase2_eligible": False,
            "bundle_created": False,
            "target25_release_authorized": False,
            "exact_member_allowlist": list(MEMBERS),
            "array_sha256": {
                name: _sha_array(value) for name, value in arrays.items()
            },
            "artifact": {
                "path": NPZ_NAME,
                "sha256": _sha_file(archive_path),
            },
            "row_count": len(z_id),
            "inputs": {
                "checkpoint_sha256": expected_checkpoint,
                "cache_set_sha256": cache_set_sha256,
                "selection_salt_receipt_sha256": selection_salt_receipt_sha256,
                "reference_archive_sha256": reference_archive_sha256,
            },
            "runtime_audit": {
                "device": str(runtime_device),
                "batch_size": int(batch_size),
                "runtime_invocations": calls,
                "rebuild_audit": rebuild_audit,
                "strict_hook_exact_bytes": True,
                "eager_reference_z_id_max_abs": maximum,
            },
            "access_audit": {
                "source_validation_weak_iq_access": True,
                "clean_iq_access": False,
                "target_access": False,
                "query_access": False,
                "received_iq_persisted": False,
                "raw_iq_persisted": False,
            },
            "selection": {
                "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
                "selected_observations_per_physical_id": 1,
                "selection_salt_sha256": salt["selection_salt_sha256"],
            },
        }
        manifest_path = staging / MANIFEST_NAME
        _write_json_new(manifest_path, manifest)
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "archive": str(root / NPZ_NAME),
        "archive_sha256": manifest["artifact"]["sha256"],
        "manifest": str(root / MANIFEST_NAME),
        "manifest_sha256": _sha_file(root / MANIFEST_NAME),
        "row_count": len(z_id),
        "runtime_invocations": calls,
        "eager_reference_z_id_max_abs": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cache-set",
        "cache-set-sha256",
        "selection-salt-receipt",
        "selection-salt-receipt-sha256",
        "checkpoint",
        "checkpoint-sha256",
        "reference-archive",
        "reference-archive-sha256",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = export_phase1_jp4_tap_archive(
        cache_set_path=args.cache_set,
        cache_set_sha256=args.cache_set_sha256,
        selection_salt_receipt_path=args.selection_salt_receipt,
        selection_salt_receipt_sha256=args.selection_salt_receipt_sha256,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        reference_archive_path=args.reference_archive,
        reference_archive_sha256=args.reference_archive_sha256,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
