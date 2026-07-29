#!/usr/bin/env python
"""Prepare, sign, and finalize the formal Phase1 input for full ablations.

The three subcommands keep the Ed25519 private key off N607:

* ``prepare`` runs beside the completed checkpoint and exact Phase1 component;
* ``sign`` signs only the detached request on the trusted local controller;
* ``finalize`` verifies the returned envelope and writes the remote path binding.

No dataset path, sample feature, training checkpoint, or private key enters the
formal deployment package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi import somph_runtime_trust as runtime_trust  # noqa: E402
from cvsrffi.checkpoint_loading import (  # noqa: E402
    build_exact_ssdg_model_from_checkpoint,
)
from cvsrffi.identity_only_forward import (  # noqa: E402
    identity_only_feature_forward,
)
from cvsrffi.phase1_adv3b02_deployment_bundle import (  # noqa: E402
    BUNDLE_MANIFEST_SCHEMA,
    CLASS_BINDING_SCHEMA,
    DETACHED_SEAL_SCHEMA,
    MANIFEST_RELATIVE_PATH,
    SIGNATURE_DOMAIN,
    SIGNATURE_ENVELOPE_SCHEMA,
    SIGNING_REQUEST_SCHEMA,
    build_unsigned_adv3b02_deployment_bundle,
    canonical_json_bytes,
    class_handle_binding_sha256,
    load_formal_adv3b02_deployment_bundle,
    runtime_structure_receipt,
    sha256_bytes,
    sha256_file,
)
from cvsrffi.phase1_center_lowrank_prototype_bundle import (  # noqa: E402
    FINAL_MANIFEST_FIELDS,
    MANIFEST_NAME as COMPONENT_MANIFEST_NAME,
    PENDING_OUTER_JOINT_SEAL,
)
from cvsrffi.phase2_prototypes import (  # noqa: E402
    attach_endpoint_accept_v1_manifest,
    verify_endpoint_accept_v1_manifest,
)


DEPLOYMENT_BINDING_SCHEMA = "cvs.full_ablation.phase1.deployment_binding.v1"
PREPARE_RECEIPT_SCHEMA = "cvs.full_ablation.phase1.deployment_prepare_receipt.v1"
SIGN_RECEIPT_SCHEMA = "cvs.full_ablation.phase1.deployment_sign_receipt.v1"
FINALIZE_RECEIPT_SCHEMA = "cvs.full_ablation.phase1.deployment_finalize_receipt.v1"
METHOD_ID = "P1-FULL"
GENERATION_CONFIG_SCHEMA = (
    "cvs.full_ablation.phase1.deployment_generation_config.v1"
)
NORMALIZATION_RECEIPT_SCHEMA = (
    "cvs.full_ablation.phase1.prototype_normalization_receipt.v1"
)
_GENERATION_CONFIG_KEYS = {
    "schema",
    "row_key",
    "run_id",
    "checkpoint_lineage_sha256",
    "completion_receipt_sha256",
    "original_prototype_pt_sha256",
    "original_prototype_json_sha256",
    "normalized_prototype_pt_sha256",
    "normalized_prototype_json_sha256",
    "prototype_normalization_status",
    "class_handle_binding_sha256",
    "component_export",
}


class FullAblationDeploymentError(ValueError):
    """Raised when the Phase1 deployment chain is incomplete or inconsistent."""


class Phase1IdentityRuntime(nn.Module):
    """Expose only normalized identity features and old-class logits."""

    def __init__(self, model: nn.Module, *, runtime_batch_size: int = 256) -> None:
        super().__init__()
        self.model = model
        self.runtime_batch_size = int(runtime_batch_size)

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = rows.size(0)
        padded = rows.new_zeros(
            (self.runtime_batch_size, rows.size(1), rows.size(2))
        )
        padded[:count].copy_(rows)
        result = identity_only_feature_forward(self.model, padded, "z_id")
        if result is None:
            raise RuntimeError("checkpoint has no identity-only z_id export")
        features, logits = result
        return features[:count], logits[:count]


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(payload)) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing deployment JSON")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullAblationDeploymentError(f"{context} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise FullAblationDeploymentError(f"{context} root must be a mapping")
    return dict(value)


def _write_torch_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise FullAblationDeploymentError("checkpoint root must be a mapping")
    if checkpoint.get("candidate_id") != METHOD_ID:
        raise FullAblationDeploymentError("checkpoint is not the frozen P1-FULL arm")
    return checkpoint


def _class_binding_source(
    path: Path,
    *,
    expected_phase1_txs: Sequence[str],
) -> tuple[str, ...]:
    payload = _load_json(path, context="class binding source")
    if (
        set(payload) != {"schema", "checkpoint_sha256", "entries", "evidence"}
        or payload.get("schema") != "cvs.phase2.d19_adv3b02_class_binding.v1"
        or not isinstance(payload.get("entries"), list)
    ):
        raise FullAblationDeploymentError("class binding source schema drift")
    expected = tuple(str(value) for value in expected_phase1_txs)
    entries = payload["entries"]
    if len(entries) != len(expected):
        raise FullAblationDeploymentError("class binding source count drift")
    handles: list[str] = []
    for index, (entry, expected_tx) in enumerate(zip(entries, expected)):
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {"class_index", "phase1_tx", "registered_class_handle"}
            or int(entry.get("class_index", -1)) != index
            or str(entry.get("phase1_tx", "")) != expected_tx
            or not str(entry.get("registered_class_handle", "")).strip()
        ):
            raise FullAblationDeploymentError("class binding TX/order drift")
        handles.append(str(entry["registered_class_handle"]))
    if len(set(handles)) != len(handles):
        raise FullAblationDeploymentError("class binding handles are not unique")
    return tuple(handles)


def _prototype_identity(
    pt_path: Path,
    json_path: Path,
    *,
    checkpoint_sha256: str,
    normalized_output_dir: Path | None,
    allow_boundary_summary_rebuild: bool,
) -> tuple[
    dict[str, Any], tuple[str, ...], Path, Path, str
]:
    try:
        package = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        package = torch.load(pt_path, map_location="cpu")
    sidecar = _load_json(json_path, context="prototype JSON")
    if not isinstance(package, Mapping) or _jsonable(package) != sidecar:
        raise FullAblationDeploymentError("prototype PT/JSON content drift")
    normalized_package = dict(package)
    normalization_status = "UNCHANGED_VALID"
    try:
        endpoint = verify_endpoint_accept_v1_manifest(normalized_package)
    except ValueError as exc:
        if (
            str(exc) != "endpoint_accept_v1 boundary hash mismatch"
            or not allow_boundary_summary_rebuild
        ):
            raise FullAblationDeploymentError(
                "prototype endpoint manifest verification failed"
            ) from exc
        before = {
            key: _jsonable(value)
            for key, value in normalized_package.items()
            if key not in {"endpoint_accept_v1", "metadata", "schema_version"}
        }
        normalized_package = attach_endpoint_accept_v1_manifest(
            normalized_package
        )
        after = {
            key: _jsonable(value)
            for key, value in normalized_package.items()
            if key not in {"endpoint_accept_v1", "metadata", "schema_version"}
        }
        if before != after:
            raise FullAblationDeploymentError(
                "endpoint summary normalization changed prototype tensors"
            )
        try:
            endpoint = verify_endpoint_accept_v1_manifest(normalized_package)
        except (TypeError, ValueError) as repaired_exc:
            raise FullAblationDeploymentError(
                "prototype endpoint summary normalization failed"
            ) from repaired_exc
        normalization_status = "REBUILT_BOUNDARY_SUMMARY_ONLY"
    except TypeError as exc:
        raise FullAblationDeploymentError(
            "prototype endpoint manifest verification failed"
        ) from exc
    if normalized_output_dir is not None:
        normalized_pt = normalized_output_dir / "phase2_zid_prototypes.pt"
        normalized_json = normalized_output_dir / "phase2_zid_prototypes.json"
        _write_torch_new(normalized_pt, normalized_package)
        _write_json_new(normalized_json, _jsonable(normalized_package))
        try:
            reloaded = torch.load(
                normalized_pt, map_location="cpu", weights_only=False
            )
        except TypeError:
            reloaded = torch.load(normalized_pt, map_location="cpu")
        reloaded_sidecar = _load_json(
            normalized_json, context="normalized prototype JSON"
        )
        if (
            not isinstance(reloaded, Mapping)
            or _jsonable(reloaded) != reloaded_sidecar
        ):
            raise FullAblationDeploymentError(
                "normalized prototype PT/JSON content drift"
            )
        verify_endpoint_accept_v1_manifest(reloaded)
    else:
        normalized_pt = pt_path
        normalized_json = json_path
    identity = endpoint.get("inference_identity")
    metadata = normalized_package.get("metadata")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(metadata, Mapping)
        or str(identity.get("source_checkpoint_sha256", "")).lower()
        != checkpoint_sha256
        or str(identity.get("candidate_id", "")) != METHOD_ID
        or int(identity.get("known_class_count", -1)) != 6
        or list(identity.get("logit_class_order", ())) != list(range(6))
        or list(metadata.get("logit_class_order", ())) != list(range(6))
    ):
        raise FullAblationDeploymentError("prototype inference identity drift")
    phase1_txs = tuple(str(value) for value in metadata.get("class_id_to_tx", ()))
    if len(phase1_txs) != 6 or len(set(phase1_txs)) != 6:
        raise FullAblationDeploymentError("prototype Phase1 TX registry drift")
    return (
        dict(identity),
        phase1_txs,
        normalized_pt,
        normalized_json,
        normalization_status,
    )


def _input_len(checkpoint: Mapping[str, Any], override: int) -> int:
    if override > 0:
        return int(override)
    args = checkpoint.get("args")
    baseline = checkpoint.get("baseline_args")
    for source in (args, baseline):
        if isinstance(source, Mapping):
            for key in ("wisig_out_len", "input_len"):
                value = source.get(key)
                if value is not None and int(value) > 0:
                    return int(value)
    raise FullAblationDeploymentError("cannot determine checkpoint input length")


@torch.no_grad()
def _run(
    module: nn.Module, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    module.eval()
    features, logits = module(rows)
    return features.detach().float().cpu(), logits.detach().float().cpu()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    header = canonical_json_bytes(
        {"shape": list(array.shape), "dtype": array.dtype.str}
    )
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _runtime_and_parity(
    checkpoint: Mapping[str, Any],
    *,
    input_len: int,
    device: torch.device,
    runtime_path: Path,
    parity_seed: int,
    parity_rows: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    model, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=input_len,
        device=device,
    )
    model.to(device).eval()
    wrapper = Phase1IdentityRuntime(model).to(device).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(parity_seed))
    if int(parity_rows) != 8:
        raise FullAblationDeploymentError(
            "formal runtime parity requires exactly the 1/8/64/256 batch set"
        )
    validated_batch_sizes = [1, 8, 64, 256]
    probes = torch.randn(
        max(validated_batch_sizes),
        2,
        int(input_len),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    trace_example = probes[: min(2, len(probes))]
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(
        wrapper, trace_example, strict=False, check_trace=False
    )
    torch.jit.save(traced, str(runtime_path))
    runtime = torch.jit.load(str(runtime_path), map_location=device).eval()
    vector_rows: list[dict[str, Any]] = []
    maximum = 0.0
    for batch_size in validated_batch_sizes:
        current = probes[:batch_size]
        eager_features, eager_logits = _run(wrapper, current)
        script_features, script_logits = _run(runtime, current)
        if (
            tuple(eager_features.shape) != (batch_size, 160)
            or tuple(script_features.shape) != (batch_size, 160)
            or tuple(eager_logits.shape) != (batch_size, 6)
            or tuple(script_logits.shape) != (batch_size, 6)
            or not torch.isfinite(eager_features).all()
            or not torch.isfinite(script_features).all()
            or not torch.isfinite(eager_logits).all()
            or not torch.isfinite(script_logits).all()
        ):
            raise FullAblationDeploymentError(
                f"runtime parity shape/finite closure failed: batch={batch_size}"
            )
        feature_delta = float(
            torch.max(torch.abs(eager_features - script_features)).item()
        )
        logit_delta = float(
            torch.max(torch.abs(eager_logits - script_logits)).item()
        )
        maximum = max(maximum, feature_delta, logit_delta)
        vector_rows.append(
            {
                "batch_size": batch_size,
                "input_sha256": _tensor_sha256(current),
                "feature_sha256": _tensor_sha256(eager_features),
                "logit_sha256": _tensor_sha256(eager_logits),
                "max_abs_feature_delta": feature_delta,
                "max_abs_logit_delta": logit_delta,
            }
        )
    if not np.isfinite(maximum) or maximum > 1.0e-5:
        raise FullAblationDeploymentError(
            f"runtime/checkpoint parity failed: max_abs={maximum}"
        )
    vector_root = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": "cvs.phase1.runtime_parity_vectors.v1",
                "seed": int(parity_seed),
                "validated_batches": vector_rows,
            }
        )
    )
    runtime_sha = sha256_file(runtime_path)
    parity = {
        "schema": "cvs.phase1.runtime_checkpoint_parity_receipt.v1",
        "runtime_sha256": runtime_sha,
        "parity_status": "PASS",
        "max_abs_output_delta": maximum,
        "parity_vector_root_sha256": vector_root,
        "validated_batch_sizes": validated_batch_sizes,
        "feature_dim": 160,
        "logit_dim": 6,
        "finite_outputs_verified": True,
        **runtime_structure_receipt(runtime_path),
    }
    return runtime_sha, parity, checkpoint_audit


def _component_manifest(
    component_dir: Path,
    *,
    checkpoint_sha256: str,
    class_binding_sha256_value: str,
) -> dict[str, Any]:
    manifest = _load_json(
        component_dir / COMPONENT_MANIFEST_NAME,
        context="Phase1 component manifest",
    )
    if (
        set(manifest) != FINAL_MANIFEST_FIELDS
        or manifest.get("checkpoint_sha256") != checkpoint_sha256
        or manifest.get("class_handle_binding_sha256")
        != class_binding_sha256_value
        or manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL
        or manifest.get("formal_phase2_eligible") is not False
    ):
        raise FullAblationDeploymentError(
            "Phase1 component checkpoint/class/formal binding drift"
        )
    return manifest


def _completion_receipt(
    path: Path,
    *,
    checkpoint_sha256: str,
    original_prototype_pt_sha256: str,
    original_prototype_json_sha256: str,
) -> dict[str, Any]:
    receipt = _load_json(path, context="Phase1 completion receipt")
    prototype_hashes = receipt.get("prototype_hashes")
    if (
        receipt.get("ablation_id") != METHOD_ID
        or receipt.get("phase1_training_complete") is not True
        or receipt.get("terminal_status") != "COMPLETE"
        or int(receipt.get("exit_code", -1)) != 0
        or str(receipt.get("selected_checkpoint_sha256", "")).lower()
        != checkpoint_sha256
        or not isinstance(prototype_hashes, Mapping)
        or str(prototype_hashes.get("prototype_path", "")).lower()
        != original_prototype_pt_sha256
        or str(prototype_hashes.get("prototype_json_path", "")).lower()
        != original_prototype_json_sha256
        or not str(receipt.get("row_key", "")).startswith("P1-FULL__")
        or not str(receipt.get("run_id", "")).strip()
    ):
        raise FullAblationDeploymentError(
            "Phase1 completion receipt checkpoint/prototype closure drift"
        )
    return receipt


def _generation_config(
    path: Path,
    *,
    checkpoint_sha256: str,
    completion_receipt_sha256: str,
    normalized_prototype_pt_sha256: str,
    normalized_prototype_json_sha256: str,
    class_binding_sha256_value: str,
) -> dict[str, Any]:
    config = _load_json(path, context="deployment generation config")
    if (
        set(config) != _GENERATION_CONFIG_KEYS
        or config.get("schema") != GENERATION_CONFIG_SCHEMA
        or config.get("checkpoint_lineage_sha256") != checkpoint_sha256
        or config.get("completion_receipt_sha256")
        != completion_receipt_sha256
        or config.get("normalized_prototype_pt_sha256")
        != normalized_prototype_pt_sha256
        or config.get("normalized_prototype_json_sha256")
        != normalized_prototype_json_sha256
        or config.get("class_handle_binding_sha256")
        != class_binding_sha256_value
        or config.get("prototype_normalization_status")
        not in {"UNCHANGED_VALID", "REBUILT_BOUNDARY_SUMMARY_ONLY"}
        or not isinstance(config.get("component_export"), Mapping)
    ):
        raise FullAblationDeploymentError(
            "deployment generation config closure drift"
        )
    return config


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError("refusing to reuse prototype normalization root")
    checkpoint_path = Path(args.checkpoint).resolve()
    prototype_pt = Path(args.prototype_pt).resolve()
    prototype_json = Path(args.prototype_json).resolve()
    completion_path = Path(args.completion_receipt).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    original_pt_sha = sha256_file(prototype_pt)
    original_json_sha = sha256_file(prototype_json)
    completion = _completion_receipt(
        completion_path,
        checkpoint_sha256=checkpoint_sha,
        original_prototype_pt_sha256=original_pt_sha,
        original_prototype_json_sha256=original_json_sha,
    )
    (
        identity,
        phase1_txs,
        normalized_pt,
        normalized_json,
        normalization_status,
    ) = _prototype_identity(
        prototype_pt,
        prototype_json,
        checkpoint_sha256=checkpoint_sha,
        normalized_output_dir=output_root / "deployment_prototype",
        allow_boundary_summary_rebuild=True,
    )
    class_handles = _class_binding_source(
        Path(args.class_binding_source).resolve(),
        expected_phase1_txs=phase1_txs,
    )
    binding_sha = class_handle_binding_sha256(class_handles)
    generation_config = {
        "schema": GENERATION_CONFIG_SCHEMA,
        "row_key": str(completion["row_key"]),
        "run_id": str(completion["run_id"]),
        "checkpoint_lineage_sha256": checkpoint_sha,
        "completion_receipt_sha256": sha256_file(completion_path),
        "original_prototype_pt_sha256": original_pt_sha,
        "original_prototype_json_sha256": original_json_sha,
        "normalized_prototype_pt_sha256": sha256_file(normalized_pt),
        "normalized_prototype_json_sha256": sha256_file(normalized_json),
        "prototype_normalization_status": normalization_status,
        "class_handle_binding_sha256": binding_sha,
        "component_export": {
            "batch_size": int(args.component_batch_size),
            "num_workers": int(args.component_num_workers),
            "min_samples_per_cell": int(args.min_samples_per_cell),
            "radius_histogram_bins": int(args.radius_histogram_bins),
            "feature": "normalized_z_id_160",
            "radius": "per_domain_class_p90_cosine_distance",
            "raw_dataset_audit_repeated": False,
            "cross_launch_data_identity_required": False,
        },
    }
    config_path = output_root / "generation_config.json"
    _write_json_new(config_path, generation_config)
    receipt = {
        "schema": NORMALIZATION_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "row_key": str(completion["row_key"]),
        "run_id": str(completion["run_id"]),
        "checkpoint_lineage_sha256": checkpoint_sha,
        "completion_receipt_path": str(completion_path),
        "completion_receipt_sha256": sha256_file(completion_path),
        "normalized_prototype_pt_path": str(normalized_pt),
        "normalized_prototype_pt_sha256": sha256_file(normalized_pt),
        "normalized_prototype_json_path": str(normalized_json),
        "normalized_prototype_json_sha256": sha256_file(normalized_json),
        "prototype_normalization_status": normalization_status,
        "class_handle_binding_sha256": binding_sha,
        "generation_config_path": str(config_path),
        "generation_config_sha256": sha256_file(config_path),
        "prototype_candidate_id": str(identity["candidate_id"]),
    }
    _write_json_new(output_root / "normalization_receipt.json", receipt)
    return receipt


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError("refusing to reuse deployment output root")
    checkpoint_path = Path(args.checkpoint).resolve()
    prototype_pt = Path(args.prototype_pt).resolve()
    prototype_json = Path(args.prototype_json).resolve()
    completion_path = Path(args.completion_receipt).resolve()
    generation_config_path = Path(args.generation_config).resolve()
    component_dir = Path(args.component_dir).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    original_pt_sha = str(
        _load_json(
            generation_config_path,
            context="deployment generation config",
        ).get("original_prototype_pt_sha256", "")
    )
    original_json_sha = str(
        _load_json(
            generation_config_path,
            context="deployment generation config",
        ).get("original_prototype_json_sha256", "")
    )
    completion = _completion_receipt(
        completion_path,
        checkpoint_sha256=checkpoint_sha,
        original_prototype_pt_sha256=original_pt_sha,
        original_prototype_json_sha256=original_json_sha,
    )
    (
        identity,
        phase1_txs,
        normalized_prototype_pt,
        normalized_prototype_json,
        prototype_normalization_status,
    ) = _prototype_identity(
        prototype_pt,
        prototype_json,
        checkpoint_sha256=checkpoint_sha,
        normalized_output_dir=None,
        allow_boundary_summary_rebuild=False,
    )
    class_handles = _class_binding_source(
        Path(args.class_binding_source).resolve(),
        expected_phase1_txs=phase1_txs,
    )
    binding_sha = class_handle_binding_sha256(class_handles)
    generation_config = _generation_config(
        generation_config_path,
        checkpoint_sha256=checkpoint_sha,
        completion_receipt_sha256=sha256_file(completion_path),
        normalized_prototype_pt_sha256=sha256_file(prototype_pt),
        normalized_prototype_json_sha256=sha256_file(prototype_json),
        class_binding_sha256_value=binding_sha,
    )
    component = _component_manifest(
        component_dir,
        checkpoint_sha256=checkpoint_sha,
        class_binding_sha256_value=binding_sha,
    )
    if component["generation_config_sha256"] != sha256_file(
        generation_config_path
    ):
        raise FullAblationDeploymentError(
            "component does not bind the deployment generation config"
        )
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FullAblationDeploymentError("requested CUDA device is unavailable")
    resolved_input_len = _input_len(checkpoint, int(args.input_len))

    work = output_root / "work"
    inputs = work / "locks"
    runtime_path = work / "runtime" / "p1_full.torchscript.pt"
    runtime_sha, parity, checkpoint_audit = _runtime_and_parity(
        checkpoint,
        input_len=resolved_input_len,
        device=device,
        runtime_path=runtime_path,
        parity_seed=int(args.parity_seed),
        parity_rows=int(args.parity_rows),
    )
    class_binding_path = inputs / "class_binding.json"
    _write_json_new(
        class_binding_path,
        {
            "schema": CLASS_BINDING_SCHEMA,
            "checkpoint_lineage_sha256": checkpoint_sha,
            "class_id_to_handle": [
                {"class_index": index, "class_handle": handle}
                for index, handle in enumerate(class_handles)
            ],
            "class_handle_binding_sha256": binding_sha,
        },
    )
    parity_path = inputs / "runtime_checkpoint_parity_receipt.json"
    parity["checkpoint_lineage_sha256"] = checkpoint_sha
    _write_json_new(parity_path, parity)
    generation_path = inputs / "generation_lock.json"
    _write_json_new(
        generation_path,
        {
            "schema": "cvs.phase1.prototype_generation_lock.v1",
            "checkpoint_lineage_sha256": checkpoint_sha,
            "component_pre_sign_content_root_sha256": component[
                "pre_sign_content_root_sha256"
            ],
            "class_handle_binding_sha256": binding_sha,
            "generation_config_sha256": component["generation_config_sha256"],
            "generation_code_sha256": component["generation_code_sha256"],
            "phase1_stream_sha256": component["phase1_stream_sha256"],
            "radius_generation_proof_sha256": component[
                "radius_generation_proof_sha256"
            ],
        },
    )
    method_path = inputs / "method_lock.json"
    _write_json_new(
        method_path,
        {
            "schema": "cvs.phase1.adv3b02_method_lock.v1",
            "method_id": METHOD_ID,
            "checkpoint_lineage_sha256": checkpoint_sha,
            "runtime_sha256": runtime_sha,
            "component_pre_sign_content_root_sha256": component[
                "pre_sign_content_root_sha256"
            ],
            "class_handle_binding_sha256": binding_sha,
            "parity_receipt_sha256": sha256_file(parity_path),
            "generation_lock_sha256": sha256_file(generation_path),
            "generation_config_sha256": component["generation_config_sha256"],
            "generation_code_sha256": component["generation_code_sha256"],
        },
    )
    package_root = output_root / "package"
    seal_path = output_root / "external" / "deployment.seal.json"
    request_path = output_root / "external" / "signing_request.json"
    bundle = build_unsigned_adv3b02_deployment_bundle(
        package_root,
        torchscript_runtime_path=runtime_path,
        component_dir=component_dir,
        class_binding_path=class_binding_path,
        parity_receipt_path=parity_path,
        generation_lock_path=generation_path,
        method_lock_path=method_path,
        detached_seal_path=seal_path,
        signing_request_path=request_path,
    )
    receipt = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": "AWAITING_EXTERNAL_SIGNATURE",
        "package_root": str(package_root),
        "detached_seal_path": str(seal_path),
        "signing_request_path": str(request_path),
        "checkpoint_lineage_sha256": checkpoint_sha,
        "prototype_pt_path": str(normalized_prototype_pt),
        "prototype_json_path": str(normalized_prototype_json),
        "prototype_pt_sha256": sha256_file(normalized_prototype_pt),
        "prototype_json_sha256": sha256_file(normalized_prototype_json),
        "prototype_normalization_status": generation_config[
            "prototype_normalization_status"
        ],
        "completion_receipt_path": str(completion_path),
        "completion_receipt_sha256": sha256_file(completion_path),
        "generation_config_path": str(generation_config_path),
        "generation_config_sha256": sha256_file(generation_config_path),
        "original_prototype_pt_sha256": generation_config[
            "original_prototype_pt_sha256"
        ],
        "original_prototype_json_sha256": generation_config[
            "original_prototype_json_sha256"
        ],
        "prototype_run_id": str(identity["run_id"]),
        "prototype_candidate_id": str(identity["candidate_id"]),
        "input_len": resolved_input_len,
        "checkpoint_load_audit": checkpoint_audit,
        **{
            key: value
            for key, value in bundle.items()
            if key not in {"manifest_path", "detached_seal_path", "signing_request_path"}
        },
    }
    _write_json_new(output_root / "prepare_receipt.json", receipt)
    return receipt


def sign(args: argparse.Namespace) -> dict[str, Any]:
    request_path = Path(args.signing_request).resolve()
    envelope_path = Path(args.signature_envelope).resolve()
    request = _load_json(request_path, context="deployment signing request")
    if (
        set(request)
        != {
            "schema",
            "signature_message_sha256",
            "unsigned_signature_envelope",
            "outer_content_root_sha256",
        }
        or request.get("schema") != SIGNING_REQUEST_SCHEMA
        or not isinstance(request.get("unsigned_signature_envelope"), Mapping)
    ):
        raise FullAblationDeploymentError("deployment signing request schema drift")
    unsigned = dict(request["unsigned_signature_envelope"])
    if (
        unsigned.get("schema") != SIGNATURE_ENVELOPE_SCHEMA
        or unsigned.get("domain") != SIGNATURE_DOMAIN
        or unsigned.get("issuer") != runtime_trust.PINNED_AUTHORITY_ISSUER
        or unsigned.get("key_id") != runtime_trust.PINNED_AUTHORITY_KEY_ID
    ):
        raise FullAblationDeploymentError("deployment signing authority drift")
    message = (
        SIGNATURE_DOMAIN.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(unsigned)
    )
    if sha256_bytes(message) != request["signature_message_sha256"]:
        raise FullAblationDeploymentError("deployment signing message drift")

    from scripts.sign_cvs_somph_authority_lock import (
        PINNED_OPENSSL_BINARY_PATH,
        _pinned_openssl_binary,
        _private_openssl_executable,
        _sign_with_openssl,
    )

    requested_openssl = (
        str(args.openssl_bin).strip()
        if str(args.openssl_bin).strip()
        else PINNED_OPENSSL_BINARY_PATH
    )
    (
        _openssl_path,
        openssl_verified_bytes,
        openssl_sha256,
        openssl_runtime_files,
    ) = _pinned_openssl_binary(requested_openssl)
    private_key_path = Path(args.authority_private_key).resolve(strict=True)
    if not private_key_path.is_file():
        raise FullAblationDeploymentError(
            "authority private key must be a regular file"
        )
    with _private_openssl_executable(
        verified_bytes=openssl_verified_bytes,
        expected_sha256=openssl_sha256,
        runtime_files=openssl_runtime_files,
    ) as private_openssl:
        signature = _sign_with_openssl(
            openssl_binary=private_openssl,
            private_key=private_key_path,
            message=message,
        )
    public_raw = bytes.fromhex(runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX)
    if (
        len(public_raw) != 32
        or hashlib.sha256(public_raw).hexdigest()
        != runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
    ):
        raise FullAblationDeploymentError("pinned deployment authority drift")
    try:
        runtime_trust.verify_ed25519(public_raw, message, signature)
    except Exception as exc:
        raise FullAblationDeploymentError(
            "private key does not match the pinned deployment authority"
        ) from exc
    envelope = {
        **unsigned,
        "signature_ed25519_hex": signature.hex(),
    }
    _write_json_new(envelope_path, envelope)
    receipt = {
        "schema": SIGN_RECEIPT_SCHEMA,
        "status": "SIGNED",
        "signing_request_sha256": sha256_file(request_path),
        "signature_envelope_sha256": sha256_file(envelope_path),
        "detached_seal_sha256": unsigned["detached_seal_sha256"],
        "outer_content_root_sha256": request["outer_content_root_sha256"],
        "openssl_binary_sha256": openssl_sha256,
    }
    if str(args.sign_receipt).strip():
        _write_json_new(Path(args.sign_receipt).resolve(), receipt)
    return receipt


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root).resolve()
    seal_path = Path(args.detached_seal).resolve()
    envelope_path = Path(args.signature_envelope).resolve()
    binding_path = Path(args.deployment_binding).resolve()
    completion_path = Path(args.completion_receipt).resolve()
    generation_config_path = Path(args.generation_config).resolve()
    prototype_pt = Path(args.prototype_pt).resolve()
    prototype_json = Path(args.prototype_json).resolve()
    manifest = _load_json(
        package_root / MANIFEST_RELATIVE_PATH,
        context="deployment manifest",
    )
    seal = _load_json(seal_path, context="deployment detached seal")
    if (
        manifest.get("schema") != BUNDLE_MANIFEST_SCHEMA
        or seal.get("schema") != DETACHED_SEAL_SCHEMA
        or seal.get("outer_content_root_sha256")
        != manifest.get("outer_content_root_sha256")
    ):
        raise FullAblationDeploymentError("deployment manifest/seal drift")
    config = _generation_config(
        generation_config_path,
        checkpoint_sha256=manifest["checkpoint_lineage_sha256"],
        completion_receipt_sha256=sha256_file(completion_path),
        normalized_prototype_pt_sha256=sha256_file(prototype_pt),
        normalized_prototype_json_sha256=sha256_file(prototype_json),
        class_binding_sha256_value=manifest[
            "class_handle_binding_sha256"
        ],
    )
    _completion_receipt(
        completion_path,
        checkpoint_sha256=manifest["checkpoint_lineage_sha256"],
        original_prototype_pt_sha256=config[
            "original_prototype_pt_sha256"
        ],
        original_prototype_json_sha256=config[
            "original_prototype_json_sha256"
        ],
    )
    if sha256_file(generation_config_path) != manifest[
        "generation_config_sha256"
    ]:
        raise FullAblationDeploymentError(
            "external prototype lock is not bound by the signed generation config"
        )
    binding = {
        "schema": DEPLOYMENT_BINDING_SCHEMA,
        "package_root": str(package_root),
        "detached_seal_path": str(seal_path),
        "detached_seal_sha256": sha256_file(seal_path),
        "signature_envelope_path": str(envelope_path),
        "signature_envelope_sha256": sha256_file(envelope_path),
        "checkpoint_lineage_sha256": manifest["checkpoint_lineage_sha256"],
        "runtime_sha256": manifest["runtime_sha256"],
        "component_pre_sign_content_root_sha256": manifest[
            "component_pre_sign_content_root_sha256"
        ],
        "class_handle_binding_sha256": manifest[
            "class_handle_binding_sha256"
        ],
        "parity_receipt_sha256": manifest["parity_receipt_sha256"],
        "generation_lock_sha256": manifest["generation_lock_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "generation_config_sha256": manifest["generation_config_sha256"],
        "generation_code_sha256": manifest["generation_code_sha256"],
        "outer_content_root_sha256": manifest["outer_content_root_sha256"],
        "phase1_completion_receipt_path": str(completion_path),
        "phase1_completion_receipt_sha256": sha256_file(completion_path),
        "generation_config_path": str(generation_config_path),
        "prototype_pt_path": str(prototype_pt),
        "prototype_pt_sha256": sha256_file(prototype_pt),
        "prototype_json_path": str(prototype_json),
        "prototype_json_sha256": sha256_file(prototype_json),
    }
    verified = load_formal_adv3b02_deployment_bundle(
        package_root,
        detached_seal_path=seal_path,
        expected_detached_seal_sha256=binding["detached_seal_sha256"],
        signature_envelope_path=envelope_path,
        expected_signature_envelope_sha256=binding[
            "signature_envelope_sha256"
        ],
        expected_checkpoint_lineage_sha256=binding[
            "checkpoint_lineage_sha256"
        ],
        expected_runtime_sha256=binding["runtime_sha256"],
        expected_component_pre_sign_content_root_sha256=binding[
            "component_pre_sign_content_root_sha256"
        ],
        expected_class_handle_binding_sha256=binding[
            "class_handle_binding_sha256"
        ],
        expected_parity_receipt_sha256=binding["parity_receipt_sha256"],
        expected_generation_lock_sha256=binding["generation_lock_sha256"],
        expected_method_lock_sha256=binding["method_lock_sha256"],
        expected_generation_config_sha256=binding[
            "generation_config_sha256"
        ],
        expected_generation_code_sha256=binding[
            "generation_code_sha256"
        ],
        expected_outer_content_root_sha256=binding[
            "outer_content_root_sha256"
        ],
    )
    if verified.formal_phase2_context.get("formal_phase2_eligible") is not True:
        raise FullAblationDeploymentError(
            "deployment bundle did not become formally Phase2 eligible"
        )
    _write_json_new(binding_path, binding)
    return {
        "schema": FINALIZE_RECEIPT_SCHEMA,
        "status": "FORMAL_PHASE2_ELIGIBLE",
        "deployment_binding": str(binding_path),
        "deployment_binding_sha256": sha256_file(binding_path),
        "checkpoint_lineage_sha256": binding["checkpoint_lineage_sha256"],
        "runtime_sha256": binding["runtime_sha256"],
        "outer_content_root_sha256": binding["outer_content_root_sha256"],
        "class_count": len(verified.class_binding["class_id_to_handle"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the P1-FULL formal Phase1 deployment chain."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--checkpoint", required=True)
    normalize_parser.add_argument("--prototype-pt", required=True)
    normalize_parser.add_argument("--prototype-json", required=True)
    normalize_parser.add_argument("--completion-receipt", required=True)
    normalize_parser.add_argument("--class-binding-source", required=True)
    normalize_parser.add_argument("--output-root", required=True)
    normalize_parser.add_argument("--component-batch-size", type=int, default=512)
    normalize_parser.add_argument("--component-num-workers", type=int, default=0)
    normalize_parser.add_argument("--min-samples-per-cell", type=int, default=2)
    normalize_parser.add_argument("--radius-histogram-bins", type=int, default=4096)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", required=True)
    prepare_parser.add_argument("--prototype-pt", required=True)
    prepare_parser.add_argument("--prototype-json", required=True)
    prepare_parser.add_argument("--component-dir", required=True)
    prepare_parser.add_argument("--class-binding-source", required=True)
    prepare_parser.add_argument("--completion-receipt", required=True)
    prepare_parser.add_argument("--generation-config", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--device", default="cpu")
    prepare_parser.add_argument("--input-len", type=int, default=0)
    prepare_parser.add_argument("--parity-seed", type=int, default=7281105)
    prepare_parser.add_argument("--parity-rows", type=int, default=8)

    sign_parser = subparsers.add_parser("sign")
    sign_parser.add_argument("--signing-request", required=True)
    sign_parser.add_argument("--authority-private-key", required=True)
    sign_parser.add_argument("--openssl-bin", default="")
    sign_parser.add_argument("--signature-envelope", required=True)
    sign_parser.add_argument("--sign-receipt", default="")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--package-root", required=True)
    finalize_parser.add_argument("--detached-seal", required=True)
    finalize_parser.add_argument("--signature-envelope", required=True)
    finalize_parser.add_argument("--deployment-binding", required=True)
    finalize_parser.add_argument("--completion-receipt", required=True)
    finalize_parser.add_argument("--generation-config", required=True)
    finalize_parser.add_argument("--prototype-pt", required=True)
    finalize_parser.add_argument("--prototype-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "normalize":
        result = normalize(args)
    elif args.command == "prepare":
        if int(args.parity_rows) != 8:
            raise FullAblationDeploymentError(
                "formal runtime parity requires --parity-rows 8"
            )
        result = prepare(args)
    elif args.command == "sign":
        result = sign(args)
    else:
        result = finalize(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
