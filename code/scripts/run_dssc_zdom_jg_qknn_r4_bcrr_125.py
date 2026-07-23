#!/usr/bin/env python3
"""Authority-backed full125 runner for DSSC-ZDOM-JG-qKNN-R4-BCRR/r1f.

The controller intentionally never accepts a caller-made support/query NPZ.
Each row is constructed through the existing p2_min_v1 cache+authority
pipeline, all five arms are sealed before the scorer opens its sidecar, and
the matrix queue invokes this same row command without a shell template.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
from cvsrffi.somph_offline_target_package import build_somph_offline_row_pair, finalize_somph_apply_package
from cvsrffi.somph_predictor_entry import run_somph_enrollment
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.somph_runtime_request import SOMPH_ENROLLMENT_REQUEST_SCHEMA
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair
from cvsrffi.stage2_diag_cosine_scorer import _read_prediction, _read_truth, _score_state
from cvsrffi.stage2_predictor_bundle import sha256_file
from cvsrffi.stage2_dssc_zdom_jg_qknn_r4_bcrr import (
    ARMS, CANDIDATE, SCENES, DSSCStateError, GEOFF_R8_COVERAGE_SHA256,
    PHASE1_ARCHIVE_MANIFEST_SHA256, PHASE1_ARCHIVE_SHA256,
    PHASE1_CHECKPOINT_SHA256, PHASE1_PARITY_RECEIPT_SHA256,
    SEALED_RUNTIME_SHA256,
    SOMPH_PACKAGE_LOCK_SHA256, adapt_support_only, attach_rank4_adapter,
    build_five_arm_states, bundle_wire_bytes, load_ground_bundle,
    canonical_method_lock,
    predict_five_arms, qknn_lock_from_method_lock, qknn_neighbor_receipt,
    resource_receipt, typed_tokens, validate_method_lock,
)

CHECKPOINT_SHA256 = PHASE1_CHECKPOINT_SHA256
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
QUERY_PER_TX = 20
LAUNCHER_RECEIPT_SCHEMA = (
    "cvs.dssc.full125.launcher_receipt.cuda_namespace.v1"
)
ROW_DEVICE_NAMESPACE_EXECUTION_SCHEMA = (
    "cvs.dssc.full125.row_device_namespace.execution.v1"
)
ROW_LOGICAL_DEVICE = "cuda:0"


class DSSCLauncherError(ValueError):
    pass


def _job_id(receiver: str, seed: int, k_shot: int, new_class_count: int) -> str:
    return (
        f"dssc_r1f_rx_{receiver}_s_{seed}_k_{k_shot}_n_{new_class_count}"
    )


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_bytes_new(path: Path, payload: bytes, *, readonly: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    if readonly:
        os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(payload).hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    return _write_bytes_new(path, _canon(value) + b"\n")


def _write_npz_new(path: Path, **arrays: np.ndarray) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return _write_bytes_new(path, stream.getvalue())


def _prediction_artifact_path(row_root: Path, state: str, arm: str) -> Path:
    if state not in {"before", "after"} or arm not in ARMS:
        raise DSSCLauncherError("prediction artifact identity drift")
    state_root = row_root / "predictions" / state
    return (
        state_root / "prediction_artifact.npz"
        if arm == "M0"
        else state_root / "arms" / f"{arm}.npz"
    )


def _score_artifact_paths(row_root: Path) -> dict[str, Path]:
    scorer = row_root / "scorer"
    result = {
        f"{arm}.base_score": scorer / f"{arm}.base_score.json" for arm in ARMS
    }
    result.update({f"{arm}.score": scorer / f"{arm}.score.json" for arm in ARMS})
    result["same_row_summary"] = scorer / "same_row_summary.json"
    return result


def _safe_receiver(receiver: str) -> str:
    return receiver.replace("-", "_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_coverage_receipt(path: str | Path) -> str:
    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise DSSCLauncherError("GEOFF/r8 coverage receipt must be a regular file")
    observed = sha256_file(source)
    if observed != GEOFF_R8_COVERAGE_SHA256:
        raise DSSCLauncherError("GEOFF/r8 coverage receipt SHA drift")
    return observed


def _exact_adv3b02(checkpoint_path: str | Path, *, device: str):
    """Reconstruct the state-dict checkpoint; no identity/TorchScript substitute."""
    import torch
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise DSSCLauncherError("r1f rejects a checkpoint SHA outside its frozen contract")
    from model_dual_cvsincnet import build_dual_model
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if type(checkpoint) is not dict or not isinstance(checkpoint.get("model"), dict):
        raise DSSCLauncherError("Phase1 checkpoint model state is missing")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint["model"].items()}
    args = dict(checkpoint.get("args") or {})
    domain = next((state[k] for k in ("dom_head.net.3.bias", "dom_head.net.3.weight", "adv_head.net.3.bias", "adv_head.net.3.weight") if k in state), None)
    if domain is None:
        raise DSSCLauncherError("cannot infer the exact dual-domain head")
    model = build_dual_model(int(args["num_classes"]), int(domain.shape[0]), model_size=str(args.get("model_size", "M")), dataset=str(args.get("dataset", "wisig")), input_len=256, sample_rate_hz=float(args.get("sample_rate_hz", 0.0)) or 25e6, id_feature_key=str(args.get("id_feature_key", "feat_joint")), dom_feature_key=str(args.get("dom_feature_key", "feat_imp")), model_variant=str(args.get("model_variant", "lite_c")), branch_ablation=str(args.get("branch_ablation", "none")), mixstyle_on=bool(args.get("use_mixstyle", False)), mixstyle_p=float(args.get("mixstyle_p", .3)), mixstyle_alpha=float(args.get("mixstyle_alpha", .1)), mixstyle_eps=float(args.get("mixstyle_eps", 1e-6)), mixstyle_layers=str(args.get("mixstyle_layers", "time_down,t1")), mixstyle_use_domain_label=bool(args.get("mixstyle_use_domain_label", True)), mixstyle_mix=str(args.get("mixstyle_mix", "crossdomain")), mixstyle_strength=float(args.get("mixstyle_strength", 1.0)), mixstyle_fallback=str(args.get("mixstyle_fallback", "random")), domain_branch_ablation=str(args.get("domain_branch_ablation", "same")), domain_enhancer=str(args.get("domain_enhancer", "rcn_stats")), domain_enhancer_strength=float(args.get("domain_enhancer_strength", .35)), fast_infer_when_no_aux=bool(args.get("fast_infer_when_no_aux", True)), arch_family=str(args.get("arch_family", "cvsincnet"))).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise DSSCLauncherError(f"strict ADV3B02 reconstruction failed: missing={list(missing)} unexpected={list(unexpected)}")
    return model.eval(), {"checkpoint_sha256": CHECKPOINT_SHA256, "missing_keys": 0, "unexpected_keys": 0, "inference_runtime": "exact_state_dict_rebuild"}


def _activate_row_device(device: str) -> dict[str, Any]:
    """Bind implicit CUDA allocations before any sealed runtime is opened."""
    import torch

    resolved = torch.device(device)
    device_count = int(torch.cuda.device_count())
    if (
        resolved.type != "cuda"
        or resolved.index is None
        or not torch.cuda.is_available()
        or resolved.index < 0
        or resolved.index >= device_count
    ):
        raise DSSCLauncherError("formal row requires an available indexed CUDA device")
    torch.cuda.set_device(resolved)
    current_device = int(torch.cuda.current_device())
    if current_device != resolved.index:
        raise DSSCLauncherError("formal row CUDA current-device binding failed")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_physical_gpu_id = (
        int(visible)
        if (
            type(visible) is str
            and visible.isdecimal()
            and str(int(visible)) == visible
        )
        else None
    )
    return {
        "schema": ROW_DEVICE_NAMESPACE_EXECUTION_SCHEMA,
        "cuda_visible_devices": visible,
        "visible_physical_gpu_id": visible_physical_gpu_id,
        "requested_logical_device": str(resolved),
        "torch_cuda_device_count": device_count,
        "torch_cuda_current_device": current_device,
    }


def _numpy_float32_tensor(values: Any, *, device: str):
    """Cross the PyTorch-2.1/NumPy-2 boundary without the ndarray C API."""
    import torch

    source = np.array(values, dtype=np.float32, order="C", copy=True)
    return (
        torch.frombuffer(source, dtype=torch.float32)
        .reshape(source.shape)
        .clone()
        .to(torch.device(device))
    )


def _tensor_float32_numpy(value: Any) -> np.ndarray:
    """Return FP32 NumPy through the list boundary used by the sealed runtime."""
    return np.asarray(value.detach().cpu().tolist(), dtype=np.float32)


def _feature(model: Any, iq: np.ndarray, *, device: str) -> np.ndarray:
    import torch
    value = _numpy_float32_tensor(iq, device=device)
    with torch.no_grad():
        result = model(value, return_aux=True)
    if not isinstance(result, Mapping) or not torch.is_tensor(result.get("z_id")):
        raise DSSCLauncherError("exact dual model did not expose z_id")
    return _tensor_float32_numpy(
        torch.nn.functional.normalize(result["z_id"], dim=1)
    )


def _id_feature(model: Any, iq: np.ndarray, *, device: str) -> np.ndarray:
    """Query path: identity backbone only; domain backbone is unreachable."""
    import torch
    from model_dual_cvsincnet import backbone_forward_compat
    value = _numpy_float32_tensor(iq, device=device)
    with torch.no_grad():
        aux = backbone_forward_compat(model.id_backbone, value, y=None, return_aux=True, domain_labels=None)
    key = str(model.id_feature_key)
    if not isinstance(aux, Mapping) or not torch.is_tensor(aux.get(key)):
        raise DSSCLauncherError("identity-only query path did not expose feat_joint")
    return _tensor_float32_numpy(
        torch.nn.functional.normalize(aux[key], dim=1)
    )


def _timed_id_feature(
    model: Any, iq: np.ndarray, *, device: str, batch_size: int = 64
) -> tuple[np.ndarray, list[float]]:
    """Return identity features plus measured per-row batch-normalized latency."""
    import torch
    source = np.asarray(iq, np.float32)
    parts: list[np.ndarray] = []
    normalized_ms: list[float] = []
    for start in range(0, len(source), batch_size):
        batch = source[start : start + batch_size]
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        tick = time.perf_counter()
        parts.append(_id_feature(model, batch, device=device))
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        per_row = 1000.0 * (time.perf_counter() - tick) / len(batch)
        normalized_ms.extend([per_row] * len(batch))
    if not parts or len(normalized_ms) != len(source):
        raise DSSCLauncherError("identity query timing row closure drift")
    return np.concatenate(parts), normalized_ms


def _feature_drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, np.float32)
    right = np.asarray(candidate, np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise DSSCLauncherError("feature drift requires matched two-dimensional rows")
    delta = right - left
    row_l2 = np.linalg.norm(delta.astype(np.float64), axis=1)
    return {
        "row_count": int(len(left)),
        "changed_row_count": int(np.count_nonzero(row_l2 > 0.0)),
        "mean_row_l2": float(np.mean(row_l2)),
        "max_row_l2": float(np.max(row_l2)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _neighbor_order_change(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        reference.get("classes") != candidate.get("classes")
        or reference.get("query_count") != candidate.get("query_count")
    ):
        raise DSSCLauncherError("neighbor receipts are not row-compatible")
    left = reference.get("orders")
    right = candidate.get("orders")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        raise DSSCLauncherError("neighbor receipt order schema drift")
    per_query = [
        any(lclass != rclass for lclass, rclass in zip(lrow, rrow))
        for lrow, rrow in zip(left, right)
    ]
    class_order_count = sum(
        int(lclass != rclass)
        for lrow, rrow in zip(left, right)
        for lclass, rclass in zip(lrow, rrow)
    )
    return {
        "query_count": len(per_query),
        "changed_query_count": int(sum(per_query)),
        "changed_query_rate": float(np.mean(per_query)),
        "changed_query_class_order_count": int(class_order_count),
    }


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise DSSCLauncherError("--gpu-ids must be a comma-separated integer list") from exc
    if not parsed or len(set(parsed)) != len(parsed) or any(item < 0 for item in parsed):
        raise DSSCLauncherError("--gpu-ids must be a nonempty unique nonnegative list")
    return parsed


def _deployed_s1_snapshot(*, source_adapter: Any, bundle: Any, ground_enabled: bool, a: argparse.Namespace) -> tuple[Any, Any]:
    """Quantized inference clone of S_B while the original adapter stays trainable for S_C."""
    import torch
    model, _ = _exact_adv3b02(a.phase1_checkpoint, device=a.device)
    deployed = attach_rank4_adapter(model, bundle if ground_enabled else None, ground_enabled=ground_enabled)
    if (
        source_adapter.coefficient_codes is None
        or source_adapter.coefficient_scale_fp16 is None
    ):
        raise DSSCLauncherError("S_B adapter lacks its audited INT8 state")
    deployed.load_quantized(
        source_adapter.coefficient_codes,
        source_adapter.coefficient_scale_fp16,
    )
    deployed.merge()
    if (
        not np.array_equal(
            deployed.coefficient_codes, source_adapter.coefficient_codes
        )
        or not np.array_equal(
            deployed.coefficient_scale_fp16,
            source_adapter.coefficient_scale_fp16,
        )
    ):
        raise DSSCLauncherError("S_B deployment clone changed INT8 codes/scale")
    return model, deployed


def _enrollment_request(package_seal_sha256: str, device: str) -> dict[str, Any]:
    return {"schema": SOMPH_ENROLLMENT_REQUEST_SCHEMA, "package_seal_sha256": package_seal_sha256, "head_output_leaf": "head_capsule.npz", "device": device, "support_batch_size": 64, **PHASE2_FULL_CONTRACT}


def _build_finalized_packages(a: argparse.Namespace, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the existing authority builder then finalize both apply packages."""
    offline, control, enroll, seals = (output / name for name in ("offline", "control", "enrollment", "apply_seals"))
    for root in (control, enroll, seals):
        root.mkdir(parents=True, exist_ok=False)
    build = build_somph_offline_row_pair(cache_set_manifest_path=a.cache_manifest, authority_bundle_root=a.authority_bundle, expected_authority_commit_sha256=a.authority_commit_sha256, phase1_checkpoint_path=a.phase1_checkpoint, sealed_feature_runtime_path=a.sealed_runtime, method_lock_path=a.package_method_lock, output_root=offline, receiver=a.receiver, seed=a.seed, k_shot=a.k_shot, new_class_count=a.new_class_count, query_per_tx=QUERY_PER_TX)
    state_runtime: dict[str, Any] = {}
    for state in ("before", "after"):
        item = build["states"][state]
        request = control / f"{state}.enrollment_request.json"
        _write_json_new(request, _enrollment_request(item["enrollment_package_seal_sha256"], a.device))
        head_root = enroll / state
        head_root.mkdir(parents=True, exist_ok=False)
        enrolled = run_somph_enrollment(request_json=request, package_root=item["enrollment_package_root"], detached_seal_path=item["enrollment_package_seal"], expected_seal_sha256=item["enrollment_package_seal_sha256"], output_root=head_root)
        head = head_root / enrolled["head_output_leaf"]
        seal = seals / f"{state}.apply.seal.json"
        applied = finalize_somph_apply_package(apply_staging_root=item["apply_staging_root"], detached_seal_path=seal, staging_authority_path=item["apply_staging_authority"], staging_authority_seal_path=item["apply_staging_authority_seal"], expected_staging_authority_seal_sha256=item["apply_staging_authority_seal_sha256"], head_capsule_path=head, expected_head_capsule_sha256=enrolled["head_capsule_sha256"], expected_head_enrollment_binding_sha256=enrolled["enrollment_binding_sha256"], authority_bundle_root=a.authority_bundle, expected_authority_commit_sha256=a.authority_commit_sha256)
        state_runtime[state] = {"enrollment": item, "enrolled": enrolled, "apply": applied, "apply_seal": seal}
    return build, state_runtime


def _registry(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(item["class_handle"] for item in manifest["registered_classes"])
    return typed_tokens(values, name="sealed opaque registry", unique=True)


def _support(payload: Mapping[str, np.ndarray], registry: tuple[str, ...], k: int) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    ranks = np.asarray(payload["support_rank_within_class"], np.int64)
    index = np.asarray(payload["support_class_indices"], np.int64)
    tokens = typed_tokens(np.asarray(payload["support_tokens"]), name="sealed support tokens", unique=True)
    iq = np.asarray(payload["support_leo_weak_iq"], np.float32)
    keep = ranks < k
    if ranks.shape != index.shape or len(iq) != len(tokens) or int(keep.sum()) != len(registry) * k or tuple(index[keep].tolist()) != tuple(i for i in range(len(registry)) for _ in range(k)):
        raise DSSCLauncherError("sealed support rank/class order is not exact K-shot registry order")
    return iq[keep], tuple(registry[i] for i in index[keep]), tuple(tokens[i] for i in np.flatnonzero(keep))


def _query(payload: Mapping[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(payload["query_leo_weak_iq"], np.float32)
    tokens = typed_tokens(np.asarray(payload["query_tokens"]), name="sealed query tokens", unique=True)
    if values.ndim < 2 or len(values) != len(tokens):
        raise DSSCLauncherError("sealed query IQ/token layout drift")
    return values, tokens


def _stage_predictions(*, state: str, scenario: str, enrollment_payload: Mapping[str, np.ndarray], query_payload: Mapping[str, np.ndarray], registry: tuple[str, ...], old_registry: tuple[str, ...], bundle: Any, qknn_lock: Any, a: argparse.Namespace) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    import torch
    started = time.perf_counter()
    if str(a.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(a.device))
    support_iq, labels, support_tokens = _support(enrollment_payload, registry, a.k_shot)
    query_iq, query_tokens = _query(query_payload)
    cache = getattr(a, "_dssc_scene_cache", {}).get(scenario) if state == "after" else None
    if state == "after" and cache is None:
        raise DSSCLauncherError("after state requires the same-scene unmerged S_B continuation")
    # The raw branch is stateless and is reconstructed for each package.  The
    # only before->after runtime continuation is the same-scene unmerged S_B
    # adapted model/adapter pair below.
    raw_model, audit = _exact_adv3b02(a.phase1_checkpoint, device=a.device)
    # Before and after packages are physically distinct.  Reuse the frozen
    # model only; always re-extract each state's own support and query rows.
    raw_support = _id_feature(raw_model, support_iq, device=a.device)
    raw_query, raw_query_ms = _timed_id_feature(
        raw_model, query_iq, device=a.device
    )
    if state == "before":
        old_indices = np.asarray([label in set(old_registry) for label in labels], bool)
        if int(old_indices.sum()) != len(old_registry) * a.k_shot:
            raise DSSCLauncherError("before state does not contain exact old support")
        fit_iq, fit_labels, fit_tokens, fit_registry = support_iq[old_indices], tuple(label for label, keep in zip(labels, old_indices) if keep), tuple(token for token, keep in zip(support_tokens, old_indices) if keep), old_registry
        # Before registration evaluates exactly S_B; all arms use the old registry.
        raw_support, raw_query = raw_support[old_indices], raw_query
    else:
        fit_iq, fit_labels, fit_tokens, fit_registry = support_iq, labels, support_tokens, registry
    ng_model = cache["ng_model"] if cache is not None else _exact_adv3b02(a.phase1_checkpoint, device=a.device)[0]
    g_model = cache["g_model"] if cache is not None else _exact_adv3b02(a.phase1_checkpoint, device=a.device)[0]
    adapt_started = time.perf_counter()
    if state == "after":
        old_indices = np.asarray([label in set(old_registry) for label in labels], bool)
        if int(old_indices.sum()) != len(old_registry) * a.k_shot:
            raise DSSCLauncherError("after state lacks the exact old support prefix")
        ng_b = cache["ng_adapter"]
        _ng, ng_receipt = adapt_support_only(ng_model, _numpy_float32_tensor(support_iq, device=a.device), labels, registry, k_shot=a.k_shot, stage="S_C", bundle=None, ground_enabled=False, support_physical_ids=support_tokens, continue_adapter=ng_b, merge=True)
        g_b = cache["g_adapter"]
        g_adapter, g_receipt = adapt_support_only(g_model, _numpy_float32_tensor(support_iq, device=a.device), labels, registry, k_shot=a.k_shot, stage="S_C", bundle=bundle, ground_enabled=True, ground_old_registry=old_registry, support_physical_ids=support_tokens, continue_adapter=g_b, merge=True)
    else:
        ng_adapter, ng_receipt = adapt_support_only(ng_model, _numpy_float32_tensor(fit_iq, device=a.device), fit_labels, fit_registry, k_shot=a.k_shot, stage="S_B", bundle=None, ground_enabled=False, support_physical_ids=fit_tokens, merge=False)
        g_adapter, g_receipt = adapt_support_only(g_model, _numpy_float32_tensor(fit_iq, device=a.device), fit_labels, fit_registry, k_shot=a.k_shot, stage="S_B", bundle=bundle, ground_enabled=True, ground_old_registry=old_registry, support_physical_ids=fit_tokens, merge=False)
    adaptation_wall_ms = 1000.0 * (time.perf_counter() - adapt_started)
    if state == "before":
        ng_deployed_model, ng_deployed_adapter = _deployed_s1_snapshot(source_adapter=ng_adapter, bundle=None, ground_enabled=False, a=a)
        g_deployed_model, g_deployed_adapter = _deployed_s1_snapshot(source_adapter=g_adapter, bundle=bundle, ground_enabled=True, a=a)
    feature_ng = ng_model if state == "after" else ng_deployed_model
    feature_g = g_model if state == "after" else g_deployed_model
    inference_adapter = g_adapter if state == "after" else g_deployed_adapter
    ng_support = _id_feature(feature_ng, fit_iq, device=a.device)
    ng_query, ng_query_ms = _timed_id_feature(
        feature_ng, query_iq, device=a.device
    )
    ground_support = _id_feature(feature_g, fit_iq, device=a.device)
    ground_query, ground_query_ms = _timed_id_feature(
        feature_g, query_iq, device=a.device
    )
    states = build_five_arm_states(raw_support_features=raw_support, ng_support_features=ng_support, ground_support_features=ground_support, support_labels=fit_labels, registered_classes=fit_registry, support_physical_ids=fit_tokens, k_shot=a.k_shot, qknn_lock=qknn_lock)
    prediction_started = time.perf_counter()
    logits = predict_five_arms(states, raw_query_features=raw_query, ng_query_features=ng_query, ground_query_features=ground_query)
    prediction_wall_ms = 1000.0 * (time.perf_counter() - prediction_started)
    head_ms_per_query = prediction_wall_ms / max(1, len(query_tokens))
    end_to_end_ms = (
        np.asarray(raw_query_ms, np.float64)
        + np.asarray(ng_query_ms, np.float64)
        + np.asarray(ground_query_ms, np.float64)
        + head_ms_per_query
    )
    peak_vram = int(torch.cuda.max_memory_allocated(torch.device(a.device))) if str(a.device).startswith("cuda") and torch.cuda.is_available() else 0
    rows = [{"arm": arm, "tokens": np.asarray(query_tokens), "scenarios": np.asarray([scenario] * len(query_tokens)), "predicted": np.asarray(fit_registry)[np.argmax(logits[arm], axis=1)]} for arm in ARMS]
    if state == "before":
        # Preserve the exact unmerged S_B adapters that generated before
        # predictions; S_C below continues these objects rather than refitting.
        a._dssc_scene_cache[scenario] = {
            "ng_model": ng_model,
            "g_model": g_model,
            "ng_adapter": ng_adapter,
            "g_adapter": g_adapter,
            "ng_s_b_receipt": ng_receipt,
            "g_s_b_receipt": g_receipt,
            "s_b_adaptation_wall_ms": adaptation_wall_ms,
            "s_b_support_rows": len(fit_iq),
        }
    s_b_ng = ng_receipt if state == "before" else cache["ng_s_b_receipt"]
    s_b_g = g_receipt if state == "before" else cache["g_s_b_receipt"]
    s_c_ng = None if state == "before" else ng_receipt
    s_c_g = None if state == "before" else g_receipt
    if int(s_b_ng["steps"]) != int(s_b_g["steps"]) or (
        s_c_ng is not None and int(s_c_ng["steps"]) != int(s_c_g["steps"])
    ):
        raise DSSCLauncherError("ground/no-ground optimizer-step closure drift")
    s_b_rows = len(fit_iq) if state == "before" else int(cache["s_b_support_rows"])
    stage_training: dict[str, Any] = {
        "S_B": {
            "steps_per_adapter": int(s_b_g["steps"]),
            "optimizer_steps_all_adapters": 2 * int(s_b_g["steps"]),
            "support_rows": int(s_b_rows),
            "full_dual_batch_forward_calls_per_adapter": 2 * int(s_b_g["steps"]) + 3,
            "full_dual_sample_forwards_all_adapters": 2
            * (2 * int(s_b_g["steps"]) + 3)
            * int(s_b_rows),
            "adaptation_wall_ms": float(
                adaptation_wall_ms
                if state == "before"
                else cache["s_b_adaptation_wall_ms"]
            ),
        }
    }
    if state == "after":
        stage_training["S_C"] = {
            "steps_per_adapter": int(s_c_g["steps"]),
            "optimizer_steps_all_adapters": 2 * int(s_c_g["steps"]),
            "support_rows": int(len(fit_iq)),
            "full_dual_batch_forward_calls_per_adapter": 2 * int(s_c_g["steps"]) + 3,
            "full_dual_sample_forwards_all_adapters": 2
            * (2 * int(s_c_g["steps"]) + 3)
            * int(len(fit_iq)),
            "adaptation_wall_ms": float(adaptation_wall_ms),
        }
    resource = dict(resource_receipt(bundle=bundle, qknn=states["M_DA"], bcrr=states["M_JOINT"][1], adapter=inference_adapter, train_receipt=g_receipt))
    profile = canonical_method_lock()["resource_profile"]
    id_mac = int(profile["id_backbone_feat_joint_mac_per_sample"])
    dual_mac = int(profile["full_dual_return_aux_mac_per_sample"])
    qknn_mac_per_query = int(len(fit_labels) * 160)
    bcrr_mac_per_query = int(len(fit_registry) * 160)
    five_arm_head_mac_per_query = 3 * qknn_mac_per_query + 2 * bcrr_mac_per_query
    stage_sample_forwards = sum(
        int(item["full_dual_sample_forwards_all_adapters"])
        for item in stage_training.values()
    )
    resource.update(
        {
            "build_ms": 1000.0 * (prediction_started - started),
            "adaptation": {
                "by_stage": stage_training,
                "optimizer_steps_S_B_plus_S_C_all_adapters": sum(
                    int(item["optimizer_steps_all_adapters"])
                    for item in stage_training.values()
                ),
                "wall_ms_S_B_plus_S_C": sum(
                    float(item["adaptation_wall_ms"])
                    for item in stage_training.values()
                ),
            },
            "latency_scope": "three_measured_identity_paths_plus_five_arm_head;batch_normalized_per_query",
            "predict_mean_ms_per_query": float(np.mean(end_to_end_ms)),
            "predict_p95_ms_per_query": float(np.percentile(end_to_end_ms, 95)),
            "identity_path_mean_ms_per_query": {
                "raw": float(np.mean(raw_query_ms)),
                "no_ground": float(np.mean(ng_query_ms)),
                "ground": float(np.mean(ground_query_ms)),
            },
            "five_arm_head_wall_ms": float(prediction_wall_ms),
            "five_arm_head_mean_ms_per_query": float(head_ms_per_query),
            "peak_vram_bytes": peak_vram,
            "forward_counts": {
                "identity_query_batch_calls_all_three_paths": int(
                    3 * ((len(query_tokens) + 63) // 64)
                ),
                "identity_query_sample_forwards_all_three_paths": int(
                    3 * len(query_tokens)
                ),
                "identity_support_sample_forwards_state_build": int(
                    3 * len(fit_labels)
                ),
                "full_dual_support_sample_forwards_S_B_plus_S_C_all_adapters": int(
                    stage_sample_forwards
                ),
            },
            "mac_scope": profile["scope"],
            "backbone_mac": {
                "input_shape": profile["input_shape"],
                "checkpoint_sha256": profile["checkpoint_sha256"],
                "id_backbone_per_sample": id_mac,
                "full_dual_return_aux_per_sample": dual_mac,
                "five_arm_query_three_identity_paths_per_query": 3 * id_mac,
                "full_dual_support_training_total": dual_mac
                * stage_sample_forwards,
            },
            "head_mac": {
                "scope": "actual_matmul_MAC_only;three_distinct_qKNN_evaluations_plus_two_BCR_evaluations",
                "qknn_per_geometry_per_query": qknn_mac_per_query,
                "bcrr_per_geometry_per_query": bcrr_mac_per_query,
                "five_arm_per_query": five_arm_head_mac_per_query,
                "five_arm_total": five_arm_head_mac_per_query
                * len(query_tokens),
                "production_M_JOINT_per_query": qknn_mac_per_query
                + bcrr_mac_per_query,
            },
            "query_mac": {
                "five_arm_eval_per_query": 3 * id_mac
                + five_arm_head_mac_per_query,
                "production_M_JOINT_per_query": id_mac
                + qknn_mac_per_query
                + bcrr_mac_per_query,
            },
            "adapter_int8_teacher_deployed": {
                "S_B_ground": s_b_g["adapter_int8_teacher_deployed"],
                "S_B_no_ground": s_b_ng["adapter_int8_teacher_deployed"],
                "S_C_ground": None
                if s_c_g is None
                else s_c_g["adapter_int8_teacher_deployed"],
                "S_C_no_ground": None
                if s_c_ng is None
                else s_c_ng["adapter_int8_teacher_deployed"],
            },
            "qknn_bcrr_internal_audits": {
                "raw": dict(states["M0"].branch_state.quantization_audit),
                "no_ground": dict(states["M_DA_NG"].branch_state.quantization_audit),
                "ground": dict(states["M_DA"].branch_state.quantization_audit),
                "M_OTHER_bcrr_fit": dict(states["M_OTHER"][1].receipt),
                "M_JOINT_bcrr_fit": dict(states["M_JOINT"][1].receipt),
            },
        }
    )
    raw_neighbor = qknn_neighbor_receipt(states["M0"], raw_query)
    ng_neighbor = qknn_neighbor_receipt(states["M_DA_NG"], ng_query)
    ground_neighbor = qknn_neighbor_receipt(states["M_DA"], ground_query)
    m0_argmax = np.argmax(logits["M0"], axis=1)
    receipt = {
        "checkpoint": audit,
        "state": state,
        "scenario": scenario,
        "support_rows_extracted_from_current_state": int(len(support_iq)),
        "query_rows_extracted_from_current_state": int(len(query_iq)),
        "fit_support_rows": int(len(fit_iq)),
        "S_B": {"ground": s_b_g, "no_ground": s_b_ng},
        "S_C": None if state == "before" else {"ground": s_c_g, "no_ground": s_c_ng},
        "ng_delta_nonzero": bool(ng_receipt["delta_norm"] > 0),
        "ground_delta_nonzero": bool(g_receipt["delta_norm"] > 0),
        "feature_drift": {
            "raw_to_no_ground": _feature_drift(raw_query, ng_query),
            "raw_to_ground": _feature_drift(raw_query, ground_query),
            "no_ground_to_ground": _feature_drift(ng_query, ground_query),
        },
        "neighbor_order": {
            "raw": raw_neighbor,
            "no_ground": ng_neighbor,
            "ground": ground_neighbor,
            "raw_to_no_ground": _neighbor_order_change(raw_neighbor, ng_neighbor),
            "raw_to_ground": _neighbor_order_change(raw_neighbor, ground_neighbor),
        },
        "argmax_change_vs_M0": {
            arm: {
                "changed_count": int(
                    np.count_nonzero(np.argmax(logits[arm], axis=1) != m0_argmax)
                ),
                "query_count": int(len(m0_argmax)),
            }
            for arm in ARMS
        },
        "resource": resource,
        "query_rows_used_for_fit": 0,
    }
    return rows, receipt


def adapt_steps(k: int, stage: str) -> int:
    return (2 if stage == "S_B" else 3) if k == 1 else 25


def _publish_predictions(root: Path, rows: Sequence[Mapping[str, Any]], *, state: str, receipt: Mapping[str, Any]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    arms = root / "arms"
    arms.mkdir()
    published: dict[str, str] = {}
    for arm in ARMS:
        chosen = [row for row in rows if row["arm"] == arm]
        arrays = {"query_tokens": np.concatenate([row["tokens"] for row in chosen]).astype(str), "scenarios": np.concatenate([row["scenarios"] for row in chosen]).astype(str), "predicted_class_handles": np.concatenate([row["predicted"] for row in chosen]).astype(str)}
        if len(set(arrays["query_tokens"].tolist())) != len(arrays["query_tokens"]):
            raise DSSCLauncherError("a prediction arm repeats an opaque query token")
        destination = root / "prediction_artifact.npz" if arm == "M0" else arms / f"{arm}.npz"
        published[arm] = _write_npz_new(destination, **arrays)
    receipt_path = root / "prediction_receipt.json"
    _write_json_new(receipt_path, {"candidate": CANDIDATE, "state": state, "arms": list(ARMS), "query_truth_present_in_predictor": False, "query_rows_used_for_fit": 0, "prediction_sha256_by_arm": published, "runtime": receipt})
    return published


def _enrich_score(path: Path, *, arm: str, score: Mapping[str, Any]) -> dict[str, Any]:
    """Truth-side enrichment after immutable predictions; never called by predictor code."""
    # The base scorer already seals all mandatory old/new, floor, forgetting,
    # scene and transmitter fields.  Preserve it unchanged; matrix aggregation
    # adds cross-arm I_syn from these same-row files.
    payload = {**score, "arm": arm, "balanced_accuracy_after": float(np.mean([item["accuracy"] for item in score["after"]["by_tx"].values()])), "min_new_after": min(item["accuracy"] for item in score["after"]["by_tx"].values() if item["role"] == "target_new"), "old_to_new_by_scene": {key: value["old_to_new_rate"] for key, value in score["after"]["by_scenario"].items()}, "new_to_old_by_scene": {key: value["new_to_old_rate"] for key, value in score["after"]["by_scenario"].items()}}
    _write_json_new(path, payload)
    return payload


def _scene_state(prediction: Path, truth: Mapping[str, Any], scene: str) -> Mapping[str, Any]:
    value = _read_prediction(prediction)
    mask = np.asarray(value["scenarios"]).astype(str) == scene
    return _score_state({"query_tokens": np.asarray(value["query_tokens"])[mask], "scenarios": np.asarray(value["scenarios"])[mask], "predicted_class_handles": np.asarray(value["predicted_class_handles"])[mask]}, truth)


def _scene_row(*, arm: str, scene: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old_before, old_after, seen_new = float(before["old_acc"]), float(after["old_acc"]), float(after["seen_new_acc"])
    old_by_tx = [item["accuracy"] for item in after["by_tx"].values() if item["role"] == "target_old"]
    new_by_tx = [item["accuracy"] for item in after["by_tx"].values() if item["role"] == "target_new"]
    all_by_tx = [item["accuracy"] for item in after["by_tx"].values()]
    forgetting = old_before - old_after
    return {"arm": arm, "scene": scene, "old_before": old_before, "old_after": old_after, "old_adaptation_gain": old_after - old_before, "seen_new": seen_new, "H": float(after["h_old_new"]), "BA": float(np.mean(all_by_tx)), "floor": float(min(all_by_tx)), "min_old": float(min(old_by_tx)), "min_new": float(min(new_by_tx)), "forgetting": forgetting, "forgetting_pp": 100.0 * forgetting, "old_to_new": float(after["by_scenario"][scene]["old_to_new_rate"]), "new_to_old": float(after["by_scenario"][scene]["new_to_old_rate"]), "per_class": after["by_tx"], "query_count": int(after["query_count"])}


def _truth_side_transitions(
    *,
    reference_path: Path,
    candidate_path: Path,
    truth: Mapping[str, Mapping[str, str]],
    scene: str,
) -> dict[str, Any]:
    reference = _read_prediction(reference_path)
    candidate = _read_prediction(candidate_path)
    ref_rows = {
        token: predicted
        for token, selected_scene, predicted in zip(
            reference["query_tokens"].tolist(),
            reference["scenarios"].tolist(),
            reference["predicted_class_handles"].tolist(),
        )
        if selected_scene == scene
    }
    candidate_rows = {
        token: predicted
        for token, selected_scene, predicted in zip(
            candidate["query_tokens"].tolist(),
            candidate["scenarios"].tolist(),
            candidate["predicted_class_handles"].tolist(),
        )
        if selected_scene == scene
    }
    if not ref_rows or set(ref_rows) != set(candidate_rows):
        raise DSSCLauncherError("truth-side transition token alignment drift")

    def summarize(role: str | None) -> dict[str, int]:
        tokens = [
            token
            for token in ref_rows
            if role is None or truth[token]["evaluation_role"] == role
        ]
        if not tokens:
            raise DSSCLauncherError("truth-side transition role is empty")
        reference_correct = np.asarray(
            [
                ref_rows[token] == truth[token]["true_class_handle"]
                for token in tokens
            ],
            bool,
        )
        candidate_correct = np.asarray(
            [
                candidate_rows[token] == truth[token]["true_class_handle"]
                for token in tokens
            ],
            bool,
        )
        wrong_to_correct = int(np.sum(~reference_correct & candidate_correct))
        correct_to_wrong = int(np.sum(reference_correct & ~candidate_correct))
        return {
            "query_count": len(tokens),
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct_decision_gain": wrong_to_correct - correct_to_wrong,
        }

    return {
        "all": summarize(None),
        "old": summarize("target_old"),
        "new": summarize("target_new"),
    }


def _score_arms(output: Path, build: Mapping[str, Any], publications: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    scorer = output / "scorer"
    scorer.mkdir(exist_ok=False)
    result: dict[str, Any] = {}
    truth = build["truth_sidecar"]
    for arm in ARMS:
        before = _prediction_artifact_path(output, "before", arm)
        after = _prediction_artifact_path(output, "after", arm)
        score = score_diag_cosine_pair(before_prediction_path=before, after_prediction_path=after, truth_sidecar_path=truth, output_path=scorer / f"{arm}.base_score.json", candidate=CANDIDATE)
        score["truth_sidecar_path"] = str(truth)
        result[arm] = _enrich_score(scorer / f"{arm}.score.json", arm=arm, score=score)
    h = lambda arm: float(result[arm]["after"]["h_old_new"])
    result["I_syn"] = h("M_JOINT") - h("M_DA") - h("M_OTHER") + h("M0")
    truth_map = _read_truth(truth)
    rows = []
    m0_after_path = output / "predictions" / "after" / "prediction_artifact.npz"
    for arm in ARMS:
        before_path = _prediction_artifact_path(output, "before", arm)
        after_path = _prediction_artifact_path(output, "after", arm)
        for scene in FORMAL_LEO_WEAK_SCENARIOS:
            row = _scene_row(arm=arm, scene=scene, before=_scene_state(before_path, truth_map, scene), after=_scene_state(after_path, truth_map, scene))
            row["vs_M0_decision_transitions"] = _truth_side_transitions(
                reference_path=m0_after_path,
                candidate_path=after_path,
                truth=truth_map,
                scene=scene,
            )
            rows.append(row)
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        by_arm = {row["arm"]: row for row in rows if row["scene"] == scene}
        syn = by_arm["M_JOINT"]["H"] - by_arm["M_DA"]["H"] - by_arm["M_OTHER"]["H"] + by_arm["M0"]["H"]
        for row in by_arm.values(): row["I_syn"] = float(syn)
    _write_json_new(scorer / "same_row_summary.json", {"candidate": CANDIDATE, "arms": result, "rows": rows, "score_rows": len(rows), "query_truth_joined_only_after_all_five_immutable_predictions": True})
    result["same_row_scene_metrics"] = rows
    return result


def run_row(a: argparse.Namespace) -> dict[str, Any]:
    if a.k_shot not in (1, 5, 10) or (a.k_shot, a.new_class_count) not in SLICES:
        raise DSSCLauncherError("row is outside frozen full125 slice set")
    device_namespace_execution = _activate_row_device(a.device)
    _validate_formal_row_device_namespace_execution(
        device_namespace_execution
    )
    output = Path(a.output_root)
    if output.exists():
        raise DSSCLauncherError("row output must be a fresh path")
    if (
        sha256_file(a.package_method_lock) != SOMPH_PACKAGE_LOCK_SHA256
        or sha256_file(a.sealed_runtime) != SEALED_RUNTIME_SHA256
        or sha256_file(a.phase1_checkpoint) != CHECKPOINT_SHA256
    ):
        raise DSSCLauncherError(
            "checkpoint/SOMPH package lock/sealed runtime SHA binding drift"
        )
    coverage_sha = _validate_coverage_receipt(a.coverage_receipt)
    bundle = load_ground_bundle(a.ground_bundle, checkpoint_sha256=CHECKPOINT_SHA256)
    dssc_lock_raw = Path(a.dssc_method_lock).read_text(encoding="utf-8")
    dssc_lock, canonical_lock_sha = validate_method_lock(dssc_lock_raw)
    lock_sha = sha256_file(a.dssc_method_lock)
    if lock_sha != canonical_lock_sha:
        raise DSSCLauncherError("DSSC lock file is not the canonical byte serialization")
    if (
        bundle.manifest.get("method_lock_sha256") != lock_sha
        or bundle.manifest.get("method_lock") != dssc_lock
    ):
        raise DSSCLauncherError("ground bundle/DSSC exact lock binding drift")
    qknn_lock = qknn_lock_from_method_lock(dssc_lock, k_shot=a.k_shot)
    output.mkdir(parents=True)
    build, runtime = _build_finalized_packages(a, output)
    build_old_labels = typed_tokens(
        build.get("old_tx_labels"), name="authority old_tx_labels", unique=True
    )
    if build_old_labels != bundle.classes:
        raise DSSCLauncherError(
            "authority old_tx_labels order differs from the Phase1 ground bundle"
        )
    be, bm, _ = load_verified_somph_predictor_bundle(runtime["before"]["enrollment"]["enrollment_package_root"], detached_seal_path=runtime["before"]["enrollment"]["enrollment_package_seal"], expected_seal_sha256=runtime["before"]["enrollment"]["enrollment_package_seal_sha256"])
    bq, ba, _ = load_verified_somph_predictor_bundle(runtime["before"]["enrollment"]["apply_staging_root"], detached_seal_path=runtime["before"]["apply_seal"], expected_seal_sha256=runtime["before"]["apply"]["package_seal_sha256"])
    ae, am, _ = load_verified_somph_predictor_bundle(runtime["after"]["enrollment"]["enrollment_package_root"], detached_seal_path=runtime["after"]["enrollment"]["enrollment_package_seal"], expected_seal_sha256=runtime["after"]["enrollment"]["enrollment_package_seal_sha256"])
    aq, aa, _ = load_verified_somph_predictor_bundle(runtime["after"]["enrollment"]["apply_staging_root"], detached_seal_path=runtime["after"]["apply_seal"], expected_seal_sha256=runtime["after"]["apply"]["package_seal_sha256"])
    old_registry, registry = _registry(bm), _registry(am)
    if _registry(ba) != old_registry or _registry(aa) != registry or registry[:len(old_registry)] != old_registry:
        raise DSSCLauncherError("sealed before/after opaque registry prefix closure drift")
    locked_k = int(bm["k_shot"])
    if locked_k != a.k_shot or len(old_registry) != len(bundle.classes):
        raise DSSCLauncherError("sealed old registry cannot bind the Phase1 ground prototype slots")
    all_rows: dict[str, list[dict[str, Any]]] = {"before": [], "after": []}
    receipts: list[Mapping[str, Any]] = []
    a._dssc_scene_cache = {}
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        for state, ep, qp, reg in (("before", be[scene], bq[scene], old_registry), ("after", ae[scene], aq[scene], registry)):
            rows, receipt = _stage_predictions(state=state, scenario=scene, enrollment_payload=ep, query_payload=qp, registry=tuple(reg), old_registry=tuple(old_registry), bundle=bundle, qknn_lock=qknn_lock, a=a)
            all_rows[state].extend(rows); receipts.append(receipt)
    prediction_root = output / "predictions"
    prediction_root.mkdir(exist_ok=False)
    publications = {state: _publish_predictions(prediction_root / state, all_rows[state], state=state, receipt={"by_scene": [x for x in receipts if x["state"] == state]}) for state in ("before", "after")}
    scores = _score_arms(output, build, publications)
    prediction_receipts = {
        state: sha256_file(prediction_root / state / "prediction_receipt.json")
        for state in ("before", "after")
    }
    score_artifacts = {
        name: sha256_file(path) for name, path in _score_artifact_paths(output).items()
    }
    receipt = {
        "candidate": CANDIDATE,
        "job_id": _job_id(a.receiver, a.seed, a.k_shot, a.new_class_count),
        "status": "ROW_ARTIFACTS_COMPLETE",
        "receiver": a.receiver,
        "seed": a.seed,
        "k_shot": a.k_shot,
        "new_class_count": a.new_class_count,
        "prediction_slice_count": 3,
        "score_rows": 15,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "phase1_archive_sha256": PHASE1_ARCHIVE_SHA256,
        "phase1_archive_manifest_sha256": PHASE1_ARCHIVE_MANIFEST_SHA256,
        "phase1_parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256,
        "cache_manifest_sha256": sha256_file(a.cache_manifest),
        "authority_commit_sha256": a.authority_commit_sha256,
        "ground_bundle_sha256": sha256_file(a.ground_bundle),
        "dssc_method_lock_sha256": lock_sha,
        "somph_package_lock_sha256": SOMPH_PACKAGE_LOCK_SHA256,
        "sealed_runtime_sha256": SEALED_RUNTIME_SHA256,
        "geoff_r8_coverage_sha256": coverage_sha,
        "qknn_lock_digest": qknn_lock.lock_digest,
        "query_truth_in_predictor": False,
        "query_rows_used_for_fit": 0,
        "device_namespace_execution": device_namespace_execution,
        "full_metrics": scores,
        "prediction_sha256_by_state_arm": publications,
        "prediction_receipt_sha256_by_state": prediction_receipts,
        "score_artifact_sha256": score_artifacts,
    }
    _write_json_new(output / "row_receipt.json", receipt)
    return receipt


def _schedule_cost(
    *, old_class_count: int, k_shot: int, new_class_count: int
) -> dict[str, int]:
    if type(old_class_count) is not int or old_class_count <= 0:
        raise DSSCLauncherError("old-class count for schedule cost must be positive")
    s_b_rows = old_class_count * k_shot
    s_c_rows = (old_class_count + new_class_count) * k_shot
    optimizer_sample_steps = 2 * (
        adapt_steps(k_shot, "S_B") * s_b_rows
        + adapt_steps(k_shot, "S_C") * s_c_rows
    )
    query_rows = (
        len(FORMAL_LEO_WEAK_SCENARIOS)
        * QUERY_PER_TX
        * (old_class_count + new_class_count)
    )
    return {
        "optimizer_step_x_support_rows_two_adapters": int(
            optimizer_sample_steps
        ),
        "query_rows_tiebreak": int(query_rows),
    }


def matrix_jobs(
    *,
    cache_root: str | Path,
    authority_root: str | Path,
    run_root: str | Path,
    old_class_count: int = 6,
) -> list[dict[str, Any]]:
    cache, authority, root = Path(cache_root), Path(authority_root), Path(run_root)
    jobs: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        leaf = f"rx_{_safe_receiver(receiver)}"
        for seed in SEEDS:
            cache_manifest = cache / leaf / f"seed_{seed}" / "cache_set.json"
            bundle = authority / f"authority_bundle_{leaf}_seed_{seed}"
            commit = bundle / "COMMIT.json"
            for k, new in SLICES:
                job_id = _job_id(receiver, seed, k, new)
                jobs.append(
                    {
                        "job_id": job_id,
                        "receiver": receiver,
                        "seed": seed,
                        "k_shot": k,
                        "new_class_count": new,
                        "old_class_count": old_class_count,
                        "cache_manifest": str(cache_manifest),
                        "cache_manifest_sha256": sha256_file(cache_manifest),
                        "authority_bundle": str(bundle),
                        "authority_commit_sha256": sha256_file(commit),
                        "output_root": str(root / "jobs" / job_id),
                        "schedule_cost": _schedule_cost(
                            old_class_count=old_class_count,
                            k_shot=k,
                            new_class_count=new,
                        ),
                    }
                )
    if len(jobs) != 125:
        raise DSSCLauncherError("frozen full125 job cardinality drift")
    return sorted(
        jobs,
        key=lambda job: (
            -int(
                job["schedule_cost"][
                    "optimizer_step_x_support_rows_two_adapters"
                ]
            ),
            -int(job["schedule_cost"]["query_rows_tiebreak"]),
            str(job["job_id"]),
        ),
    )


def _validate_completed_row_receipt(
    job: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_hashes: Mapping[str, str],
    launcher_device_namespace: Mapping[str, Any],
) -> None:
    expected = {
        "candidate": CANDIDATE,
        "job_id": job["job_id"],
        "status": "ROW_ARTIFACTS_COMPLETE",
        "receiver": job["receiver"],
        "seed": job["seed"],
        "k_shot": job["k_shot"],
        "new_class_count": job["new_class_count"],
        "prediction_slice_count": 3,
        "score_rows": 15,
        "checkpoint_sha256": expected_hashes["checkpoint_sha256"],
        "phase1_archive_sha256": expected_hashes["phase1_archive_sha256"],
        "phase1_archive_manifest_sha256": expected_hashes[
            "phase1_archive_manifest_sha256"
        ],
        "phase1_parity_receipt_sha256": expected_hashes[
            "phase1_parity_receipt_sha256"
        ],
        "cache_manifest_sha256": job["cache_manifest_sha256"],
        "authority_commit_sha256": job["authority_commit_sha256"],
        "ground_bundle_sha256": expected_hashes["ground_bundle_sha256"],
        "dssc_method_lock_sha256": expected_hashes[
            "dssc_method_lock_sha256"
        ],
        "somph_package_lock_sha256": expected_hashes[
            "somph_package_lock_sha256"
        ],
        "sealed_runtime_sha256": expected_hashes["sealed_runtime_sha256"],
        "geoff_r8_coverage_sha256": expected_hashes[
            "geoff_r8_coverage_sha256"
        ],
        "qknn_lock_digest": qknn_lock_from_method_lock(
            canonical_method_lock(), k_shot=int(job["k_shot"])
        ).lock_digest,
        "query_truth_in_predictor": False,
        "query_rows_used_for_fit": 0,
    }
    if type(receipt) is not dict or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise DSSCLauncherError("row receipt identity/hash closure drift")
    _validate_row_launcher_device_namespace_binding(
        receipt.get("device_namespace_execution"),
        launcher_device_namespace,
    )
    row_root = Path(str(job["output_root"]))
    if not row_root.is_dir() or row_root.is_symlink():
        raise DSSCLauncherError("completed row root is absent or not a regular directory")
    publications = receipt.get("prediction_sha256_by_state_arm")
    if (
        type(publications) is not dict
        or set(publications) != {"before", "after"}
        or any(
            type(publications[state]) is not dict
            or set(publications[state]) != set(ARMS)
            for state in ("before", "after")
        )
    ):
        raise DSSCLauncherError("row receipt prediction publication schema drift")
    old_class_count = job.get("old_class_count")
    if type(old_class_count) is not int or old_class_count <= 0:
        raise DSSCLauncherError("row old-class cardinality is absent")
    prediction_identity: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for state in ("before", "after"):
        expected_classes = old_class_count + (
            0 if state == "before" else int(job["new_class_count"])
        )
        expected_rows = expected_classes * QUERY_PER_TX * len(SCENES)
        for arm, digest in publications[state].items():
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise DSSCLauncherError("row receipt prediction SHA drift")
            path = _prediction_artifact_path(row_root, state, arm)
            if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
                raise DSSCLauncherError("row prediction artifact/hash closure drift")
            prediction = _read_prediction(path)
            tokens = tuple(prediction["query_tokens"].tolist())
            scenes = tuple(prediction["scenarios"].tolist())
            predicted = tuple(prediction["predicted_class_handles"].tolist())
            if (
                len(tokens) != expected_rows
                or len(set(tokens)) != expected_rows
                or any(not value for value in tokens)
                or any(not value for value in predicted)
                or any(scenes.count(scene) != expected_classes * QUERY_PER_TX for scene in SCENES)
            ):
                raise DSSCLauncherError("row prediction token/scene/cardinality closure drift")
            identity = (tokens, scenes)
            previous = prediction_identity.setdefault(state, identity)
            if previous != identity:
                raise DSSCLauncherError("row arm prediction identity/order drift")

    prediction_receipts = receipt.get("prediction_receipt_sha256_by_state")
    if type(prediction_receipts) is not dict or set(prediction_receipts) != {
        "before",
        "after",
    }:
        raise DSSCLauncherError("row prediction receipt publication schema drift")
    for state, digest in prediction_receipts.items():
        path = row_root / "predictions" / state / "prediction_receipt.json"
        if (
            type(digest) is not str
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise DSSCLauncherError("row prediction receipt artifact/hash drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("candidate") != CANDIDATE
            or payload.get("state") != state
            or payload.get("arms") != list(ARMS)
            or payload.get("query_truth_present_in_predictor") is not False
            or payload.get("query_rows_used_for_fit") != 0
            or payload.get("prediction_sha256_by_arm") != publications[state]
        ):
            raise DSSCLauncherError("row prediction receipt content drift")

    score_hashes = receipt.get("score_artifact_sha256")
    score_paths = _score_artifact_paths(row_root)
    if type(score_hashes) is not dict or set(score_hashes) != set(score_paths):
        raise DSSCLauncherError("row score artifact publication schema drift")
    for name, path in score_paths.items():
        digest = score_hashes[name]
        if (
            type(digest) is not str
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise DSSCLauncherError("row score artifact/hash closure drift")

    metrics = receipt.get("full_metrics")
    if type(metrics) is not dict or set(metrics) != set(ARMS) | {
        "I_syn",
        "same_row_scene_metrics",
    }:
        raise DSSCLauncherError("row full-metric schema drift")
    truth_sha: str | None = None
    for arm in ARMS:
        base_path = score_paths[f"{arm}.base_score"]
        score_path = score_paths[f"{arm}.score"]
        base = json.loads(base_path.read_text(encoding="utf-8"))
        enriched = json.loads(score_path.read_text(encoding="utf-8"))
        if enriched != metrics[arm]:
            raise DSSCLauncherError("row score/full-metric content drift")
        if (
            base.get("candidate") != CANDIDATE
            or base.get("before_prediction_sha256") != publications["before"][arm]
            or base.get("after_prediction_sha256") != publications["after"][arm]
            or enriched.get("score_artifact_sha256")
            != score_hashes[f"{arm}.base_score"]
            or enriched.get("before_prediction_sha256")
            != publications["before"][arm]
            or enriched.get("after_prediction_sha256")
            != publications["after"][arm]
        ):
            raise DSSCLauncherError("row score/prediction binding drift")
        current_truth_sha = base.get("truth_sidecar_sha256")
        if type(current_truth_sha) is not str or (
            truth_sha is not None and current_truth_sha != truth_sha
        ):
            raise DSSCLauncherError("row score truth-sidecar binding drift")
        truth_sha = current_truth_sha
    expected_syn = (
        float(metrics["M_JOINT"]["after"]["h_old_new"])
        - float(metrics["M_DA"]["after"]["h_old_new"])
        - float(metrics["M_OTHER"]["after"]["h_old_new"])
        + float(metrics["M0"]["after"]["h_old_new"])
    )
    if float(metrics["I_syn"]) != expected_syn:
        raise DSSCLauncherError("row aggregate I_syn drift")
    rows = metrics["same_row_scene_metrics"]
    if (
        type(rows) is not list
        or len(rows) != len(ARMS) * len(SCENES)
        or {(row.get("arm"), row.get("scene")) for row in rows}
        != {(arm, scene) for arm in ARMS for scene in SCENES}
        or any(
            row.get("query_count")
            != (old_class_count + int(job["new_class_count"])) * QUERY_PER_TX
            or "forgetting" not in row
            for row in rows
        )
    ):
        raise DSSCLauncherError("row same-row metric cardinality/schema drift")
    summary = json.loads(score_paths["same_row_summary"].read_text(encoding="utf-8"))
    if summary != {
        "candidate": CANDIDATE,
        "arms": {key: metrics[key] for key in (*ARMS, "I_syn")},
        "rows": rows,
        "score_rows": len(rows),
        "query_truth_joined_only_after_all_five_immutable_predictions": True,
    }:
        raise DSSCLauncherError("row same-row summary/full-metric binding drift")


def _validate_launcher_receipt(
    root: Path,
    job: Mapping[str, Any],
    *,
    allowed_gpu_ids: Sequence[int],
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    launcher_root = root / "launcher"
    receipt_path = launcher_root / f"{job_id}.launcher_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise DSSCLauncherError("completed subprocess has no launcher receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    device_namespace = receipt.get("device_namespace")
    physical_gpu_id = (
        device_namespace.get("physical_gpu_id")
        if type(device_namespace) is dict
        else None
    )
    if (
        receipt.get("schema") != LAUNCHER_RECEIPT_SCHEMA
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("job_id") != job_id
        or receipt.get("status") != "ROW_PROCESS_COMPLETE"
        or receipt.get("returncode") != 0
        or type(physical_gpu_id) is not int
        or physical_gpu_id not in allowed_gpu_ids
        or device_namespace != _row_device_namespace(physical_gpu_id)
        or receipt.get("exception") is not None
        or receipt.get("schedule_cost") != job["schedule_cost"]
        or not isinstance(receipt.get("duration_seconds"), (int, float))
        or not np.isfinite(receipt["duration_seconds"])
        or receipt["duration_seconds"] < 0
    ):
        raise DSSCLauncherError(
            "launcher receipt identity/exit/GPU namespace closure drift"
        )
    for stream in ("stdout", "stderr"):
        expected_path = launcher_root / f"{job_id}.{stream}.log"
        actual_path = Path(str(receipt.get(f"{stream}_path", "")))
        if (
            actual_path != expected_path
            or not expected_path.is_file()
            or expected_path.is_symlink()
            or receipt.get(f"{stream}_sha256") != sha256_file(expected_path)
        ):
            raise DSSCLauncherError("launcher log artifact/hash closure drift")
    return dict(device_namespace)


def _row_device_namespace(physical_gpu_id: int) -> dict[str, Any]:
    if type(physical_gpu_id) is not int or physical_gpu_id < 0:
        raise DSSCLauncherError(
            "row physical GPU ID must be a nonnegative integer"
        )
    return {
        "physical_gpu_id": physical_gpu_id,
        "logical_device": ROW_LOGICAL_DEVICE,
        "cuda_visible_devices": str(physical_gpu_id),
    }


def _expected_row_device_namespace_execution(
    physical_gpu_id: int,
) -> dict[str, Any]:
    _row_device_namespace(physical_gpu_id)
    return {
        "schema": ROW_DEVICE_NAMESPACE_EXECUTION_SCHEMA,
        "cuda_visible_devices": str(physical_gpu_id),
        "visible_physical_gpu_id": physical_gpu_id,
        "requested_logical_device": ROW_LOGICAL_DEVICE,
        "torch_cuda_device_count": 1,
        "torch_cuda_current_device": 0,
    }


def _validate_formal_row_device_namespace_execution(
    evidence: Mapping[str, Any],
) -> None:
    physical_gpu_id = (
        evidence.get("visible_physical_gpu_id")
        if type(evidence) is dict
        else None
    )
    if (
        type(physical_gpu_id) is not int
        or evidence
        != _expected_row_device_namespace_execution(physical_gpu_id)
    ):
        raise DSSCLauncherError(
            "formal row CUDA namespace execution evidence drift"
        )


def _validate_row_launcher_device_namespace_binding(
    row_execution: Any,
    launcher_namespace: Mapping[str, Any],
) -> None:
    physical_gpu_id = (
        launcher_namespace.get("physical_gpu_id")
        if type(launcher_namespace) is dict
        else None
    )
    if (
        type(physical_gpu_id) is not int
        or launcher_namespace != _row_device_namespace(physical_gpu_id)
        or row_execution
        != _expected_row_device_namespace_execution(physical_gpu_id)
    ):
        raise DSSCLauncherError(
            "launcher/row CUDA namespace execution binding drift"
        )


def _row_subprocess_environment(physical_gpu_id: int) -> dict[str, str]:
    device_namespace = _row_device_namespace(physical_gpu_id)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device_namespace[
        "cuda_visible_devices"
    ]
    return environment


def _row_subprocess_command(
    job: Mapping[str, Any], a: argparse.Namespace
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "row",
        "--cache-manifest",
        str(job["cache_manifest"]),
        "--authority-bundle",
        str(job["authority_bundle"]),
        "--authority-commit-sha256",
        str(job["authority_commit_sha256"]),
        "--phase1-checkpoint",
        a.phase1_checkpoint,
        "--sealed-runtime",
        a.sealed_runtime,
        "--package-method-lock",
        a.package_method_lock,
        "--dssc-method-lock",
        a.dssc_method_lock,
        "--ground-bundle",
        a.ground_bundle,
        "--coverage-receipt",
        a.coverage_receipt,
        "--output-root",
        str(job["output_root"]),
        "--receiver",
        str(job["receiver"]),
        "--seed",
        str(job["seed"]),
        "--k-shot",
        str(job["k_shot"]),
        "--new-class-count",
        str(job["new_class_count"]),
        "--device",
        ROW_LOGICAL_DEVICE,
    ]


def run_matrix(a: argparse.Namespace) -> dict[str, Any]:
    root = Path(a.run_root)
    if root.exists():
        raise DSSCLauncherError("matrix run root must be new and cannot be overwritten")
    gpu_ids = _parse_gpu_ids(a.gpu_ids)
    if (
        sha256_file(a.package_method_lock) != SOMPH_PACKAGE_LOCK_SHA256
        or sha256_file(a.sealed_runtime) != SEALED_RUNTIME_SHA256
        or sha256_file(a.phase1_checkpoint) != CHECKPOINT_SHA256
    ):
        raise DSSCLauncherError(
            "matrix checkpoint/package lock/sealed runtime SHA binding drift"
        )
    coverage_sha = _validate_coverage_receipt(a.coverage_receipt)
    lock_raw = Path(a.dssc_method_lock).read_text(encoding="utf-8")
    _lock, canonical_lock_sha = validate_method_lock(lock_raw)
    if sha256_file(a.dssc_method_lock) != canonical_lock_sha:
        raise DSSCLauncherError("matrix DSSC lock is not canonical bytes")
    bundle = load_ground_bundle(
        a.ground_bundle, checkpoint_sha256=CHECKPOINT_SHA256
    )
    if (
        bundle.manifest.get("method_lock") != _lock
        or bundle.manifest.get("method_lock_sha256") != canonical_lock_sha
    ):
        raise DSSCLauncherError("matrix ground bundle/DSSC lock binding drift")
    jobs = matrix_jobs(
        cache_root=a.cache_root,
        authority_root=a.authority_root,
        run_root=root,
        old_class_count=len(bundle.classes),
    )
    root.mkdir(parents=True)
    expected_hashes = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "phase1_archive_sha256": PHASE1_ARCHIVE_SHA256,
        "phase1_archive_manifest_sha256": PHASE1_ARCHIVE_MANIFEST_SHA256,
        "phase1_parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256,
        "ground_bundle_sha256": sha256_file(a.ground_bundle),
        "dssc_method_lock_sha256": canonical_lock_sha,
        "somph_package_lock_sha256": SOMPH_PACKAGE_LOCK_SHA256,
        "sealed_runtime_sha256": SEALED_RUNTIME_SHA256,
        "geoff_r8_coverage_sha256": coverage_sha,
    }
    manifest = {
        "candidate": CANDIDATE,
        "phase1_checkpoint_sha256": CHECKPOINT_SHA256,
        "phase1_archive_sha256": PHASE1_ARCHIVE_SHA256,
        "phase1_archive_manifest_sha256": PHASE1_ARCHIVE_MANIFEST_SHA256,
        "phase1_parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256,
        "ground_bundle_sha256": expected_hashes["ground_bundle_sha256"],
        "dssc_method_lock_sha256": canonical_lock_sha,
        "somph_package_lock_sha256": SOMPH_PACKAGE_LOCK_SHA256,
        "sealed_runtime_sha256": SEALED_RUNTIME_SHA256,
        "geoff_r8_coverage_sha256": coverage_sha,
        "jobs": jobs,
        "job_count": 125,
        "prediction_slice_count": 375,
        "score_row_count": 1875,
        "schedule_policy": {
            "name": "LPT_dynamic_queue",
            "primary_cost": "2_adapters_x_steps_x_support_rows",
            "tiebreak": "query_rows_then_job_id",
            "submission_order": "descending_cost",
            "job_id_changed_by_schedule": False,
        },
        "gpu_ids": list(gpu_ids),
        "worker_count": len(gpu_ids),
        "max_workers_per_gpu_for_this_run": 1,
        "gpu_ids_are_external_preflight_safe_slots": True,
        "shell_template": False,
        "query_truth_in_predictor": False,
    }
    _write_json_new(root / "matrix_manifest.json", manifest)
    available: queue.Queue[int] = queue.Queue()
    for physical_gpu in gpu_ids:
        available.put(physical_gpu)

    def one(job: Mapping[str, Any]) -> tuple[str, int]:
        physical_gpu = available.get()
        job_id = str(job["job_id"])
        launcher_root = root / "launcher"
        stdout_path = launcher_root / f"{job_id}.stdout.log"
        stderr_path = launcher_root / f"{job_id}.stderr.log"
        receipt_path = launcher_root / f"{job_id}.launcher_receipt.json"
        start_utc = _utc_now()
        start_tick = time.perf_counter()
        returncode = 125
        exception_text: str | None = None
        try:
            command = _row_subprocess_command(job, a)
            environment = _row_subprocess_environment(physical_gpu)
            launcher_root.mkdir(parents=True, exist_ok=True)
            out_fd = os.open(
                stdout_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
            err_fd = os.open(
                stderr_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
            with os.fdopen(out_fd, "wb") as stdout_handle, os.fdopen(
                err_fd, "wb"
            ) as stderr_handle:
                returncode = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=environment,
                ).returncode
        except BaseException as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            returncode = 125
        finally:
            end_utc = _utc_now()
            for log_path in (stdout_path, stderr_path):
                if log_path.is_file():
                    os.chmod(log_path, stat.S_IREAD)
            launcher_receipt = {
                "schema": LAUNCHER_RECEIPT_SCHEMA,
                "candidate": CANDIDATE,
                "job_id": job_id,
                "status": "ROW_PROCESS_COMPLETE"
                if returncode == 0
                else "TECHNICAL_FAILURE",
                "device_namespace": _row_device_namespace(physical_gpu),
                "start_utc": start_utc,
                "end_utc": end_utc,
                "duration_seconds": float(time.perf_counter() - start_tick),
                "returncode": int(returncode),
                "stdout_path": str(stdout_path),
                "stdout_sha256": sha256_file(stdout_path)
                if stdout_path.is_file()
                else None,
                "stderr_path": str(stderr_path),
                "stderr_sha256": sha256_file(stderr_path)
                if stderr_path.is_file()
                else None,
                "exception": exception_text,
                "schedule_cost": job["schedule_cost"],
            }
            try:
                _write_json_new(receipt_path, launcher_receipt)
            finally:
                available.put(physical_gpu)
        return job_id, int(returncode)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        results = dict(future.result() for future in as_completed(futures))
    if any(results.values()):
        _write_json_new(root / "matrix_exit.json", {"candidate": CANDIDATE, "returncodes": results, "complete": False})
        raise DSSCLauncherError("one or more full125 row subprocesses failed")
    receipts = []
    try:
        for job in jobs:
            launcher_device_namespace = _validate_launcher_receipt(
                root, job, allowed_gpu_ids=gpu_ids
            )
            receipt_path = Path(str(job["output_root"])) / "row_receipt.json"
            if not receipt_path.is_file():
                raise DSSCLauncherError("completed subprocess has no row receipt")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            _validate_completed_row_receipt(
                job,
                receipt,
                expected_hashes=expected_hashes,
                launcher_device_namespace=launcher_device_namespace,
            )
            receipts.append(receipt)
    except BaseException as exc:
        _write_json_new(
            root / "matrix_exit.json",
            {
                "candidate": CANDIDATE,
                "returncodes": results,
                "complete": False,
                "receipt_validation_error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    aggregate = {"candidate": CANDIDATE, "status": "ARTIFACTS_COMPLETE", "job_count": len(receipts), "prediction_slice_count": sum(int(item["prediction_slice_count"]) for item in receipts), "score_row_count": sum(int(item["score_rows"]) for item in receipts), "phase1_archive_sha256": PHASE1_ARCHIVE_SHA256, "phase1_archive_manifest_sha256": PHASE1_ARCHIVE_MANIFEST_SHA256, "phase1_parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256, "geoff_r8_coverage_sha256": GEOFF_R8_COVERAGE_SHA256, "rows": [{key: item[key] for key in ("receiver", "seed", "k_shot", "new_class_count", "full_metrics")} for item in receipts]}
    if aggregate["job_count"] != 125 or aggregate["prediction_slice_count"] != 375 or aggregate["score_row_count"] != 1875:
        raise DSSCLauncherError("full125 artifact cardinality closure failed")
    _write_json_new(root / "aggregate_index.json", aggregate)
    _write_json_new(root / "matrix_exit.json", {"candidate": CANDIDATE, "returncodes": results, "complete": True, "aggregate_index": str(root / "aggregate_index.json")})
    return {"run_root": str(root), "returncodes": results, "expected_prediction_slices": 375, "expected_score_rows": 1875}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    def common(x: argparse.ArgumentParser) -> None:
        x.add_argument("--phase1-checkpoint", required=True); x.add_argument("--sealed-runtime", required=True); x.add_argument("--package-method-lock", required=True); x.add_argument("--dssc-method-lock", required=True); x.add_argument("--ground-bundle", required=True); x.add_argument("--coverage-receipt", required=True)
    row = sub.add_parser("row"); common(row)
    row.add_argument("--cache-manifest", required=True); row.add_argument("--authority-bundle", required=True); row.add_argument("--authority-commit-sha256", required=True); row.add_argument("--output-root", required=True); row.add_argument("--receiver", required=True); row.add_argument("--seed", type=int, required=True); row.add_argument("--k-shot", type=int, required=True); row.add_argument("--new-class-count", type=int, required=True); row.add_argument("--device", required=True)
    matrix = sub.add_parser("matrix"); common(matrix)
    matrix.add_argument("--cache-root", required=True); matrix.add_argument("--authority-root", required=True); matrix.add_argument("--run-root", required=True); matrix.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    return p


def main() -> int:
    args = parser().parse_args()
    result = run_row(args) if args.mode == "row" else run_matrix(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
