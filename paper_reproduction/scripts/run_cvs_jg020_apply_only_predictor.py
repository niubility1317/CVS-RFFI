#!/usr/bin/env python
"""Apply sealed JG020 runtimes/head to truth-free LEO_weak query IQ.

This process has no support or truth argument and performs no prototype fitting,
threshold fitting, model selection, optimizer step or batch-global assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact  # noqa: E402
from cvsrffi.stage2_predictor_bundle import iq_row_sha256  # noqa: E402
from paper_reproduction.cvs_aligned.jg020_stage2c import (  # noqa: E402
    APPLY_PROFILE,
    FORMAL_SCENARIOS,
    HEAD_SCHEMA,
    RECEIPT_SCHEMA,
    apply_head_streams,
    descriptor_by_role,
    load_npz_member,
    numpy_from_torch_compat,
    open_regular_member_same_fd,
    preflight_package,
    sha256_file,
    torch_tensor_from_numpy_compat,
    validate_locked_candidate,
)


QUERY_FIELDS = {
    "query_leo_weak_iq",
    "query_tokens",
    "query_overlay_tokens",
    "query_satellite_seeds",
    "query_post_channel_iq_sha256",
    "manifest_json",
}


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_member(root: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        value = json.loads(handle.read().decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JG020 JSON package member must be an object")
    return value


def _load_runtime(root: Path, descriptor: Mapping[str, Any], *, device: torch.device) -> torch.jit.ScriptModule:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        try:
            runtime = torch.jit.load(handle, map_location=device)
        except Exception as exc:
            raise ValueError(f"failed to load sealed JG020 runtime: {descriptor['artifact_role']}") from exc
    return runtime.eval()


def _validate_query_arrays(arrays: Mapping[str, np.ndarray], *, scenario: str) -> None:
    if set(arrays) != QUERY_FIELDS:
        raise ValueError("JG020 query NPZ exact field drift")
    manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
    expected = {
        "schema": "cvs.phase2.unlabeled_query_iq.v2",
        "scenario": scenario,
        "query_truth_included": False,
        "query_role_included": False,
        "query_true_batch_class_count_included": False,
        "query_class_quota_included": False,
        "query_ordering_hint_included": False,
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    if manifest != expected:
        raise ValueError("JG020 query embedded truth-free manifest drift")
    iq = np.asarray(arrays["query_leo_weak_iq"], dtype=np.float32)
    hashes = np.asarray(arrays["query_post_channel_iq_sha256"]).astype(str)
    if iq.ndim != 3 or iq.shape[1] != 2 or len(iq) != len(hashes):
        raise ValueError("JG020 query IQ layout drift")
    observed = np.asarray([iq_row_sha256(row) for row in iq])
    if not np.array_equal(observed, hashes):
        raise ValueError("JG020 query sample-level post-channel hash drift")


@torch.no_grad()
def _forward(runtime: torch.jit.ScriptModule, rows: np.ndarray, *, device: torch.device, batch_size: int):
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for start in range(0, len(rows), int(batch_size)):
        batch = torch_tensor_from_numpy_compat(
            np.asarray(rows[start : start + int(batch_size)], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        result = runtime(batch)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("JG020 runtime output must be (features, logits)")
        feature, logit = result
        if not torch.is_tensor(feature) or not torch.is_tensor(logit):
            raise ValueError("JG020 runtime returned non-tensor output")
        features.append(numpy_from_torch_compat(feature.float(), dtype=np.dtype(np.float32)))
        logits.append(numpy_from_torch_compat(logit.float(), dtype=np.dtype(np.float32)))
    return np.concatenate(features).astype(np.float32), np.concatenate(logits).astype(np.float32)


def predict(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root).resolve(strict=True)
    document, preopen = preflight_package(
        package_root,
        detached_seal=args.detached_seal,
        expected_seal_sha256=args.expected_seal_sha256,
        expected_profile=APPLY_PROFILE,
    )
    roles = descriptor_by_role(document)
    lock = validate_locked_candidate(_load_json_member(package_root, roles["candidate_lock"]))
    receipt = _load_json_member(package_root, roles["enrollment_receipt"])
    required_receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "candidate_id": lock["candidate_id"],
        "receiver": lock["receiver"],
        "seed": lock["seed"],
        "k_shot": lock["k_shot"],
        "new_class_count": lock["new_class_count"],
        "trainable_parameters": 6_400,
        "adapt_epochs": 5,
        "training_compute_mode": "frozen_backbone_cached_joint_inputs",
        "adapter_alpha": 1.0,
        "trust_decision": "locked_k10_full_delta",
        "k1_trust_gate_enabled": False,
        "adapt_fit_class_count": 6,
        "prototype_fit_class_count": 6 + lock["new_class_count"],
        "optimizer_input_stage": "preincrement_registered_old_only",
        "registered_support_labels_used": True,
        "new_support_gradient_used": False,
        "adapter_retrained_at_registration": False,
        "query_role_used_by_optimizer": False,
        "per_sample_old_new_role_branch_used": False,
        "query_rows_used_for_training": 0,
        "query_path_argument_exists": False,
        "old_new_query_role_used": False,
        "query_class_quota_used": False,
        "dense_query_graph_used": False,
        "persistent_state_within_cap": True,
        "candidate_lock_sha256": document["candidate_lock_sha256"],
    }
    failed = [key for key, value in required_receipt.items() if receipt.get(key) != value]
    if failed:
        raise ValueError(f"JG020 enrollment receipt contract failed: {failed}")
    if receipt.get("optimizer_steps", 51) > 50 or receipt.get("persistent_state_bytes", 1 << 30) > 256 * 1024:
        raise ValueError("JG020 enrollment resource receipt exceeds the preferred cap")
    mapping_audit = receipt.get("direct_class_mapping_audit", {})
    if mapping_audit.get("direct_logit_to_class_handle_order_bound") is not True or mapping_audit.get(
        "old_class_order_sha256"
    ) != lock["old_class_order_sha256"]:
        raise ValueError("JG020 direct-logit class-order evidence is missing")
    if receipt.get("package_root_sha256") != document["lineage"]["enrollment_package_root_sha256"]:
        raise ValueError("JG020 apply/enrollment package lineage drift")
    if receipt.get("full_backbone_forward_sample_equivalents", 0) <= 0 or receipt.get(
        "cached_small_path_forward_sample_equivalents", 0
    ) <= 0:
        raise ValueError("JG020 cached-training compute evidence is missing")
    if receipt.get("full_backbone_forward_avoided_sample_equivalents") != receipt.get(
        "cached_small_path_forward_sample_equivalents"
    ):
        raise ValueError("JG020 cached-training avoided-forward evidence drift")

    head = load_npz_member(package_root, roles["prototype_head"])
    head_manifest = json.loads(str(np.asarray(head["manifest_json"]).item()))
    if head_manifest.get("schema") != HEAD_SCHEMA or head_manifest.get("role_symmetric_rule") is not True:
        raise ValueError("JG020 prototype head support-only/symmetric contract drift")
    class_handles = np.asarray(head["class_handles"]).astype(str).tolist()
    if class_handles != [item["class_handle"] for item in document["registered_classes"]]:
        raise ValueError("JG020 prototype head/class registry drift")

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    candidate_runtime = _load_runtime(package_root, roles["candidate_runtime"], device=device)
    identity_runtime = _load_runtime(package_root, roles["identity_runtime"], device=device)
    direct_runtime = _load_runtime(package_root, roles["direct_runtime"], device=device)
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    query_tokens: list[np.ndarray] = []
    scenario_values: list[np.ndarray] = []
    streams = {name: [] for name in (
        "candidate_after", "candidate_before", "identity_after", "identity_before", "direct"
    )}
    reference_tokens: np.ndarray | None = None
    candidate_seconds = identity_seconds = direct_seconds = 0.0
    total_queries = 0
    for scenario in FORMAL_SCENARIOS:
        arrays = load_npz_member(package_root, roles[f"query:{scenario}"])
        _validate_query_arrays(arrays, scenario=scenario)
        iq = np.asarray(arrays["query_leo_weak_iq"], dtype=np.float32)
        tokens = np.asarray(arrays["query_tokens"]).astype(str)
        if reference_tokens is None:
            reference_tokens = tokens
        elif not np.array_equal(reference_tokens, tokens):
            raise ValueError("JG020 physical query ordering drifts across scenarios")
        started = time.perf_counter()
        candidate_features, _candidate_logits = _forward(
            candidate_runtime, iq, device=device, batch_size=args.batch_size
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_seconds += time.perf_counter() - started
        started = time.perf_counter()
        identity_features, _identity_logits = _forward(
            identity_runtime, iq, device=device, batch_size=args.batch_size
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        identity_seconds += time.perf_counter() - started
        started = time.perf_counter()
        _direct_features, direct_logits = _forward(
            direct_runtime, iq, device=device, batch_size=args.batch_size
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        direct_seconds += time.perf_counter() - started
        predicted = apply_head_streams(
            scenario=scenario,
            candidate_features=candidate_features,
            identity_features=identity_features,
            direct_logits=direct_logits,
            head=head,
        )
        for name in streams:
            streams[name].append(predicted[name])
        query_tokens.append(tokens)
        scenario_values.append(np.asarray([scenario] * len(tokens)))
        total_queries += len(tokens)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "predictions.cvspred"
    publication = publish_prediction_artifact(
        prediction_path,
        stage="Stage2-C",
        row_id=f"JG_R8_LR020_rx20-1_seed713101_n{lock['new_class_count']}_k10",
        receiver=lock["receiver"],
        k_shot=lock["k_shot"],
        candidate_lock_sha256=document["candidate_lock_sha256"],
        package_root_sha256=document["package_root_sha256"],
        package_seal_sha256=args.expected_seal_sha256,
        query_tokens=np.concatenate(query_tokens),
        scenarios=np.concatenate(scenario_values),
        candidate_after=np.concatenate(streams["candidate_after"]),
        candidate_before=np.concatenate(streams["candidate_before"]),
        identity_after=np.concatenate(streams["identity_after"]),
        identity_before=np.concatenate(streams["identity_before"]),
        direct=np.concatenate(streams["direct"]),
        shared_view_counts=np.ones(total_queries, dtype=np.uint8),
    )
    predictor_receipt = {
        "schema": "cvs.phase2.jg020_apply_receipt.v1",
        "status": "PASS",
        "candidate_id": lock["candidate_id"],
        "receiver": lock["receiver"],
        "seed": lock["seed"],
        "k_shot": lock["k_shot"],
        "new_class_count": lock["new_class_count"],
        "query_count_per_scenario": (
            int(len(reference_tokens)) if reference_tokens is not None else 0
        ),
        "scenario_count": len(FORMAL_SCENARIOS),
        "query_view_count": 1,
        "candidate_backbone_forward_sample_count": total_queries,
        "candidate_latency_ms_per_query": 1000.0 * candidate_seconds / max(total_queries, 1),
        "identity_baseline_latency_ms_per_query": 1000.0 * identity_seconds / max(total_queries, 1),
        "direct_baseline_latency_ms_per_query": 1000.0 * direct_seconds / max(total_queries, 1),
        "peak_cuda_memory_bytes_all_three_eval_streams": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "adaptive_view_triggering_enabled": False,
        "fft96_enabled": False,
        "prototype_fit_inside_predictor": False,
        "optimizer_inside_predictor": False,
        "query_truth_access": False,
        "query_role_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "dense_query_graph_used": False,
        "per_sample_all_registered_classes": True,
        "enrollment_receipt_sha256": roles["enrollment_receipt"]["sha256"],
        "package_root_sha256": document["package_root_sha256"],
        "package_seal_sha256": args.expected_seal_sha256,
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "prediction_immutable_state": publication["immutable_state"],
        "preopen_audit": preopen,
    }
    receipt_path = output_dir / "predictor_receipt.json"
    _write_json_new(receipt_path, predictor_receipt)
    return {
        "status": "PASS",
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "predictor_receipt": str(receipt_path),
        "predictor_receipt_sha256": sha256_file(receipt_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(predict(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
