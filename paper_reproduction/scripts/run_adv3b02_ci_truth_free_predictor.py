#!/usr/bin/env python3
"""Fit a support-only CI head and publish truth-free Stage2-C predictions."""

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
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    _materialize_npz,
    _validate_query_arrays,
    _validate_support_arrays,
    open_regular_member_same_fd,
    preflight_stage2_predictor_package,
    sha256_file,
)
from paper_reproduction.cvs_aligned.adv3b02_ci_heads import (  # noqa: E402
    METHODS,
    fit_incremental_head,
    predict_incremental_head,
    prototype_baseline,
)


def _write_json_new(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tensor(value: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    array = np.ascontiguousarray(value)
    if dtype == torch.float32:
        array = array.astype(np.float32, copy=False)
        base = torch.frombuffer(memoryview(array), dtype=torch.float32)
    elif dtype == torch.int64:
        array = array.astype(np.int64, copy=False)
        base = torch.frombuffer(memoryview(array), dtype=torch.int64)
    else:
        raise TypeError(dtype)
    return base.reshape(array.shape).clone().to(device)


def _numpy(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().cpu().contiguous()
    dtype_by_torch = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
        torch.int64: np.dtype(np.int64),
    }
    if tensor.dtype not in dtype_by_torch:
        raise TypeError(f"unsupported state dtype: {tensor.dtype}")
    raw = bytes(tensor.view(torch.uint8).tolist())
    return np.frombuffer(raw, dtype=dtype_by_torch[tensor.dtype]).copy().reshape(tuple(tensor.shape))


def _load_runtime(
    package_root: Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.jit.ScriptModule:
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    descriptor = roles["checkpoint"]
    with open_regular_member_same_fd(package_root, descriptor["relative_path"]) as handle:
        runtime = torch.jit.load(handle, map_location=device)
    return runtime.eval()


@torch.no_grad()
def _forward(
    runtime: torch.jit.ScriptModule,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(rows), int(batch_size)):
        batch = _tensor(
            np.asarray(rows[start : start + int(batch_size)], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        output = runtime(batch)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError("ADV3B02 runtime output must be (z_id, logits)")
        feature, logit = output
        if not torch.is_tensor(feature) or not torch.is_tensor(logit):
            raise ValueError("ADV3B02 runtime returned a non-tensor output")
        features.append(feature.float())
        logits.append(logit.float())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return torch.cat(features), torch.cat(logits), elapsed


def _selected_support(arrays: Mapping[str, np.ndarray], *, k_shot: int):
    labels = np.asarray(arrays["support_pool_class_indices"], dtype=np.int64)
    ranks = np.asarray(arrays["support_pool_rank_within_class"], dtype=np.int64)
    selected = np.flatnonzero(ranks < int(k_shot))
    expected = int(np.unique(labels).size) * int(k_shot)
    if len(selected) != expected:
        raise ValueError("nested support selection count drift")
    return (
        np.asarray(arrays["support_pool_leo_weak_iq"], dtype=np.float32)[selected],
        labels[selected],
        np.asarray(arrays["support_pool_tokens"]).astype(str)[selected],
    )


def _handles(indices: torch.Tensor, class_handles: list[str]) -> np.ndarray:
    values = indices.detach().cpu().tolist()
    if any(int(value) < 0 or int(value) >= len(class_handles) for value in values):
        raise ValueError("predicted class index is outside the registry")
    return np.asarray([class_handles[int(value)] for value in values])


def predict(args: argparse.Namespace) -> dict[str, Any]:
    method = str(args.method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    package_root = Path(args.package_root).resolve(strict=True)
    manifest, _seal, package_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=args.detached_seal,
        expected_seal_sha256=args.expected_seal_sha256,
    )
    if manifest["stage"] != "stage2c":
        raise ValueError("class-incremental predictor requires Stage2-C")
    if int(args.k_shot) > int(manifest["support_pool_max_k"]):
        raise ValueError("K exceeds the sealed support pool")
    old_count = int(args.old_class_count)
    class_handles = [item["class_handle"] for item in manifest["registered_classes"]]
    if not 0 < old_count < len(class_handles):
        raise ValueError("old class count is inconsistent with the class registry")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    runtime = _load_runtime(package_root, manifest, device=device)
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    roles = {item["artifact_role"]: item for item in manifest["members"]}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    loss_trace: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    state_payload: dict[str, np.ndarray] = {}
    fitted_by_scenario = {}
    support_features_by_scenario: dict[str, torch.Tensor] = {}
    support_labels_by_scenario: dict[str, torch.Tensor] = {}
    support_forward_samples = query_forward_samples = 0
    support_forward_seconds = query_forward_seconds = 0.0
    reference_support_tokens: np.ndarray | None = None
    opened_roles = ["checkpoint"]

    # Enrollment phase: query members have not been opened at this point.
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        support_arrays, support_manifest = _materialize_npz(
            package_root, roles[f"support:{scenario}"]
        )
        _validate_support_arrays(
            support_arrays,
            support_manifest,
            scenario=scenario,
            class_count=int(manifest["registered_class_count"]),
            max_k=int(manifest["support_pool_max_k"]),
        )
        opened_roles.append(f"support:{scenario}")
        support_iq, support_labels_np, support_tokens = _selected_support(
            support_arrays, k_shot=int(args.k_shot)
        )
        if reference_support_tokens is None:
            reference_support_tokens = support_tokens
        elif not np.array_equal(reference_support_tokens, support_tokens):
            raise ValueError("physical support ID ordering drift across scenarios")
        support_features, _support_logits, support_seconds = _forward(
            runtime, support_iq, device=device, batch_size=int(args.batch_size)
        )
        support_labels = _tensor(
            support_labels_np, dtype=torch.int64, device=device
        ).long()
        fitted = fit_incremental_head(
            method,
            support_features,
            support_labels,
            old_count=old_count,
            seed=int(args.seed) + scenario_index * 1009,
            steps=int(args.head_steps),
        )
        fitted_by_scenario[scenario] = fitted
        support_features_by_scenario[scenario] = support_features
        support_labels_by_scenario[scenario] = support_labels
        for row in fitted.loss_trace:
            loss_trace.append({"scenario": scenario, **row})
        resources.append({"scenario": scenario, **fitted.resource})
        for phase, state in (("before", fitted.before_state), ("after", fitted.after_state)):
            for key, value in state.items():
                state_payload[f"{scenario}.{phase}.{key}"] = _numpy(value)
        support_forward_samples += len(support_iq)
        support_forward_seconds += support_seconds

    # The enrolled state is written and hash-locked before any query member opens.
    state_path = output_dir / "enrolled_head_states.npz"
    with state_path.open("xb") as handle:
        np.savez(handle, **state_payload)
    enrolled_head_sha256 = sha256_file(state_path)
    enrollment_receipt_path = output_dir / "enrollment_receipt.json"
    _write_json_new(enrollment_receipt_path, {
        "schema": "cvs.phase2.adv3b02_ci_enrollment_receipt.v1",
        "status": "PASS",
        "method": method,
        "receiver": manifest["receiver"],
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "new_class_count": int(manifest["new_class_count"]),
        "query_members_opened_before_head_lock": False,
        "opened_roles_before_head_lock": list(opened_roles),
        "enrolled_head_state_sha256": enrolled_head_sha256,
        "query_rows_used_for_training": 0,
    })

    # Apply phase: heads are already fixed; only per-sample all-class inference remains.
    streams = {name: [] for name in (
        "candidate_after", "candidate_before", "identity_after", "identity_before", "direct"
    )}
    token_rows: list[np.ndarray] = []
    scenario_rows: list[np.ndarray] = []
    reference_query_tokens: np.ndarray | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        query, query_manifest = _materialize_npz(
            package_root, roles[f"query:{scenario}"]
        )
        _validate_query_arrays(query, query_manifest, scenario=scenario)
        opened_roles.append(f"query:{scenario}")
        query_iq = np.asarray(query["query_leo_weak_iq"], dtype=np.float32)
        query_tokens = np.asarray(query["query_tokens"]).astype(str)
        if reference_query_tokens is None:
            reference_query_tokens = query_tokens
        elif not np.array_equal(reference_query_tokens, query_tokens):
            raise ValueError("physical query ID ordering drift across scenarios")
        if reference_support_tokens is not None and set(reference_support_tokens) & set(query_tokens):
            raise ValueError("support/query opaque token overlap")
        query_features, direct_logits, query_seconds = _forward(
            runtime, query_iq, device=device, batch_size=int(args.batch_size)
        )
        fitted = fitted_by_scenario[scenario]
        support_features = support_features_by_scenario[scenario]
        support_labels = support_labels_by_scenario[scenario]
        candidate_before, candidate_after = predict_incremental_head(fitted, query_features)
        old_support_mask = support_labels < old_count
        identity_before = prototype_baseline(
            support_features[old_support_mask], support_labels[old_support_mask], query_features,
            class_count=old_count,
        )
        identity_after = prototype_baseline(
            support_features, support_labels, query_features,
            class_count=len(class_handles),
        )
        if direct_logits.ndim != 2 or direct_logits.shape[1] != old_count:
            raise ValueError("strict direct ADV3B02 logits/class order drift")
        direct = direct_logits.argmax(1)
        streams["candidate_after"].append(_handles(candidate_after, class_handles))
        streams["candidate_before"].append(_handles(candidate_before, class_handles[:old_count]))
        streams["identity_after"].append(_handles(identity_after, class_handles))
        streams["identity_before"].append(_handles(identity_before, class_handles[:old_count]))
        streams["direct"].append(_handles(direct, class_handles[:old_count]))
        token_rows.append(query_tokens)
        scenario_rows.append(np.asarray([scenario] * len(query_tokens)))
        query_forward_samples += len(query_iq)
        query_forward_seconds += query_seconds

    prediction_path = output_dir / "prediction_artifact.cvspred"
    publication = publish_prediction_artifact(
        prediction_path,
        stage="Stage2-C",
        row_id=str(args.row_id),
        receiver=str(manifest["receiver"]),
        k_shot=int(args.k_shot),
        candidate_lock_sha256=str(manifest["candidate_lock_sha256"]),
        package_root_sha256=str(manifest["package_root_sha256"]),
        package_seal_sha256=str(args.expected_seal_sha256),
        query_tokens=np.concatenate(token_rows),
        scenarios=np.concatenate(scenario_rows),
        candidate_after=np.concatenate(streams["candidate_after"]),
        candidate_before=np.concatenate(streams["candidate_before"]),
        identity_after=np.concatenate(streams["identity_after"]),
        identity_before=np.concatenate(streams["identity_before"]),
        direct=np.concatenate(streams["direct"]),
        shared_view_counts=np.ones(sum(len(v) for v in token_rows), dtype=np.uint8),
    )
    _write_json_new(output_dir / "loss_trace.json", loss_trace)
    receipt = {
        "schema": "cvs.phase2.adv3b02_ci_predictor_receipt.v1",
        "status": "PROTOCOL_VALID",
        "method": method,
        "method_claim_boundary": "cvs_aligned_adv3b02_feature_head_extension",
        "row_id": str(args.row_id),
        "receiver": manifest["receiver"],
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "new_class_count": int(manifest["new_class_count"]),
        "old_class_count": old_count,
        "registered_class_count": len(class_handles),
        "head_hyperparameter_source": "paper_derived_locked_no_confirmation_query_tuning",
        "head_steps_per_phase": int(args.head_steps),
        "backbone": "ADV3B02",
        "backbone_frozen": True,
        "fft96_enabled": False,
        "query_view_count": 1,
        "support_view_count": 1,
        "support_backbone_forward_samples": support_forward_samples,
        "query_backbone_forward_samples": query_forward_samples,
        "support_forward_seconds": support_forward_seconds,
        "query_forward_seconds": query_forward_seconds,
        "candidate_resources_by_scenario": resources,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
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
        "query_rows_used_for_training": 0,
        "query_labels_available_to_predictor": False,
        "dense_query_graph_used": False,
        "prototype_fit_inside_query_forward": False,
        "package_preopen_audit": package_audit,
        "runtime_open_role_ledger": opened_roles,
        "query_members_opened_before_head_lock": False,
        "enrollment_receipt_sha256": sha256_file(enrollment_receipt_path),
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": str(args.expected_seal_sha256),
        "enrolled_head_state_sha256": enrolled_head_sha256,
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "prediction_immutable_state": publication["immutable_state"],
    }
    receipt_path = output_dir / "predictor_receipt.json"
    _write_json_new(receipt_path, receipt)
    return {
        "status": "PROTOCOL_VALID",
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
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--old-class-count", type=int, default=6)
    parser.add_argument("--k-shot", type=int, choices=(1, 5, 10, 20), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--head-steps", type=int, default=10)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(predict(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
