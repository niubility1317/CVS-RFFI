"""Strict Phase2 predictor entry: validate the request before any package open."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cvsrffi.phase2_runtime_contract import validate_predictor_request
from cvsrffi.stage2_predictor_bundle import (
    load_verified_stage2_predictor_bundle,
    preflight_stage2_predictor_package,
)
from cvsrffi.stage2_predictor_runtime import (
    load_json_artifact_same_fd,
    load_torchscript_backbone_same_fd,
    predict_all_streams,
)


def _request_descriptor(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": item["relative_path"],
        "sha256": item["sha256"],
        "size_bytes": item["size_bytes"],
        "artifact_role": item["artifact_role"],
        "schema": item["schema"],
    }


def _cross_check_request_manifest(
    request: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    for key in (
        "stage",
        "receiver",
        "candidate_lock_sha256",
        "package_root_sha256",
        "registered_class_count",
        "registered_classes",
    ):
        if request[key] != manifest[key]:
            raise ValueError(f"predictor request/package mismatch: {key}")
    if int(request["satellite_seed"]) != int(manifest["seed"]):
        raise ValueError("predictor request/package mismatch: satellite_seed")
    if int(request["k_shot"]) > int(manifest["support_pool_max_k"]):
        raise ValueError("predictor request K exceeds sealed support pool")
    members = {item["artifact_role"]: item for item in manifest["members"]}
    request_roles = {
        "checkpoint_artifact": "checkpoint",
        "adapter_artifact": "adapter",
        "head_artifact": "head",
    }
    for request_field, role in request_roles.items():
        if role not in members or request[request_field] != _request_descriptor(members[role]):
            raise ValueError(f"predictor request artifact descriptor mismatch: {request_field}")
    for request_field, prefix in (
        ("support_artifacts", "support:"),
        ("query_artifacts", "query:"),
    ):
        expected = [
            _request_descriptor(members[prefix + scenario])
            for scenario in request["scenarios"]
        ]
        if request[request_field] != expected:
            raise ValueError(
                f"predictor request artifact descriptor mismatch: {request_field}"
            )
    return members


def prepare_role_blind_prediction(
    request: Mapping[str, Any],
    *,
    predictor_package_root: str | Path,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Return five prediction streams; request validation is the first operation."""

    # This must remain the first operation.  The negative spy test enforces that
    # no package manifest, seal, NPZ, IQ row, or runtime artifact is opened first.
    validate_predictor_request(request)

    package_root = Path(predictor_package_root)
    manifest, _seal, preopen_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    members = _cross_check_request_manifest(request, manifest)
    if (
        request["phase2_runtime_isolation_evidence"]["sealed_inference_package_sha256"]
        != str(expected_seal_sha256).lower()
    ):
        raise ValueError("predictor request/seal trust-root mismatch")
    support_by_scenario, query_by_scenario, loaded_manifest, materialize_audit = (
        load_verified_stage2_predictor_bundle(
            package_root,
            detached_seal_path=detached_seal_path,
            expected_seal_sha256=expected_seal_sha256,
        )
    )
    if loaded_manifest != manifest:
        raise ValueError("predictor package manifest changed after pre-open audit")

    tta_config = load_json_artifact_same_fd(package_root, members["tta_policy"])
    if tta_config != request["tta_policy"]:
        raise ValueError("predictor request/TTA artifact content mismatch")
    adapter_config = load_json_artifact_same_fd(package_root, members["adapter"])
    head_config = load_json_artifact_same_fd(package_root, members["head"])
    model = load_torchscript_backbone_same_fd(
        package_root, members["checkpoint"], device=device
    )
    payload_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "query_tokens",
            "scenarios",
            "candidate_after",
            "candidate_before",
            "identity_after",
            "identity_before",
            "direct",
            "shared_view_counts",
        )
    }
    resources_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in request["scenarios"]:
        predictions, scenario_resource = predict_all_streams(
            model,
            support_by_scenario[scenario],
            query_by_scenario[scenario],
            k_shot=int(request["k_shot"]),
            registered_class_count=int(manifest["registered_class_count"]),
            new_class_count=int(manifest["new_class_count"]),
            adapter_config=adapter_config,
            head_config=head_config,
            tta_config=tta_config,
            device=device,
            batch_size=int(batch_size),
        )
        query_tokens = np.asarray(
            query_by_scenario[scenario]["query_tokens"]
        ).astype(str)
        payload_parts["query_tokens"].append(query_tokens)
        payload_parts["scenarios"].append(np.asarray([scenario] * len(query_tokens)))
        for name, values in predictions.items():
            payload_parts[name].append(np.asarray(values))
        resources_by_scenario[scenario] = scenario_resource
    payload = {
        name: np.concatenate(parts, axis=0) for name, parts in payload_parts.items()
    }
    row_count = len(payload["query_tokens"])
    shared_counts = np.asarray(payload["shared_view_counts"], dtype=np.int64)
    resource_receipt = {
        "schema": "cvs.phase2.predictor_resource_receipt.v2",
        "scenarios": list(request["scenarios"]),
        "by_scenario": resources_by_scenario,
        "candidate_query_latency_ms": float(
            np.mean(
                [value["candidate_query_latency_ms"] for value in resources_by_scenario.values()]
            )
        ),
        "mean_backbone_forwards": float(np.mean(shared_counts)),
        "p95_backbone_forwards": float(
            np.percentile(shared_counts, 95, method="higher")
        ),
        "view1_rate": float(np.mean(shared_counts == 1)),
        "view3_rate": float(np.mean(shared_counts == 3)),
        "view5_rate": float(np.mean(shared_counts == 5)),
        "support_enrollment_backbone_forwards": int(
            sum(value["support_enrollment_backbone_forwards"] for value in resources_by_scenario.values())
        ),
        "query_backbone_forwards": int(np.sum(shared_counts)),
        "fft_descriptor_count": int(
            sum(value["fft_descriptor_count"] for value in resources_by_scenario.values())
        ),
        "trainable_parameters": int(adapter_config["trainable_parameters"]),
        "adapt_epochs": int(adapter_config["adapt_epochs"]),
        "persistent_state_bytes": int(adapter_config["persistent_state_bytes"]),
        "shared_view_budget_for_all_streams": True,
        "direct_uses_base_view_only": True,
    }
    metadata = {
        "schema": "cvs.phase2.prediction_metadata.v2",
        "request_id": request["request_id"],
        "row_id": request["row_id"],
        "stage": request["stage"],
        "receiver": request["receiver"],
        "scenarios": list(request["scenarios"]),
        "k_shot": int(request["k_shot"]),
        "candidate_lock_sha256": request["candidate_lock_sha256"],
        "package_root_sha256": request["package_root_sha256"],
        "predictor_package_seal_sha256": str(expected_seal_sha256).lower(),
        "runtime_code_sha256": request["runtime_code_sha256"],
        "checkpoint_sha256": request["checkpoint_artifact"]["sha256"],
        "adapter_sha256": request["adapter_artifact"]["sha256"],
        "head_sha256": request["head_artifact"]["sha256"],
        "tta_policy_sha256": request["tta_policy_sha256"],
        "prediction_row_count": row_count,
        "query_truth_access": False,
        "query_role_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "shared_view_budget_for_all_streams": True,
    }
    audit = {
        "preopen": preopen_audit,
        "materialization": materialize_audit,
        "resource_receipt": resource_receipt,
    }
    return payload, metadata, audit
