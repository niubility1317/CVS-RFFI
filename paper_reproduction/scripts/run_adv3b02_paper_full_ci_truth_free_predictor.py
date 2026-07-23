#!/usr/bin/env python3
"""Run paper-mechanism CSIL/MoPC-HR with a trainable sealed ADV3B02."""

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

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
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
from model_dual_cvsincnet import backbone_forward_compat  # noqa: E402
from paper_reproduction.cvs_aligned.adv3b02_ci_heads import prototype_baseline  # noqa: E402
from paper_reproduction.cvs_aligned.adv3b02_paper_full_ci import (  # noqa: E402
    METHODS as LEGACY_METHODS,
    fit_paper_full,
    predict_after as predict_after_legacy,
    predict_before as predict_before_legacy,
)
from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import (  # noqa: E402
    METHODS as OFFICIAL_METHODS,
    fit_official_repo,
    predict_after as predict_after_official,
    predict_before as predict_before_official,
)


METHODS = LEGACY_METHODS + OFFICIAL_METHODS


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
    elif dtype == torch.int64:
        array = array.astype(np.int64, copy=False)
    else:
        raise TypeError(dtype)
    # N607 currently pairs NumPy 2.2.5 with Torch 2.1.0; its NumPy C bridge
    # rejects genuine ndarrays. The buffer protocol avoids that ABI boundary.
    # clone() owns the storage before the local NumPy array leaves scope.
    tensor = torch.frombuffer(memoryview(array), dtype=dtype)
    return tensor.reshape(array.shape).clone().to(device=device, dtype=dtype)


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


def _load_exact_backbone(
    package_root: Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
):
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    descriptor = roles["checkpoint"]
    with open_regular_member_same_fd(package_root, descriptor["relative_path"]) as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    exact, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=256,
        device=device,
    )
    if not hasattr(exact, "id_backbone"):
        raise ValueError("ADV3B02 checkpoint misses id_backbone")
    feature_key = str(getattr(exact, "id_feature_key", "feat_joint"))

    def feature_fn(backbone: torch.nn.Module, rows: torch.Tensor):
        auxiliary = backbone_forward_compat(
            backbone,
            rows,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        feature = auxiliary.get(feature_key)
        if not torch.is_tensor(feature):
            feature = auxiliary.get("feat_joint")
        logits = auxiliary.get("logits")
        if not torch.is_tensor(feature) or not torch.is_tensor(logits):
            raise ValueError("ADV3B02 identity backbone output drift")
        return feature.float(), logits.float()

    return exact.id_backbone.to(device), feature_fn, audit


def _load_base_state(
    package_root: Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    old_count: int,
) -> dict[str, Any]:
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    descriptor = roles["head"]
    with open_regular_member_same_fd(package_root, descriptor["relative_path"]) as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    schema = state.get("schema")
    if schema == "cvs.adv3b02.official_repo_base_state.v2":
        if int(state.get("base_sample_count", 0)) != 8400:
            raise ValueError("official-repo base state requires exactly 8400 rows")
        if (
            int(state.get("csil_base_train_sample_count", 0)) != 5879
            or int(state.get("fisher_sample_count", 0)) != 2521
            or state.get("source_train_fisher_disjoint") is not True
        ):
            raise ValueError("official CSIL 70/30 train/Fisher split drift")
        if not isinstance(state.get("csil"), dict) or not isinstance(
            state.get("mopc_hr"), dict
        ):
            raise ValueError("official-repo method base states are missing")
        return {
            "csil": state["csil"],
            "mopc_hr": state["mopc_hr"],
            "receipt": {
                "schema": schema,
                "checkpoint_sha256": state.get("checkpoint_sha256"),
                "base_sample_count": int(state["base_sample_count"]),
                "base_class_counts": list(state.get("base_class_counts", [])),
                "csil_base_train_sample_count": int(
                    state["csil_base_train_sample_count"]
                ),
                "fisher_sample_count": int(state["fisher_sample_count"]),
                "fisher_class_counts": list(state.get("fisher_class_counts", [])),
                "source_train_fisher_disjoint": True,
                "source_receiver_labels": list(
                    state.get("source_receiver_labels", [])
                ),
                "official_repo_commits": dict(
                    state.get("official_repo_commits", {})
                ),
                "raw_exemplars_stored": bool(
                    state.get("raw_exemplars_stored", True)
                ),
            },
        }
    if schema != "cvs.adv3b02.paper_full_base_state.v1":
        raise ValueError("paper-full base state schema drift")
    if len(state.get("old_class_labels", [])) != int(old_count):
        raise ValueError("paper-full base state old-class count drift")
    fingerprints = state.get("old_fingerprints")
    prototypes = state.get("old_prototypes")
    fisher = state.get("fisher")
    if (
        not torch.is_tensor(fingerprints)
        or not torch.is_tensor(prototypes)
        or not isinstance(fisher, dict)
    ):
        raise ValueError("paper-full base state tensor surface drift")
    return {
        "old_fingerprints": fingerprints.to(device),
        "old_prototypes": prototypes.to(device),
        "fisher": {name: value.to(device) for name, value in fisher.items()},
        "receipt": {
            "schema": state["schema"],
            "checkpoint_sha256": state.get("checkpoint_sha256"),
            "base_sample_count": int(state.get("base_sample_count", 0)),
            "source_receiver_labels": list(state.get("source_receiver_labels", [])),
            "raw_exemplars_stored": bool(state.get("raw_exemplars_stored", True)),
        },
    }


@torch.no_grad()
def _forward_direct(
    backbone: torch.nn.Module,
    feature_fn,
    rows: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    logits = []
    for start in range(0, len(rows), int(batch_size)):
        feature, logit = feature_fn(backbone, rows[start : start + int(batch_size)])
        features.append(feature)
        logits.append(logit)
    return torch.cat(features), torch.cat(logits)


def predict(args: argparse.Namespace) -> dict[str, Any]:
    method = str(args.method)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    package_root = Path(args.package_root).resolve(strict=True)
    manifest, _seal, package_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=args.detached_seal,
        expected_seal_sha256=args.expected_seal_sha256,
    )
    if manifest["stage"] != "stage2c":
        raise ValueError("paper-full CI predictor requires Stage2-C")
    old_count = int(args.old_class_count)
    class_handles = [item["class_handle"] for item in manifest["registered_classes"]]
    if not 0 < old_count < len(class_handles):
        raise ValueError("old class count is inconsistent with the class registry")
    if int(args.k_shot) > int(manifest["support_pool_max_k"]):
        raise ValueError("K exceeds sealed support pool")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    base_state = _load_base_state(
        package_root, manifest, device=device, old_count=old_count
    )
    opened_roles = ["checkpoint", "head"]
    fitted_by_scenario = {}
    support_by_scenario = {}
    resources = []
    loss_trace = []
    reference_support_tokens = None
    checkpoint_audits = []
    training_started = time.perf_counter()

    # Enrollment: only checkpoint and support members are open.
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        arrays, support_manifest = _materialize_npz(
            package_root, roles[f"support:{scenario}"]
        )
        _validate_support_arrays(
            arrays,
            support_manifest,
            scenario=scenario,
            class_count=len(class_handles),
            max_k=int(manifest["support_pool_max_k"]),
        )
        opened_roles.append(f"support:{scenario}")
        iq_np, labels_np, tokens = _selected_support(arrays, k_shot=int(args.k_shot))
        if reference_support_tokens is None:
            reference_support_tokens = tokens
        elif not np.array_equal(reference_support_tokens, tokens):
            raise ValueError("physical support ordering drift across scenarios")
        support_x = _tensor(iq_np, dtype=torch.float32, device=device)
        support_y = _tensor(labels_np, dtype=torch.int64, device=device).long()
        backbone, feature_fn, audit = _load_exact_backbone(
            package_root, manifest, device=device
        )
        checkpoint_audits.append({"scenario": scenario, **audit})
        started = time.perf_counter()
        if method in OFFICIAL_METHODS:
            fitted = fit_official_repo(
                method,
                backbone,
                support_x,
                support_y,
                feature_fn=feature_fn,
                old_count=old_count,
                seed=int(args.seed) + scenario_index * 1009,
                base_state=base_state,
            )
        else:
            fitted = fit_paper_full(
                method,
                backbone,
                support_x,
                support_y,
                feature_fn=feature_fn,
                old_count=old_count,
                seed=int(args.seed) + scenario_index * 1009,
                base_state=base_state,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resource = {
            "scenario": scenario,
            **fitted.resource,
            "adaptation_seconds": time.perf_counter() - started,
        }
        fitted_by_scenario[scenario] = fitted
        support_by_scenario[scenario] = (support_x, support_y, feature_fn)
        resources.append(resource)
        loss_trace.extend({"scenario": scenario, **row} for row in fitted.loss_trace)

    # Serialize/hash-lock all trainable model and prototype state before query opens.
    state_path = output_dir / "enrolled_models.pt"
    with state_path.open("xb") as handle:
        torch.save(
            {
                scenario: fitted_by_scenario[scenario].serializable_state()
                for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    model_state_sha256 = sha256_file(state_path)
    enrollment_receipt_path = output_dir / "enrollment_receipt.json"
    _write_json_new(
        enrollment_receipt_path,
        {
            "schema": "cvs.phase2.adv3b02_paper_full_ci_enrollment_receipt.v1",
            "status": "PASS",
            "method": method,
            "receiver": manifest["receiver"],
            "seed": int(args.seed),
            "k_shot": int(args.k_shot),
            "new_class_count": int(manifest["new_class_count"]),
            "query_members_opened_before_model_lock": False,
            "opened_roles_before_model_lock": list(opened_roles),
            "enrolled_model_state_sha256": model_state_sha256,
            "query_rows_used_for_training": 0,
            "checkpoint_load_audits": checkpoint_audits,
            "base_state_receipt": base_state["receipt"],
        },
    )

    streams = {
        name: []
        for name in (
            "candidate_after",
            "candidate_before",
            "identity_after",
            "identity_before",
            "direct",
        )
    }
    token_rows = []
    scenario_rows = []
    reference_query_tokens = None
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
            raise ValueError("physical query ordering drift across scenarios")
        if reference_support_tokens is not None and set(reference_support_tokens) & set(query_tokens):
            raise ValueError("support/query opaque token overlap")
        query_x = _tensor(query_iq, dtype=torch.float32, device=device)
        fitted = fitted_by_scenario[scenario]
        support_x, support_y, feature_fn = support_by_scenario[scenario]
        if method in OFFICIAL_METHODS:
            candidate_before = predict_before_official(fitted, query_x)
            candidate_after = predict_after_official(fitted, query_x)
        else:
            candidate_before = predict_before_legacy(fitted, query_x)
            candidate_after = predict_after_legacy(fitted, query_x)
        support_features, _ = _forward_direct(
            fitted.teacher_backbone,
            feature_fn,
            support_x,
            batch_size=int(args.batch_size),
        )
        query_features, direct_logits = _forward_direct(
            fitted.teacher_backbone,
            feature_fn,
            query_x,
            batch_size=int(args.batch_size),
        )
        old_mask = support_y < old_count
        identity_before = prototype_baseline(
            support_features[old_mask],
            support_y[old_mask],
            query_features,
            class_count=old_count,
        )
        identity_after = prototype_baseline(
            support_features,
            support_y,
            query_features,
            class_count=len(class_handles),
        )
        if direct_logits.ndim != 2 or int(direct_logits.shape[1]) != old_count:
            raise ValueError("direct ADV3B02 class order drift")
        streams["candidate_after"].append(_handles(candidate_after, class_handles))
        streams["candidate_before"].append(_handles(candidate_before, class_handles[:old_count]))
        streams["identity_after"].append(_handles(identity_after, class_handles))
        streams["identity_before"].append(_handles(identity_before, class_handles[:old_count]))
        streams["direct"].append(_handles(direct_logits.argmax(1), class_handles[:old_count]))
        token_rows.append(query_tokens)
        scenario_rows.append(np.asarray([scenario] * len(query_tokens)))

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
        "schema": "cvs.phase2.adv3b02_paper_full_ci_predictor_receipt.v1",
        "status": "FORMAL_COMPARISON_BASELINE",
        "method": method,
        "method_claim_boundary": "formal_paper_method_comparison_baseline",
        "row_id": str(args.row_id),
        "receiver": manifest["receiver"],
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "new_class_count": int(manifest["new_class_count"]),
        "old_class_count": old_count,
        "registered_class_count": len(class_handles),
        "backbone": "ADV3B02",
        "backbone_frozen": False,
        "candidate_resources_by_scenario": resources,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "total_enrollment_seconds": time.perf_counter() - training_started,
        "comparison_method_protocol_scope": (
            "stage2_main_method_protocol_exempt_new_class_leo_required"
        ),
        "new_class_support_channel_policy": "leo_satellite_required",
        "new_class_query_channel_policy": "leo_satellite_required",
        "base_source_reference_access_allowed": True,
        "base_state_receipt": base_state["receipt"],
        "fixed_received_iq_reused_across_epochs": True,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "query_rows_used_for_training": 0,
        "query_labels_available_to_predictor": False,
        "dense_query_graph_used": False,
        "query_members_opened_before_model_lock": False,
        "runtime_open_role_ledger": opened_roles,
        "package_preopen_audit": package_audit,
        "enrolled_model_state_sha256": model_state_sha256,
        "enrollment_receipt_sha256": sha256_file(enrollment_receipt_path),
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "prediction_immutable_state": publication["immutable_state"],
    }
    receipt_path = output_dir / "predictor_receipt.json"
    _write_json_new(receipt_path, receipt)
    return {
        "status": receipt["status"],
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
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(predict(parse_args()), ensure_ascii=False, sort_keys=True))
