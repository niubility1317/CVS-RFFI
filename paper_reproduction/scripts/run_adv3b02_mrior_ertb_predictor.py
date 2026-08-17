#!/usr/bin/env python3
"""Run ERTB-IDR registration on an MRIOR-SDA adapted ADV3B02 backbone.

The predictor keeps the four-state interpretation explicit: the MRIOR-SDA
backbone is the ``DA1_REG0`` state, and the D92 E0_FULL_ONLY registration
head produces ``DA1_REG1``.  Query IQ is opened only after every scenario's
support-derived state has been fitted and locked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from cvsrffi.stage2_d42_unified_shrinkage_lda import (  # noqa: E402
    D42UnifiedShrinkageLDAConfig,
    fit_d42_unified_shrinkage_lda,
    predict_d42_unified_shrinkage_lda,
)
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    registered_feature,
)
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact  # noqa: E402
from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (  # noqa: E402
    load_verified_mrior_preadapt_artifact,
)
from scripts.probe_d92_registration_balanced_covariance import (  # noqa: E402
    build_d92_fit,
    load_ground_basis,
)
from paper_reproduction.scripts.run_adv3b02_paper_full_ci_truth_free_predictor import (  # noqa: E402
    _forward_direct,
    _load_exact_backbone,
    _materialize_npz,
    _read_mrior_preadapt_bindings,
    _restore_mrior_preadapted_backbone,
    _selected_support,
    _tensor,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS as PACKAGE_SCENARIOS,
    _validate_query_arrays,
    _validate_support_arrays,
    preflight_stage2_predictor_package,
)


METHOD = "mrior_sda_then_ertb_idr"
OLD_CLASS_COUNT = 6
EXPECTED_NEW_COUNTS = (5, 10, 20)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _class_handles(manifest: Mapping[str, Any]) -> list[str]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list) or len(rows) <= OLD_CLASS_COUNT:
        raise ValueError("ERTB class registry is missing")
    handles = [str(item.get("class_handle", "")) for item in rows]
    if any(not value for value in handles) or len(set(handles)) != len(handles):
        raise ValueError("ERTB class registry drift")
    if [int(item.get("class_index", -1)) for item in rows] != list(range(len(rows))):
        raise ValueError("ERTB class index order drift")
    return handles


def _load_bindings(
    path: Path,
    *,
    receiver: str,
    seed: int,
    k_shot: int,
) -> dict[str, Any]:
    bindings = _read_mrior_preadapt_bindings(path)
    loaded: dict[str, Any] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        entry = bindings[scenario]
        result = load_verified_mrior_preadapt_artifact(
            entry["artifact_root"],
            expected_input_binding_sha256=entry["expected_input_binding_sha256"],
            expected_method_lock_sha256=entry["expected_method_lock_sha256"],
        )
        binding = result.input_binding
        if (
            str(binding.receiver) != str(receiver)
            or int(binding.seed) != int(seed)
            or int(binding.k_shot) != int(k_shot)
            or str(binding.scene) != str(scenario)
        ):
            raise ValueError("MRIOR domain-adaptation row binding drift")
        loaded[scenario] = {
            "result": result,
            "artifact_root": entry["artifact_root"],
            "input_binding_sha256": binding.canonical_sha256,
            "method_lock_sha256": result.method_lock_sha256,
            "manifest_sha256": _sha256_file(entry["artifact_root"] / "manifest.json"),
            "state_sha256": _sha256_file(entry["artifact_root"] / "mrior_preadapt_state.pt"),
        }
    return loaded


def _prototype_predictions(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    handles: list[str],
) -> np.ndarray:
    means = []
    for handle in handles:
        rows = support_features[support_labels == handle]
        if rows.size == 0:
            raise ValueError("ERTB prototype support class is absent")
        mean = rows.mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1.0e-8)
        means.append(mean.astype(np.float32))
    prototypes = np.stack(means)
    values = np.asarray(query_features, dtype=np.float32)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)
    return np.asarray(handles, dtype=str)[np.argmax(values @ prototypes.T, axis=1)]


def _handles_from_values(values: np.ndarray, handles: list[str]) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if any(value not in handles for value in result.tolist()):
        raise ValueError("ERTB prediction references an unregistered class")
    return result


def predict(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.old_class_count) != OLD_CLASS_COUNT:
        raise ValueError("ERTB old-class count is locked to 6")
    package_root = Path(args.package_root).resolve(strict=True)
    manifest, _seal, package_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=args.detached_seal,
        expected_seal_sha256=args.expected_seal_sha256,
    )
    if manifest.get("stage") != "stage2c":
        raise ValueError("ERTB predictor requires a Stage2-C package")
    if int(manifest["seed"]) != int(args.seed):
        raise ValueError("ERTB package seed mismatch")
    if int(manifest["new_class_count"]) not in EXPECTED_NEW_COUNTS:
        raise ValueError("ERTB E0_FULL_ONLY requires new-count 5/10/20")
    if tuple(PACKAGE_SCENARIOS) != tuple(FORMAL_LEO_WEAK_SCENARIOS):
        raise ValueError("ERTB scenario registry drift")
    handles = _class_handles(manifest)
    if int(args.expected_total_capacity) != len(handles):
        raise ValueError("ERTB class capacity drift")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    bindings = _load_bindings(
        Path(args.mrior_bindings).resolve(strict=True),
        receiver=str(manifest["receiver"]),
        seed=int(args.seed),
        k_shot=int(args.k_shot),
    )
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    base_backbone, feature_fn, backbone_audit = _load_exact_backbone(
        package_root, manifest, device=device, verify_checkpoint_member=True
    )
    ground_basis, ground_weights, ground_audit = load_ground_basis(
        Path(args.ground_component_dir), str(args.ground_manifest_sha256), 288
    )
    import cvsrffi.stage2_d42_unified_shrinkage_lda as d42

    d92_fit, component_records, transform_records = build_d92_fit(
        d42, ground_basis, ground_weights, ground_audit
    )
    sklearn_version = str(d42.sklearn.__version__)
    original_fit = d42._fit_equal_prior_lda
    original_runtime = d42.SKLEARN_RUNTIME_VERSION
    fit_results: dict[str, Any] = {}
    backbones: dict[str, tuple[torch.nn.Module, Any, dict[str, Any]]] = {}
    support_features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fit_audits: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    reference_support_tokens: np.ndarray | None = None
    try:
        d42._fit_equal_prior_lda = d92_fit
        d42.SKLEARN_RUNTIME_VERSION = sklearn_version
        for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
            arrays, support_manifest = _materialize_npz(
                package_root, roles[f"support:{scenario}"]
            )
            _validate_support_arrays(
                arrays,
                support_manifest,
                scenario=scenario,
                class_count=len(handles),
                max_k=int(manifest["support_pool_max_k"]),
            )
            iq_np, labels_np, tokens = _selected_support(
                arrays, k_shot=int(args.k_shot)
            )
            if reference_support_tokens is None:
                reference_support_tokens = tokens
            elif not np.array_equal(reference_support_tokens, tokens):
                raise ValueError("ERTB support token ordering drift")
            adapted = copy.deepcopy(base_backbone)
            _restore_mrior_preadapted_backbone(adapted, bindings[scenario]["result"].model_state)
            support_x = _tensor(iq_np, dtype=torch.float32, device=device)
            support_y = _tensor(labels_np, dtype=torch.int64, device=device).long()
            base_features, _ = _forward_direct(
                adapted, feature_fn, support_x, batch_size=int(args.batch_size)
            )
            feature_np = registered_feature(
                iq_np, base_features.detach().cpu().numpy().astype(np.float32)
            )
            labels_handles = np.asarray(handles, dtype=str)[labels_np]
            old_mask = labels_np < OLD_CLASS_COUNT
            new_mask = ~old_mask
            started = time.perf_counter()
            fitted = fit_d42_unified_shrinkage_lda(
                feature_np[old_mask],
                labels_handles[old_mask],
                tuple(handles[:OLD_CLASS_COUNT]),
                feature_np[new_mask],
                labels_handles[new_mask],
                tuple(handles[OLD_CLASS_COUNT:]),
                seed=int(args.seed) + scenario_index,
                device=device,
                config=D42UnifiedShrinkageLDAConfig(
                    sklearn_runtime_version=sklearn_version
                ),
            )
            fit_results[scenario] = fitted
            backbones[scenario] = (adapted, feature_fn, backbone_audit)
            support_features[scenario] = (feature_np, labels_handles, tokens)
            fit_audits.append(
                {
                    "scenario": scenario,
                    "before_state_sha256": hashlib.sha256(
                        repr(fitted.before_state).encode("utf-8")
                    ).hexdigest(),
                    "after_state_sha256": hashlib.sha256(
                        repr(fitted.state).encode("utf-8")
                    ).hexdigest(),
                    "geometry": dict(fitted.geometry_audit),
                }
            )
            resources.append(
                {
                    "scenario": scenario,
                    **dict(fitted.resource_audit),
                    "mrior_domain_adaptation_state_sha256": bindings[scenario][
                        "state_sha256"
                    ],
                    "registration_seconds": time.perf_counter() - started,
                }
            )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42.SKLEARN_RUNTIME_VERSION = original_runtime
    if set(fit_results) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise ValueError("ERTB support fit did not close all scenarios")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    enrollment_receipt = {
        "schema": "cvs.phase2.adv3b02_mrior_ertb_enrollment_receipt.v1",
        "status": "PASS",
        "method": METHOD,
        "four_state_before_registration": "DA1_REG0",
        "four_state_after_registration": "DA1_REG1",
        "receiver": manifest["receiver"],
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "new_class_count": int(manifest["new_class_count"]),
        "query_rows_used_for_training": 0,
        "mrior_domain_adaptation_lineage": {
            scenario: {
                "artifact_root": str(bindings[scenario]["artifact_root"]),
                "manifest_sha256": bindings[scenario]["manifest_sha256"],
                "state_sha256": bindings[scenario]["state_sha256"],
                "input_binding_sha256": bindings[scenario]["input_binding_sha256"],
                "method_lock_sha256": bindings[scenario]["method_lock_sha256"],
            }
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "ertb_idr_arm": "E0_FULL_ONLY",
        "d92_registration_state": "support_only_D42_full_old_new_equal_prior_LDA",
        "query_members_opened_before_model_lock": False,
        "fit_audits": fit_audits,
    }
    enrollment_path = output_dir / "enrollment_receipt.json"
    _write_json_new(enrollment_path, enrollment_receipt)

    streams = {name: [] for name in ("candidate_after", "candidate_before", "identity_after", "identity_before", "direct")}
    query_tokens_rows = []
    scenario_rows = []
    query_opened = False
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        query, query_manifest = _materialize_npz(
            package_root, roles[f"query:{scenario}"]
        )
        _validate_query_arrays(query, query_manifest, scenario=scenario)
        query_opened = True
        iq_np = np.asarray(query["query_leo_weak_iq"], dtype=np.float32)
        tokens = np.asarray(query["query_tokens"]).astype(str)
        adapted, feature_fn, _audit = backbones[scenario]
        query_x = _tensor(iq_np, dtype=torch.float32, device=device)
        query_base, query_logits = _forward_direct(
            adapted, feature_fn, query_x, batch_size=int(args.batch_size)
        )
        query_features = registered_feature(
            iq_np, query_base.detach().cpu().numpy().astype(np.float32)
        )
        fitted = fit_results[scenario]
        candidate_before = predict_d42_unified_shrinkage_lda(
            fitted.before_state, query_features
        )
        candidate_after = predict_d42_unified_shrinkage_lda(
            fitted.state, query_features
        )
        support_np, support_labels, _support_tokens = support_features[scenario]
        identity_before = _prototype_predictions(
            support_np,
            support_labels,
            query_features,
            handles[:OLD_CLASS_COUNT],
        )
        identity_after = _prototype_predictions(
            support_np,
            support_labels,
            query_features,
            handles,
        )
        direct = np.asarray(handles[:OLD_CLASS_COUNT], dtype=str)[
            np.asarray(query_logits.detach().cpu().numpy()).argmax(axis=1)
        ]
        streams["candidate_before"].append(
            _handles_from_values(candidate_before, handles[:OLD_CLASS_COUNT])
        )
        streams["candidate_after"].append(_handles_from_values(candidate_after, handles))
        streams["identity_before"].append(
            _handles_from_values(identity_before, handles[:OLD_CLASS_COUNT])
        )
        streams["identity_after"].append(_handles_from_values(identity_after, handles))
        streams["direct"].append(_handles_from_values(direct, handles[:OLD_CLASS_COUNT]))
        query_tokens_rows.append(tokens)
        scenario_rows.append(np.asarray([scenario] * len(tokens)))
    if not query_opened:
        raise ValueError("ERTB query stage did not open")

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
        query_tokens=np.concatenate(query_tokens_rows),
        scenarios=np.concatenate(scenario_rows),
        candidate_after=np.concatenate(streams["candidate_after"]),
        candidate_before=np.concatenate(streams["candidate_before"]),
        identity_after=np.concatenate(streams["identity_after"]),
        identity_before=np.concatenate(streams["identity_before"]),
        direct=np.concatenate(streams["direct"]),
        shared_view_counts=np.ones(sum(len(value) for value in query_tokens_rows), dtype=np.uint8),
    )
    receipt = {
        "schema": "cvs.phase2.adv3b02_mrior_ertb_predictor_receipt.v1",
        "status": "FORMAL_COMPARISON_MRIOR_DOMAIN_ADAPTATION",
        "method": METHOD,
        "method_claim_boundary": "ERTB_IDR_on_MRIOR_SDA_DA1_REG1",
        "four_state_labels": {"before_registration": "DA1_REG0", "after_registration": "DA1_REG1"},
        "row_id": str(args.row_id),
        "receiver": manifest["receiver"],
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "new_class_count": int(manifest["new_class_count"]),
        "registered_class_count": len(handles),
        "backbone": "ADV3B02_MRIOR_SDA",
        "ertb_idr_arm": "E0_FULL_ONLY",
        "candidate_lock_sha256": manifest["candidate_lock_sha256"],
        "package_preopen_audit": package_audit,
        "query_members_opened_before_model_lock": False,
        "query_opened_after_model_lock": True,
        "query_rows_used_for_training": 0,
        "enrollment_receipt_sha256": _sha256_file(enrollment_path),
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "prediction_immutable_state": publication["immutable_state"],
        "mrior_domain_adaptation_state_sha256_by_scenario": {
            scenario: bindings[scenario]["state_sha256"]
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "resources_by_scenario": resources,
    }
    receipt_path = output_dir / "predictor_receipt.json"
    _write_json_new(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "predictor_receipt": str(receipt_path),
        "predictor_receipt_sha256": _sha256_file(receipt_path),
        "enrollment_receipt": str(enrollment_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--mrior-bindings", type=Path, required=True)
    parser.add_argument("--ground-component-dir", type=Path, required=True)
    parser.add_argument("--ground-manifest-sha256", required=True)
    parser.add_argument("--old-class-count", type=int, default=OLD_CLASS_COUNT)
    parser.add_argument("--expected-total-capacity", type=int, required=True)
    parser.add_argument("--k-shot", type=int, choices=(1, 5, 10, 20), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(predict(parse_args()), ensure_ascii=False, sort_keys=True))
