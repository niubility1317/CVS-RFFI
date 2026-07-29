#!/usr/bin/env python3
"""Run one frozen Phase2 ablation row from a reusable feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.stage2_ablation_factory import (
    get_stage2_arm,
    resolved_stage2_config_hash,
)
from cvsrffi.stage2_ablation_feature_cache import load_feature_cache
from cvsrffi.stage2_ablation_row_executor import execute_feature_row


REQUEST_SCHEMA = "cvs.full_ablation.phase2.row_request.v1"
_REQUEST_KEYS = {
    "schema",
    "ablation_id",
    "row_id",
    "receiver",
    "stage_scope",
    "k_shot",
    "new_class_count",
    "support_seed",
    "query_seed",
    "new_class_draw_seed",
    "phase2_data_status",
    "capsule_id",
    "split_id",
    "phase1_bundle_sha256",
    "phase1_prototype_sha256",
    "candidate_lock_sha256",
    "effective_config_hash",
    "feature_cache_payload",
    "feature_cache_payload_sha256",
    "feature_cache_manifest",
    "feature_cache_manifest_sha256",
    "output_root",
    "seed",
    "device",
    "shared_view_count",
}


class Stage2AblationRowRequestError(ValueError):
    """Raised when a sealed row request is incomplete or inconsistent."""


def _load_request(path: str | Path) -> dict[str, Any]:
    request_path = Path(path)
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(request, Mapping)
        or set(request) != _REQUEST_KEYS
        or request.get("schema") != REQUEST_SCHEMA
    ):
        raise Stage2AblationRowRequestError("row request exact schema drift")
    return dict(request)


def run_request(path: str | Path) -> dict[str, Any]:
    request = _load_request(path)
    ablation_id = str(request["ablation_id"])
    spec = get_stage2_arm(ablation_id)
    expected_config_hash = resolved_stage2_config_hash(ablation_id)
    if request["effective_config_hash"] != expected_config_hash:
        raise Stage2AblationRowRequestError(
            "row request effective config hash drift"
        )
    cache = load_feature_cache(
        request["feature_cache_payload"],
        request["feature_cache_manifest"],
        expected_payload_sha256=request["feature_cache_payload_sha256"],
        expected_manifest_sha256=request[
            "feature_cache_manifest_sha256"
        ],
    )
    manifest = cache["manifest"]
    if (
        str(request["receiver"]) != str(manifest["receiver"])
        or int(request["seed"]) != int(manifest["method_seed"])
        or str(request["stage_scope"]) != spec.stage
        or str(manifest.get("stage_scope")) != spec.stage
        or int(request["k_shot"]) != int(manifest["k_shot"])
        or int(request["new_class_count"])
        != len(manifest["new_classes"])
        or int(request["support_seed"])
        != int(manifest["support_seed"])
        or int(request["query_seed"]) != int(manifest["query_seed"])
        or int(request["new_class_draw_seed"])
        != int(manifest["new_class_draw_seed"])
        or request["phase2_data_status"]
        != manifest["phase2_data_status"]
        or request["capsule_id"] != manifest["capsule_id"]
        or request["split_id"] != manifest["split_id"]
        or request["phase1_bundle_sha256"]
        != manifest["phase1_bundle_sha256"]
        or request["phase1_prototype_sha256"]
        != manifest["phase1_prototype_sha256"]
    ):
        raise Stage2AblationRowRequestError(
            "row request/cache identity mismatch"
        )
    if spec.stage == "stage2c" and len(cache["new_classes"]) not in {
        5,
        10,
        20,
    }:
        raise Stage2AblationRowRequestError(
            "Stage2-C cache lacks a registered new-class set"
        )
    receipt = execute_feature_row(
        ablation_id=ablation_id,
        row_id=str(request["row_id"]),
        receiver=str(request["receiver"]),
        candidate_lock_sha256=str(request["candidate_lock_sha256"]),
        package_root_sha256=str(manifest["package_root_sha256"]),
        package_seal_sha256=str(manifest["package_seal_sha256"]),
        input_identity={
            "stage_scope": str(request["stage_scope"]),
            "k_shot": int(request["k_shot"]),
            "new_class_count": int(request["new_class_count"]),
            "method_seed": int(request["seed"]),
            "support_seed": int(request["support_seed"]),
            "query_seed": int(request["query_seed"]),
            "new_class_draw_seed": int(
                request["new_class_draw_seed"]
            ),
            "phase2_data_status": str(
                request["phase2_data_status"]
            ),
            "capsule_id": str(request["capsule_id"]),
            "split_id": str(request["split_id"]),
            "phase1_bundle_sha256": str(
                request["phase1_bundle_sha256"]
            ),
            "phase1_prototype_sha256": str(
                request["phase1_prototype_sha256"]
            ),
            "feature_cache_payload_sha256": str(
                request["feature_cache_payload_sha256"]
            ),
            "feature_cache_manifest_sha256": str(
                request["feature_cache_manifest_sha256"]
            ),
        },
        output_root=request["output_root"],
        seed=int(request["seed"]),
        device=str(request["device"]),
        shared_view_count=int(request["shared_view_count"]),
        feature_cache_bytes=Path(
            request["feature_cache_payload"]
        ).stat().st_size,
        deployment_state_bytes=int(
            manifest["deployment_state_bytes"]
        ),
        old_classes=cache["old_classes"],
        new_classes=cache["new_classes"],
        scenario_payloads=cache["scenario_payloads"],
        deployment_prototypes_by_scenario=cache[
            "deployment_prototypes_by_scenario"
        ],
        ground_basis=cache["ground_basis"],
        ground_spectral_weights=cache["ground_spectral_weights"],
        ground_audit=cache["ground_audit"],
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one truth-inaccessible Phase2 ablation row from a sealed "
            "reusable feature cache."
        )
    )
    parser.add_argument("--request", required=True)
    return parser


def main() -> int:
    receipt = run_request(_parser().parse_args().request)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
