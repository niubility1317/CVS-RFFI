#!/usr/bin/env python
"""Build and verify a source-only candidate lock for formal Stage2-C rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.scripts.run_cvs_stage2c_effective8_formal_plan import (
    validate_execution_manifest,
)


FORMAL_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
FORMAL_CONFIRMATION_SEEDS = (713101, 713102, 713103, 713104, 713105)
FORMAL_K = (1, 5, 10, 20)
FORMAL_NEW_COUNTS = (5, 10, 20)
CODE_ARTIFACTS = (
    "paper_reproduction/scripts/benchmark_cvs_adaptive_rxlight_tta.py",
    "paper_reproduction/scripts/build_cvs_stage2c_candidate_lock.py",
    "paper_reproduction/scripts/summarize_cvs_stage2c_locked_matrix.py",
    "paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py",
    "paper_reproduction/cvs_aligned/k1_symmetric_head.py",
    "paper_reproduction/cvs_aligned/adaptive_rxlight_tta.py",
    "code/cvsrffi/leo_weak_cache.py",
    "code/scripts/build_cvs_leo_weak_iq_cache.py",
    "code/scripts/train_apply_phase1_iq_preadapter_20260703.py",
    "paper_reproduction/scripts/build_cvs_stage2c_effective8_formal_plan.py",
    "paper_reproduction/scripts/run_cvs_stage2c_effective8_formal_plan.py",
    "paper_reproduction/scripts/collect_cvs_stage2c_formal_outputs.py",
    "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json",
)

PHASE2_SAMPLE_VIEW_POLICY = "leo_weak_only_no_clean_access"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_head(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "use_alignment": bool(payload["use_alignment"]),
        "prototype_rule": str(payload["prototype_rule"]),
        "ridge": None if payload["ridge"] is None else float(payload["ridge"]),
    }


def build_candidate_lock(
    *,
    candidate_id: str,
    checkpoint: Path,
    adapter_state: Path,
    promotion_manifest: Path,
    class_split_manifest: Path,
    execution_plan_manifest: Path,
) -> dict[str, Any]:
    manifest = json.loads(promotion_manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("method") != "ground_source_effective_feature_lora_v1":
        raise ValueError("formal candidate lock requires the effective8 ground LoRA")
    if (
        manifest.get("source_validation_pass") is not True
        or manifest.get("source_only") is not True
        or manifest.get("target_receiver_data_used_for_training") is not False
        or manifest.get("clean_samples_used_for_training") is not False
        or manifest.get("formal_training_view") != "leo_weak_only"
        or manifest.get("proxy_data_used_for_training") is not False
        or manifest.get("stage") != "Phase1_offline_ground_adapter_training"
        or manifest.get("training_input_stage")
        != "phase1_offline_prechannel_export"
        or manifest.get("phase2_sample_view_policy")
        != PHASE2_SAMPLE_VIEW_POLICY
        or manifest.get("clean_sample_access") is not False
        or manifest.get("clean_derived_signal_access") is not False
        or not manifest.get("source_leo_weak_cache_set_manifest_sha256")
    ):
        raise ValueError("ground-source candidate is not validated leo_weak-only")
    if str(manifest.get("checkpoint_sha256", "")) != _sha256_file(checkpoint):
        raise ValueError("candidate checkpoint hash mismatch")
    if str(manifest.get("adapter_state_sha256", "")) != _sha256_file(adapter_state):
        raise ValueError("candidate adapter hash mismatch")
    validation_path = Path(str(manifest.get("source_validation_manifest", "")))
    if (
        not validation_path.is_file()
        or _sha256_file(validation_path)
        != str(manifest.get("source_validation_manifest_sha256", ""))
    ):
        raise ValueError("source validation path/hash is invalid")
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    permissions = dict(validation.get("permissions", {}))
    head_lock = dict(validation.get("symmetric_head_lock", {}))
    nested_k_lock = dict(validation.get("nested_k_source_lock", {}))
    validation_gates = dict(validation.get("gates", {}))
    if (
        validation.get("source_validation_pass") is not True
        or validation.get("clean_samples_used_for_validation") is not False
        or validation.get("validation_input_stage")
        != "phase1_offline_prechannel_export"
        or validation.get("phase2_sample_view_policy")
        != PHASE2_SAMPLE_VIEW_POLICY
        or validation.get("clean_sample_access") is not False
        or validation.get("clean_derived_signal_access") is not False
        or not validation.get("source_leo_weak_cache_set_manifest_sha256")
        or not validation_gates
        or not all(bool(value) for value in validation_gates.values())
        or validation.get("failed_gates")
        or permissions.get("target_support_used") is not False
        or permissions.get("target_query_features_used") is not False
        or permissions.get("target_query_labels_used") is not False
        or head_lock.get("selection_source")
        != "disjoint_source_receiver_holdout_k1_episodes"
        or head_lock.get("target_support_used_for_selection") is not False
        or head_lock.get("target_query_features_used") is not False
        or head_lock.get("old_new_role_oracle_used") is not False
        or head_lock.get("class_quota_used") is not False
        or head_lock.get("support_view_policy")
        != "three_leo_weak_scenario_base_views"
        or int(head_lock.get("support_receive_views_per_physical_sample", -1)) != 3
        or tuple(int(value) for value in head_lock.get("allowed_k", ())) != FORMAL_K
        or tuple(int(value) for value in nested_k_lock.get("k_values", ()))
        != FORMAL_K
        or nested_k_lock.get("target_rows_used") is not False
        or nested_k_lock.get("role_labels_used") is not False
        or nested_k_lock.get("class_quota_used") is not False
        or tuple(str(value) for value in validation.get("scenarios", ()))
        != (
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        )
    ):
        raise ValueError("source validation is not a legal formal candidate lock source")
    calibration = dict(validation.get("calibration", {}))
    thresholds = dict(calibration.get("selected", {})).get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("source validation did not freeze adaptive TTA thresholds")
    stats = dict(validation.get("source_feature_statistics", {}))
    stats_path = Path(str(stats.get("path", "")))
    if (
        not stats_path.is_file()
        or _sha256_file(stats_path) != str(stats.get("sha256", ""))
        or stats.get("target_rows_used") is not False
        or stats.get("feature_kind") != "normalized_z_id_plus_fft96_weight2"
        or int(stats.get("fft_dim", -1)) != 96
        or float(stats.get("fft_weight", -1.0)) != 2.0
    ):
        raise ValueError("source feature statistics are not immutable/source-only")
    source_train_cache_path = Path(
        str(manifest.get("source_leo_weak_cache_set_manifest", ""))
    )
    source_validation_cache_path = Path(
        str(validation.get("source_leo_weak_cache_set_manifest", ""))
    )
    immutable_cache_artifacts = (
        (
            "source_train",
            source_train_cache_path,
            str(manifest.get("source_leo_weak_cache_set_manifest_sha256", "")),
        ),
        (
            "source_validation",
            source_validation_cache_path,
            str(validation.get("source_leo_weak_cache_set_manifest_sha256", "")),
        ),
    )
    for cache_name, cache_path, expected_hash in immutable_cache_artifacts:
        if (
            not cache_path.is_file()
            or len(expected_hash) != 64
            or _sha256_file(cache_path) != expected_hash
        ):
            raise ValueError(f"immutable {cache_name} LEO_weak cache-set drift")
    split = json.loads(class_split_manifest.read_text(encoding="utf-8-sig"))
    old_labels = [str(value) for value in split.get("target_old_tx_labels", [])]
    nested_raw = dict(split.get("nested_target_new_tx_labels", {}))
    nested_new = {
        str(count): [str(value) for value in nested_raw.get(str(count), [])]
        for count in FORMAL_NEW_COUNTS
    }
    if not old_labels or len(old_labels) != len(set(old_labels)):
        raise ValueError("class split has invalid target-old labels")
    for count in FORMAL_NEW_COUNTS:
        labels = nested_new[str(count)]
        if len(labels) != count or len(labels) != len(set(labels)):
            raise ValueError(f"class split has invalid new-{count} labels")
        if set(labels) & set(old_labels):
            raise ValueError("target-old and target-new labels overlap")
    if (
        nested_new["10"][:5] != nested_new["5"]
        or nested_new["20"][:10] != nested_new["10"]
    ):
        raise ValueError("new-class label lists must be ordered nested prefixes")
    direct_mapping_path = Path(
        str(split.get("direct_adv3b02_class_mapping_source", ""))
    )
    direct_mapping_sha256 = str(
        split.get("direct_adv3b02_class_mapping_sha256", "")
    )
    direct_class_labels = [
        str(value) for value in split.get("direct_adv3b02_class_id_to_tx", [])
    ]
    direct_mapping_payload = (
        json.loads(direct_mapping_path.read_text(encoding="utf-8-sig"))
        if direct_mapping_path.is_file()
        else {}
    )
    mapped_labels = [
        str(value)
        for value in direct_mapping_payload.get(
            "class_id_to_tx",
            direct_mapping_payload.get("direct_adv3b02_class_id_to_tx", []),
        )
    ]
    if (
        not direct_mapping_path.is_file()
        or _sha256_file(direct_mapping_path) != direct_mapping_sha256
        or direct_class_labels != old_labels
        or mapped_labels != direct_class_labels
    ):
        raise ValueError("class split has invalid strict ADV3B02 class mapping")
    execution_plan = validate_execution_manifest(
        json.loads(execution_plan_manifest.read_text(encoding="utf-8-sig"))
    )
    for raw_contract in execution_plan["target_cache_contracts"]:
        contract = dict(raw_contract)
        spec_path = Path(str(contract.get("cache_build_spec", "")))
        if (
            not spec_path.is_file()
            or _sha256_file(spec_path)
            != str(contract.get("cache_build_spec_sha256", ""))
        ):
            raise ValueError("generated target-cache build spec/hash drift")
    for raw_contract in execution_plan["stage2_config_contracts"]:
        contract = dict(raw_contract)
        config_path = Path(str(contract.get("config", "")))
        if (
            not config_path.is_file()
            or _sha256_file(config_path)
            != str(contract.get("config_file_sha256", ""))
        ):
            raise ValueError("generated Stage2 config file/hash drift")
        config_payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if _canonical_sha256(config_payload) != str(
            contract.get("config_content_sha256", "")
        ):
            raise ValueError("generated Stage2 config content/hash drift")
    code_hashes = {
        relative: _sha256_file(REPO_ROOT / relative) for relative in CODE_ARTIFACTS
    }
    locked_candidate = {
        "candidate_id": str(candidate_id),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256_file(checkpoint)},
        "adapter_state": {
            "path": str(adapter_state),
            "sha256": _sha256_file(adapter_state),
        },
        "promotion_manifest": {
            "path": str(promotion_manifest),
            "sha256": _sha256_file(promotion_manifest),
        },
        "source_validation": {
            "path": str(validation_path),
            "sha256": _sha256_file(validation_path),
        },
        "source_feature_statistics": {
            "path": str(stats_path),
            "sha256": _sha256_file(stats_path),
            "feature_kind": "normalized_z_id_plus_fft96_weight2",
        },
        "source_leo_weak_cache_sets": {
            "source_train": {
                "path": str(source_train_cache_path),
                "sha256": _sha256_file(source_train_cache_path),
            },
            "source_validation": {
                "path": str(source_validation_cache_path),
                "sha256": _sha256_file(source_validation_cache_path),
            },
            "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
        },
        "class_split": {
            "path": str(class_split_manifest),
            "sha256": _sha256_file(class_split_manifest),
            "target_old_tx_labels": old_labels,
            "nested_target_new_tx_labels": nested_new,
            "direct_adv3b02_class_mapping_source": str(direct_mapping_path),
            "direct_adv3b02_class_mapping_sha256": direct_mapping_sha256,
            "direct_adv3b02_class_id_to_tx": direct_class_labels,
        },
        "execution_plan": {
            "path": str(execution_plan_manifest),
            "sha256": _sha256_file(execution_plan_manifest),
            "expected_counts": dict(execution_plan["expected_counts"]),
            "formal_matrix_contract": dict(
                execution_plan["formal_matrix_contract"]
            ),
            "target_cache_contracts": list(
                execution_plan["target_cache_contracts"]
            ),
            "stage2_config_contracts": list(
                execution_plan["stage2_config_contracts"]
            ),
        },
        "feature_pipeline": {
            "primary": "ADV3B02_feat_joint",
            "auxiliary": "FFT_logmag_96",
            "auxiliary_weight": 2.0,
        },
        "head": {
            "mode": "symmetric_locked",
            "selected": _selected_head(dict(head_lock["selected"])),
            "selection_source": str(head_lock["selection_source"]),
            "support_view_policy": "three_leo_weak_scenario_base_views",
        },
        "adaptive_tta": {
            "policy": "lazy_adaptive_1to3to5",
            "thresholds": thresholds,
            "selection_source": "disjoint_source_receiver_holdout_only",
        },
        "formal_matrix": {
            "target_receivers": list(FORMAL_RECEIVERS),
            "k_values": list(FORMAL_K),
            "new_class_counts": list(FORMAL_NEW_COUNTS),
            "scenarios": [
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            ],
            "support_pool_max_k": 20,
            "query_per_tx": 20,
            "minimum_independent_confirmation_seeds": 5,
            "confirmation_seeds": list(FORMAL_CONFIRMATION_SEEDS),
        },
        "permissions": {
            "target_support_used_for_selection": False,
            "target_query_features_used_for_selection": False,
            "target_query_labels_used_for_selection": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
            "clean_samples_used": False,
            "clean_derived_signals_used": False,
            "query_fit_used": False,
        },
        "code_artifacts_sha256": code_hashes,
    }
    return {
        "schema": "cvs_stage2c_source_candidate_lock_v1",
        "locked_candidate": locked_candidate,
        "locked_candidate_sha256": _canonical_sha256(locked_candidate),
    }


def verify_candidate_lock(
    lock_path: Path,
    *,
    checkpoint: Path,
    adapter_state: Path,
    promotion_manifest: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    if lock.get("schema") != "cvs_stage2c_source_candidate_lock_v1":
        raise ValueError("candidate lock schema mismatch")
    candidate = dict(lock.get("locked_candidate", {}))
    if _canonical_sha256(candidate) != str(lock.get("locked_candidate_sha256", "")):
        raise ValueError("candidate lock payload hash mismatch")
    expected_files = {
        "checkpoint": checkpoint,
        "adapter_state": adapter_state,
        "promotion_manifest": promotion_manifest,
    }
    for field, path in expected_files.items():
        entry = dict(candidate.get(field, {}))
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise ValueError(f"candidate lock {field} hash mismatch")
    for field in (
        "source_validation",
        "source_feature_statistics",
        "class_split",
        "execution_plan",
    ):
        entry = dict(candidate.get(field, {}))
        artifact_path = Path(str(entry.get("path", "")))
        if (
            not artifact_path.is_file()
            or _sha256_file(artifact_path) != str(entry.get("sha256", ""))
        ):
            raise ValueError(f"candidate lock immutable artifact drift: {field}")
    locked_plan_entry = dict(candidate["execution_plan"])
    current_execution_plan = validate_execution_manifest(
        json.loads(
            Path(str(locked_plan_entry["path"])).read_text(encoding="utf-8-sig")
        )
    )
    if (
        dict(current_execution_plan["formal_matrix_contract"])
        != dict(locked_plan_entry.get("formal_matrix_contract", {}))
        or list(current_execution_plan["target_cache_contracts"])
        != list(locked_plan_entry.get("target_cache_contracts", []))
        or list(current_execution_plan["stage2_config_contracts"])
        != list(locked_plan_entry.get("stage2_config_contracts", []))
    ):
        raise ValueError("candidate lock execution-plan structure drift")
    for field, entry in dict(
        candidate.get("source_leo_weak_cache_sets", {})
    ).items():
        if field in {
            "phase2_sample_view_policy",
            "clean_sample_access",
            "clean_derived_signal_access",
        }:
            continue
        cache_entry = dict(entry)
        cache_path = Path(str(cache_entry.get("path", "")))
        if (
            not cache_path.is_file()
            or _sha256_file(cache_path) != str(cache_entry.get("sha256", ""))
        ):
            raise ValueError(f"candidate lock immutable cache drift: {field}")
    for relative, expected_hash in dict(
        candidate.get("code_artifacts_sha256", {})
    ).items():
        if _sha256_file(REPO_ROOT / relative) != str(expected_hash):
            raise ValueError(f"candidate lock code drift: {relative}")
    matrix = dict(candidate.get("formal_matrix", {}))
    receiver = [str(value) for value in config.get("target_receiver_labels", [])]
    k_shot = int(config.get("k_shot", -1))
    new_count = len(config.get("target_new_tx_labels", []))
    seed = int(config.get("seed", -1))
    if len(receiver) != 1 or receiver[0] not in matrix.get("target_receivers", []):
        raise ValueError("target receiver is outside the locked formal matrix")
    if k_shot not in [int(value) for value in matrix.get("k_values", [])]:
        raise ValueError("K is outside the locked formal matrix")
    if new_count not in [int(value) for value in matrix.get("new_class_counts", [])]:
        raise ValueError("new-class count is outside the locked formal matrix")
    if seed not in [int(value) for value in matrix.get("confirmation_seeds", [])]:
        raise ValueError("seed is outside the locked formal confirmation set")
    if int(config.get("support_pool_max_k", -1)) != int(
        matrix.get("support_pool_max_k", -2)
    ):
        raise ValueError("support_pool_max_k differs from the candidate lock")
    if int(config.get("query_per_tx", -1)) != int(matrix.get("query_per_tx", -2)):
        raise ValueError("query_per_tx differs from the candidate lock")
    matching_cache_contracts = [
        dict(value)
        for value in locked_plan_entry.get("target_cache_contracts", [])
        if str(value.get("receiver", "")) == receiver[0]
        and int(value.get("seed", -1)) == seed
    ]
    if len(matching_cache_contracts) != 1 or str(
        config.get("leo_weak_cache_set_manifest", "")
    ) != str(matching_cache_contracts[0].get("cache_set_manifest", "")):
        raise ValueError("target LEO_weak cache path differs from the receiver/seed lock")
    cache_spec_path = Path(
        str(matching_cache_contracts[0].get("cache_build_spec", ""))
    )
    if (
        not cache_spec_path.is_file()
        or _sha256_file(cache_spec_path)
        != str(matching_cache_contracts[0].get("cache_build_spec_sha256", ""))
    ):
        raise ValueError("target LEO_weak cache build-spec drift after candidate lock")
    matching_config_contracts = [
        dict(value)
        for value in locked_plan_entry.get("stage2_config_contracts", [])
        if str(value.get("receiver", "")) == receiver[0]
        and int(value.get("seed", -1)) == seed
        and int(value.get("new_class_count", -1)) == new_count
        and int(value.get("k_shot", -1)) == k_shot
    ]
    if len(matching_config_contracts) != 1 or _canonical_sha256(config) != str(
        matching_config_contracts[0].get("config_content_sha256", "")
    ):
        raise ValueError("Stage2 row config differs from the locked generated config")
    split = dict(candidate.get("class_split", {}))
    if [str(value) for value in config.get("target_old_tx_labels", [])] != [
        str(value) for value in split.get("target_old_tx_labels", [])
    ]:
        raise ValueError("target-old labels differ from the locked class split")
    expected_new = list(
        dict(split.get("nested_target_new_tx_labels", {})).get(str(new_count), [])
    )
    if [str(value) for value in config.get("target_new_tx_labels", [])] != [
        str(value) for value in expected_new
    ]:
        raise ValueError("target-new labels differ from the locked nested class split")
    for field in (
        "direct_adv3b02_class_mapping_source",
        "direct_adv3b02_class_mapping_sha256",
        "direct_adv3b02_class_id_to_tx",
    ):
        if config.get(field) != split.get(field):
            raise ValueError(f"strict ADV3B02 mapping differs from candidate lock: {field}")
    if dict(candidate.get("feature_pipeline", {})) != {
        "primary": "ADV3B02_feat_joint",
        "auxiliary": "FFT_logmag_96",
        "auxiliary_weight": 2.0,
    }:
        raise ValueError("candidate feature pipeline drift")
    required_protocol = {
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "old_new_role_oracle_used": False,
        "class_quota_used": False,
        "query_fit_used": False,
    }
    failed = [
        key
        for key, expected in required_protocol.items()
        if config.get(key) != expected
    ]
    if failed:
        raise ValueError(f"candidate lock formal protocol drift: {failed}")
    return lock


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter_state", type=Path, required=True)
    parser.add_argument("--promotion_manifest", type=Path, required=True)
    parser.add_argument("--class_split_manifest", type=Path, required=True)
    parser.add_argument("--execution_plan_manifest", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError(f"refusing to overwrite candidate lock: {args.out_json}")
    lock = build_candidate_lock(
        candidate_id=str(args.candidate_id),
        checkpoint=args.checkpoint,
        adapter_state=args.adapter_state,
        promotion_manifest=args.promotion_manifest,
        class_split_manifest=args.class_split_manifest,
        execution_plan_manifest=args.execution_plan_manifest,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
