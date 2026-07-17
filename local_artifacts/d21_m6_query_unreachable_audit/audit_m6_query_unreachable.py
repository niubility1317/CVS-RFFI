from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "cvs.stage2c.m6_support_only_crop.v1"
ENROLLMENT_SCHEMA = "cvs.phase2.somph_predictor_bundle.v1"
PACKAGE_ROLE = "stage2c_support_only_calibration"
STATE_CAP_BYTES = 256 * 1024
FORBIDDEN_TOKENS = ("query", "truth", "scorer", "apply_only_staging")
ALLOWED_MEMBER_KINDS = {
    "sealed_feature_runtime",
    "support_leo_weak",
    "method_lock",
    "overlay_provenance",
    "base_checkpoint",
}
EXACT_WHITELISTS = {
    "A_tail_idproj": (
        "model.id_backbone.cls_head.id_proj.0.weight",
        "model.id_backbone.cls_head.id_proj.0.bias",
    ),
    "B_input_proj": (
        "model.id_backbone.t_proj.weight",
        "model.id_backbone.t_proj.bias",
        "model.id_backbone.f_proj.weight",
        "model.id_backbone.f_proj.bias",
        "model.id_backbone.freq_stats_proj.0.weight",
        "model.id_backbone.freq_stats_proj.0.bias",
        "model.id_backbone.pa_stats_proj.0.weight",
        "model.id_backbone.pa_stats_proj.0.bias",
    ),
    "C_tail_gate": (
        "model.id_backbone.cls_head.id_gate.0.weight",
        "model.id_backbone.cls_head.id_gate.0.bias",
    ),
}
REQUIRED_FALSE_FIELDS = (
    "query_access",
    "query_fit",
    "query_truth_opened",
    "query_role_oracle_access",
    "query_true_batch_class_count_access",
    "query_class_quota_access",
    "query_batch_global_assignment",
)


class AuditError(ValueError):
    pass


def _reject(condition: bool, message: str) -> None:
    if condition:
        raise AuditError(message)


def _safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and not os.path.isabs(value)


def validate_support_only_manifest(document: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata only. No package member is opened by this function."""
    _reject(document.get("schema") != SCHEMA, "schema must be the M6 support-only crop schema")
    _reject(document.get("package_role") != PACKAGE_ROLE, "package_role must be support-only calibration")
    allowed_top = {
        "schema", "package_role", "stage", "receiver", "seed", "k_shot",
        "seen_new_count", "scenarios", "members", "calibration", "phase2_contract",
        "package_sha256", "runtime_sha256",
    }
    unexpected = sorted(set(document) - allowed_top)
    _reject(bool(unexpected), f"unexpected top-level keys: {unexpected}")
    _reject(document.get("stage") != "stage2c", "only Stage2-C is allowed")
    members = document.get("members")
    _reject(not isinstance(members, list) or not members, "members must be a non-empty list")
    support_count = 0
    for index, member in enumerate(members):
        _reject(not isinstance(member, dict), f"member[{index}] must be an object")
        _reject(set(member) - {"kind", "relative_path", "sha256", "size_bytes", "scenario", "npz_members"}, f"member[{index}] has unexpected keys")
        kind = str(member.get("kind", ""))
        relative = str(member.get("relative_path", ""))
        _reject(kind not in ALLOWED_MEMBER_KINDS, f"member[{index}] kind is not support-only allowlisted")
        _reject(not _safe_relative_path(relative), f"member[{index}] path must be relative and traversal-free")
        joined = f"{kind}|{relative}|{'|'.join(map(str, member.get('npz_members', [])))}".lower()
        _reject(any(token in joined for token in FORBIDDEN_TOKENS), f"member[{index}] exposes forbidden evaluation material")
        if kind == "support_leo_weak":
            support_count += 1
            npz_members = set(member.get("npz_members", []))
            _reject("support_leo_weak_iq" not in npz_members, "support member lacks received LEO_weak IQ")
            _reject("support_class_indices" not in npz_members, "support member lacks registered support labels")
    _reject(support_count < 1, "at least one support_leo_weak member is required")

    contract = document.get("phase2_contract")
    _reject(not isinstance(contract, dict), "phase2_contract is required")
    for field in REQUIRED_FALSE_FIELDS:
        _reject(contract.get(field) is not False, f"{field} must be explicitly false")
    _reject(contract.get("phase2_sample_view_policy") != "leo_weak_only_no_clean_access", "LEO_weak-only policy missing")
    for field in ("clean_sample_access", "clean_derived_signal_access", "phase2_clean_dataset_reachable", "phase2_clean_cache_reachable"):
        _reject(contract.get(field) is not False, f"{field} must be explicitly false")

    calibration = document.get("calibration")
    _reject(not isinstance(calibration, dict), "calibration resource lock is required")
    whitelist_id = calibration.get("selected_whitelist_id")
    _reject(whitelist_id not in EXACT_WHITELISTS, "selected whitelist id is not preregistered")
    exact_names = tuple(calibration.get("exact_layer_names", ()))
    _reject(exact_names != EXACT_WHITELISTS[whitelist_id], "exact layer names do not match preregistered whitelist")
    _reject(any("*" in name or "?" in name or "[" in name for name in exact_names), "fuzzy layer expressions are forbidden")
    updated = calibration.get("updated_original_parameters")
    _reject(not isinstance(updated, int) or updated <= 0 or updated >= 50000, "updated original parameters must be 1..49,999")
    epochs = calibration.get("epochs")
    steps = calibration.get("optimizer_steps")
    _reject(not isinstance(epochs, int) or epochs < 1 or epochs > 5, "epochs must be 1..5")
    _reject(not isinstance(steps, int) or steps < 1 or steps > 50, "optimizer_steps must be 1..50")
    _reject(str(calibration.get("optimizer")) != "SGD", "optimizer must be SGD")
    _reject(float(calibration.get("momentum", -1.0)) != 0.0, "SGD momentum must be zero")
    _reject(calibration.get("optimizer_state_persisted") is not False, "optimizer state must not persist")
    _reject(calibration.get("patch_dtype") != "float16", "delta patch must be float16")
    patch_bytes = calibration.get("fp16_patch_bytes")
    head_bytes = calibration.get("head_bytes")
    _reject(not isinstance(patch_bytes, int) or patch_bytes < 0, "invalid patch bytes")
    _reject(not isinstance(head_bytes, int) or head_bytes < 0, "invalid head bytes")
    _reject(patch_bytes + head_bytes > STATE_CAP_BYTES, "patch plus head exceeds 256KB")
    return {
        "status": "PASS",
        "schema": SCHEMA,
        "member_count": len(members),
        "support_member_count": support_count,
        "whitelist_id": whitelist_id,
        "updated_original_parameters": updated,
        "epochs": epochs,
        "optimizer_steps": steps,
        "patch_plus_head_bytes": patch_bytes + head_bytes,
        "query_content_opened": False,
    }


def validate_enrollment_manifest(document: dict[str, Any]) -> dict[str, Any]:
    """Validate the existing sealed after/enrollment_only manifest without opening members."""
    _reject(document.get("schema") != ENROLLMENT_SCHEMA, "unexpected enrollment manifest schema")
    _reject(document.get("profile") != "enrollment_only", "profile must be enrollment_only")
    _reject(document.get("registration_state") != "after", "registration state must be after")
    _reject(document.get("stage") != "stage2c", "stage must be stage2c")
    _reject(document.get("phase2_sample_view_policy") != "leo_weak_only_no_clean_access", "LEO_weak-only policy missing")
    required_false = (
        "clean_sample_access", "clean_derived_signal_access",
        "phase2_clean_dataset_reachable", "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable", "phase2_source_sample_access",
        "phase2_source_cache_access", "phase2_source_label_access",
        "phase2_source_derived_signal_access", "phase2_source_replay",
        "phase2_external_source_adapter_access", "phase2_query_role_oracle_access",
        "phase2_query_true_batch_class_count_access", "phase2_query_class_quota_access",
        "phase2_query_batch_global_assignment", "phase2_query_post_reception_view_fit_access",
    )
    for field in required_false:
        _reject(document.get(field) is not False, f"{field} must be explicitly false")
    _reject(document.get("phase2_query_decision_policy") != "per_sample_all_registered_classes", "per-sample decision lock missing")
    expected = {
        ("feature_runtime", "sealed_feature_runtime.pt"),
        ("method_lock", "method_lock.json"),
        ("overlay_provenance", "overlay_provenance.json"),
        ("support:leo_clear_weak", "support_leo_clear_weak.npz"),
        ("support:leo_low_elev_weak", "support_leo_low_elev_weak.npz"),
        ("support:leo_rain_weak", "support_leo_rain_weak.npz"),
    }
    members = document.get("members")
    _reject(not isinstance(members, list), "members must be a list")
    observed: set[tuple[str, str]] = set()
    for index, member in enumerate(members):
        _reject(not isinstance(member, dict), f"member[{index}] must be an object")
        kind = str(member.get("kind", ""))
        relative = str(member.get("relative_path", ""))
        _reject(not _safe_relative_path(relative), f"member[{index}] has unsafe path")
        joined = f"{kind}|{relative}|{'|'.join(map(str, member.get('npz_members', [])))}".lower()
        _reject(any(token in joined for token in FORBIDDEN_TOKENS), f"member[{index}] exposes forbidden evaluation material")
        observed.add((kind, relative))
    _reject(observed != expected or len(members) != len(expected), "enrollment members must equal the exact six-member support-only allowlist")
    return {
        "status": "PASS",
        "schema": ENROLLMENT_SCHEMA,
        "member_count": len(members),
        "exact_support_only_allowlist": True,
        "query_content_opened": False,
    }


def validate_resource_audit(document: dict[str, Any]) -> dict[str, Any]:
    exact = tuple(document.get("exact_model_parameter_whitelist", ()))
    _reject(set(exact) != set(EXACT_WHITELISTS["A_tail_idproj"]), "M6 exact original-layer whitelist drift")
    _reject(document.get("updated_original_parameters_after_merge") != 25760, "updated original parameter count must be 25,760")
    _reject(int(document.get("adaptation_epochs_per_fold", 999)) > 5, "epoch cap exceeded")
    _reject(int(document.get("adaptation_steps_per_fold", 999)) > 50, "step cap exceeded")
    optimizer = str(document.get("optimizer", ""))
    _reject("SGD" not in optimizer or "momentum=0" not in optimizer.replace(".0", ""), "optimizer must be SGD with zero momentum")
    _reject(document.get("optimizer_state_persisted") is not False, "optimizer state persisted")
    patch_bytes = document.get("selected_fp16_factor_patch_bytes_upper_bound")
    head_bytes = document.get("int8_registered_head_bytes_pre_registered_upper_bound")
    _reject(not isinstance(patch_bytes, int) or not isinstance(head_bytes, int), "patch/head bytes must be explicit integers")
    _reject(patch_bytes + head_bytes > STATE_CAP_BYTES, "patch plus head exceeds 256KB")
    _reject(document.get("merged_inference_added_MAC") != 0, "merged inference must add zero MAC")
    return {"status": "PASS", "updated_original_parameters": 25760, "patch_plus_head_bytes": patch_bytes + head_bytes, "query_content_opened": False}


def validate_query_unreachable_proof(document: dict[str, Any]) -> dict[str, Any]:
    required_false = (
        "query_access", "query_fit", "query_truth_opened", "query_iq_access",
        "query_token_access", "truth_sidecar_access", "score_operation_available",
        "prediction_artifact_emitted", "query_calibration", "query_selection",
        "query_early_stop", "query_rollback", "query_candidate_ranking",
        "phase2_query_role_oracle_access", "phase2_query_true_batch_class_count_access",
        "phase2_query_class_quota_access", "phase2_query_batch_global_assignment",
    )
    for field in required_false:
        _reject(document.get(field) is not False, f"proof field {field} must be false")
    _reject(document.get("observed_equals_allowlist") is not True, "observed access set escaped allowlist")
    _reject(document.get("input_manifest_member_allowlist_exact") is not True, "manifest member allowlist was not exact")
    _reject(document.get("input_manifest_extra_member_rejected") is not True, "extra-member rejection not evidenced")
    return {"status": "PASS", "checked_false_fields": len(required_false), "query_content_opened": False}


def audit_runner_source(path: Path) -> dict[str, Any]:
    """Static audit only; never imports or executes the M6 runner."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    options: list[str] = []
    literal_open_targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    options.append(argument.value)
        if isinstance(node, ast.Call):
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name in {"open", "load", "read_text", "read_bytes"}:
                for argument in node.args[:1]:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        literal_open_targets.append(argument.value)
    forbidden_options = [option for option in options if any(token in option.lower() for token in FORBIDDEN_TOKENS)]
    broad_options = [option for option in options if option in {"--input-dir", "--capsule-root", "--predictor-root", "--scorer-root"}]
    forbidden_literals = [value for value in literal_open_targets if any(token in value.lower() for token in FORBIDDEN_TOKENS)]
    _reject(bool(forbidden_options), f"runner exposes forbidden CLI options: {forbidden_options}")
    _reject(bool(broad_options), f"runner exposes broad package roots: {broad_options}")
    _reject(bool(forbidden_literals), f"runner contains literal forbidden open targets: {forbidden_literals}")
    _reject("--support-package" not in options and "--support-manifest" not in options and "--enrollment-root" not in options, "runner must accept only a cropped support package/manifest entry")
    required_markers = (
        "query_access", "query_fit", "query_truth_opened", "query_role_oracle_access",
        "query_true_batch_class_count_access", "query_class_quota_access",
        "query_batch_global_assignment", "exact_model_parameter_whitelist",
        "adaptation_steps_per_fold",
        "selected_fp16_factor_patch_bytes_upper_bound",
        "int8_registered_head_bytes_pre_registered_upper_bound", "forbidden", "members",
    )
    missing_markers = [marker for marker in required_markers if marker not in source]
    _reject(bool(missing_markers), f"runner lacks fail-closed schema markers: {missing_markers}")
    return {
        "status": "PASS",
        "runner": str(path.resolve()),
        "cli_options": options,
        "forbidden_cli_options": [],
        "broad_root_options": [],
        "forbidden_literal_open_targets": [],
        "query_content_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--runner")
    parser.add_argument("--resource-audit")
    parser.add_argument("--query-proof")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.support_manifest).resolve()
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_result = validate_enrollment_manifest(manifest_document) if manifest_document.get("schema") == ENROLLMENT_SCHEMA else validate_support_only_manifest(manifest_document)
    result = {"manifest": manifest_result, "runner": None, "resource": None, "proof": None, "query_content_opened": False}
    if args.runner:
        result["runner"] = audit_runner_source(Path(args.runner).resolve())
    if args.resource_audit:
        result["resource"] = validate_resource_audit(json.loads(Path(args.resource_audit).resolve().read_text(encoding="utf-8")))
    if args.query_proof:
        result["proof"] = validate_query_unreachable_proof(json.loads(Path(args.query_proof).resolve().read_text(encoding="utf-8")))
    Path(args.output).resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
