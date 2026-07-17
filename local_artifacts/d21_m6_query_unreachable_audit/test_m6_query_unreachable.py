from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("audit_m6_query_unreachable.py")
SPEC = importlib.util.spec_from_file_location("m6_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def valid_manifest():
    return {
        "schema": AUDIT.SCHEMA,
        "package_role": AUDIT.PACKAGE_ROLE,
        "stage": "stage2c",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "seen_new_count": 5,
        "scenarios": ["leo_clear_weak"],
        "members": [
            {"kind": "sealed_feature_runtime", "relative_path": "runtime/sealed_feature_runtime.pt"},
            {
                "kind": "support_leo_weak",
                "relative_path": "support/leo_clear_weak.npz",
                "scenario": "leo_clear_weak",
                "npz_members": ["support_leo_weak_iq", "support_class_indices", "support_tokens"],
            },
            {"kind": "method_lock", "relative_path": "method_lock.json"},
        ],
        "calibration": {
            "selected_whitelist_id": "B_input_proj",
            "exact_layer_names": list(AUDIT.EXACT_WHITELISTS["B_input_proj"]),
            "updated_original_parameters": 22080,
            "epochs": 5,
            "optimizer_steps": 5,
            "optimizer": "SGD",
            "momentum": 0.0,
            "optimizer_state_persisted": False,
            "patch_dtype": "float16",
            "fp16_patch_bytes": 44160,
            "head_bytes": 28600,
        },
        "phase2_contract": {
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            **{field: False for field in AUDIT.REQUIRED_FALSE_FIELDS},
        },
    }


def test_valid_support_only_manifest_passes():
    assert AUDIT.validate_support_only_manifest(valid_manifest())["status"] == "PASS"


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("support_leo_weak", "query/leo_clear_weak.npz"),
        ("support_leo_weak", "truth_sidecar.json"),
        ("scorer", "score.json"),
        ("support_leo_weak", "apply_only_staging/sample.npz"),
    ],
)
def test_forbidden_evaluation_members_fail(kind, path):
    document = valid_manifest()
    document["members"].append({"kind": kind, "relative_path": path, "npz_members": ["support_leo_weak_iq", "support_class_indices"]})
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_support_only_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 6), ("optimizer_steps", 51), ("optimizer", "Adam"), ("momentum", 0.9), ("updated_original_parameters", 50000), ("optimizer_state_persisted", True)],
)
def test_resource_or_optimizer_violation_fails(field, value):
    document = valid_manifest()
    document["calibration"][field] = value
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_support_only_manifest(document)


def test_patch_plus_head_over_cap_fails():
    document = valid_manifest()
    document["calibration"]["fp16_patch_bytes"] = AUDIT.STATE_CAP_BYTES
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_support_only_manifest(document)


def test_fuzzy_or_unregistered_layer_fails():
    document = valid_manifest()
    document["calibration"]["exact_layer_names"][0] = "model.id_backbone.*.weight"
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_support_only_manifest(document)


@pytest.mark.parametrize("field", AUDIT.REQUIRED_FALSE_FIELDS)
def test_every_query_guard_must_be_explicit_false(field):
    document = valid_manifest()
    document["phase2_contract"][field] = True
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_support_only_manifest(document)


def test_runner_source_rejects_query_cli(tmp_path):
    source = tmp_path / "bad_runner.py"
    source.write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--support-package')\np.add_argument('--query-dir')\n", encoding="utf-8")
    with pytest.raises(AUDIT.AuditError):
        AUDIT.audit_runner_source(source)


def test_runner_source_rejects_broad_capsule_root(tmp_path):
    source = tmp_path / "bad_runner.py"
    source.write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--support-package')\np.add_argument('--capsule-root')\n", encoding="utf-8")
    with pytest.raises(AUDIT.AuditError):
        AUDIT.audit_runner_source(source)


def valid_enrollment_manifest():
    return {
        "schema": AUDIT.ENROLLMENT_SCHEMA,
        "profile": "enrollment_only",
        "registration_state": "after",
        "stage": "stage2c",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        **{field: False for field in (
            "clean_sample_access", "clean_derived_signal_access",
            "phase2_clean_dataset_reachable", "phase2_clean_cache_reachable",
            "phase2_clean_control_flow_reachable", "phase2_source_sample_access",
            "phase2_source_cache_access", "phase2_source_label_access",
            "phase2_source_derived_signal_access", "phase2_source_replay",
            "phase2_external_source_adapter_access", "phase2_query_role_oracle_access",
            "phase2_query_true_batch_class_count_access", "phase2_query_class_quota_access",
            "phase2_query_batch_global_assignment", "phase2_query_post_reception_view_fit_access",
        )},
        "members": [],
    }


def _complete_enrollment():
    document = valid_enrollment_manifest()
    document["members"] = [
        {"kind": "feature_runtime", "relative_path": "sealed_feature_runtime.pt"},
        {"kind": "method_lock", "relative_path": "method_lock.json"},
        {"kind": "overlay_provenance", "relative_path": "overlay_provenance.json"},
        {"kind": "support:leo_clear_weak", "relative_path": "support_leo_clear_weak.npz"},
        {"kind": "support:leo_low_elev_weak", "relative_path": "support_leo_low_elev_weak.npz"},
        {"kind": "support:leo_rain_weak", "relative_path": "support_leo_rain_weak.npz"},
    ]
    return document


def test_existing_enrollment_schema_exact_allowlist_passes():
    assert AUDIT.validate_enrollment_manifest(_complete_enrollment())["status"] == "PASS"


@pytest.mark.parametrize(
    ("kind", "path"),
    [("query", "query_leo_clear_weak.npz"), ("truth", "truth_sidecar.json"), ("scorer", "scorer.json")],
)
def test_existing_enrollment_schema_rejects_extra_evaluation_member(kind, path):
    document = _complete_enrollment()
    document["members"].append({"kind": kind, "relative_path": path})
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_enrollment_manifest(document)


def test_existing_enrollment_schema_rejects_path_traversal():
    document = _complete_enrollment()
    document["members"][3]["relative_path"] = "../support_leo_clear_weak.npz"
    with pytest.raises(AUDIT.AuditError):
        AUDIT.validate_enrollment_manifest(document)


def _load_actual_m6_runner():
    path = Path(__file__).parents[1] / "d21_m6_support_fold_lowrank" / "run_m6_support_fold_lowrank.py"
    spec = importlib.util.spec_from_file_location("actual_m6_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def test_actual_m6_runner_static_query_unreachable_audit_passes():
    path, _ = _load_actual_m6_runner()
    assert AUDIT.audit_runner_source(path)["status"] == "PASS"


@pytest.mark.parametrize(
    ("kind", "path"),
    [("query", "query.npz"), ("truth", "truth_sidecar.json"), ("scorer", "scorer.json")],
)
def test_actual_m6_member_guard_rejects_extra_evaluation_member(kind, path):
    _, runner = _load_actual_m6_runner()
    document = _complete_enrollment()
    document["members"].append({"kind": kind, "relative_path": path})
    with pytest.raises(RuntimeError, match="allowlist mismatch"):
        runner._member_map(document)
