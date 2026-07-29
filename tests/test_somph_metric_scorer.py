from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import somph_metric_scorer as scorer
from cvsrffi.stage2_metric_scorer import (
    canonical_json_bytes,
    sha256_file,
)
from cvsrffi.somph_prediction_artifact import publish_somph_prediction_artifact


SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
OLD = list(scorer.FORMAL_OLD_TX_LABELS)
NEW = list(scorer.FORMAL_NEW20_TX_LABELS[:5])


def _qid(value: str) -> str:
    return f"qid_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _cls(value: str) -> str:
    return f"cls_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _truth_rows(*, stage: str, query_per_tx: int = 20) -> list[dict]:
    labels = OLD if stage == "stage2b" else [*OLD, *NEW]
    rows = []
    for class_index, label in enumerate(labels):
        role = "target_old" if label in OLD else "target_new"
        for rank in range(query_per_tx):
            rows.append({
                "query_token": _qid(f"{label}-{rank}"),
                "true_class_index": class_index,
                "true_class_handle": _cls(label),
                "transmitter_label": label,
                "evaluation_role": role,
                "receiver_label": "8-8",
                "physical_sample_id": f"sample-{label}-{rank}",
            })
    return rows


def _binding(
    *,
    state: str,
    stage: str,
    new_count: int,
    protocol_policy_sha256: str,
    resource_sha256: str,
) -> dict:
    labels = OLD
    if stage == "stage2c" and state == "after_registration":
        labels = [*OLD, *NEW]
    registry_snapshot_sha256 = hashlib.sha256(
        canonical_json_bytes([_cls(label) for label in labels])
    ).hexdigest()
    return {
        "stage": "Stage2-B" if stage == "stage2b" else "Stage2-C",
        "registration_state": state,
        "row_id": f"row_{hashlib.sha256(b'formal-row').hexdigest()}",
        "receiver": "8-8",
        "seed": 713106,
        "k_shot": 5,
        "registered_class_count": len(labels),
        "registry_snapshot_sha256": registry_snapshot_sha256,
        "method_lock_sha256": "1" * 64,
        "row_manifest_sha256": "2" * 64,
        "stage_input_binding_sha256": ("3" if state == "before_registration" else "4") * 64,
        "package_root_sha256": ("5" if state == "before_registration" else "6") * 64,
        "package_seal_sha256": ("7" if state == "before_registration" else "8") * 64,
        "feature_runtime_sha256": "9" * 64,
        "head_capsule_sha256": ("a" if state == "before_registration" else "b") * 64,
        "protocol_policy_sha256": protocol_policy_sha256,
    }


def _publish(
    path: Path,
    *,
    state: str,
    stage: str,
    rows: list[dict],
    token_rows: list[dict],
    protocol_policy_sha256: str,
    resource_sha256: str,
    override_predictions: dict[str, str] | None = None,
) -> dict:
    tokens: list[str] = []
    scenarios: list[str] = []
    predictions: list[str] = []
    truth_by_token = {row["query_token"]: row for row in rows}
    for scenario in SCENARIOS:
        for row in token_rows:
            token = row["query_token"]
            tokens.append(token)
            scenarios.append(scenario)
            predictions.append(
                (override_predictions or {}).get(token, truth_by_token[token]["true_class_handle"])
            )
    return publish_somph_prediction_artifact(
        path,
        query_tokens=np.asarray(tokens),
        scenarios=np.asarray(scenarios),
        predicted_class_handles=np.asarray(predictions),
        backbone_forward_counts=np.ones(len(tokens), dtype=np.uint8),
        **_binding(
            state=state,
            stage=stage,
            new_count=0 if stage == "stage2b" else 5,
            protocol_policy_sha256=protocol_policy_sha256,
            resource_sha256=resource_sha256,
        ),
    )


def _case(
    root: Path,
    *,
    stage: str = "stage2c",
    query_per_tx: int = 20,
    preopen_clean_access: bool = False,
    pair_query_mismatch: bool = False,
) -> dict:
    root.mkdir()
    rows = _truth_rows(stage=stage, query_per_tx=query_per_tx)
    old_rows = [row for row in rows if row["evaluation_role"] == "target_old"]
    policy_path = root / "protocol_policy.json"
    _write_json(policy_path, dict(scorer._PHASE2_CONTRACT))
    protocol_policy_sha = sha256_file(policy_path)
    states = ["before_registration"]
    if stage == "stage2c":
        states.append("after_registration")
    state_evidence: dict[str, dict] = {}
    resource_hashes: dict[str, str] = {}
    for state in states:
        artifact_stage = (
            "stage2b"
            if stage == "stage2c" and state == "before_registration"
            else stage
        )
        placeholder = _binding(
            state=state,
            stage=artifact_stage,
            new_count=0 if stage == "stage2b" else 5,
            protocol_policy_sha256=protocol_policy_sha,
            resource_sha256="0" * 64,
        )
        resource_path = root / f"{state}_resource_audit.json"
        _write_json(resource_path, {
            "schema": scorer.RESOURCE_AUDIT_SCHEMA,
            "status": "PASS",
            "head_capsule_sha256": placeholder["head_capsule_sha256"],
            "trainable_parameters": 0,
            "updated_original_parameters": 0,
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "optimizer_state_bytes": 0,
            "optimizer_state_deployment_required": False,
            "query_rows_used_for_fit": 0,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "candidate_state_bytes_fp16": 25_440,
            "active_scenario_state_bytes_fp16": 25_440,
            "candidate_state_cap_bytes": 262_144,
            "candidate_extra_macs_per_query": 13_142,
            "capsule_array_bytes_including_registry_and_audit": 78_492,
            "base_checkpoint_state_bytes": 1_000_000,
            "base_backbone_macs_per_forward": 2_000_000,
            "total_deployment_state_bytes": 1_025_440,
            "total_macs_per_query": 2_013_142,
        })
        resource_sha = sha256_file(resource_path)
        resource_hashes[state] = resource_sha
        preopen_path = root / f"{state}_preopen_audit.json"
        _write_json(preopen_path, {
            "schema": scorer.PREOPEN_AUDIT_SCHEMA,
            "status": "PASS",
            "package_root_sha256": placeholder["package_root_sha256"],
            "package_seal_sha256": placeholder["package_seal_sha256"],
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "all_members_preoverlaid_leo_weak": True,
            "clean_sample_access": preopen_clean_access,
            "clean_derived_signal_access": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "leo_weak_member_sha256_by_scenario": {
                scenario: hashlib.sha256(
                    f"{state}-{scenario}".encode("utf-8")
                ).hexdigest()
                for scenario in SCENARIOS
            },
        })
        runtime_path = root / f"{state}_runtime_access_audit.json"
        _write_json(runtime_path, {
            "schema": scorer.RUNTIME_ACCESS_AUDIT_SCHEMA,
            "status": "PASS",
            "opened_input_paths": [
                f"E:/sealed/{state}/leo_clear_weak.npz",
                f"E:/sealed/{state}/leo_low_elev_weak.npz",
                f"E:/sealed/{state}/leo_rain_weak.npz",
            ],
            "allowed_input_roots": [f"E:/sealed/{state}"],
            "forbidden_open_count": 0,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "truth_sidecar_access": False,
            "scoring_manifest_access": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
        })
        state_evidence[state] = {
            "method_lock_sha256": placeholder["method_lock_sha256"],
            "row_manifest_sha256": placeholder["row_manifest_sha256"],
            "stage_input_binding_sha256": placeholder["stage_input_binding_sha256"],
            "package_root_sha256": placeholder["package_root_sha256"],
            "package_seal_sha256": placeholder["package_seal_sha256"],
            "feature_runtime_sha256": placeholder["feature_runtime_sha256"],
            "head_capsule_sha256": placeholder["head_capsule_sha256"],
            "leo_weak_member_sha256_by_scenario": {
                scenario: hashlib.sha256(
                    f"{state}-{scenario}".encode("utf-8")
                ).hexdigest()
                for scenario in SCENARIOS
            },
            "resource_audit_json": resource_path.name,
            "resource_audit_sha256": resource_sha,
            "preopen_audit_json": preopen_path.name,
            "preopen_audit_sha256": sha256_file(preopen_path),
            "runtime_access_audit_json": runtime_path.name,
            "runtime_access_audit_sha256": sha256_file(runtime_path),
        }
    pair_path = None
    pair_sha = None
    if stage == "stage2c":
        old_physical_digest = hashlib.sha256(
            canonical_json_bytes(
                [
                    row["physical_sample_id"]
                    for row in sorted(old_rows, key=lambda row: row["query_token"])
                ]
            )
        ).hexdigest()
        pair_path = root / "registration_pair.json"
        _write_json(pair_path, {
            "schema": scorer.REGISTRATION_PAIR_SCHEMA,
            "row_manifest_sha256": "2" * 64,
            "before_binding_sha256": "3" * 64,
            "after_binding_sha256": "4" * 64,
            "old_support_physical_ids_sha256_before": "a" * 64,
            "old_support_physical_ids_sha256_after": "a" * 64,
            "old_query_physical_ids_sha256_before": old_physical_digest,
            "old_query_physical_ids_sha256_after": (
                "c" * 64 if pair_query_mismatch else old_physical_digest
            ),
        })
        pair_sha = sha256_file(pair_path)
    before = _publish(
        root / "before.cvspred",
        state="before_registration",
        stage="stage2b" if stage == "stage2c" else stage,
        rows=rows,
        token_rows=old_rows,
        protocol_policy_sha256=protocol_policy_sha,
        resource_sha256=resource_hashes["before_registration"],
    )
    after = None
    if stage == "stage2c":
        after = _publish(
            root / "after.cvspred",
            state="after_registration",
            stage=stage,
            rows=rows,
            token_rows=rows,
            protocol_policy_sha256=protocol_policy_sha,
            resource_sha256=resource_hashes["after_registration"],
        )
    artifacts = {
        "before_registration": before,
        **({"after_registration": after} if after is not None else {}),
    }
    for state, publication in artifacts.items():
        state_evidence[state]["prediction_artifact_sha256"] = publication[
            "artifact_sha256"
        ]
        state_evidence[state]["prediction_seal_sha256"] = publication["seal_sha256"]
    evidence_path = root / "evidence_manifest.json"
    _write_json(evidence_path, {
        "schema": scorer.EVIDENCE_MANIFEST_SCHEMA,
        "stage": stage,
        "protocol_policy_json": policy_path.name,
        "protocol_policy_sha256": protocol_policy_sha,
        "scenarios": list(SCENARIOS),
        "satellite_seed_by_scenario": {
            scenario: 713100 + index for index, scenario in enumerate(SCENARIOS)
        },
        "state_evidence": state_evidence,
        "registration_pair_json": pair_path.name if pair_path is not None else None,
        "registration_pair_sha256": pair_sha,
    })
    evidence_sha = sha256_file(evidence_path)
    truth_path = root / "truth_sidecar.json"
    _write_json(truth_path, {
        "schema": scorer.SOMPH_TRUTH_SIDECAR_SCHEMA,
        "stage": stage,
        "receiver": "8-8",
        "seed": 713106,
        "rows": rows,
    })
    manifest_path = root / "scoring_manifest.json"
    _write_json(manifest_path, {
        "schema": scorer.SCORING_MANIFEST_SCHEMA,
        "stage": stage,
        "receiver": "8-8",
        "seed": 713106,
        "k_shot": 5,
        "new_class_count": 0 if stage == "stage2b" else 5,
        "expected_query_per_tx": query_per_tx,
        "scenarios": list(SCENARIOS),
        "old_tx_labels": OLD,
        "new_tx_labels": [] if stage == "stage2b" else NEW,
        "truth_sidecar_json": truth_path.name,
        "truth_sidecar_sha256": sha256_file(truth_path),
        "evidence_manifest_json": evidence_path.name,
        "evidence_manifest_sha256": evidence_sha,
        "scorer_output_must_not_feed_predictor": True,
    })
    return {
        "root": root,
        "rows": rows,
        "before": before,
        "after": after,
        "protocol_policy_sha256": protocol_policy_sha,
        "evidence_sha256": evidence_sha,
        "evidence": evidence_path,
        "resource_hashes": resource_hashes,
        "truth": truth_path,
        "manifest": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _refresh_postrun_evidence(case: dict) -> None:
    evidence_path = Path(case["evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for state, key in (
        ("before_registration", "before"),
        ("after_registration", "after"),
    ):
        publication = case.get(key)
        if publication is None:
            continue
        evidence["state_evidence"][state]["prediction_artifact_sha256"] = publication[
            "artifact_sha256"
        ]
        evidence["state_evidence"][state]["prediction_seal_sha256"] = publication[
            "seal_sha256"
        ]
    _write_json(evidence_path, evidence)
    case["evidence_sha256"] = sha256_file(evidence_path)
    manifest_path = Path(case["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_manifest_sha256"] = case["evidence_sha256"]
    _write_json(manifest_path, manifest)
    case["manifest_sha256"] = sha256_file(manifest_path)


def _score_pair(case: dict):
    return scorer.score_somph_registration_pair(
        case["before"]["path"],
        case["after"]["path"],
        case["manifest"],
        expected_before_artifact_sha256=case["before"]["artifact_sha256"],
        expected_before_seal_sha256=case["before"]["seal_sha256"],
        expected_after_artifact_sha256=case["after"]["artifact_sha256"],
        expected_after_seal_sha256=case["after"]["seal_sha256"],
        expected_scoring_manifest_sha256=case["manifest_sha256"],
    )


def test_pair_scorer_reports_before_after_per_class_and_confusion(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    rows_payload, predictions_payload, receipt = _score_pair(case)
    assert len(rows_payload["rows"]) == 3
    row = rows_payload["rows"][0]
    assert row["old_acc_before_increment"] == 1.0
    assert row["old_acc_after_increment"] == 1.0
    assert row["average_forgetting"] == 0.0
    assert row["seen_new_acc"] == 1.0
    assert row["H_old_new"] == 1.0
    assert row["min_old_class_acc_after"] == 1.0
    assert row["min_seen_new_class_acc"] == 1.0
    assert set(row["old_class_acc_after"]) == set(OLD)
    assert set(row["seen_new_class_acc"]) == set(NEW)
    assert row["old_to_new_count"] == row["new_to_old_count"] == 0
    assert set(row["before_old_confusion_matrix_counts"]) == set(OLD)
    assert set(row["after_all_confusion_matrix_counts"]) == set([*OLD, *NEW])
    assert sum(
        sum(values.values())
        for values in row["after_all_confusion_matrix_counts"].values()
    ) == 220
    assert len(predictions_payload["predictions"]) == 3 * (120 + 220)
    assert receipt["truth_join_after_all_predictions_verified"] is True
    assert receipt["scorer_output_must_not_feed_predictor"] is True
    assert receipt["phase2_protocol_evidence_status"] == (
        "STRUCTURAL_ONLY_REAL_INPUT_RECOMPUTE_REQUIRED"
    )
    assert receipt["status"] == "LOCAL_PROTOCOL_REPAIR_REQUIRED"
    assert receipt["formal_launch_authority"] is False
    assert receipt["formal_metric_claim_allowed"] is False


def test_stage2b_is_old_only_and_has_no_seen_new_metric(tmp_path: Path) -> None:
    case = _case(tmp_path / "case", stage="stage2b")
    rows, predictions, receipt = scorer.score_somph_stage2b(
        case["before"]["path"],
        case["manifest"],
        expected_prediction_artifact_sha256=case["before"]["artifact_sha256"],
        expected_prediction_seal_sha256=case["before"]["seal_sha256"],
        expected_scoring_manifest_sha256=case["manifest_sha256"],
    )
    assert all(row["seen_new_acc"] is None for row in rows["rows"])
    assert all(item["evaluation_role"] == "target_old" for item in predictions["predictions"])
    assert receipt["new_class_count"] == 0


def test_pair_rejects_unregistered_prediction_handle(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    after_path = Path(case["after"]["path"])
    os.chmod(after_path, 0o600)
    after_path.unlink()
    rows = case["rows"]
    replacement = _publish(
        after_path,
        state="after_registration",
        stage="stage2c",
        rows=rows,
        token_rows=rows,
        protocol_policy_sha256=case["protocol_policy_sha256"],
        resource_sha256=case["resource_hashes"]["after_registration"],
        override_predictions={rows[0]["query_token"]: _cls("unregistered")},
    )
    case["after"] = replacement
    _refresh_postrun_evidence(case)
    with pytest.raises(scorer.SomphScoringError, match="unregistered class handle"):
        _score_pair(case)


def test_pair_rejects_before_query_set_drift(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    before_path = Path(case["before"]["path"])
    os.chmod(before_path, 0o600)
    before_path.unlink()
    old_rows = [row for row in case["rows"] if row["evaluation_role"] == "target_old"][:-1]
    case["before"] = _publish(
        before_path,
        state="before_registration",
        stage="stage2b",
        rows=case["rows"],
        token_rows=old_rows,
        protocol_policy_sha256=case["protocol_policy_sha256"],
        resource_sha256=case["resource_hashes"]["before_registration"],
    )
    _refresh_postrun_evidence(case)
    with pytest.raises(scorer.SomphScoringError, match="matched old queries only"):
        _score_pair(case)


def test_before_artifact_cannot_predict_a_new_registration_handle(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    before_path = Path(case["before"]["path"])
    os.chmod(before_path, 0o600)
    before_path.unlink()
    old_rows = [row for row in case["rows"] if row["evaluation_role"] == "target_old"]
    case["before"] = _publish(
        before_path,
        state="before_registration",
        stage="stage2b",
        rows=case["rows"],
        token_rows=old_rows,
        protocol_policy_sha256=case["protocol_policy_sha256"],
        resource_sha256=case["resource_hashes"]["before_registration"],
        override_predictions={old_rows[0]["query_token"]: _cls(NEW[0])},
    )
    _refresh_postrun_evidence(case)
    with pytest.raises(scorer.SomphScoringError, match="outside its registration state"):
        _score_pair(case)


def test_both_artifacts_verify_before_truth_is_opened(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    Path(case["truth"]).write_text("not-json", encoding="utf-8")
    os.chmod(case["after"]["path"], 0o600)
    with pytest.raises(scorer.SomphScoringError, match="prediction verification failed"):
        _score_pair(case)


def test_evidence_bundle_rejects_clean_access_before_truth_join(tmp_path: Path) -> None:
    case = _case(tmp_path / "case", preopen_clean_access=True)
    Path(case["truth"]).write_text("not-json", encoding="utf-8")
    with pytest.raises(scorer.SomphScoringError, match="pre-open forbidden access"):
        _score_pair(case)


def test_registration_pair_entity_rejects_physical_query_mismatch(tmp_path: Path) -> None:
    case = _case(tmp_path / "case", pair_query_mismatch=True)
    with pytest.raises(scorer.SomphScoringError, match="old query mismatch"):
        _score_pair(case)


def test_scorer_side_rejects_per_tx_query_coverage_drift(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    truth = json.loads(Path(case["truth"]).read_text(encoding="utf-8"))
    truth["rows"].pop()
    _write_json(Path(case["truth"]), truth)
    manifest = json.loads(Path(case["manifest"]).read_text(encoding="utf-8"))
    manifest["truth_sidecar_sha256"] = sha256_file(case["truth"])
    _write_json(Path(case["manifest"]), manifest)
    case["manifest_sha256"] = sha256_file(case["manifest"])
    with pytest.raises(scorer.SomphScoringError, match="per-TX query coverage drift"):
        _score_pair(case)


def test_scoring_outputs_are_exclusive(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    rows, predictions, receipt = _score_pair(case)
    output = tmp_path / "output"
    paths = {
        "formal_rows_path": output / "formal_rows.json",
        "formal_predictions_path": output / "formal_predictions.json",
        "scoring_receipt_path": output / "scoring_receipt.json",
    }
    scorer.write_somph_scoring_outputs_exclusive(
        **paths,
        formal_rows=rows,
        formal_predictions=predictions,
        scoring_receipt=receipt,
    )
    with pytest.raises(FileExistsError):
        scorer.write_somph_scoring_outputs_exclusive(
            **paths,
            formal_rows=rows,
            formal_predictions=predictions,
            scoring_receipt=receipt,
        )


def test_cli_executes_separate_pair_scoring_flow(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    output = tmp_path / "output"
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "score_cvs_somph_predictions.py"
    )
    command = [
        sys.executable,
        str(script),
        "--mode", "stage2c",
        "--before-prediction-artifact", case["before"]["path"],
        "--expected-before-artifact-sha256", case["before"]["artifact_sha256"],
        "--expected-before-seal-sha256", case["before"]["seal_sha256"],
        "--after-prediction-artifact", case["after"]["path"],
        "--expected-after-artifact-sha256", case["after"]["artifact_sha256"],
        "--expected-after-seal-sha256", case["after"]["seal_sha256"],
        "--scoring-manifest", str(case["manifest"]),
        "--expected-scoring-manifest-sha256", case["manifest_sha256"],
        "--formal-rows", str(output / "formal_rows.json"),
        "--formal-predictions", str(output / "formal_predictions.json"),
        "--scoring-receipt", str(output / "scoring_receipt.json"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / "formal_rows.json").is_file()
    assert (output / "formal_predictions.json").is_file()
    receipt = json.loads((output / "scoring_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "LOCAL_PROTOCOL_REPAIR_REQUIRED"
