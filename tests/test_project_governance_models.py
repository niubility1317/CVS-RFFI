from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

from tools.project_governance.config import load_config
from tools.project_governance.models import (
    AccessStatus,
    ApprovalState,
    AssetKind,
    AssetRecord,
    DeletionCandidate,
    ExecutionState,
    ExperimentRecord,
    ExperimentState,
    GitOwnership,
    GitOwnershipRecord,
    HashStatus,
    Location,
    RetentionClass,
    RetentionDecision,
    ScanBundle,
    ScopeResult,
)


def test_fixed_governance_vocabularies_are_stable():
    assert [item.value for item in Location] == ["LOCAL", "N607"]
    assert [item.value for item in AssetKind] == [
        "file",
        "directory",
        "symlink",
        "junction",
        "other",
    ]
    assert [item.value for item in AccessStatus] == ["OK", "SCAN_ERROR"]
    assert [item.value for item in HashStatus] == [
        "SHA256",
        "METADATA_ONLY",
        "NOT_HASHED_SIZE_LIMIT",
        "ERROR",
    ]
    assert [item.value for item in ExperimentState] == [
        "ACTIVE_LIVE",
        "OPEN_INCOMPLETE",
        "COMPLETE_EVIDENCE",
        "HISTORICAL_ARCHIVE",
        "ORPHAN_REVIEW",
        "SCAN_ERROR",
    ]
    assert [item.value for item in GitOwnership] == [
        "TRACKED_GIT",
        "UNTRACKED_IN_GIT_WORKTREE",
        "IGNORED_REGENERABLE",
        "NON_GIT_EVIDENCE",
        "REMOTE_NON_GIT",
        "MIRROR_PENDING",
        "GIT_STATE_ERROR",
    ]
    assert [item.value for item in RetentionClass] == [
        "KEEP_IMMUTABLE",
        "KEEP_ACTIVE",
        "KEEP_UNTIL_PUBLISHED",
        "HISTORICAL_ARCHIVE",
        "REGENERABLE_CACHE",
        "REVIEW_REQUIRED",
        "DELETE_CANDIDATE",
    ]
    assert [item.value for item in ApprovalState] == ["AWAITING_USER_APPROVAL"]
    assert [item.value for item in ExecutionState] == ["NOT_AUTHORIZED"]

    assert ExperimentState.ACTIVE_LIVE.value == "ACTIVE_LIVE"
    assert GitOwnership.NON_GIT_EVIDENCE.value == "NON_GIT_EVIDENCE"
    assert RetentionClass.DELETE_CANDIDATE.value == "DELETE_CANDIDATE"
    assert ApprovalState.AWAITING_USER_APPROVAL.value == "AWAITING_USER_APPROVAL"
    assert ExecutionState.NOT_AUTHORIZED.value == "NOT_AUTHORIZED"
    assert AssetKind.JUNCTION.value == "junction"


def test_records_are_frozen_and_optional_evidence_is_none():
    asset = AssetRecord(
        asset_id="asset:LOCAL:TYPE10_7:runs/A",
        scan_id="SCAN_1",
        location=Location.LOCAL,
        root_id="TYPE10_7",
        relative_path="runs/A",
        display_name="A",
        escaped_name="A",
        asset_kind=AssetKind.DIRECTORY,
        size_bytes=None,
        mtime_utc=None,
        access_status=AccessStatus.OK,
        hash_status=HashStatus.METADATA_ONLY,
        sha256=None,
    )
    assert is_dataclass(asset)
    assert asset.experiment_id is None
    assert asset.git_ownership is None
    assert asset.evidence_role is None
    assert asset.retention_class is None
    assert asset.recommended_action == "REVIEW"
    assert asset.decision_reason == "UNCLASSIFIED"
    with pytest.raises(FrozenInstanceError):
        asset.display_name = "changed"

    scope = ScopeResult(
        scan_id="SCAN_1",
        location=Location.LOCAL,
        root_id="TYPE10_7",
        relative_path="runs",
        status="VERIFIED",
    )
    ownership = GitOwnershipRecord(
        asset_id=asset.asset_id,
        ownership=GitOwnership.NON_GIT_EVIDENCE,
    )
    experiment = ExperimentRecord(
        experiment_id="RUN_A",
        run_id="RUN_A",
        experiment_state=ExperimentState.OPEN_INCOMPLETE,
    )
    decision = RetentionDecision(
        asset_id=asset.asset_id,
        retention_class=RetentionClass.REVIEW_REQUIRED,
        rule_code="INSUFFICIENT_EVIDENCE",
    )
    candidate = DeletionCandidate(
        candidate_id="DELETE_1",
        location=Location.LOCAL,
        absolute_path="E:/type10-7/runs/A",
        asset_kind=AssetKind.DIRECTORY,
        size_bytes=0,
        reason="No deletion evidence yet",
    )
    bundle = ScanBundle(scan_id="SCAN_1")

    for record in (scope, ownership, experiment, decision, candidate, bundle):
        assert is_dataclass(record)
        assert record.__dataclass_params__.frozen is True

    assert candidate.approval_state is ApprovalState.AWAITING_USER_APPROVAL
    assert candidate.execution_state is ExecutionState.NOT_AUTHORIZED
    assert candidate.approved_scope is None
    with pytest.raises(FrozenInstanceError):
        candidate.reason = "changed"


def test_asset_record_field_order_keeps_required_contract():
    assert [field.name for field in fields(AssetRecord)] == [
        "asset_id",
        "scan_id",
        "location",
        "root_id",
        "relative_path",
        "display_name",
        "escaped_name",
        "asset_kind",
        "size_bytes",
        "mtime_utc",
        "access_status",
        "hash_status",
        "sha256",
        "experiment_id",
        "git_ownership",
        "evidence_role",
        "retention_class",
        "recommended_action",
        "decision_reason",
    ]


def test_versioned_inventory_config_is_exact_and_loadable():
    path = Path(__file__).resolve().parents[1] / "configs" / "project_governance_inventory_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {
        "schema_version": 1,
        "local": {
            "root_id": "TYPE10_7",
            "root": "E:/type10-7",
            "carrier_surfaces": [
                "automation_reports/CV-SincNet",
                "code/snapshots",
                "local_artifacts",
                "remote_artifacts",
                "runs",
                "logs",
                "outputs",
                "server_log_backups",
                "runner_staging",
                "github_publish/CVS-RFFI-repo",
            ],
        },
        "n607": {
            "root_id": "N607_CVS_SINCNET",
            "root": "/home/szu2070436088/2510044040/CV-SincNet",
            "carrier_surfaces": [
                "automation_reports",
                "runs",
                "logs",
                "releases",
                "remote_artifacts",
                "snapshots",
                "code",
            ],
        },
        "discovery": {
            "control_evidence_max_depth": 3,
            "hash_max_bytes": 10485760,
            "text_read_max_bytes": 2097152,
        },
        "output": {
            "git_file_max_bytes": 10485760,
            "git_scan_max_bytes": 52428800,
        },
    }
    config = load_config(path)
    assert config.schema_version == 1
    assert config.local.root_id == "TYPE10_7"
    assert config.n607.root_id == "N607_CVS_SINCNET"
    assert config.discovery.hash_max_bytes == 10485760


def test_load_config_validates_location_roots_and_carrier_boundaries(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs" / "project_governance_inventory_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))

    payload["schema_version"] = 99
    invalid_version = tmp_path / "invalid-version.json"
    invalid_version.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_config(invalid_version)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["local"]["carrier_surfaces"] = ["../outside"]
    invalid_carrier = tmp_path / "invalid-carrier.json"
    invalid_carrier.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="carrier"):
        load_config(invalid_carrier)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["discovery"]["hash_max_bytes"] = 0
    invalid_limit = tmp_path / "invalid-limit.json"
    invalid_limit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_config(invalid_limit)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["local"]["root"] = "E:/other"
    invalid_root = tmp_path / "invalid-root.json"
    invalid_root.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="LOCAL"):
        load_config(invalid_root, location=Location.LOCAL)


def test_missing_optional_carrier_surfaces_are_retained_as_not_present(tmp_path):
    payload = {
        "schema_version": 1,
        "local": {
            "root_id": "TYPE10_7",
            "root": "E:/type10-7",
            "carrier_surfaces": [
                "automation_reports/CV-SincNet",
                "code/snapshots",
                "local_artifacts",
                "remote_artifacts",
                "runs",
                "logs",
                "outputs",
                "server_log_backups",
                "runner_staging",
                "github_publish/CVS-RFFI-repo",
            ],
        },
        "n607": {
            "root_id": "N607_CVS_SINCNET",
            "root": "/home/szu2070436088/2510044040/CV-SincNet",
            "carrier_surfaces": [
                "automation_reports",
                "runs",
                "logs",
                "releases",
                "remote_artifacts",
                "snapshots",
                "code",
            ],
        },
        "discovery": {
            "control_evidence_max_depth": 3,
            "hash_max_bytes": 10485760,
            "text_read_max_bytes": 2097152,
        },
        "output": {
            "git_file_max_bytes": 10485760,
            "git_scan_max_bytes": 52428800,
        },
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(path, location=Location.LOCAL, probe_local_paths=False)
    assert [surface.relative_path for surface in config.local.carrier_surfaces] == payload["local"][
        "carrier_surfaces"
    ]
    statuses = {surface.relative_path: surface.status for surface in config.local.carrier_surfaces}
    assert set(statuses.values()) == {"NOT_PRESENT"}


def test_uncollected_evidence_collections_are_none_not_empty_tuples():
    ownership = GitOwnershipRecord(
        asset_id="asset:LOCAL:TYPE10_7:runs/A",
        ownership=GitOwnership.NON_GIT_EVIDENCE,
    )
    experiment = ExperimentRecord(experiment_id="RUN_A")
    decision = RetentionDecision(
        asset_id="asset:LOCAL:TYPE10_7:runs/A",
        retention_class=RetentionClass.REVIEW_REQUIRED,
        rule_code="INSUFFICIENT_EVIDENCE",
    )
    bundle = ScanBundle(scan_id="SCAN_1")

    assert ownership.linked_worktrees is None
    scope = ScopeResult(
        scan_id="SCAN_1",
        location=Location.LOCAL,
        root_id="TYPE10_7",
        relative_path="runs",
        status="NOT_PRESENT",
    )
    assert scope.asset_ids is None
    assert experiment.local_artifact_paths is None
    assert experiment.n607_artifact_paths is None
    assert experiment.expected_artifacts is None
    assert experiment.observed_artifacts is None
    assert experiment.closure_gaps is None
    assert decision.evidence_asset_ids is None
    assert bundle.assets is None
    assert bundle.scope_results is None
    assert bundle.git_ownership is None
    assert bundle.experiments is None
    assert bundle.retention_decisions is None
    assert bundle.deletion_candidates is None


def test_measured_empty_evidence_collections_are_distinct_from_none():
    scope = ScopeResult(
        scan_id="SCAN_1",
        location=Location.LOCAL,
        root_id="TYPE10_7",
        relative_path="runs",
        status="VERIFIED",
        asset_ids=(),
    )
    ownership = GitOwnershipRecord(
        asset_id="asset:LOCAL:TYPE10_7:runs/A",
        ownership=GitOwnership.NON_GIT_EVIDENCE,
        linked_worktrees=(),
    )
    experiment = ExperimentRecord(
        experiment_id="RUN_A",
        local_artifact_paths=(),
        n607_artifact_paths=(),
        expected_artifacts=(),
        observed_artifacts=(),
        closure_gaps=(),
    )
    decision = RetentionDecision(
        asset_id="asset:LOCAL:TYPE10_7:runs/A",
        retention_class=RetentionClass.REVIEW_REQUIRED,
        rule_code="INSUFFICIENT_EVIDENCE",
        evidence_asset_ids=(),
    )
    bundle = ScanBundle(
        scan_id="SCAN_1",
        assets=(),
        scope_results=(),
        git_ownership=(),
        experiments=(),
        retention_decisions=(),
        deletion_candidates=(),
    )

    assert scope.asset_ids == ()
    assert GitOwnershipRecord(
        asset_id=ownership.asset_id,
        ownership=ownership.ownership,
    ).linked_worktrees is None
    assert ownership.linked_worktrees == ()
    assert experiment.local_artifact_paths == ()
    assert experiment.n607_artifact_paths == ()
    assert experiment.expected_artifacts == ()
    assert experiment.observed_artifacts == ()
    assert experiment.closure_gaps == ()
    assert RetentionDecision(
        asset_id=decision.asset_id,
        retention_class=decision.retention_class,
        rule_code=decision.rule_code,
    ).evidence_asset_ids is None
    assert decision.evidence_asset_ids == ()
    assert bundle.assets == ()
    assert bundle.scope_results == ()
    assert bundle.git_ownership == ()
    assert bundle.experiments == ()
    assert bundle.retention_decisions == ()
    assert bundle.deletion_candidates == ()
    assert ScanBundle(scan_id="SCAN_1").assets is None


def test_load_config_rejects_replaced_or_reordered_carrier_surfaces(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs" / "project_governance_inventory_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["local"]["carrier_surfaces"] = ["missing/optional"]
    path = tmp_path / "replaced-carriers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="carrier_surfaces"):
        load_config(path, location=Location.LOCAL, probe_local_paths=False)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["n607"]["carrier_surfaces"] = list(reversed(payload["n607"]["carrier_surfaces"]))
    path = tmp_path / "reordered-carriers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="carrier_surfaces"):
        load_config(path, location=Location.N607, probe_local_paths=False)
