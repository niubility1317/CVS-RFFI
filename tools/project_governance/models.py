"""Immutable domain records and stable vocabularies for asset governance.

The governance pipeline passes these records between read-only collectors,
indexers and report emitters.  Enum values are wire-format values: changing
one would make an existing inventory incompatible with later readers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Location(str, Enum):
    LOCAL = "LOCAL"
    N607 = "N607"


class AssetKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    JUNCTION = "junction"
    OTHER = "other"


class AccessStatus(str, Enum):
    OK = "OK"
    SCAN_ERROR = "SCAN_ERROR"


class HashStatus(str, Enum):
    SHA256 = "SHA256"
    METADATA_ONLY = "METADATA_ONLY"
    NOT_HASHED_SIZE_LIMIT = "NOT_HASHED_SIZE_LIMIT"
    ERROR = "ERROR"


class ExperimentState(str, Enum):
    ACTIVE_LIVE = "ACTIVE_LIVE"
    OPEN_INCOMPLETE = "OPEN_INCOMPLETE"
    COMPLETE_EVIDENCE = "COMPLETE_EVIDENCE"
    HISTORICAL_ARCHIVE = "HISTORICAL_ARCHIVE"
    ORPHAN_REVIEW = "ORPHAN_REVIEW"
    SCAN_ERROR = "SCAN_ERROR"


class GitOwnership(str, Enum):
    TRACKED_GIT = "TRACKED_GIT"
    UNTRACKED_IN_GIT_WORKTREE = "UNTRACKED_IN_GIT_WORKTREE"
    IGNORED_REGENERABLE = "IGNORED_REGENERABLE"
    NON_GIT_EVIDENCE = "NON_GIT_EVIDENCE"
    REMOTE_NON_GIT = "REMOTE_NON_GIT"
    MIRROR_PENDING = "MIRROR_PENDING"
    GIT_STATE_ERROR = "GIT_STATE_ERROR"


class RetentionClass(str, Enum):
    KEEP_IMMUTABLE = "KEEP_IMMUTABLE"
    KEEP_ACTIVE = "KEEP_ACTIVE"
    KEEP_UNTIL_PUBLISHED = "KEEP_UNTIL_PUBLISHED"
    HISTORICAL_ARCHIVE = "HISTORICAL_ARCHIVE"
    REGENERABLE_CACHE = "REGENERABLE_CACHE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    DELETE_CANDIDATE = "DELETE_CANDIDATE"


class ApprovalState(str, Enum):
    AWAITING_USER_APPROVAL = "AWAITING_USER_APPROVAL"


class ExecutionState(str, Enum):
    NOT_AUTHORIZED = "NOT_AUTHORIZED"


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    scan_id: str
    location: Location
    root_id: str
    relative_path: str
    display_name: str
    escaped_name: str
    asset_kind: AssetKind
    size_bytes: int | None
    mtime_utc: str | None
    access_status: AccessStatus
    hash_status: HashStatus
    sha256: str | None
    experiment_id: str | None = None
    git_ownership: GitOwnership | None = None
    evidence_role: str | None = None
    retention_class: RetentionClass | None = None
    recommended_action: str = "REVIEW"
    decision_reason: str = "UNCLASSIFIED"


@dataclass(frozen=True)
class ScopeResult:
    scan_id: str
    location: Location
    root_id: str
    relative_path: str
    status: str
    asset_ids: tuple[str, ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class GitOwnershipRecord:
    asset_id: str
    ownership: GitOwnership
    repository_root: str | None = None
    common_git_dir: str | None = None
    branch: str | None = None
    head_commit: str | None = None
    status_summary: str | None = None
    linked_worktrees: tuple[str, ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    run_id: str | None = None
    experiment_state: ExperimentState = ExperimentState.ORPHAN_REVIEW
    phase: str | None = None
    method_or_candidate: str | None = None
    report_path: str | None = None
    local_artifact_paths: tuple[str, ...] | None = None
    n607_artifact_paths: tuple[str, ...] | None = None
    git_commit: str | None = None
    process_evidence: Any | None = None
    prediction_count: int | None = None
    score_count: int | None = None
    expected_artifacts: tuple[str, ...] | None = None
    observed_artifacts: tuple[str, ...] | None = None
    closure_gaps: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RetentionDecision:
    asset_id: str
    retention_class: RetentionClass
    rule_code: str
    reason: str | None = None
    evidence_asset_ids: tuple[str, ...] | None = None
    recommended_action: str = "REVIEW"


@dataclass(frozen=True)
class DeletionCandidate:
    candidate_id: str
    location: Location
    absolute_path: str
    asset_kind: AssetKind
    size_bytes: int | None
    reason: str
    evidence: tuple[str, ...] | None = None
    dependencies: tuple[str, ...] | None = None
    recoverability: str | None = None
    estimated_space_reclaim: int | None = None
    approval_state: ApprovalState = ApprovalState.AWAITING_USER_APPROVAL
    approved_scope: str | None = None
    execution_state: ExecutionState = ExecutionState.NOT_AUTHORIZED


@dataclass(frozen=True)
class ScanBundle:
    scan_id: str
    schema_version: int = 1
    operator: str | None = None
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    assets: tuple[AssetRecord, ...] | None = None
    scope_results: tuple[ScopeResult, ...] | None = None
    git_ownership: tuple[GitOwnershipRecord, ...] | None = None
    experiments: tuple[ExperimentRecord, ...] | None = None
    retention_decisions: tuple[RetentionDecision, ...] | None = None
    deletion_candidates: tuple[DeletionCandidate, ...] | None = None


__all__ = [
    "AccessStatus",
    "ApprovalState",
    "AssetKind",
    "AssetRecord",
    "DeletionCandidate",
    "ExecutionState",
    "ExperimentRecord",
    "ExperimentState",
    "GitOwnership",
    "GitOwnershipRecord",
    "HashStatus",
    "Location",
    "RetentionClass",
    "RetentionDecision",
    "ScanBundle",
    "ScopeResult",
]
