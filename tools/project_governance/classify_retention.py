"""Classify already-indexed assets without changing or inspecting them.

This module deliberately accepts only caller-supplied facts.  It does not
walk paths, read asset bytes, query Git, contact N607, or perform an approval
or execution action.  A deletion result is therefore only a row for a later,
explicit user-approval workflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .models import (
    AccessStatus,
    ApprovalState,
    AssetRecord,
    DeletionCandidate,
    ExecutionState,
    ExperimentState,
    GitOwnership,
    HashStatus,
    RetentionClass,
    RetentionDecision,
)


RETENTION_PRIORITY = (
    "ERROR_OR_CONFLICT",
    "PROTECTED_EVIDENCE",
    "ACTIVE_OR_OPEN_EXPERIMENT",
    "CURRENT_PUBLICATION_DEPENDENCY",
    "VERIFIED_HISTORICAL_ARCHIVE",
    "PROVEN_REGENERABLE_CACHE",
    "FULLY_PROVEN_DELETE_CANDIDATE",
    "INSUFFICIENT_EVIDENCE",
)

_PROTECTED_EVIDENCE_ROLES = frozenset(
    {
        "dataset",
        "checkpoint",
        "formal_report",
        "report",
        "log",
        "metrics",
        "metric",
        "prediction",
        "score",
        "receipt",
        "manifest",
        "run_output",
    }
)
_CACHE_ROLES = frozenset({"cache", "generated_cache", "cache_output"})
_CURRENT_GIT_OWNERSHIP = frozenset(
    {
        GitOwnership.TRACKED_GIT,
        GitOwnership.UNTRACKED_IN_GIT_WORKTREE,
    }
)
_RETAINED_CANONICAL_CLASSES = frozenset(
    {
        RetentionClass.KEEP_IMMUTABLE,
        RetentionClass.KEEP_ACTIVE,
        RetentionClass.KEEP_UNTIL_PUBLISHED,
        RetentionClass.HISTORICAL_ARCHIVE,
    }
)


@dataclass(frozen=True)
class RetentionEvidence:
    """Explicit, already-collected facts for one asset's retention decision.

    ``evidence_asset_ids`` identifies the records supporting the supplied
    booleans.  Its values are carried into the decision; this module does not
    infer them from names, paths, timestamps, Git state, or the filesystem.
    """

    evidence_asset_ids: tuple[str, ...] = ()
    protected_evidence: bool = False
    active_process: bool | None = None
    experiment_state: ExperimentState | None = None
    current_publication_dependency: bool | None = None
    current_document_dependency: bool | None = None
    current_release_dependency: bool | None = None
    current_review_dependency: bool | None = None
    terminal_evidence: bool = False
    archive_marker: bool = False
    cache_designated: bool = False
    generator_recorded: bool = False
    source_dependencies_recorded: bool = False
    rebuild_command_recorded: bool = False
    generator_evidence_asset_ids: tuple[str, ...] = ()
    source_dependency_evidence_asset_ids: tuple[str, ...] = ()
    rebuild_command_evidence_asset_ids: tuple[str, ...] = ()
    deletion_candidate_requested: bool = False
    provenance_known: bool = False
    purpose_known: bool = False
    git_worktree_dependency: bool | None = None
    report_dependency: bool | None = None
    manifest_dependency: bool | None = None
    receipt_dependency: bool | None = None
    recoverability: str | None = None
    estimated_space_reclaim: int | None = None
    absolute_path: str | None = None
    canonical_copy_asset_id: str | None = None
    canonical_copy_sha256: str | None = None
    canonical_copy_verified: bool = False
    canonical_copy_retention_class: RetentionClass | None = None
    read_error: bool = False
    parse_error: bool = False
    git_error: bool = False
    evidence_conflict: bool = False
    old_mtime_signal: bool = False
    unusual_name_signal: bool = False


def _normalized_role(asset: AssetRecord) -> str:
    return (asset.evidence_role or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _canonical_asset_id(value: object) -> str | None:
    """Accept only non-empty asset IDs already supplied in canonical form."""

    if not isinstance(value, str):
        return None
    canonical_value = value.strip()
    return canonical_value if canonical_value and value == canonical_value else None


def _evidence_ids(asset: AssetRecord, facts: RetentionEvidence) -> tuple[str, ...]:
    identifiers = [asset.asset_id]
    for evidence_ids in (
        facts.evidence_asset_ids,
        facts.generator_evidence_asset_ids,
        facts.source_dependency_evidence_asset_ids,
        facts.rebuild_command_evidence_asset_ids,
    ):
        if isinstance(evidence_ids, tuple):
            identifiers.extend(
                canonical_asset_id
                for evidence_asset_id in evidence_ids
                if (canonical_asset_id := _canonical_asset_id(evidence_asset_id)) is not None
            )
    if (canonical_copy_asset_id := _canonical_asset_id(facts.canonical_copy_asset_id)) is not None:
        identifiers.append(canonical_copy_asset_id)
    return tuple(dict.fromkeys(identifiers))


def _has_error_or_conflict(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return any(
        (
            asset.access_status is not AccessStatus.OK,
            asset.hash_status is HashStatus.ERROR,
            asset.git_ownership is GitOwnership.GIT_STATE_ERROR,
            facts.experiment_state is ExperimentState.SCAN_ERROR,
            facts.read_error,
            facts.parse_error,
            facts.git_error,
            facts.evidence_conflict,
        )
    )


def _is_protected(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return facts.protected_evidence or _normalized_role(asset) in _PROTECTED_EVIDENCE_ROLES


def _is_active_or_open(facts: RetentionEvidence) -> bool:
    return facts.active_process is True or facts.experiment_state in {
        ExperimentState.ACTIVE_LIVE,
        ExperimentState.OPEN_INCOMPLETE,
    }


def _has_current_publication_dependency(facts: RetentionEvidence) -> bool:
    return any(
        (
            facts.current_publication_dependency is True,
            facts.current_document_dependency is True,
            facts.current_release_dependency is True,
            facts.current_review_dependency is True,
        )
    )


def _is_verified_historical_archive(facts: RetentionEvidence) -> bool:
    return (
        facts.experiment_state is ExperimentState.HISTORICAL_ARCHIVE
        and facts.terminal_evidence
        and facts.archive_marker
    )


def _has_nonself_evidence_ids(asset: AssetRecord, evidence_asset_ids: tuple[str, ...]) -> bool:
    asset_id = _canonical_asset_id(asset.asset_id)
    return (
        asset_id is not None
        and isinstance(evidence_asset_ids, tuple)
        and bool(evidence_asset_ids)
        and all(
            (canonical_asset_id := _canonical_asset_id(evidence_asset_id)) is not None
            and canonical_asset_id != asset_id
            for evidence_asset_id in evidence_asset_ids
        )
    )


def _is_proven_regenerable(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return (
        facts.generator_recorded
        and facts.source_dependencies_recorded
        and facts.rebuild_command_recorded
        and _has_nonself_evidence_ids(asset, facts.generator_evidence_asset_ids)
        and _has_nonself_evidence_ids(asset, facts.source_dependency_evidence_asset_ids)
        and _has_nonself_evidence_ids(asset, facts.rebuild_command_evidence_asset_ids)
    )


def _is_regenerable_cache(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return (facts.cache_designated or _normalized_role(asset) in _CACHE_ROLES) and _is_proven_regenerable(asset, facts)


def _has_current_dependency(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return any(
        (
            _is_active_or_open(facts),
            asset.git_ownership in _CURRENT_GIT_OWNERSHIP,
            facts.git_worktree_dependency is True,
            facts.report_dependency is True,
            facts.manifest_dependency is True,
            facts.receipt_dependency is True,
            _has_current_publication_dependency(facts),
        )
    )


def _has_explicit_clear_dependencies(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return (
        facts.active_process is False
        and facts.current_publication_dependency is False
        and facts.current_document_dependency is False
        and facts.current_release_dependency is False
        and facts.current_review_dependency is False
        and facts.git_worktree_dependency is False
        and facts.report_dependency is False
        and facts.manifest_dependency is False
        and facts.receipt_dependency is False
        and asset.git_ownership is not None
        and asset.git_ownership not in _CURRENT_GIT_OWNERSHIP
        and (asset.experiment_id is None or facts.experiment_state is not None)
        and facts.experiment_state not in {
            ExperimentState.ORPHAN_REVIEW,
            ExperimentState.HISTORICAL_ARCHIVE,
        }
    )


def _is_sha256(value: str | None) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _has_retained_canonical_copy(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    asset_id = _canonical_asset_id(asset.asset_id)
    canonical_copy_asset_id = _canonical_asset_id(facts.canonical_copy_asset_id)
    return (
        asset_id is not None
        and canonical_copy_asset_id is not None
        and canonical_copy_asset_id != asset_id
        and facts.canonical_copy_verified
        and facts.canonical_copy_retention_class in _RETAINED_CANONICAL_CLASSES
        and asset.hash_status is HashStatus.SHA256
        and _is_sha256(asset.sha256)
        and _is_sha256(facts.canonical_copy_sha256)
        and asset.sha256.casefold() == facts.canonical_copy_sha256.casefold()
    )


def _has_recoverability_record(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return (
        isinstance(facts.recoverability, str)
        and bool(facts.recoverability.strip())
        and type(facts.estimated_space_reclaim) is int
        and facts.estimated_space_reclaim >= 0
        and isinstance(facts.absolute_path, str)
        and bool(facts.absolute_path.strip())
    )


def _is_fully_proven_delete_candidate(asset: AssetRecord, facts: RetentionEvidence) -> bool:
    return (
        facts.deletion_candidate_requested
        and not _has_error_or_conflict(asset, facts)
        and not _is_protected(asset, facts)
        and not _has_current_dependency(asset, facts)
        and _has_explicit_clear_dependencies(asset, facts)
        and facts.provenance_known
        and facts.purpose_known
        and (_is_proven_regenerable(asset, facts) or _has_retained_canonical_copy(asset, facts))
        and _has_recoverability_record(asset, facts)
    )


def _decision(
    asset: AssetRecord,
    facts: RetentionEvidence,
    retention_class: RetentionClass,
    rule_code: str,
    reason: str,
) -> RetentionDecision:
    return RetentionDecision(
        asset_id=asset.asset_id,
        retention_class=retention_class,
        rule_code=rule_code,
        reason=reason,
        evidence_asset_ids=_evidence_ids(asset, facts),
    )


def classify_retention(asset: AssetRecord, facts: RetentionEvidence) -> RetentionDecision:
    """Return the first applicable conservative retention decision.

    The ordered rules are exposed as :data:`RETENTION_PRIORITY`.  A cleanup
    heuristic such as age, a zero-byte asset, an unusual name, or untracked
    ownership never supplies a positive deletion predicate.
    """

    if _has_error_or_conflict(asset, facts):
        return _decision(
            asset,
            facts,
            RetentionClass.REVIEW_REQUIRED,
            "ERROR_OR_CONFLICT",
            "A scan, read, parse, Git, or evidence conflict prevents automated retention.",
        )
    if _is_protected(asset, facts):
        return _decision(
            asset,
            facts,
            RetentionClass.KEEP_IMMUTABLE,
            "PROTECTED_EVIDENCE",
            "The asset is explicitly classified as protected evidence.",
        )
    if _is_active_or_open(facts):
        return _decision(
            asset,
            facts,
            RetentionClass.KEEP_ACTIVE,
            "ACTIVE_OR_OPEN_EXPERIMENT",
            "The asset is associated with an active process or an open experiment.",
        )
    if _has_current_publication_dependency(facts):
        return _decision(
            asset,
            facts,
            RetentionClass.KEEP_UNTIL_PUBLISHED,
            "CURRENT_PUBLICATION_DEPENDENCY",
            "The asset is referenced by a current publication, release, or review dependency.",
        )
    if _is_verified_historical_archive(facts):
        return _decision(
            asset,
            facts,
            RetentionClass.HISTORICAL_ARCHIVE,
            "VERIFIED_HISTORICAL_ARCHIVE",
            "Explicit terminal evidence and an archive marker retain the historical archive.",
        )
    if _is_regenerable_cache(asset, facts) and not facts.deletion_candidate_requested:
        return _decision(
            asset,
            facts,
            RetentionClass.REGENERABLE_CACHE,
            "PROVEN_REGENERABLE_CACHE",
            "The cache has recorded generator, source-dependency, and rebuild-command evidence.",
        )
    if _is_fully_proven_delete_candidate(asset, facts):
        return _decision(
            asset,
            facts,
            RetentionClass.DELETE_CANDIDATE,
            "FULLY_PROVEN_DELETE_CANDIDATE",
            "All required non-execution deletion predicates are explicitly recorded.",
        )
    return _decision(
        asset,
        facts,
        RetentionClass.REVIEW_REQUIRED,
        "INSUFFICIENT_EVIDENCE",
        "The supplied facts do not prove a safe retention transition or deletion candidate.",
    )


def classify_retentions(
    assets: Iterable[AssetRecord], evidence_by_asset_id: Mapping[str, RetentionEvidence]
) -> tuple[RetentionDecision, ...]:
    """Classify assets only when their explicit evidence record is supplied."""

    return tuple(
        classify_retention(asset, facts)
        for asset in assets
        if (facts := evidence_by_asset_id.get(asset.asset_id)) is not None
    )


def build_deletion_candidates(
    assets: Iterable[AssetRecord], evidence_by_asset_id: Mapping[str, RetentionEvidence]
) -> tuple[DeletionCandidate, ...]:
    """Build approval-only rows for fully proven candidates; never execute them."""

    classified_assets: list[tuple[AssetRecord, RetentionEvidence, RetentionDecision]] = []
    seen_asset_ids: set[str] = set()
    for asset in assets:
        if asset.asset_id in seen_asset_ids:
            continue
        seen_asset_ids.add(asset.asset_id)
        facts = evidence_by_asset_id.get(asset.asset_id)
        if facts is None:
            continue
        decision = classify_retention(asset, facts)
        classified_assets.append((asset, facts, decision))

    batch_candidate_asset_ids = {
        canonical_asset_id
        for asset, facts, decision in classified_assets
        if decision.retention_class is RetentionClass.DELETE_CANDIDATE
        and _is_fully_proven_delete_candidate(asset, facts)
        and (canonical_asset_id := _canonical_asset_id(asset.asset_id)) is not None
    }
    candidates: list[DeletionCandidate] = []
    for asset, facts, decision in classified_assets:
        asset_id = _canonical_asset_id(asset.asset_id)
        if asset_id not in batch_candidate_asset_ids:
            continue
        uses_canonical_copy = _has_retained_canonical_copy(asset, facts) and not _is_proven_regenerable(asset, facts)
        canonical_copy_asset_id = _canonical_asset_id(facts.canonical_copy_asset_id)
        if uses_canonical_copy and canonical_copy_asset_id in batch_candidate_asset_ids:
            continue
        candidates.append(
            DeletionCandidate(
                candidate_id=f"deletion-candidate:{asset.asset_id}",
                location=asset.location,
                absolute_path=facts.absolute_path or "",
                asset_kind=asset.asset_kind,
                size_bytes=asset.size_bytes,
                reason=decision.reason or "Fully proven deletion candidate awaiting user approval.",
                evidence=decision.evidence_asset_ids,
                dependencies=tuple(
                    identifier
                    for identifier in (decision.evidence_asset_ids or ())
                    if identifier != asset.asset_id
                ),
                recoverability=facts.recoverability,
                estimated_space_reclaim=facts.estimated_space_reclaim,
                approval_state=ApprovalState.AWAITING_USER_APPROVAL,
                approved_scope=None,
                execution_state=ExecutionState.NOT_AUTHORIZED,
            )
        )
    return tuple(candidates)


__all__ = [
    "RETENTION_PRIORITY",
    "RetentionEvidence",
    "build_deletion_candidates",
    "classify_retention",
    "classify_retentions",
]
