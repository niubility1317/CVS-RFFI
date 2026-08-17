"""Build a conservative experiment index from already indexed small evidence.

This module deliberately receives its universe of paths from ``AssetRecord``
instances and caller-supplied roots.  It never walks a run directory, probes a
process table, or reads checkpoint-like data.  Metrics are retained solely as
opaque evidence references; no score is interpreted, compared, or promoted.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AccessStatus, AssetKind, AssetRecord, ExperimentRecord, ExperimentState, Location
from .paths import normalize_relative_path


MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
_BLOCKED_SUFFIXES = {".bin", ".ckpt", ".npy", ".npz", ".pkl", ".pickle", ".pt", ".pth"}
_TEXT_SUFFIXES = {".json", ".md", ".receipt", ".txt"}
_TERMINAL_VALUES = {"COMPLETE", "COMPLETED", "DONE", "FINISHED", "TERMINAL", "TRUE"}
_TRUE_VALUES = {"1", "TRUE", "YES", "Y", "ARCHIVED"}
_KEY_ALIASES = {
    "run_id": ("run_id", "runid", "experiment_id", "experimentid"),
    "git_commit": ("git_commit", "gitcommit", "commit", "git_sha", "gitsha", "commit_id", "commitid"),
    "run_root": ("run_root", "runroot", "output_root", "outputroot", "artifact_root", "artifactroot", "run_dir", "rundir"),
    "expected_artifacts": ("expected_artifacts", "expectedartifacts", "expected_outputs", "expectedoutputs"),
    "terminal": ("terminal", "status", "state", "completion_status", "completionstatus"),
    "archive": ("archive", "archived", "archive_marker", "archivemarker"),
    "phase": ("phase", "stage"),
    "method_or_candidate": ("method_or_candidate", "methodorcandidate", "candidate", "method"),
}


@dataclass(frozen=True)
class EvidenceClaim:
    """One parsed, source-addressable statement without a performance judgement."""

    source_asset_id: str
    field: str
    value: Any
    confidence: str
    parse_status: str


@dataclass(frozen=True)
class ProcessEvidence:
    """A caller-provided process snapshot; the indexer never queries processes itself."""

    pid: int | None
    cwd: str | None
    cmdline: str | None
    run_root: str | None = None


class ExperimentIndex(dict[str, ExperimentRecord]):
    """Mapping-like result with the raw, non-lossy evidence claim ledger."""

    def __init__(
        self,
        records: Mapping[str, ExperimentRecord] | None = None,
        *,
        claims: Iterable[EvidenceClaim] = (),
        claims_by_experiment: Mapping[str, Sequence[EvidenceClaim]] | None = None,
    ) -> None:
        super().__init__(records or {})
        self.claims = tuple(claims)
        self.claims_by_experiment = {
            key: tuple(value) for key, value in (claims_by_experiment or {}).items()
        }


@dataclass
class _Evidence:
    asset: AssetRecord
    path: Path | None
    kind: str | None
    claims: list[EvidenceClaim]
    issues: list[str]


class _UnionFind:
    def __init__(self, count: int) -> None:
        self._parents = list(range(count))

    def find(self, value: int) -> int:
        while self._parents[value] != value:
            self._parents[value] = self._parents[self._parents[value]]
            value = self._parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parents[right_root] = left_root


def _normalize_run_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFC", value).strip()
    return normalized or None


def _run_key(value: str) -> str:
    return value.casefold()


def _path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def _same_path(left: Path | str, right: Path | str) -> bool:
    return _path_key(left) == _path_key(right)


def _is_under(path: Path, root: Path) -> bool:
    path_key = _path_key(path)
    root_key = _path_key(root)
    return path_key == root_key or path_key.startswith(root_key + os.sep)


def _root_for(asset: AssetRecord, root_paths: Mapping[Any, str | os.PathLike[str]]) -> Path | None:
    root = root_paths.get(asset.root_id)
    if root is None:
        root = root_paths.get(asset.location)
    if root is None:
        root = root_paths.get(asset.location.value)
    return Path(root).resolve(strict=False) if root is not None else None


def _path_from_asset(asset: AssetRecord, root_paths: Mapping[Any, str | os.PathLike[str]]) -> Path | None:
    root = _root_for(asset, root_paths)
    if root is None:
        return None
    try:
        relative = normalize_relative_path(asset.relative_path)
    except ValueError:
        return None
    candidate = (root / Path(relative)).resolve(strict=False)
    return candidate if _is_under(candidate, root) else None


def _evidence_kind(asset: AssetRecord) -> str | None:
    role = (asset.evidence_role or "").casefold()
    name = asset.display_name.casefold()
    suffix = Path(name).suffix
    if suffix in _BLOCKED_SUFFIXES or role in {"checkpoint", "pickle", "numpy", "pytorch"}:
        return None
    if suffix not in _TEXT_SUFFIXES:
        return None
    if "prediction" in role or "prediction" in name:
        return None
    if "metrics" in role or "metric" in role or "score" in role or "metrics" in name or "scores" in name:
        return "metrics"
    if "manifest" in role or "manifest" in name:
        return "manifest"
    if "receipt" in role or "receipt" in name:
        return "receipt"
    if "report" in role or "report" in name:
        return "report"
    if role in {"text", "evidence_text"} and suffix in _TEXT_SUFFIXES:
        return "text"
    return None


def _decode_text(payload: bytes) -> str:
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = payload.decode("utf-16")
    else:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("gb18030")
    if "\ufffd" in text or text.startswith("\ufeff"):
        raise ValueError("text decode is not lossless")
    return text


def _read_text(asset: AssetRecord, path: Path | None) -> str:
    if asset.size_bytes is not None and asset.size_bytes > MAX_EVIDENCE_BYTES:
        raise OverflowError("indexed evidence exceeds 2 MiB")
    if path is None:
        raise OSError("asset root path is unavailable")
    with path.open("rb") as handle:
        payload = handle.read(MAX_EVIDENCE_BYTES + 1)
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise OverflowError("evidence exceeds 2 MiB")
    return _decode_text(payload)


def _canonical_key(value: str) -> str | None:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) or None


def _field_for_key(value: str) -> str | None:
    key = _canonical_key(value)
    if key is None:
        return None
    for field, aliases in _KEY_ALIASES.items():
        if key in aliases:
            return field
    return None


def _claim(asset: AssetRecord, field: str, value: Any, *, confidence: str = "EXPLICIT") -> EvidenceClaim:
    return EvidenceClaim(asset.asset_id, field, value, confidence, "PARSED")


def _claims_from_mapping(asset: AssetRecord, payload: Mapping[str, Any]) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for raw_key, value in payload.items():
        field = _field_for_key(str(raw_key))
        if field is None:
            continue
        if field == "expected_artifacts":
            values = value if isinstance(value, (list, tuple)) else (value,)
            for item in values:
                if isinstance(item, str) and item.strip():
                    claims.append(_claim(asset, field, item.strip()))
            continue
        if isinstance(value, (str, bool, int, float)):
            claims.append(_claim(asset, field, value))
    return claims


def _claims_from_text(asset: AssetRecord, text: str) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    in_expected_artifacts = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if in_expected_artifacts and line.startswith("-"):
            value = line[1:].strip()
            if value:
                claims.append(_claim(asset, "expected_artifacts", value))
            continue
        match = re.match(r"^(?:[-*]\s*)?([A-Za-z][A-Za-z0-9_ -]*)\s*[:=]\s*(.*)$", line)
        if match is None:
            in_expected_artifacts = False
            continue
        field = _field_for_key(match.group(1))
        value = match.group(2).strip().strip("`\"'")
        in_expected_artifacts = field == "expected_artifacts" and not value
        if field is None or not value:
            continue
        if field == "expected_artifacts":
            for item in value.split(","):
                candidate = item.strip()
                if candidate:
                    claims.append(_claim(asset, field, candidate))
        else:
            claims.append(_claim(asset, field, value))
    return claims


def _parse_evidence(asset: AssetRecord, path: Path | None) -> _Evidence:
    kind = _evidence_kind(asset)
    evidence = _Evidence(asset=asset, path=path, kind=kind, claims=[], issues=[])
    if kind is None:
        return evidence
    if asset.experiment_id:
        evidence.claims.append(_claim(asset, "run_id", asset.experiment_id, confidence="CALLER_EXPLICIT"))
    if asset.access_status is AccessStatus.SCAN_ERROR:
        evidence.issues.append("UNREADABLE_EVIDENCE")
        return evidence
    if asset.asset_kind is not AssetKind.FILE:
        evidence.issues.append("UNREADABLE_EVIDENCE")
        return evidence
    if kind == "metrics":
        evidence.claims.append(
            EvidenceClaim(asset.asset_id, "metrics_reference", asset.asset_id, "OPAQUE", "OPAQUE_REFERENCE")
        )
        return evidence
    try:
        text = _read_text(asset, path)
    except OverflowError:
        evidence.issues.append("EVIDENCE_SIZE_LIMIT")
        return evidence
    except (OSError, UnicodeError, ValueError):
        evidence.issues.append("UNREADABLE_EVIDENCE")
        return evidence
    try:
        if path is not None and path.suffix.casefold() == ".json":
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError("JSON evidence must be an object")
            evidence.claims.extend(_claims_from_mapping(asset, payload))
        else:
            evidence.claims.extend(_claims_from_text(asset, text))
    except (json.JSONDecodeError, ValueError):
        evidence.issues.append("MALFORMED_EVIDENCE")
    return evidence


def _value_set(claims: Iterable[EvidenceClaim], field: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for claim in claims:
        if claim.field == field and claim.value not in values:
            values.append(claim.value)
    return tuple(values)


def _resolve_claim_path(value: Any, asset: AssetRecord, root_paths: Mapping[Any, str | os.PathLike[str]]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    root = _root_for(asset, root_paths)
    if root is None:
        return None
    try:
        relative = normalize_relative_path(value)
    except ValueError:
        return None
    resolved = (root / Path(relative)).resolve(strict=False)
    return resolved if _is_under(resolved, root) else None


def _explicit_run_roots(item: _Evidence, root_paths: Mapping[Any, str | os.PathLike[str]]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in _value_set(item.claims, "run_root"):
        resolved = _resolve_claim_path(value, item.asset, root_paths)
        if resolved is not None and not any(_same_path(resolved, seen) for seen in roots):
            roots.append(resolved)
    return tuple(roots)


def _expected_artifact_paths(
    item: _Evidence, root_paths: Mapping[Any, str | os.PathLike[str]]
) -> tuple[Path, ...]:
    """Resolve only explicitly declared expected artifacts for one evidence item."""

    resolved_paths: list[Path] = []
    run_roots = _explicit_run_roots(item, root_paths)
    for value in _value_set(item.claims, "expected_artifacts"):
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            candidates = (candidate.resolve(strict=False),)
        else:
            try:
                relative = normalize_relative_path(value)
            except ValueError:
                continue
            if run_roots:
                candidates = tuple(
                    (root / Path(relative)).resolve(strict=False)
                    for root in run_roots
                    if _is_under((root / Path(relative)).resolve(strict=False), root)
                )
            else:
                root_relative = _resolve_claim_path(value, item.asset, root_paths)
                candidates = (root_relative,) if root_relative is not None else ()
        for resolved in candidates:
            if resolved is not None and not any(_same_path(resolved, seen) for seen in resolved_paths):
                resolved_paths.append(resolved)
    return tuple(resolved_paths)


def _direct_binding_tokens(
    item: _Evidence, root_paths: Mapping[Any, str | os.PathLike[str]]
) -> frozenset[str]:
    """Return only run IDs and exact paths that are explicitly declared."""

    tokens: set[str] = set()
    for value in _value_set(item.claims, "run_id"):
        run_id = _normalize_run_id(value)
        if run_id is not None:
            tokens.add(f"run:{_run_key(run_id)}")
    for root in _explicit_run_roots(item, root_paths):
        tokens.add(f"path:{_path_key(root)}")
    for artifact in _expected_artifact_paths(item, root_paths):
        tokens.add(f"path:{_path_key(artifact)}")
    return frozenset(tokens)


def _read_failure_blocks_classification(items: Sequence[_Evidence]) -> bool:
    """Return whether unreadable required evidence prevents a conservative state."""

    read_failures = {"UNREADABLE_EVIDENCE", "MALFORMED_EVIDENCE", "EVIDENCE_SIZE_LIMIT"}
    reports = [item for item in items if item.kind == "report"]
    if any(read_failures.intersection(item.issues) for item in reports):
        return True
    has_readable_report = any(not read_failures.intersection(item.issues) for item in reports)
    return not has_readable_report and any(
        item.kind in {"manifest", "receipt"} and read_failures.intersection(item.issues)
        for item in items
    )


def _process_from_value(value: ProcessEvidence | Mapping[str, Any]) -> ProcessEvidence:
    if isinstance(value, ProcessEvidence):
        return value
    return ProcessEvidence(
        pid=value.get("pid"),
        cwd=value.get("cwd"),
        cmdline=value.get("cmdline"),
        run_root=value.get("run_root"),
    )


def _cmdline_mentions(cmdline: str | None, run_root: Path) -> bool:
    if not cmdline:
        return False
    variants = {
        str(run_root),
        run_root.as_posix(),
        str(run_root).replace("\\", "/"),
        str(run_root).replace("/", "\\"),
    }
    for variant in variants:
        if not variant:
            continue
        start = 0
        while True:
            offset = cmdline.find(variant, start)
            if offset < 0:
                break
            before = cmdline[offset - 1] if offset else ""
            after_offset = offset + len(variant)
            after = cmdline[after_offset] if after_offset < len(cmdline) else ""
            before_is_boundary = not before or before.isspace() or before in "\"'=(:,["
            after_is_boundary = not after or after.isspace() or after in "\"'),;]"
            if before_is_boundary and after_is_boundary:
                return True
            start = offset + len(variant)
    return False


def _active_process_for(
    run_roots: Sequence[Path], process_evidence: Sequence[ProcessEvidence]
) -> tuple[ProcessEvidence | None, bool]:
    active_reference = False
    for process in process_evidence:
        for root in run_roots:
            cwd_matches = bool(process.cwd) and _same_path(process.cwd, root)
            cmdline_matches = _cmdline_mentions(process.cmdline, root)
            supplied_root_matches = (
                process.run_root is None or _same_path(process.run_root, root)
            )
            if cwd_matches or cmdline_matches:
                active_reference = True
            if (
                isinstance(process.pid, int)
                and process.pid > 0
                and cwd_matches
                and cmdline_matches
                and supplied_root_matches
            ):
                return process, True
    return None, active_reference


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().upper() in _TRUE_VALUES


def _terminal(values: Iterable[Any]) -> bool:
    for value in values:
        if value is True:
            return True
        if isinstance(value, str) and value.strip().upper() in _TERMINAL_VALUES:
            return True
    return False


def _low_confidence_name_candidate(asset: AssetRecord, known_run_ids: Iterable[str]) -> bool:
    candidate = asset.relative_path.casefold()
    return any(run_id.casefold() in candidate for run_id in known_run_ids)


def _record_key(preferred: str | None, fallback_asset_id: str, occupied: set[str]) -> str:
    base = preferred or f"ORPHAN:{fallback_asset_id}"
    key = base
    suffix = 2
    while key in occupied:
        key = f"{base}#{suffix}"
        suffix += 1
    occupied.add(key)
    return key


def index_experiments(
    indexed_assets: Iterable[AssetRecord],
    *,
    root_paths: Mapping[Any, str | os.PathLike[str]],
    process_evidence: Iterable[ProcessEvidence | Mapping[str, Any]] = (),
) -> ExperimentIndex:
    """Associate only explicitly indexed evidence and return conservative run states.

    ``root_paths`` is a caller-owned mapping from each indexed root identifier
    to an explicit local path.  The indexer reads exact allowlisted files only;
    it never discovers neighbouring files or follows unindexed artifacts.
    """

    assets = tuple(indexed_assets)
    paths = tuple(_path_from_asset(asset, root_paths) for asset in assets)
    evidence = [_parse_evidence(asset, path) for asset, path in zip(assets, paths)]
    processes = tuple(_process_from_value(item) for item in process_evidence)

    union = _UnionFind(len(assets))
    by_run_id: dict[str, list[int]] = defaultdict(list)
    by_root: dict[str, list[int]] = defaultdict(list)
    root_bindings: list[tuple[int, Path]] = []
    expected_artifact_bindings: list[tuple[int, Path]] = []

    for index, item in enumerate(evidence):
        for value in _value_set(item.claims, "run_id"):
            run_id = _normalize_run_id(value)
            if run_id is not None:
                by_run_id[_run_key(run_id)].append(index)
        for run_root in _explicit_run_roots(item, root_paths):
            root_bindings.append((index, run_root))
            by_root[_path_key(run_root)].append(index)
        for artifact_path in _expected_artifact_paths(item, root_paths):
            expected_artifact_bindings.append((index, artifact_path))

    for grouped in (*by_run_id.values(), *by_root.values()):
        for member in grouped[1:]:
            union.union(grouped[0], member)

    for evidence_index, root in root_bindings:
        for asset_index, path in enumerate(paths):
            if path is not None and _is_under(path, root):
                union.union(evidence_index, asset_index)

    for evidence_index, expected_path in expected_artifact_bindings:
        for asset_index, path in enumerate(paths):
            if path is not None and _same_path(path, expected_path):
                union.union(evidence_index, asset_index)

    # A matching commit is only enough when a manifest or receipt already
    # carries a direct run/path binding.  Commit text alone never joins runs.
    by_commit: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(evidence):
        commits = [str(value).strip() for value in _value_set(item.claims, "git_commit") if str(value).strip()]
        if commits:
            by_commit[commits[0]].append(index)
    for members in by_commit.values():
        for binding_member in members:
            if evidence[binding_member].kind not in {"manifest", "receipt"}:
                continue
            binding_tokens = _direct_binding_tokens(evidence[binding_member], root_paths)
            if not binding_tokens:
                continue
            for member in members:
                if member == binding_member:
                    continue
                if binding_tokens.intersection(_direct_binding_tokens(evidence[member], root_paths)):
                    union.union(binding_member, member)

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(assets)):
        grouped_indices[union.find(index)].append(index)

    all_claims = tuple(claim for item in evidence for claim in item.claims)
    records: dict[str, ExperimentRecord] = {}
    claims_by_experiment: dict[str, tuple[EvidenceClaim, ...]] = {}
    occupied_keys: set[str] = set()
    known_run_ids = tuple(
        run_id
        for item in evidence
        for value in _value_set(item.claims, "run_id")
        if (run_id := _normalize_run_id(value)) is not None
    )

    for members in grouped_indices.values():
        group_items = [evidence[index] for index in members]
        group_assets = [assets[index] for index in members]
        group_paths = [paths[index] for index in members]
        group_claims = tuple(claim for item in group_items for claim in item.claims)
        run_ids = []
        for value in _value_set(group_claims, "run_id"):
            run_id = _normalize_run_id(value)
            if run_id is not None and run_id not in run_ids:
                run_ids.append(run_id)
        report_run_ids = [
            _normalize_run_id(claim.value)
            for item in group_items
            if item.kind == "report"
            for claim in item.claims
            if claim.field == "run_id" and _normalize_run_id(claim.value) is not None
        ]
        preferred_run_id = report_run_ids[0] if report_run_ids else (run_ids[0] if run_ids else None)
        key = _record_key(preferred_run_id, group_assets[0].asset_id, occupied_keys)

        reports = [item for item in group_items if item.kind == "report"]
        gaps: list[str] = []
        issues = {issue for item in group_items for issue in item.issues}
        if _read_failure_blocks_classification(group_items):
            gaps.append("UNREADABLE_EVIDENCE")
        if "EVIDENCE_SIZE_LIMIT" in issues:
            gaps.append("EVIDENCE_SIZE_LIMIT")
        if len(run_ids) > 1:
            gaps.append("CONFLICTING_RUN_ID")
        commits = [str(value).strip() for value in _value_set(group_claims, "git_commit") if str(value).strip()]
        if len(commits) > 1:
            gaps.append("CONFLICTING_GIT_COMMIT")

        run_roots: list[Path] = []
        for item in group_items:
            for resolved in _explicit_run_roots(item, root_paths):
                if not any(_same_path(resolved, seen) for seen in run_roots):
                    run_roots.append(resolved)
        expected_paths: list[Path] = []
        for item in group_items:
            for resolved in _expected_artifact_paths(item, root_paths):
                if not any(_same_path(resolved, seen) for seen in expected_paths):
                    expected_paths.append(resolved)

        observed_paths = [
            expected
            for expected in expected_paths
            if any(path is not None and _same_path(expected, path) for path in group_paths)
        ]
        if expected_paths and len(observed_paths) != len(expected_paths):
            gaps.append("MISSING_EXPECTED_ARTIFACT")
        matched_process, active_reference = _active_process_for(run_roots, processes)
        active_live = matched_process is not None

        terminal = any(_terminal(_value_set(item.claims, "terminal")) for item in reports)
        archived = any(_truthy(value) for value in _value_set(group_claims, "archive"))
        has_explicit_binding = bool(run_ids or run_roots)

        if "UNREADABLE_EVIDENCE" in gaps:
            state = ExperimentState.SCAN_ERROR
        elif active_live:
            state = ExperimentState.ACTIVE_LIVE
        elif any(gap.startswith("CONFLICTING_") for gap in gaps):
            state = ExperimentState.ORPHAN_REVIEW
        elif reports and terminal and len(observed_paths) == len(expected_paths) and expected_paths:
            state = ExperimentState.HISTORICAL_ARCHIVE if archived and not active_reference else ExperimentState.COMPLETE_EVIDENCE
        elif reports or has_explicit_binding:
            state = ExperimentState.OPEN_INCOMPLETE
        else:
            state = ExperimentState.ORPHAN_REVIEW
            if _low_confidence_name_candidate(group_assets[0], known_run_ids):
                gaps.append("LOW_CONFIDENCE_NAME_ONLY")
            else:
                gaps.append("INSUFFICIENT_BINDING")

        local_paths = tuple(
            str(path) for asset, path in zip(group_assets, group_paths) if asset.location is Location.LOCAL and path is not None
        )
        n607_paths = tuple(
            str(path) for asset, path in zip(group_assets, group_paths) if asset.location is Location.N607 and path is not None
        )
        report_path = next((str(item.path) for item in reports if item.path is not None), None)
        phase_values = _value_set(group_claims, "phase")
        candidate_values = _value_set(group_claims, "method_or_candidate")
        predictions = sum(
            1
            for asset in group_assets
            if "prediction" in (asset.evidence_role or "").casefold() or "prediction" in asset.display_name.casefold()
        )
        scores = sum(1 for item in group_items if item.kind == "metrics")
        records[key] = ExperimentRecord(
            experiment_id=key,
            run_id=preferred_run_id,
            experiment_state=state,
            phase=str(phase_values[0]) if phase_values else None,
            method_or_candidate=str(candidate_values[0]) if candidate_values else None,
            report_path=report_path,
            local_artifact_paths=local_paths or None,
            n607_artifact_paths=n607_paths or None,
            git_commit=commits[0] if len(commits) == 1 else None,
            process_evidence=matched_process,
            prediction_count=predictions or None,
            score_count=scores or None,
            expected_artifacts=tuple(str(path) for path in expected_paths) or None,
            observed_artifacts=tuple(str(path) for path in observed_paths) or None,
            closure_gaps=tuple(dict.fromkeys(gaps)) or None,
        )
        claims_by_experiment[key] = group_claims

    return ExperimentIndex(records, claims=all_claims, claims_by_experiment=claims_by_experiment)


class ExperimentIndexer:
    """Small object wrapper for callers that retain fixed roots/process snapshots."""

    def __init__(
        self,
        *,
        root_paths: Mapping[Any, str | os.PathLike[str]],
        process_evidence: Iterable[ProcessEvidence | Mapping[str, Any]] = (),
    ) -> None:
        self._root_paths = dict(root_paths)
        self._process_evidence = tuple(process_evidence)

    def index(self, indexed_assets: Iterable[AssetRecord]) -> ExperimentIndex:
        return index_experiments(
            indexed_assets,
            root_paths=self._root_paths,
            process_evidence=self._process_evidence,
        )


__all__ = [
    "EvidenceClaim",
    "ExperimentIndex",
    "ExperimentIndexer",
    "MAX_EVIDENCE_BYTES",
    "ProcessEvidence",
    "index_experiments",
]
