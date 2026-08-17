"""Bounded, read-only collection of local project asset metadata."""

from __future__ import annotations

import hashlib
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from .config import DiscoveryConfig, LocationConfig
from .models import AccessStatus, AssetKind, AssetRecord, HashStatus, Location, ScopeResult
from .paths import escaped_display_name, normalize_relative_path, stable_asset_id


_CONTROL_SUFFIXES = {".json", ".md", ".markdown", ".py", ".sh", ".toml", ".yaml", ".yml"}
_PROTECTED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".h5", ".mat", ".tar", ".zip", ".7z"}
_SUMMARY_DIRECTORY_NAMES = {"prediction", "predictions", "score", "scores"}


def _is_reparse_point(entry: os.DirEntry[str], metadata: os.stat_result) -> bool:
    """Identify Windows junction-like entries without resolving their targets."""

    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


class LocalCollector:
    """Collect a fixed local root without following links or changing files."""

    def __init__(
        self, location_config: LocationConfig, discovery_config: DiscoveryConfig, *, scan_id: str
    ) -> None:
        if location_config.location is not Location.LOCAL:
            raise ValueError("LocalCollector requires a LOCAL configuration")
        if location_config.root_id != "TYPE10_7":
            raise ValueError("LOCAL root_id must be TYPE10_7")
        self._location = location_config
        self._discovery = discovery_config
        self._scan_id = scan_id
        self._records: dict[str, AssetRecord] = {}
        self._tags: dict[str, set[str]] = {}
        self._scopes: list[ScopeResult] = []

    def collect(self) -> tuple[tuple[AssetRecord, ...], tuple[ScopeResult, ...]]:
        """Return immutable record tuples for the configured root and carrier surfaces."""

        root = Path(self._location.root)
        self._scan_direct(root, "", "ROOT_DIRECT", descend_units=False)
        for surface in self._location.carrier_surfaces:
            relative_path = normalize_relative_path(surface.relative_path)
            self._scan_direct(
                root.joinpath(*relative_path.split("/")),
                relative_path,
                f"CARRIER_DIRECT:{relative_path}",
                descend_units=True,
            )
        return tuple(self._records.values()), tuple(self._scopes)

    def _scan_direct(
        self, directory: Path, relative_directory: str, tag: str, *, descend_units: bool
    ) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name.casefold())
        except FileNotFoundError:
            self._scopes.append(
                ScopeResult(
                    scan_id=self._scan_id,
                    location=Location.LOCAL,
                    root_id=self._location.root_id,
                    relative_path=relative_directory,
                    status="NOT_PRESENT",
                    asset_ids=(),
                )
            )
            return
        except OSError as exc:
            self._record_scan_error(relative_directory, str(exc))
            self._scopes.append(
                ScopeResult(
                    scan_id=self._scan_id,
                    location=Location.LOCAL,
                    root_id=self._location.root_id,
                    relative_path=relative_directory,
                    status="SCAN_ERROR",
                    error=str(exc),
                )
            )
            return

        asset_ids: list[str] = []
        units: list[tuple[Path, str]] = []
        for entry in entries:
            relative_path = self._join_relative(relative_directory, entry.name)
            record = self._record_entry(entry, relative_path, tag)
            asset_ids.append(record.asset_id)
            if descend_units and record.asset_kind is AssetKind.DIRECTORY:
                units.append((Path(entry.path), relative_path))
        self._scopes.append(
            ScopeResult(
                scan_id=self._scan_id,
                location=Location.LOCAL,
                root_id=self._location.root_id,
                relative_path=relative_directory,
                status="VERIFIED",
                asset_ids=tuple(asset_ids),
            )
        )
        for unit_path, unit_relative_path in units:
            self._scan_control_evidence(unit_path, unit_relative_path, depth=1)

    def _scan_control_evidence(self, directory: Path, relative_directory: str, *, depth: int) -> None:
        if depth > self._discovery.control_evidence_max_depth:
            return
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name.casefold())
        except OSError as exc:
            self._record_scan_error(relative_directory, str(exc))
            self._scopes.append(
                ScopeResult(
                    scan_id=self._scan_id,
                    location=Location.LOCAL,
                    root_id=self._location.root_id,
                    relative_path=relative_directory,
                    status="SCAN_ERROR",
                    error=str(exc),
                )
            )
            return

        for entry in entries:
            relative_path = self._join_relative(relative_directory, entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                self._record_scan_error(relative_path, str(exc))
                continue
            if entry.is_symlink():
                self._add_record(self._asset_record(entry, relative_path, AssetKind.SYMLINK, metadata), "LINK")
                continue
            if _is_reparse_point(entry, metadata):
                self._add_record(self._asset_record(entry, relative_path, AssetKind.JUNCTION, metadata), "JUNCTION")
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name.casefold() in _SUMMARY_DIRECTORY_NAMES:
                    self._add_record(
                        self._asset_record(entry, relative_path, AssetKind.DIRECTORY, metadata),
                        "PREDICTION_SCORE_SUMMARY",
                    )
                self._scan_control_evidence(Path(entry.path), relative_path, depth=depth + 1)
                continue
            if entry.is_file(follow_symlinks=False) and self._is_control_evidence(entry.name):
                self._add_record(
                    self._asset_record(entry, relative_path, AssetKind.FILE, metadata),
                    "CONTROL_EVIDENCE",
                )

    def _record_entry(self, entry: os.DirEntry[str], relative_path: str, tag: str) -> AssetRecord:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            return self._record_scan_error(relative_path, str(exc))
        if entry.is_symlink():
            kind = AssetKind.SYMLINK
        elif _is_reparse_point(entry, metadata):
            kind = AssetKind.JUNCTION
        elif entry.is_dir(follow_symlinks=False):
            kind = AssetKind.DIRECTORY
        elif entry.is_file(follow_symlinks=False):
            kind = AssetKind.FILE
        else:
            kind = AssetKind.OTHER
        return self._add_record(self._asset_record(entry, relative_path, kind, metadata), tag)

    def _asset_record(
        self, entry: os.DirEntry[str], relative_path: str, kind: AssetKind, metadata: os.stat_result
    ) -> AssetRecord:
        hash_status, digest = self._hash_metadata(Path(entry.path), entry.name, kind, metadata.st_size)
        display_name = normalize_relative_path(entry.name)
        return AssetRecord(
            asset_id=stable_asset_id(Location.LOCAL, self._location.root_id, relative_path),
            scan_id=self._scan_id,
            location=Location.LOCAL,
            root_id=self._location.root_id,
            relative_path=normalize_relative_path(relative_path),
            display_name=display_name,
            escaped_name=escaped_display_name(display_name),
            asset_kind=kind,
            size_bytes=metadata.st_size,
            mtime_utc=datetime.fromtimestamp(metadata.st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            access_status=AccessStatus.OK,
            hash_status=hash_status,
            sha256=digest,
        )

    def _hash_metadata(
        self, path: Path, name: str, kind: AssetKind, size_bytes: int
    ) -> tuple[HashStatus, str | None]:
        suffix = Path(name).suffix.casefold()
        if kind is not AssetKind.FILE or suffix in _PROTECTED_SUFFIXES or not self._is_control_evidence(name):
            return HashStatus.METADATA_ONLY, None
        if size_bytes > self._discovery.hash_max_bytes:
            return HashStatus.NOT_HASHED_SIZE_LIMIT, None
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return HashStatus.SHA256, digest.hexdigest()
        except OSError:
            return HashStatus.ERROR, None

    def _record_scan_error(self, relative_path: str, error: str) -> AssetRecord:
        if not relative_path:
            relative_path = "root"
        normalized = normalize_relative_path(relative_path)
        display_name = normalized.rsplit("/", 1)[-1]
        record = AssetRecord(
            asset_id=stable_asset_id(Location.LOCAL, self._location.root_id, normalized),
            scan_id=self._scan_id,
            location=Location.LOCAL,
            root_id=self._location.root_id,
            relative_path=normalized,
            display_name=display_name,
            escaped_name=escaped_display_name(display_name),
            asset_kind=AssetKind.DIRECTORY,
            size_bytes=None,
            mtime_utc=None,
            access_status=AccessStatus.SCAN_ERROR,
            hash_status=HashStatus.ERROR,
            sha256=None,
            decision_reason=error,
        )
        return self._add_record(record, "SCAN_ERROR")

    def _add_record(self, record: AssetRecord, tag: str) -> AssetRecord:
        identity = record.asset_id
        tags = self._tags.setdefault(identity, set())
        tags.add(tag)
        existing = self._records.get(identity)
        selected = record if existing is None or record.access_status is AccessStatus.SCAN_ERROR else existing
        merged = AssetRecord(
            asset_id=selected.asset_id,
            scan_id=selected.scan_id,
            location=selected.location,
            root_id=selected.root_id,
            relative_path=selected.relative_path,
            display_name=selected.display_name,
            escaped_name=selected.escaped_name,
            asset_kind=selected.asset_kind,
            size_bytes=existing.size_bytes if existing is not None and selected.size_bytes is None else selected.size_bytes,
            mtime_utc=existing.mtime_utc if existing is not None and selected.mtime_utc is None else selected.mtime_utc,
            access_status=selected.access_status,
            hash_status=selected.hash_status,
            sha256=selected.sha256,
            experiment_id=selected.experiment_id,
            git_ownership=selected.git_ownership,
            evidence_role="|".join(sorted(tags)),
            retention_class=selected.retention_class,
            recommended_action=selected.recommended_action,
            decision_reason=selected.decision_reason,
        )
        self._records[identity] = merged
        return merged

    @staticmethod
    def _is_control_evidence(name: str) -> bool:
        normalized_name = name.casefold()
        return Path(normalized_name).suffix in _CONTROL_SUFFIXES or "manifest" in normalized_name or "receipt" in normalized_name

    @staticmethod
    def _join_relative(parent: str, name: str) -> str:
        return normalize_relative_path(f"{parent}/{name}" if parent else name)


__all__ = ["LocalCollector"]
