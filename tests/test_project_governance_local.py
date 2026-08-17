from __future__ import annotations

import ast
import codecs
import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace

from tools.project_governance.collect_local import LocalCollector
from tools.project_governance.config import CarrierSurface, DiscoveryConfig, LocationConfig
from tools.project_governance.models import AccessStatus, AssetKind, HashStatus, Location


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                relative_path = Path(entry.path).relative_to(root).as_posix()
                snapshot[relative_path] = (metadata.st_size, metadata.st_mtime_ns)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
    return snapshot


def _freeze_mtimes(root: Path) -> None:
    fixed_ns = 1_700_000_000_000_000_000
    for directory, _, filenames in os.walk(root, topdown=False, followlinks=False):
        for filename in filenames:
            path = Path(directory) / filename
            if not path.is_symlink():
                os.utime(path, ns=(fixed_ns, fixed_ns))
        os.utime(directory, ns=(fixed_ns, fixed_ns))


def _fixture_config(root: Path) -> LocationConfig:
    return LocationConfig(
        location=Location.LOCAL,
        root_id="TYPE10_7",
        root=str(root),
        carrier_surfaces=(
            CarrierSurface(relative_path="runs", status="PRESENT"),
            CarrierSurface(relative_path="missing", status="NOT_PRESENT"),
        ),
    )


def test_collects_fixture_metadata_without_following_links_or_mutating(tmp_path, monkeypatch):
    root = tmp_path / "fixture-root"
    runs = root / "runs"
    root.mkdir()
    runs.mkdir()
    (root / "normal.md").write_text("normal control evidence", encoding="utf-8")
    (root / "empty.bin").write_bytes(b"")
    (root / "cafe\u0301-\u2603-receipt.md").write_text("unicode control evidence", encoding="utf-8")
    manifest = runs / "manifest.json"
    manifest.write_text('{"fixture": true}', encoding="utf-8")
    checkpoint = runs / "checkpoint.pth"
    checkpoint.write_bytes(b"x")
    with checkpoint.open("r+b") as stream:
        stream.truncate(10 * 1024 * 1024 + 1)
    large_receipt = runs / "large-receipt.md"
    large_receipt.write_bytes(b"x")
    with large_receipt.open("r+b") as stream:
        stream.truncate(10 * 1024 * 1024 + 1)
    denied = runs / "denied"
    denied.mkdir()
    (denied / "private.json").write_text("not readable", encoding="utf-8")
    junction = runs / "junction"
    junction.mkdir()
    (junction / "hidden.json").write_text("must not be traversed", encoding="utf-8")
    outside = root / "outside"
    outside.mkdir()
    (outside / "external.json").write_text("must not be traversed", encoding="utf-8")
    os.symlink(outside, runs / "link", target_is_directory=True)
    unit = runs / "run-1"
    (unit / "predictions").mkdir(parents=True)
    (unit / "predictions" / "result.csv").write_text("not a control file", encoding="utf-8")
    (unit / "a" / "b").mkdir(parents=True)
    (unit / "a" / "b" / "receipt.toml").write_text("ok = true", encoding="utf-8")
    (unit / "a" / "b" / "c").mkdir()
    (unit / "a" / "b" / "c" / "too_deep.json").write_text("not scanned", encoding="utf-8")
    _freeze_mtimes(root)
    before = _snapshot_tree(root)

    from tools.project_governance import collect_local

    original_scandir = collect_local.os.scandir

    def guarded_scandir(path):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(os.fspath(denied)):
            raise PermissionError("fixture access denial")
        return original_scandir(path)

    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(collect_local, "_is_reparse_point", lambda entry, metadata: entry.name == "junction")
        fixture_patch.setattr(collect_local.os, "scandir", guarded_scandir)
        records, scopes = LocalCollector(
            _fixture_config(root),
            DiscoveryConfig(
                control_evidence_max_depth=3,
                hash_max_bytes=10 * 1024 * 1024,
                text_read_max_bytes=2 * 1024 * 1024,
            ),
            scan_id="FIXTURE_SCAN",
        ).collect()

    after = _snapshot_tree(root)
    assert after == before
    by_path = {record.relative_path: record for record in records}
    zero = by_path["empty.bin"]
    manifest_record = by_path["runs/manifest.json"]
    checkpoint_record = by_path["runs/checkpoint.pth"]
    large_receipt_record = by_path["runs/large-receipt.md"]
    denied_record = by_path["runs/denied"]
    abnormal_record = by_path["caf\u00e9-\u2603-receipt.md"]

    assert zero.retention_class is None
    assert zero.recommended_action == "REVIEW"
    assert checkpoint_record.hash_status is HashStatus.METADATA_ONLY
    assert large_receipt_record.hash_status is HashStatus.NOT_HASHED_SIZE_LIMIT
    assert manifest_record.hash_status is HashStatus.SHA256
    assert manifest_record.sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert denied_record.access_status is AccessStatus.SCAN_ERROR
    assert codecs.decode(abnormal_record.escaped_name.encode("ascii"), "unicode_escape") == abnormal_record.display_name
    assert not any(record.relative_path.startswith("runs/link/") for record in records)
    assert "runs/junction/hidden.json" not in by_path
    assert "runs/run-1/predictions" in by_path
    assert "runs/run-1/predictions/result.csv" not in by_path
    assert "runs/run-1/a/b/receipt.toml" in by_path
    assert "runs/run-1/a/b/c/too_deep.json" not in by_path
    assert any(scope.relative_path == "runs/denied" and scope.status == "SCAN_ERROR" for scope in scopes)
    assert any(scope.relative_path == "missing" and scope.status == "NOT_PRESENT" for scope in scopes)


def test_missing_configured_root_is_a_scan_error_but_missing_carriers_remain_not_present(tmp_path):
    root = tmp_path / "missing-root"

    records, scopes = LocalCollector(
        _fixture_config(root),
        DiscoveryConfig(
            control_evidence_max_depth=3,
            hash_max_bytes=10 * 1024 * 1024,
            text_read_max_bytes=2 * 1024 * 1024,
        ),
        scan_id="MISSING_ROOT_SCAN",
    ).collect()

    statuses = {scope.relative_path: scope.status for scope in scopes}
    assert statuses == {"": "SCAN_ERROR", "runs": "NOT_PRESENT", "missing": "NOT_PRESENT"}
    assert any(
        record.relative_path == "root" and record.access_status is AccessStatus.SCAN_ERROR
        for record in records
    )


def test_collect_local_has_no_destructive_call_sites():
    source = (
        Path(__file__).resolve().parents[1] / "tools" / "project_governance" / "collect_local.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"unlink", "remove", "rmdir", "rmtree", "rename", "replace", "chmod", "chown", "kill"}
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not any(attribute.endswith(name) for attribute in called_attributes for name in banned)


def test_deduplicates_case_insensitive_local_identity_and_merges_coverage_tags(tmp_path):
    root = tmp_path / "fixture-root"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "manifest.json").write_text("{}", encoding="utf-8")
    config = LocationConfig(
        location=Location.LOCAL,
        root_id="TYPE10_7",
        root=str(root),
        carrier_surfaces=(
            CarrierSurface(relative_path="runs", status="PRESENT"),
            CarrierSurface(relative_path="RUNS", status="PRESENT"),
        ),
    )

    records, _ = LocalCollector(
        config,
        DiscoveryConfig(
            control_evidence_max_depth=3,
            hash_max_bytes=10 * 1024 * 1024,
            text_read_max_bytes=2 * 1024 * 1024,
        ),
        scan_id="DEDUP_SCAN",
    ).collect()

    manifests = [record for record in records if record.relative_path.casefold() == "runs/manifest.json"]
    assert len(manifests) == 1
    assert manifests[0].evidence_role == "CARRIER_DIRECT:RUNS|CARRIER_DIRECT:runs"


def test_does_not_scan_carrier_surfaces_that_are_links_or_junctions(tmp_path, monkeypatch):
    root = tmp_path / "fixture-root"
    root.mkdir()
    linked_target = root / "linked-target"
    linked_target.mkdir()
    (linked_target / "secret.json").write_text("external", encoding="utf-8")
    os.symlink(linked_target, root / "linked", target_is_directory=True)
    junction = root / "junction"
    junction.mkdir()
    (junction / "secret.json").write_text("external", encoding="utf-8")
    config = LocationConfig(
        location=Location.LOCAL,
        root_id="TYPE10_7",
        root=str(root),
        carrier_surfaces=(
            CarrierSurface(relative_path="linked", status="PRESENT"),
            CarrierSurface(relative_path="junction", status="PRESENT"),
        ),
    )

    from tools.project_governance import collect_local

    original_lstat = collect_local.os.lstat
    original_reparse_check = collect_local._is_reparse_point

    def junction_lstat(path, *args, **kwargs):
        metadata = original_lstat(path, *args, **kwargs)
        if os.path.normcase(os.fspath(path)) == os.path.normcase(os.fspath(junction)):
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
            )
        return metadata

    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(collect_local.os, "lstat", junction_lstat)
        fixture_patch.setattr(
            collect_local,
            "_is_reparse_point",
            lambda entry, metadata: entry.name == "junction" or original_reparse_check(entry, metadata),
        )
        records, _ = LocalCollector(
            config,
            DiscoveryConfig(
                control_evidence_max_depth=3,
                hash_max_bytes=10 * 1024 * 1024,
                text_read_max_bytes=2 * 1024 * 1024,
            ),
            scan_id="CARRIER_LINK_SCAN",
        ).collect()

    by_path = {record.relative_path: record for record in records}
    assert by_path["linked"].asset_kind is AssetKind.SYMLINK
    assert by_path["junction"].asset_kind is AssetKind.JUNCTION
    assert "linked/secret.json" not in by_path
    assert "junction/secret.json" not in by_path


def test_prediction_summary_does_not_collect_control_files_below_it(tmp_path):
    root = tmp_path / "fixture-root"
    prediction_config = root / "runs" / "run-1" / "predictions" / "config.json"
    prediction_config.parent.mkdir(parents=True)
    prediction_config.write_text('{"must_not_be_read": true}', encoding="utf-8")

    records, _ = LocalCollector(
        _fixture_config(root),
        DiscoveryConfig(
            control_evidence_max_depth=3,
            hash_max_bytes=10 * 1024 * 1024,
            text_read_max_bytes=2 * 1024 * 1024,
        ),
        scan_id="PREDICTION_BOUNDARY_SCAN",
    ).collect()

    by_path = {record.relative_path: record for record in records}
    assert "runs/run-1/predictions" in by_path
    assert "runs/run-1/predictions/config.json" not in by_path


def test_hash_read_failure_emits_scan_error_record(tmp_path, monkeypatch):
    root = tmp_path / "fixture-root"
    manifest = root / "runs" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"fixture": true}', encoding="utf-8")

    from tools.project_governance import collect_local

    original_open = collect_local.Path.open

    def denied_open(path, *args, **kwargs):
        if Path(path) == manifest and args and args[0] == "rb":
            raise PermissionError("fixture hash denial")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(collect_local.Path, "open", denied_open)
        records, _ = LocalCollector(
            _fixture_config(root),
            DiscoveryConfig(
                control_evidence_max_depth=3,
                hash_max_bytes=10 * 1024 * 1024,
                text_read_max_bytes=2 * 1024 * 1024,
            ),
            scan_id="HASH_ERROR_SCAN",
        ).collect()

    manifest_record = {record.relative_path: record for record in records}["runs/manifest.json"]
    assert manifest_record.access_status is AccessStatus.SCAN_ERROR
    assert manifest_record.hash_status is HashStatus.ERROR
