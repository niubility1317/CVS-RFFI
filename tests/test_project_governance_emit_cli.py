from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import runpy
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.project_governance.emit import ReportEmitter
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


def _asset(
    relative_path: str,
    *,
    location: Location = Location.LOCAL,
    access_status: AccessStatus = AccessStatus.OK,
    evidence_role: str | None = "report",
    size_bytes: int = 12,
) -> AssetRecord:
    root_id = "TYPE10_7" if location is Location.LOCAL else "N607_CVS_SINCNET"
    return AssetRecord(
        asset_id=f"asset:{location.value}:{root_id}:{relative_path}",
        scan_id="EMIT_FIXTURE",
        location=location,
        root_id=root_id,
        relative_path=relative_path,
        display_name=relative_path.rsplit("/", 1)[-1],
        escaped_name=relative_path.rsplit("/", 1)[-1],
        asset_kind=AssetKind.FILE,
        size_bytes=size_bytes,
        mtime_utc="2026-08-17T00:00:00Z",
        access_status=access_status,
        hash_status=HashStatus.SHA256,
        sha256=hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
        experiment_id="RUN_B",
        git_ownership=GitOwnership.TRACKED_GIT if location is Location.LOCAL else None,
        evidence_role=evidence_role,
        retention_class=RetentionClass.KEEP_IMMUTABLE,
        recommended_action="KEEP",
        decision_reason="FIXTURE_EVIDENCE",
    )


def _bundle(*, include_error: bool = True, long_path: str | None = None) -> ScanBundle:
    assets = [
        _asset("zeta/report.md"),
        _asset("alpha/manifest.json"),
        _asset("runs/receipt.json", location=Location.N607),
    ]
    if include_error:
        assets.append(
            replace(
                _asset("denied/error.json", evidence_role="text"),
                access_status=AccessStatus.SCAN_ERROR,
                hash_status=HashStatus.ERROR,
                sha256=None,
            )
        )
    if long_path is not None:
        assets.append(_asset(long_path, size_bytes=len(long_path)))
    return ScanBundle(
        scan_id="EMIT_FIXTURE",
        schema_version=1,
        operator="test-operator",
        started_at_utc="2026-08-17T01:02:03Z",
        completed_at_utc="2026-08-17T01:02:04Z",
        assets=tuple(assets),
        scope_results=(
            ScopeResult("EMIT_FIXTURE", Location.LOCAL, "TYPE10_7", "runs", "VERIFIED"),
            ScopeResult("EMIT_FIXTURE", Location.N607, "N607_CVS_SINCNET", "logs", "SCAN_ERROR", error="read denied"),
        ),
        git_ownership=(
            GitOwnershipRecord(
                asset_id=assets[0].asset_id,
                ownership=GitOwnership.TRACKED_GIT,
                repository_root="E:/type10-7",
                branch="codex/project-governance-20260813",
                head_commit="abc123",
                status_summary="clean",
            ),
        ),
        experiments=(
            ExperimentRecord(
                experiment_id="RUN_B",
                run_id="RUN_B",
                experiment_state=ExperimentState.OPEN_INCOMPLETE,
                phase="governance-fixture",
                method_or_candidate="fixture",
                report_path="docs/report.md",
                local_artifact_paths=("runs/B/prediction.json",),
                n607_artifact_paths=("runs/B/receipt.json",),
                closure_gaps=("missing score",),
            ),
            ExperimentRecord(
                experiment_id="RUN_A",
                run_id="RUN_A",
                experiment_state=ExperimentState.COMPLETE_EVIDENCE,
                phase="governance-fixture",
                method_or_candidate="fixture",
                prediction_count=2,
                score_count=2,
            ),
        ),
        retention_decisions=(
            RetentionDecision(
                asset_id=assets[0].asset_id,
                retention_class=RetentionClass.KEEP_IMMUTABLE,
                rule_code="PROTECTED_EVIDENCE",
                reason="fixture",
            ),
        ),
        deletion_candidates=(
            DeletionCandidate(
                candidate_id="DELETE_B",
                location=Location.LOCAL,
                absolute_path="E:/type10-7/cache/B",
                asset_kind=AssetKind.DIRECTORY,
                size_bytes=42,
                reason="fixture candidate",
                approval_state=ApprovalState.AWAITING_USER_APPROVAL,
                execution_state=ExecutionState.NOT_AUTHORIZED,
            ),
        ),
    )


def _attempt_metadata(
    label: str,
    child_pid: int,
    *,
    proxy_child_pids: tuple[int, ...] = (),
    returncode: int | None = 0,
    timed_out: bool = False,
    child_exited: bool = True,
    proxy_children_exited: bool = True,
    disconnect_status: str = "VERIFIED",
    lingering_connections: tuple[dict[str, object], ...] = (),
    stderr_tail: str = "",
) -> dict[str, object]:
    return {
        "label": label,
        "child_pid": child_pid,
        "proxy_child_pids": proxy_child_pids,
        "returncode": returncode,
        "timed_out": timed_out,
        "child_exited": child_exited,
        "proxy_children_exited": proxy_children_exited,
        "disconnect_status": disconnect_status,
        "lingering_connections": lingering_connections,
        "stderr_tail": stderr_tail,
    }


def _metadata() -> dict[str, object]:
    return {
        "local_root": "E:/type10-7",
        "n607_root": "/home/szu2070436088/2510044040/CV-SincNet",
        "local_scopes": ["runs", "logs"],
        "n607_scopes": ["runs", "logs"],
        "implementation_git_head": "abc123",
        "git_tracked_diff_state": "clean",
        "collector_versions": {"local": "1", "n607": "1"},
        "n607_requested": True,
        "n607_outcome": "VERIFIED",
        "n607_route": "DIRECT",
        "n607_preflight": "DIRECT_READY",
        "n607_disconnect": "VERIFIED",
        "n607_attempts": (
            _attempt_metadata("PREFLIGHT", 4100, proxy_child_pids=(4101,)),
            _attempt_metadata("DIRECT", 4200),
        ),
        "n607_active_training_observed": False,
        "n607_scan_error_count": 0,
    }


def test_emitter_produces_stable_small_outputs_with_required_encodings(tmp_path):
    emitter = ReportEmitter(
        _bundle(),
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=_metadata(),
    )

    result = emitter.emit()
    output = tmp_path / "git" / "EMIT_FIXTURE"
    expected = {
        "report.md",
        "asset_inventory_local.csv",
        "asset_inventory_n607.csv",
        "experiment_index.csv",
        "git_ownership.csv",
        "retention_decisions.csv",
        "deletion_candidates.csv",
        "asset_inventory_full.json",
        "scan_receipt.json",
    }

    assert {path.name for path in output.iterdir()} == expected
    assert output.joinpath("asset_inventory_local.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert not output.joinpath("asset_inventory_full.json").read_bytes().startswith(b"\xef\xbb\xbf")
    assert not output.joinpath("scan_receipt.json").read_bytes().startswith(b"\xef\xbb\xbf")
    assert json.loads(output.joinpath("asset_inventory_full.json").read_text(encoding="utf-8"))["scan_id"] == "EMIT_FIXTURE"
    assert result.git_output_dir == output

    local_csv = output.joinpath("asset_inventory_local.csv").read_text(encoding="utf-8-sig")
    assert local_csv.index("alpha/manifest.json") < local_csv.index("zeta/report.md")
    receipt = json.loads(output.joinpath("scan_receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 1
    assert receipt["scan_id"] == "EMIT_FIXTURE"
    assert receipt["counts"]["assets"] == 4
    assert receipt["scan_error_counts"]["assets"] == 1
    assert receipt["scan_error_counts"]["scopes"] == 1
    assert receipt["source_asset_mutations"] == 0
    assert receipt["moves"] == receipt["overwrites"] == receipt["deletions"] == 0
    assert receipt["authorized_deletion_rows"] == 0
    assert all(row["approval_state"] == "AWAITING_USER_APPROVAL" for row in receipt["deletion_rows"])
    assert all(path["sha256"] and path["bytes"] >= 0 for path in receipt["files"])


@pytest.mark.parametrize(
    "formula_path",
    (
        "=1+1",
        "+SUM(A1:A2)",
        "-2+3",
        "@cmd",
        "\t=1+1",
        "\v=1+1",
        "\f=1+1",
        "\u00a0=1+1",
    ),
)
def test_emitter_neutralizes_text_formula_prefixes_in_csv_only(tmp_path, formula_path):
    bundle = replace(_bundle(), assets=(_asset(formula_path),))

    result = ReportEmitter(
        bundle,
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=_metadata(),
    ).emit()
    csv_path = result.git_output_dir / "asset_inventory_local.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    full = json.loads(
        result.git_output_dir.joinpath("asset_inventory_full.json").read_text(encoding="utf-8")
    )

    assert rows[0]["relative_path"] == "'" + formula_path
    assert full["assets"][0]["relative_path"] == formula_path


def test_emitter_is_immutable_on_second_scan_id_and_receipt_is_last(tmp_path):
    writes: list[str] = []

    class RecordingEmitter(ReportEmitter):
        def _write_exclusive(self, path, payload, *, encoding="utf-8", newline=""):
            writes.append(path.name)
            return super()._write_exclusive(
                path, payload, encoding=encoding, newline=newline
            )

    emitter = RecordingEmitter(
        _bundle(),
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=_metadata(),
    )
    first = emitter.emit()
    before = {path.name: path.read_bytes() for path in first.git_output_dir.iterdir()}

    with pytest.raises(FileExistsError):
        ReportEmitter(
            _bundle(),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        ).emit()

    after = {path.name: path.read_bytes() for path in first.git_output_dir.iterdir()}
    assert after == before
    assert writes[-1] == "scan_receipt.json"


@pytest.mark.parametrize(
    "scan_id",
    (
        "C:",
        "C:outside",
        "D:relative",
        "COM¹",
        "COM¹.txt",
        "COM²",
        "COM³",
        "LPT¹",
        "LPT².txt",
        "LPT³",
        "CONIN$",
        "CONOUT$",
        "CONIN$.txt",
    ),
)
def test_emitter_rejects_windows_drive_relative_scan_ids_before_writing(tmp_path, scan_id):
    original = _bundle()
    bundle = replace(
        original,
        scan_id=scan_id,
        assets=tuple(replace(asset, scan_id=scan_id) for asset in (original.assets or ())),
        scope_results=tuple(
            replace(scope, scan_id=scan_id) for scope in (original.scope_results or ())
        ),
    )

    with pytest.raises(ValueError, match="scan_id"):
        ReportEmitter(
            bundle,
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()
    assert not (tmp_path / "external").exists()


@pytest.mark.parametrize(
    "status", (AccessStatus.SCAN_ERROR, "SCAN_ERROR ", " scan_error", "UNKNOWN")
)
def test_emitter_rejects_noncanonical_scope_status_before_writing(tmp_path, status):
    bundle = _bundle()
    scope = replace((bundle.scope_results or ())[0], status=status)

    with pytest.raises(ValueError, match="scope_results"):
        ReportEmitter(
            replace(bundle, scope_results=(scope,)),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()
    assert not (tmp_path / "external").exists()


@pytest.mark.parametrize(
    ("git_parts", "external_parts"),
    (
        (("shared",), ("shared",)),
        (("git",), ("git", "external")),
        (("external", "git"), ("external",)),
    ),
)
def test_emitter_rejects_overlapping_output_roots_before_writing(
    tmp_path, git_parts, external_parts
):
    output_root = tmp_path.joinpath(*git_parts)
    external_output_root = tmp_path.joinpath(*external_parts)

    with pytest.raises(ValueError, match="output roots"):
        ReportEmitter(
            _bundle(),
            output_root=output_root,
            external_output_root=external_output_root,
            metadata=_metadata(),
        )

    assert not output_root.exists()
    assert not external_output_root.exists()


@pytest.mark.parametrize(
    ("collection", "field", "value"),
    (
        ("assets", "location", "LOCAL"),
        ("assets", "asset_kind", "file"),
        ("assets", "access_status", "SCAN_ERROR"),
        ("assets", "hash_status", "SHA256"),
        ("assets", "git_ownership", "TRACKED_GIT"),
        ("assets", "retention_class", "KEEP_IMMUTABLE"),
        ("scope_results", "location", "LOCAL"),
        ("git_ownership", "ownership", "GIT_STATE_ERROR"),
        ("experiments", "experiment_state", "SCAN_ERROR"),
        ("retention_decisions", "retention_class", "KEEP_IMMUTABLE"),
        ("deletion_candidates", "location", "LOCAL"),
        ("deletion_candidates", "asset_kind", "directory"),
    ),
)
def test_emitter_rejects_untyped_nested_enum_values_before_writing(
    tmp_path, collection, field, value
):
    bundle = _bundle()
    record = (getattr(bundle, collection) or ())[0]
    invalid = replace(record, **{field: value})

    with pytest.raises(ValueError, match=collection):
        ReportEmitter(
            replace(bundle, **{collection: (invalid,)}),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()


def test_emitter_rejects_wrong_nested_record_type_before_writing(tmp_path):
    with pytest.raises(ValueError, match="assets"):
        ReportEmitter(
            replace(_bundle(), assets=({"access_status": "SCAN_ERROR"},)),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()


@pytest.mark.parametrize("collection", ("assets", "scope_results"))
def test_emitter_rejects_nested_scan_id_mismatch_before_writing(tmp_path, collection):
    bundle = _bundle()
    record = (getattr(bundle, collection) or ())[0]
    invalid = replace(record, scan_id="OTHER_SCAN")

    with pytest.raises(ValueError, match="scan_id"):
        ReportEmitter(
            replace(bundle, **{collection: (invalid,)}),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()


def test_emitter_reuses_one_timestamp_when_bundle_times_are_missing(tmp_path):
    bundle = replace(_bundle(), started_at_utc=None, completed_at_utc=None)
    result = ReportEmitter(
        bundle,
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=_metadata(),
    ).emit()
    full = json.loads(result.git_output_dir.joinpath("asset_inventory_full.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        result.git_output_dir.joinpath("scan_receipt.json").read_text(encoding="utf-8")
    )

    assert full["started_at_utc"] == full["completed_at_utc"]
    assert receipt["started_at_utc"] == receipt["completed_at_utc"] == receipt["emitted_at_utc"]
    assert receipt["started_at_utc"].endswith("Z")


def test_emitter_rejects_invalid_timestamp_before_creating_output(tmp_path):
    bundle = replace(_bundle(), started_at_utc="not-an-iso-timestamp")

    with pytest.raises(ValueError, match="ISO 8601"):
        ReportEmitter(
            bundle,
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()


def test_emitter_retains_partial_output_without_receipt_on_failure(tmp_path):
    class FailingEmitter(ReportEmitter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._writes = 0

        def _write_exclusive(self, path, payload, *, encoding, newline=""):
            self._writes += 1
            if self._writes == 2:
                raise OSError("fixture write failure")
            return super()._write_exclusive(path, payload, encoding=encoding, newline=newline)

    with pytest.raises(OSError, match="fixture write failure"):
        FailingEmitter(
            _bundle(),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        ).emit()
    output = tmp_path / "git" / "EMIT_FIXTURE"
    assert output.exists()
    assert not output.joinpath("scan_receipt.json").exists()
    assert len(tuple(output.iterdir())) == 1


def test_emitter_routes_complete_oversized_tables_without_truncating_evidence(tmp_path):
    long_path = "runs/" + ("x" * 100) + ".json"
    original = _bundle(long_path=long_path)
    bundle = replace(
        original,
        assets=tuple(original.assets or ())
        + tuple(_asset(f"bulk/{index:03d}.json") for index in range(40)),
    )
    emitter = ReportEmitter(
        bundle,
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=_metadata(),
        git_file_max_bytes=10_000,
        git_scan_max_bytes=100_000,
    )

    result = emitter.emit()
    git_output = tmp_path / "git" / "EMIT_FIXTURE"
    external_output = tmp_path / "external" / "EMIT_FIXTURE"
    receipt = json.loads(git_output.joinpath("scan_receipt.json").read_text(encoding="utf-8"))

    assert result.external_output_dir == external_output
    assert external_output.exists()
    assert any(path.suffix == ".csv" for path in external_output.iterdir())
    assert any(path.name.endswith(".summary.json") for path in git_output.iterdir())
    assert any(path.name.endswith(".part.csv") for path in git_output.iterdir())
    assert all(path.stat().st_size <= 10_000 for path in git_output.iterdir())
    assert receipt["external_files"]
    assert all(Path(item["path"]).is_absolute() for item in receipt["external_files"])
    full_external_csv = next(path for path in external_output.iterdir() if path.name == "asset_inventory_local.csv")
    assert long_path in full_external_csv.read_text(encoding="utf-8-sig")


def test_emitter_fails_when_a_csv_row_cannot_fit_a_git_shard(tmp_path):
    long_path = "runs/" + ("x" * 512) + ".json"

    with pytest.raises(ValueError, match="CSV row"):
        ReportEmitter(
            _bundle(long_path=long_path),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
            git_file_max_bytes=128,
            git_scan_max_bytes=256,
        ).emit()

    output = tmp_path / "git" / "EMIT_FIXTURE"
    external_output = tmp_path / "external" / "EMIT_FIXTURE"
    expected_external = {
        "report.md",
        "asset_inventory_local.csv",
        "asset_inventory_n607.csv",
        "experiment_index.csv",
        "git_ownership.csv",
        "retention_decisions.csv",
        "deletion_candidates.csv",
        "asset_inventory_full.json",
    }
    assert output.exists()
    assert {path.name for path in external_output.iterdir()} == expected_external
    assert not output.joinpath("scan_receipt.json").exists()


def test_emitter_treats_embedded_newline_as_one_csv_row_for_shard_limit(tmp_path):
    multiline = "runs/" + "\n".join("x" * 120 for _ in range(8)) + ".json"
    bundle = replace(
        _bundle(),
        assets=tuple(_bundle().assets or ()) + (_asset(multiline, location=Location.N607),),
    )

    with pytest.raises(ValueError, match="CSV row"):
        ReportEmitter(
            bundle,
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
            git_file_max_bytes=900,
            git_scan_max_bytes=256,
        ).emit()

    assert not (tmp_path / "git" / "EMIT_FIXTURE" / "scan_receipt.json").exists()


def test_emitter_preserves_embedded_newline_in_complete_csv_shards(tmp_path):
    multiline = "runs/line one\nline two.json"
    original = _bundle()
    bundle = replace(
        original,
        assets=tuple(original.assets or ())
        + (_asset(multiline, location=Location.N607),)
        + tuple(_asset(f"bulk/{index:03d}.json") for index in range(40)),
    )
    result = ReportEmitter(
        bundle,
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=_metadata(),
        git_file_max_bytes=10_000,
        git_scan_max_bytes=100_000,
    ).emit()

    external_csv = (result.external_output_dir or Path()) / "asset_inventory_n607.csv"
    with external_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        external_rows = list(csv.reader(stream))
    shard_rows: list[list[str]] = []
    for shard in sorted(result.git_output_dir.glob("asset_inventory_n607.*.part.csv")):
        with shard.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == external_rows[0]
        shard_rows.extend(rows[1:])

    assert sorted(shard_rows) == sorted(external_rows[1:])
    assert any(row[4] == multiline for row in shard_rows)


def test_emitter_fails_instead_of_writing_an_oversized_receipt(tmp_path):
    bundle = ScanBundle(
        scan_id="SMALL_RECEIPT_LIMIT",
        operator="fixture",
        started_at_utc="2026-08-17T01:02:03Z",
        completed_at_utc="2026-08-17T01:02:04Z",
    )

    with pytest.raises(ValueError, match="receipt exceeds"):
        ReportEmitter(
            bundle,
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
            git_file_max_bytes=512,
            git_scan_max_bytes=256,
        ).emit()

    output = tmp_path / "git" / "SMALL_RECEIPT_LIMIT"
    assert output.exists()
    assert not output.joinpath("scan_receipt.json").exists()


def test_emitter_refuses_terminal_receipt_when_git_shards_exceed_scan_limit(tmp_path):
    base = _bundle(include_error=False)
    bundle = replace(
        base,
        assets=tuple(
            _asset(f"bulk/{index:03d}-{'x' * 80}.json")
            for index in range(80)
        ),
    )

    with pytest.raises(ValueError, match="git output exceeds scan threshold"):
        ReportEmitter(
            bundle,
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
            git_file_max_bytes=8192,
            git_scan_max_bytes=1024,
        ).emit()

    assert (tmp_path / "external" / "EMIT_FIXTURE" / "asset_inventory_full.json").exists()
    assert not (tmp_path / "git" / "EMIT_FIXTURE" / "scan_receipt.json").exists()


@pytest.mark.parametrize(
    "updates",
    (
        {"approval_state": "APPROVED"},
        {"execution_state": "AUTHORIZED"},
        {"approved_scope": "DELETE_B"},
    ),
)
def test_emitter_rejects_any_authorized_deletion_candidate_before_writing(tmp_path, updates):
    original = _bundle()
    candidate = replace((original.deletion_candidates or ())[0], **updates)
    bundle = replace(original, deletion_candidates=(candidate,))

    with pytest.raises(ValueError, match="deletion candidate"):
        ReportEmitter(
            bundle,
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=_metadata(),
        )

    assert not (tmp_path / "git").exists()


def test_emitter_requires_complete_terminal_receipt_metadata(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        ReportEmitter(
            _bundle(),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=None,
        )

    assert not (tmp_path / "git").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n607_route", " DIRECT "),
        ("n607_preflight", " NOT_PROVIDED "),
        ("n607_disconnect", " VERIFIED "),
        ("n607_route", "arbitrary-route"),
        ("n607_preflight", "not_provided"),
        ("n607_preflight", "VERIFIED"),
        ("n607_disconnect", "CLEAN"),
    ),
)
def test_emitter_rejects_uncontrolled_requested_n607_states_before_writing(
    tmp_path, field, value
):
    metadata = _metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match="N607|metadata"):
        ReportEmitter(
            _bundle(),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=metadata,
        )

    assert not (tmp_path / "git").exists()


@pytest.mark.parametrize(
    ("outcome", "route", "preflight", "disconnect", "attempts"),
    (
        (
            "VERIFIED",
            "DIRECT",
            "DIRECT_READY",
            "VERIFIED",
            (
                _attempt_metadata("PREFLIGHT", 4100, proxy_child_pids=(4101,)),
                _attempt_metadata("DIRECT", 4200),
            ),
        ),
        (
            "VERIFIED",
            "LAB_BRIDGE",
            "DIRECT_PATH_UNAVAILABLE",
            "VERIFIED",
            (
                _attempt_metadata("PREFLIGHT", 4100, proxy_child_pids=(4101,)),
                _attempt_metadata("LAB_BRIDGE", 4200, proxy_child_pids=(4201,)),
            ),
        ),
        (
            "FAILED",
            "NO_ROUTE",
            "FAILED",
            "VERIFIED",
            (_attempt_metadata("PREFLIGHT", 4100, proxy_child_pids=(4101,), returncode=1),),
        ),
        (
            "UNKNOWN",
            "NO_ROUTE",
            "UNKNOWN",
            "UNKNOWN",
            (
                _attempt_metadata(
                    "PREFLIGHT",
                    4100,
                    proxy_child_pids=(4101,),
                    returncode=None,
                    timed_out=True,
                    child_exited=False,
                    proxy_children_exited=False,
                    disconnect_status="UNKNOWN",
                ),
            ),
        ),
    ),
)
def test_emitter_accepts_collector_n607_state_vocabulary(
    tmp_path, outcome, route, preflight, disconnect, attempts
):
    metadata = _metadata()
    metadata.update(
        {
            "n607_outcome": outcome,
            "n607_route": route,
            "n607_preflight": preflight,
            "n607_disconnect": disconnect,
            "n607_attempts": attempts,
        }
    )

    result = ReportEmitter(
        _bundle(),
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=metadata,
    ).emit()
    receipt = json.loads(
        result.git_output_dir.joinpath("scan_receipt.json").read_text(encoding="utf-8")
    )

    assert receipt["n607_evidence"]["requested"] is True
    assert receipt["n607_evidence"]["outcome"] == outcome
    assert receipt["n607_evidence"]["route"] == route
    assert receipt["n607_evidence"]["preflight"] == preflight
    assert receipt["n607_evidence"]["disconnect"] == disconnect
    assert [attempt["label"] for attempt in receipt["n607_evidence"]["attempts"]] == [
        attempt["label"] for attempt in attempts
    ]


@pytest.mark.parametrize(
    ("route", "preflight"),
    (
        ("DIRECT", "DIRECT_PATH_UNAVAILABLE"),
        ("LAB_BRIDGE", "DIRECT_READY"),
    ),
)
def test_emitter_rejects_inconsistent_requested_n607_route_and_preflight(
    tmp_path, route, preflight
):
    metadata = _metadata()
    metadata.update({"n607_route": route, "n607_preflight": preflight})

    with pytest.raises(ValueError, match="N607"):
        ReportEmitter(
            _bundle(),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=metadata,
        )

    assert not (tmp_path / "git").exists()


@pytest.mark.parametrize(
    "case",
    ("zero_pid", "oversized_stderr", "unapproved_endpoint", "missing_field", "missing_proxy"),
)
def test_emitter_rejects_malformed_n607_attempt_evidence_before_writing(tmp_path, case):
    metadata = _metadata()
    attempts = [dict(attempt) for attempt in metadata["n607_attempts"]]
    if case == "zero_pid":
        attempts[1]["child_pid"] = 0
    elif case == "oversized_stderr":
        attempts[1]["stderr_tail"] = "x" * 8193
    elif case == "unapproved_endpoint":
        attempts[1]["lingering_connections"] = (
            {"pid": 4200, "endpoint": "203.0.113.1:22", "state": "ESTABLISHED"},
        )
    elif case == "missing_field":
        attempts[1].pop("returncode")
    else:
        attempts[0]["proxy_child_pids"] = ()
    metadata["n607_attempts"] = tuple(attempts)

    with pytest.raises(ValueError, match="N607|n607"):
        ReportEmitter(
            _bundle(),
            output_root=tmp_path / "git",
            external_output_root=tmp_path / "external",
            metadata=metadata,
        )

    assert not (tmp_path / "git").exists()


def test_emitter_accepts_explicit_not_requested_n607_evidence(tmp_path):
    metadata = _metadata()
    metadata.update(
        {
            "n607_requested": False,
            "n607_outcome": "NOT_REQUESTED",
            "n607_route": "NOT_REQUESTED",
            "n607_preflight": "NOT_REQUESTED",
            "n607_disconnect": "NOT_REQUESTED",
            "n607_attempts": (),
            "n607_active_training_observed": False,
            "n607_scan_error_count": 0,
        }
    )

    result = ReportEmitter(
        _bundle(),
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=metadata,
    ).emit()
    receipt = json.loads(result.git_output_dir.joinpath("scan_receipt.json").read_text(encoding="utf-8"))

    assert receipt["n607_evidence"]["requested"] is False
    assert receipt["n607_evidence"] == {
        "requested": False,
        "outcome": "NOT_REQUESTED",
        "route": "NOT_REQUESTED",
        "preflight": "NOT_REQUESTED",
        "disconnect": "NOT_REQUESTED",
        "active_training_observed": False,
        "attempts": [],
    }


def test_emitter_receipt_keeps_remote_evidence_and_zero_execution_fields(tmp_path):
    result = ReportEmitter(
        _bundle(), output_root=tmp_path / "git", external_output_root=tmp_path / "external", metadata=_metadata()
    ).emit()
    receipt = json.loads(result.git_output_dir.joinpath("scan_receipt.json").read_text(encoding="utf-8"))

    assert receipt["roots"]["local"] == "E:/type10-7"
    assert receipt["roots"]["n607"] == "/home/szu2070436088/2510044040/CV-SincNet"
    assert receipt["n607_evidence"] == {
        "requested": True,
        "outcome": "VERIFIED",
        "route": "DIRECT",
        "preflight": "DIRECT_READY",
        "disconnect": "VERIFIED",
        "active_training_observed": False,
        "attempts": [
            {
                **_attempt_metadata("PREFLIGHT", 4100, proxy_child_pids=(4101,)),
                "proxy_child_pids": [4101],
                "lingering_connections": [],
            },
            {
                **_attempt_metadata("DIRECT", 4200),
                "proxy_child_pids": [],
                "lingering_connections": [],
            },
        ],
    }
    assert receipt["deletion_rows"][0]["execution_state"] == "NOT_AUTHORIZED"
    report = result.git_output_dir.joinpath("report.md").read_text(encoding="utf-8")
    assert "\u8bc1\u636e\u4f18\u5148" in report
    assert "\u5b9e\u9645\u79fb\u52a8\u3001\u8986\u76d6\u3001\u5220\u9664\u6570\u91cf\u4e3a0" in report
    assert "\u664b\u7ea7" not in report
    assert "promotion" not in report.casefold()


def test_emitter_reports_all_remote_errors_closure_gaps_and_approval_evidence(tmp_path):
    original = _bundle()
    candidate = replace(
        (original.deletion_candidates or ())[0],
        evidence=("asset:evidence",),
        dependencies=("asset:dependency",),
        recoverability="REGENERABLE_FROM_RETAINED_SOURCE",
        estimated_space_reclaim=42,
    )
    bundle = replace(original, deletion_candidates=(candidate,))
    metadata = _metadata()
    metadata["n607_scan_error_count"] = 3

    result = ReportEmitter(
        bundle,
        output_root=tmp_path / "git",
        external_output_root=tmp_path / "external",
        metadata=metadata,
    ).emit()
    receipt = json.loads(result.git_output_dir.joinpath("scan_receipt.json").read_text(encoding="utf-8"))
    report = result.git_output_dir.joinpath("report.md").read_text(encoding="utf-8")

    assert receipt["scan_error_counts"]["n607_records"] == 3
    assert "RUN_B:missing score" in report
    assert "asset:evidence" in report
    assert "asset:dependency" in report
    assert "REGENERABLE_FROM_RETAINED_SOURCE" in report


def test_emitter_has_no_destructive_execution_surface():
    assert not any(
        hasattr(ReportEmitter, name)
        for name in ("delete", "move", "overwrite", "cleanup", "execute_deletions")
    )


def _cli_fixture_config(root: Path):
    from tools.project_governance.config import (
        CarrierSurface,
        DiscoveryConfig,
        GovernanceConfig,
        LocationConfig,
        OutputConfig,
    )

    return GovernanceConfig(
        schema_version=1,
        local=LocationConfig(
            location=Location.LOCAL,
            root_id="TYPE10_7",
            root=str(root),
            carrier_surfaces=(CarrierSurface("runs", "PRESENT"),),
        ),
        n607=LocationConfig(
            location=Location.N607,
            root_id="N607_CVS_SINCNET",
            root="/home/szu2070436088/2510044040/CV-SincNet",
            carrier_surfaces=(CarrierSurface("runs", "NOT_PRESENT"),),
        ),
        discovery=DiscoveryConfig(
            control_evidence_max_depth=2,
            hash_max_bytes=1024 * 1024,
            text_read_max_bytes=1024 * 1024,
        ),
        output=OutputConfig(git_file_max_bytes=1024 * 1024, git_scan_max_bytes=4 * 1024 * 1024),
    )


def _cli_args(tmp_path: Path, *, include_n607: bool = False, print_plan: bool = False):
    return SimpleNamespace(
        command="scan",
        config=str(tmp_path / "fixture-config.json"),
        scan_id="CLI_FIXTURE",
        output_root=str(tmp_path / "governance-git"),
        external_output_root=str(tmp_path / "governance-external"),
        operator="fixture-operator",
        include_n607=include_n607,
        print_plan=print_plan,
    )


def _fake_git_runner_factory(root: Path):
    from tools.project_governance.collect_git import CommandResult

    calls: list[tuple[str, tuple[str, ...]]] = []

    def runner(cwd, args, *, input=b""):
        cwd_text = os.fspath(cwd)
        args = tuple(args)
        calls.append((cwd_text, args))
        if args == ("rev-parse", "--show-toplevel"):
            return CommandResult(0, os.fsencode(str(root) + "\n"), b"")
        if args == ("worktree", "list", "--porcelain"):
            return CommandResult(0, os.fsencode(f"worktree {root}\n"), b"")
        if args == ("rev-parse", "--git-common-dir"):
            return CommandResult(0, os.fsencode(str(root / ".git") + "\n"), b"")
        if args == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return CommandResult(0, b"main\n", b"")
        if args == ("rev-parse", "HEAD"):
            return CommandResult(0, b"fixture-head\n", b"")
        if args[:2] == ("status", "--porcelain=v2"):
            return CommandResult(0, b"", b"")
        if args[:2] == ("ls-files", "--stage"):
            paths = args[args.index("--") + 1 :]
            payload = b"".join(b"100644\t" + os.fsencode(path) + b"\x00" for path in paths)
            return CommandResult(0, payload, b"")
        if args[:2] == ("check-ignore", "-z"):
            return CommandResult(1, b"", b"")
        raise AssertionError(f"unexpected fake git command: {args!r}")

    return runner, calls


def _fake_n607_result(scan_id: str):
    from tools.project_governance.collect_n607 import (
        AttemptReceipt,
        N607CollectionResult,
        N607Receipt,
        RemoteOutcome,
    )

    asset_id = "asset:N607:N607_CVS_SINCNET:runs/fixture/report.md"
    records = (
        {
            "schema_version": 1,
            "scan_id": scan_id,
            "record_type": "SERVER_INFO",
            "hostname": "fixture-n607",
            "server_time_utc": "2026-08-17T00:00:00Z",
            "root": "/home/szu2070436088/2510044040/CV-SincNet",
        },
        {
            "schema_version": 1,
            "scan_id": scan_id,
            "record_type": "ASSET",
            "asset_id": asset_id,
            "location": "N607",
            "root_id": "N607_CVS_SINCNET",
            "relative_path": "runs/fixture/report.md",
            "display_name": "report.md",
            "escaped_name": "report.md",
            "asset_kind": "file",
            "size_bytes": 12,
            "mtime_utc": "2026-08-17T00:00:00Z",
            "access_status": "OK",
            "hash_status": "SHA256",
            "sha256": "0" * 64,
            "evidence_role": "report",
        },
        {
            "schema_version": 1,
            "scan_id": scan_id,
            "record_type": "SCOPE",
            "location": "N607",
            "root_id": "N607_CVS_SINCNET",
            "relative_path": "",
            "status": "VERIFIED",
            "asset_ids": [asset_id],
        },
        {
            "schema_version": 1,
            "scan_id": scan_id,
            "record_type": "SCOPE",
            "location": "N607",
            "root_id": "N607_CVS_SINCNET",
            "relative_path": "runs",
            "status": "VERIFIED",
            "asset_ids": [],
        },
        {
            "schema_version": 1,
            "scan_id": scan_id,
            "record_type": "PROCESS",
            "pid": 123,
            "ppid": 1,
            "cwd": "/home/szu2070436088/2510044040/CV-SincNet/runs/fixture",
            "cmdline": "python3 train_fixture.py",
            "training_like": False,
        },
        {
            "schema_version": 1,
            "scan_id": scan_id,
            "record_type": "COLLECTION_COMPLETE",
            "record_count": 5,
            "scan_error_count": 0,
        },
    )
    return N607CollectionResult(
        records=records,
        receipt=N607Receipt(
            outcome=RemoteOutcome.VERIFIED,
            route="DIRECT",
            preflight_status="DIRECT_READY",
            disconnect_status="VERIFIED",
            attempts=(
                AttemptReceipt(
                    label="PREFLIGHT",
                    child_pid=4100,
                    proxy_child_pids=(4101,),
                    returncode=0,
                    timed_out=False,
                    child_exited=True,
                    proxy_children_exited=True,
                    disconnect_status="VERIFIED",
                    lingering_connections=(),
                    stderr_tail="",
                ),
                AttemptReceipt(
                    label="DIRECT",
                    child_pid=4200,
                    proxy_child_pids=(),
                    returncode=0,
                    timed_out=False,
                    child_exited=True,
                    proxy_children_exited=True,
                    disconnect_status="VERIFIED",
                    lingering_connections=(),
                    stderr_tail="",
                ),
            ),
        ),
    )


def test_cli_requires_scan_and_explicit_scope_arguments():
    from tools.project_governance.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["scan"])
    parsed = parse_args(
        [
            "scan",
            "--config",
            "config.json",
            "--scan-id",
            "SCAN_1",
            "--output-root",
            "git-output",
            "--external-output-root",
            "external-output",
            "--operator",
            "codex",
        ]
    )
    assert parsed.include_n607 is False
    assert parsed.print_plan is False


@pytest.mark.parametrize("flag", ("--delete", "--cleanup", "--move", "--overwrite", "--kill", "--admin"))
def test_cli_rejects_mutation_and_admin_flags(flag):
    from tools.project_governance.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "scan",
                "--config",
                "config.json",
                "--scan-id",
                "SCAN_1",
                "--output-root",
                "git-output",
                "--external-output-root",
                "external-output",
                "--operator",
                "codex",
                flag,
            ]
        )


def test_cli_default_never_constructs_or_calls_n607(tmp_path):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "report.md").write_text("run_id: RUN_LOCAL\n", encoding="utf-8")
    config = _cli_fixture_config(root)
    git_runner, calls = _fake_git_runner_factory(root)
    outcome = run_scan(
        _cli_args(tmp_path),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda *_: (_ for _ in ()).throw(AssertionError("N607 contacted")),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert outcome.exit_code == 0
    assert outcome.remote_contacted is False
    assert calls
    assert not (tmp_path / "governance-external" / "CLI_FIXTURE").exists()


def test_cli_print_plan_has_no_scan_git_network_or_output_side_effects(tmp_path, capsys):
    from tools.project_governance.cli import print_plan

    root = tmp_path / "local"
    root.mkdir()
    config = _cli_fixture_config(root)
    args = _cli_args(tmp_path, include_n607=True, print_plan=True)
    plan = print_plan(args, config=config)
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert plan["n607_contact"] is True
    assert payload["local_root"] == str(root)
    assert payload["n607_contact"] is True
    assert payload["output_targets"]["git"].endswith("CLI_FIXTURE")
    assert not (tmp_path / "governance-git").exists()
    assert not (tmp_path / "governance-external").exists()


def test_cli_fixture_scan_emits_joined_inventory_and_approval_only_rows(tmp_path):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    (root / "runs" / "fixture").mkdir(parents=True)
    (root / "runs" / "fixture" / "report.md").write_text(
        "run_id: RUN_FIXTURE\nphase: governance\n", encoding="utf-8"
    )
    (root / "runs" / "fixture" / "receipt.json").write_text(
        '{"run_id":"RUN_FIXTURE","terminal":true}\n', encoding="utf-8"
    )
    (root / ".git").mkdir()
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)

    class FakeN607Collector:
        def collect(self):
            return _fake_n607_result("CLI_FIXTURE")

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    first = run_scan(
        _cli_args(tmp_path, include_n607=True),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )
    assert first.exit_code == 0
    output = Path(first.output_dir)
    expected = {
        "report.md",
        "asset_inventory_local.csv",
        "asset_inventory_n607.csv",
        "experiment_index.csv",
        "git_ownership.csv",
        "retention_decisions.csv",
        "deletion_candidates.csv",
        "asset_inventory_full.json",
        "scan_receipt.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    full = json.loads(output.joinpath("asset_inventory_full.json").read_text(encoding="utf-8"))
    assert {row["asset_id"] for row in full["assets"]} >= {
        "asset:LOCAL:TYPE10_7:runs/fixture/report.md",
        "asset:N607:N607_CVS_SINCNET:runs/fixture/report.md",
    }
    experiments = list(csv.DictReader(output.joinpath("experiment_index.csv").open("r", encoding="utf-8-sig", newline="")))
    assert any(row["experiment_id"] == "RUN_FIXTURE" for row in experiments)
    receipt = json.loads(output.joinpath("scan_receipt.json").read_text(encoding="utf-8"))
    assert receipt["authorized_deletion_rows"] == 0
    assert receipt["implementation"] == {
        "git_head": "fixture-head",
        "tracked_diff_state": "CLEAN",
    }
    assert receipt["n607_evidence"] == {
        "requested": True,
        "outcome": "VERIFIED",
        "route": "DIRECT",
        "preflight": "DIRECT_READY",
        "disconnect": "VERIFIED",
        "active_training_observed": False,
        "attempts": [
            {
                "label": "PREFLIGHT",
                "child_pid": 4100,
                "proxy_child_pids": [4101],
                "returncode": 0,
                "timed_out": False,
                "child_exited": True,
                "proxy_children_exited": True,
                "disconnect_status": "VERIFIED",
                "lingering_connections": [],
                "stderr_tail": "",
            },
            {
                "label": "DIRECT",
                "child_pid": 4200,
                "proxy_child_pids": [],
                "returncode": 0,
                "timed_out": False,
                "child_exited": True,
                "proxy_children_exited": True,
                "disconnect_status": "VERIFIED",
                "lingering_connections": [],
                "stderr_tail": "",
            },
        ],
    }
    assert all(item["sha256"] for item in receipt["files"])
    assert len(full["retention_decisions"]) == len(full["assets"])
    assert all(
        candidate["approval_state"] == "AWAITING_USER_APPROVAL"
        and candidate["execution_state"] == "NOT_AUTHORIZED"
        and candidate["approved_scope"] is None
        for candidate in full["deletion_candidates"]
    )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("outcome_name", "preflight", "disconnect", "expected_exit"),
    (
        ("FAILED", "FAILED", "VERIFIED", 2),
        ("UNKNOWN", "UNKNOWN", "UNKNOWN", 3),
    ),
)
def test_cli_maps_remote_failure_evidence_to_the_fixed_exit_codes(
    tmp_path, outcome_name, preflight, disconnect, expected_exit
):
    from tools.project_governance.cli import run_scan
    from tools.project_governance.collect_n607 import (
        N607CollectionResult,
        N607Receipt,
        RemoteOutcome,
    )

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "report.md").write_text("run_id: RUN_LOCAL\n", encoding="utf-8")
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)
    remote_result = N607CollectionResult(
        records=(),
        receipt=N607Receipt(
            outcome=RemoteOutcome(outcome_name),
            route=None,
            preflight_status=preflight,
            disconnect_status=disconnect,
            attempts=(),
        ),
    )

    class FakeN607Collector:
        def collect(self):
            return remote_result

    outcome = run_scan(
        _cli_args(tmp_path, include_n607=True),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert outcome.exit_code == expected_exit
    assert outcome.remote_contacted is True
    assert outcome.remote_outcome == outcome_name


@pytest.mark.parametrize(
    ("scenario", "expected_exit", "expected_labels"),
    (
        ("PREFLIGHT_FAILED", 2, ["PREFLIGHT"]),
        ("BRIDGE_LINGERING", 3, ["PREFLIGHT", "LAB_BRIDGE"]),
    ),
)
def test_cli_receipt_preserves_attempt_evidence_for_non_success_routes(
    tmp_path, scenario, expected_exit, expected_labels
):
    from tools.project_governance.cli import run_scan
    from tools.project_governance.collect_n607 import (
        APPROVED_BRIDGE_HOST,
        AttemptReceipt,
        ConnectionEvidence,
        N607CollectionResult,
        N607Receipt,
        RemoteOutcome,
    )

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)
    preflight = AttemptReceipt(
        label="PREFLIGHT",
        child_pid=5100,
        proxy_child_pids=(5101,),
        returncode=1 if scenario == "PREFLIGHT_FAILED" else 0,
        timed_out=False,
        child_exited=True,
        proxy_children_exited=True,
        disconnect_status="VERIFIED",
        lingering_connections=(),
        stderr_tail="fixture preflight evidence",
    )
    if scenario == "PREFLIGHT_FAILED":
        receipt = N607Receipt(
            outcome=RemoteOutcome.FAILED,
            route=None,
            preflight_status="FAILED",
            disconnect_status="VERIFIED",
            attempts=(preflight,),
        )
    else:
        bridge = AttemptReceipt(
            label="LAB_BRIDGE",
            child_pid=5200,
            proxy_child_pids=(5201,),
            returncode=0,
            timed_out=False,
            child_exited=True,
            proxy_children_exited=False,
            disconnect_status="UNKNOWN",
            lingering_connections=(
                ConnectionEvidence(
                    pid=5201,
                    endpoint=APPROVED_BRIDGE_HOST,
                    state="ESTABLISHED",
                ),
            ),
            stderr_tail="fixture bridge evidence",
        )
        receipt = N607Receipt(
            outcome=RemoteOutcome.UNKNOWN,
            route="LAB_BRIDGE",
            preflight_status="DIRECT_PATH_UNAVAILABLE",
            disconnect_status="UNKNOWN",
            attempts=(preflight, bridge),
        )

    class FakeN607Collector:
        def collect(self):
            return N607CollectionResult(records=(), receipt=receipt)

    args = _cli_args(tmp_path, include_n607=True)
    args.scan_id = f"CLI_{scenario}"
    outcome = run_scan(
        args,
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert outcome.exit_code == expected_exit
    persisted = json.loads(Path(outcome.output_dir, "scan_receipt.json").read_text(encoding="utf-8"))[
        "n607_evidence"
    ]
    assert [attempt["label"] for attempt in persisted["attempts"]] == expected_labels
    if scenario == "BRIDGE_LINGERING":
        assert persisted["attempts"][-1]["lingering_connections"] == [
            {"pid": 5201, "endpoint": APPROVED_BRIDGE_HOST, "state": "ESTABLISHED"}
        ]


def test_cli_rejects_an_unsafe_output_component_before_collecting(tmp_path):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    root.mkdir()
    config = _cli_fixture_config(root)
    args = _cli_args(tmp_path)
    args.scan_id = "../unsafe"

    outcome = run_scan(
        args,
        config=config,
        git_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Git called")),
        n607_collector_factory=lambda *_: (_ for _ in ()).throw(AssertionError("N607 called")),
    )

    assert outcome.exit_code == 4
    assert outcome.output_dir is None
    assert not (tmp_path / "governance-git").exists()
    assert not (tmp_path / "governance-external").exists()


def test_cli_entrypoint_has_no_destructive_process_control_in_package():
    package_root = Path(__file__).resolve().parents[1] / "tools" / "project_governance"
    # ``str.replace`` and dataclass replacement are ordinary immutable value
    # operations used by the established package.  The safety boundary here
    # is process control and destructive filesystem calls, which must never
    # appear in the package-level CLI orchestration surface.
    banned = {"Popen", "terminate", "kill", "send_signal", "unlink", "rmtree"}
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Attribute) and node.attr in banned
            for node in ast.walk(tree)
        ), source_path


def _n607_result_with_error_source(scan_id: str, source: str):
    from tools.project_governance.collect_n607 import N607CollectionResult

    original = _fake_n607_result(scan_id)
    records = [dict(record) for record in original.records]
    completion = records[-1]
    if source == "asset":
        asset = next(record for record in records if record["record_type"] == "ASSET")
        asset.update(access_status="SCAN_ERROR", hash_status="ERROR", sha256=None)
    elif source == "scope":
        scope = next(record for record in records if record["record_type"] == "SCOPE")
        scope.update(status="SCAN_ERROR", error="fixture scope error")
    elif source in {"raw", "scope_and_raw", "mismatched_raw"}:
        if source == "scope_and_raw":
            scope = next(record for record in records if record["record_type"] == "SCOPE")
            scope.update(status="SCAN_ERROR", error="fixture scope error")
        records.insert(
            -1,
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "SCAN_ERROR",
                "location": "N607",
                "root_id": "N607_CVS_SINCNET",
                "relative_path": "runs",
                "operation": "scandir",
                "error_type": "OSError",
                "error": "fixture remote error",
            },
        )
        completion["record_count"] += 1
        completion["scan_error_count"] = 0 if source == "mismatched_raw" else 1
    else:
        raise AssertionError(f"unknown fixture error source: {source}")
    return N607CollectionResult(records=tuple(records), receipt=original.receipt)


@pytest.mark.parametrize("source", ("asset", "scope", "raw"))
def test_cli_remote_scan_errors_never_return_success(tmp_path, source):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "report.md").write_text("run_id: RUN_LOCAL\n", encoding="utf-8")
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)

    class FakeN607Collector:
        def collect(self):
            return _n607_result_with_error_source("CLI_FIXTURE", source)

    outcome = run_scan(
        _cli_args(tmp_path, include_n607=True),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert outcome.exit_code == 2
    assert outcome.remote_error_count == 1
    if source == "raw":
        full = json.loads(
            Path(outcome.output_dir, "asset_inventory_full.json").read_text(encoding="utf-8")
        )
        preserved = [
            json.loads(scope["error"])
            for scope in full["scope_results"]
            if scope["status"] == "SCAN_ERROR" and scope["error"] is not None
        ]
        assert {
            "record_type": "SCAN_ERROR",
            "operation": "scandir",
            "error_type": "OSError",
            "error": "fixture remote error",
        } in preserved


def test_cli_preserves_the_protocol_scan_error_count_without_double_counting(tmp_path):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)

    class FakeN607Collector:
        def collect(self):
            return _n607_result_with_error_source("CLI_FIXTURE", "scope_and_raw")

    outcome = run_scan(
        _cli_args(tmp_path, include_n607=True),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert outcome.exit_code == 2
    assert outcome.remote_error_count == 1
    receipt = json.loads(Path(outcome.output_dir, "scan_receipt.json").read_text(encoding="utf-8"))
    assert receipt["scan_error_counts"]["n607_records"] == 1


def test_cli_rejects_a_remote_scan_error_count_that_does_not_close(tmp_path):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)

    class FakeN607Collector:
        def collect(self):
            return _n607_result_with_error_source("CLI_FIXTURE", "mismatched_raw")

    outcome = run_scan(
        _cli_args(tmp_path, include_n607=True),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert outcome.exit_code == 3
    assert outcome.remote_outcome == "UNKNOWN"


def test_cli_selects_the_exact_implementation_repository_head(tmp_path):
    from tools.project_governance.cli import _implementation_state
    from tools.project_governance.collect_git import RepositoryRecord

    implementation = tmp_path / "implementation"
    other = tmp_path / "other"
    records = (
        RepositoryRecord(
            repository_root=str(implementation),
            head_commit="implementation-head",
            status_summary="",
        ),
        RepositoryRecord(
            repository_root=str(other),
            head_commit="unrelated-head",
            status_summary="unrelated dirty state",
        ),
    )

    assert _implementation_state(records, implementation_repository=implementation) == (
        "implementation-head",
        "CLEAN",
    )


@pytest.mark.parametrize(("include_n607", "expected_exit"), ((False, 2), (True, 3)))
def test_cli_post_scan_emission_failure_never_claims_a_prescan_gate(
    tmp_path, monkeypatch, include_n607, expected_exit
):
    from tools.project_governance.cli import run_scan
    from tools.project_governance.collect_n607 import N607CollectionResult, N607Receipt, RemoteOutcome

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    config = _cli_fixture_config(root)
    git_runner, calls = _fake_git_runner_factory(root)

    class FakeN607Collector:
        def collect(self):
            return N607CollectionResult(
                records=(),
                receipt=N607Receipt(
                    outcome=RemoteOutcome.UNKNOWN,
                    route=None,
                    preflight_status="UNKNOWN",
                    disconnect_status="UNKNOWN",
                    attempts=(),
                ),
            )

    monkeypatch.setattr(ReportEmitter, "emit", lambda self: (_ for _ in ()).throw(OSError("disk fault")))
    outcome = run_scan(
        _cli_args(tmp_path, include_n607=include_n607),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=lambda _config, _scan_id: FakeN607Collector(),
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert calls
    assert outcome.exit_code == expected_exit
    assert outcome.message is not None and "after scanning" in outcome.message


def test_cli_does_not_retry_a_factory_that_raises_typeerror_internally(tmp_path):
    from tools.project_governance.cli import run_scan

    root = tmp_path / "local"
    (root / "runs").mkdir(parents=True)
    config = _cli_fixture_config(root)
    git_runner, _ = _fake_git_runner_factory(root)
    calls: list[int] = []

    def broken_factory(*args):
        calls.append(len(args))
        raise TypeError("internal factory failure")

    outcome = run_scan(
        _cli_args(tmp_path, include_n607=True),
        config=config,
        git_runner=git_runner,
        repository_seeds=(root,),
        implementation_repository=root,
        n607_collector_factory=broken_factory,
        clock=lambda: "2026-08-17T00:00:00Z",
    )

    assert calls == [2]
    assert outcome.exit_code == 3


def test_top_level_runner_rejects_unapproved_commands_without_invoking_processes():
    entrypoint = Path(__file__).resolve().parents[1] / "tools" / "project_governance_inventory.py"
    module = runpy.run_path(str(entrypoint), run_name="project_governance_inventory_test")
    calls: list[tuple[object, ...]] = []
    runner = module["ProductionCommandRunner"](
        popen_factory=lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="approved"):
        runner.run(
            ("ssh", "unapproved-target"),
            input_text=None,
            timeout_seconds=1,
            label="DIRECT",
        )

    assert calls == []


def test_top_level_runner_streams_only_an_approved_preflight_through_fakes():
    entrypoint = Path(__file__).resolve().parents[1] / "tools" / "project_governance_inventory.py"
    module = runpy.run_path(str(entrypoint), run_name="project_governance_inventory_test")
    from tools.project_governance.collect_n607 import DEFAULT_PREFLIGHT_SCRIPT

    class FakeProcess:
        def __init__(self):
            self.pid = 321
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"preflight line\\n")
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    class FakeTracker:
        proxy_child_pids = (654,)

        def wait_for_exit(self):
            return True

        def close(self):
            return None

    seen: list[bytes] = []
    runner = module["ProductionCommandRunner"](
        popen_factory=lambda *args, **kwargs: FakeProcess(),
        tracker_factory=lambda pid, required: FakeTracker(),
    )
    result = runner.run(
        (
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DEFAULT_PREFLIGHT_SCRIPT),
        ),
        input_text=None,
        timeout_seconds=1,
        label="PREFLIGHT",
        stdout_line_handler=seen.append,
    )

    assert result.child_pid == 321
    assert result.proxy_child_pids == (654,)
    assert result.child_exited is True
    assert result.proxy_children_exited is True
    assert result.stdout_lines == (b"preflight line\\n",)
    assert seen == [b"preflight line\\n"]


def test_top_level_runner_bounds_each_stream_read_and_retains_only_a_marked_stderr_tail():
    entrypoint = Path(__file__).resolve().parents[1] / "tools" / "project_governance_inventory.py"
    module = runpy.run_path(str(entrypoint), run_name="project_governance_inventory_bounded_test")
    from tools.project_governance.collect_n607 import (
        DEFAULT_PREFLIGHT_SCRIPT,
        MAX_NDJSON_LINE_BYTES,
    )

    class RecordingStream:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.read_sizes: list[int] = []

        def readline(self, size=-1):
            self.read_sizes.append(size)
            if not self.payload:
                return b""
            limit = len(self.payload) if size is None or size < 0 else min(size, len(self.payload))
            newline = self.payload.find(b"\n", 0, limit)
            take = newline + 1 if newline >= 0 else limit
            chunk, self.payload = self.payload[:take], self.payload[take:]
            return chunk

    class FakeProcess:
        def __init__(self):
            self.pid = 4321
            self.stdin = io.BytesIO()
            self.stdout = RecordingStream(b"bounded preflight line\n")
            self.stderr = RecordingStream(b"\xff" * 20000)
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    class FakeTracker:
        proxy_child_pids = (4322,)

        def wait_for_exit(self):
            return True

        def close(self):
            return None

    process = FakeProcess()
    runner = module["ProductionCommandRunner"](
        popen_factory=lambda *args, **kwargs: process,
        tracker_factory=lambda pid, required: FakeTracker(),
    )
    result = runner.run(
        (
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DEFAULT_PREFLIGHT_SCRIPT),
        ),
        input_text=None,
        timeout_seconds=1,
        label="PREFLIGHT",
    )

    expected_read_limit = MAX_NDJSON_LINE_BYTES + 3
    assert process.stdout.read_sizes and set(process.stdout.read_sizes) == {expected_read_limit}
    assert process.stderr.read_sizes and set(process.stderr.read_sizes) == {8193}
    assert "truncated" in result.stderr_tail
    assert len(result.stderr_tail.encode("utf-8")) <= 8192
