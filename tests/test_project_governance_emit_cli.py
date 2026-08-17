from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

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
        "n607_route": "DIRECT",
        "n607_preflight": "DIRECT_READY",
        "n607_disconnect": "VERIFIED",
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


def test_emitter_is_immutable_on_second_scan_id_and_receipt_is_last(tmp_path):
    emitter = ReportEmitter(
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
    assert list(before).index("scan_receipt.json") == len(before) - 1


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
    ("route", "preflight", "disconnect"),
    (
        ("DIRECT", "DIRECT_READY", "VERIFIED"),
        ("LAB_BRIDGE", "DIRECT_PATH_UNAVAILABLE", "VERIFIED"),
        ("NO_ROUTE", "FAILED", "VERIFIED"),
        ("NO_ROUTE", "UNKNOWN", "UNKNOWN"),
    ),
)
def test_emitter_accepts_collector_n607_state_vocabulary(
    tmp_path, route, preflight, disconnect
):
    metadata = _metadata()
    metadata.update(
        {
            "n607_route": route,
            "n607_preflight": preflight,
            "n607_disconnect": disconnect,
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

    assert receipt["n607_evidence"] == {
        "requested": True,
        "route": route,
        "preflight": preflight,
        "disconnect": disconnect,
    }


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


def test_emitter_accepts_explicit_not_requested_n607_evidence(tmp_path):
    metadata = _metadata()
    metadata.update(
        {
            "n607_requested": False,
            "n607_route": "NOT_REQUESTED",
            "n607_preflight": "NOT_REQUESTED",
            "n607_disconnect": "NOT_REQUESTED",
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
    assert set(receipt["n607_evidence"].values()) >= {False, "NOT_REQUESTED"}


def test_emitter_receipt_keeps_remote_evidence_and_zero_execution_fields(tmp_path):
    result = ReportEmitter(
        _bundle(), output_root=tmp_path / "git", external_output_root=tmp_path / "external", metadata=_metadata()
    ).emit()
    receipt = json.loads(result.git_output_dir.joinpath("scan_receipt.json").read_text(encoding="utf-8"))

    assert receipt["roots"]["local"] == "E:/type10-7"
    assert receipt["roots"]["n607"] == "/home/szu2070436088/2510044040/CV-SincNet"
    assert receipt["n607_evidence"] == {
        "requested": True,
        "route": "DIRECT",
        "preflight": "DIRECT_READY",
        "disconnect": "VERIFIED",
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
