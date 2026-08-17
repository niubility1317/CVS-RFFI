from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.project_governance.index_experiments import EvidenceClaim, ProcessEvidence, index_experiments
from tools.project_governance.models import (
    AccessStatus,
    AssetKind,
    AssetRecord,
    ExperimentState,
    HashStatus,
    Location,
)
from tools.project_governance.paths import stable_asset_id


ROOT_ID = "FIXTURE"
FIXTURE_MTIME = "2026-08-17T00:00:00Z"


def _write(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _asset(
    relative_path: str,
    *,
    evidence_role: str | None = None,
    access_status: AccessStatus = AccessStatus.OK,
    asset_kind: AssetKind = AssetKind.FILE,
    size_bytes: int | None = None,
    experiment_id: str | None = None,
) -> AssetRecord:
    return AssetRecord(
        asset_id=stable_asset_id(Location.LOCAL, ROOT_ID, relative_path),
        scan_id="EXPERIMENT_FIXTURE_SCAN",
        location=Location.LOCAL,
        root_id=ROOT_ID,
        relative_path=relative_path,
        display_name=Path(relative_path).name,
        escaped_name=Path(relative_path).name,
        asset_kind=asset_kind,
        size_bytes=size_bytes,
        mtime_utc=FIXTURE_MTIME,
        access_status=access_status,
        hash_status=HashStatus.METADATA_ONLY,
        sha256=None,
        experiment_id=experiment_id,
        evidence_role=evidence_role,
    )


def _record_containing(index, suffix: str):
    return next(
        record
        for record in index.values()
        if any(path.replace("\\", "/").endswith(suffix) for path in (record.local_artifact_paths or ()))
    )


@pytest.fixture()
def evidence_fixture(tmp_path: Path):
    root = tmp_path / "indexed-assets"
    root.mkdir()

    run_root = root / "runs" / "RUN_A"
    run_root.mkdir(parents=True)
    _write(
        root,
        "runs/RUN_A/report.md",
        "\n".join(
            (
                "run_id: RUN_A",
                "git_commit: abc123",
                "run_root: runs/RUN_A",
                "status: COMPLETE",
                "expected_artifacts:",
                "- predictions.json",
                "- scores.json",
                "",
            )
        ),
    )
    _write(
        root,
        "runs/RUN_A/manifest.json",
        json.dumps(
            {
                "run_id": "RUN_A",
                "git_commit": "abc123",
                "run_root": "runs/RUN_A",
                "receipt_for": "runs/RUN_A",
            }
        ),
    )
    _write(root, "runs/RUN_A/predictions.json", '{"opaque_prediction_artifact": true}\n')
    _write(root, "runs/RUN_A/scores.json", '{"accuracy": 0.99, "loss": 0.01}\n')

    _write(
        root,
        "runs/RUN_ARCHIVE/report.md",
        "\n".join(
            (
                "run_id: RUN_ARCHIVE",
                "run_root: runs/RUN_ARCHIVE",
                "status: COMPLETE",
                "archive: true",
                "expected_artifacts:",
                "- predictions.json",
                "",
            )
        ),
    )
    _write(root, "runs/RUN_ARCHIVE/predictions.json", '{"opaque_prediction_artifact": true}\n')

    _write(
        root,
        "runs/RUN_OPEN/report.md",
        "\n".join(
            (
                "run_id: RUN_OPEN",
                "run_root: runs/RUN_OPEN",
                "status: COMPLETE",
                "expected_artifacts:",
                "- predictions.json",
                "",
            )
        ),
    )

    _write(root, "runs/RUN_A_review/predictions.json", '{"opaque_prediction_artifact": true}\n')
    _write(root, "lost/predictions.json", '{"opaque_prediction_artifact": true}\n')

    _write(
        root,
        "runs/RUN_CONFLICT/report.md",
        "\n".join(("run_id: RUN_CONFLICT", "run_root: runs/RUN_CONFLICT", "status: COMPLETE", "")),
    )
    _write(
        root,
        "runs/RUN_CONFLICT/manifest.json",
        json.dumps({"run_id": "RUN_OTHER", "run_root": "runs/RUN_CONFLICT"}),
    )

    _write(root, "runs/RUN_UNREADABLE/report.md", "run_id: RUN_UNREADABLE\n")
    _write(root, "runs/RUN_A/checkpoint.pth", "not a checkpoint\n")
    _write(root, "runs/RUN_A/oversized-manifest.json", '{"run_id": "MUST_NOT_BE_READ"}\n')
    _write(root, "runs/RUN_A/untrusted-report.pdf", "run_id: MUST_NOT_BE_READ\n")

    assets = (
        _asset("runs/RUN_A", asset_kind=AssetKind.DIRECTORY),
        _asset("runs/RUN_A/report.md", evidence_role="report"),
        _asset("runs/RUN_A/manifest.json", evidence_role="manifest"),
        _asset("runs/RUN_A/predictions.json", evidence_role="prediction"),
        _asset("runs/RUN_A/scores.json", evidence_role="metrics_summary"),
        _asset("runs/RUN_ARCHIVE/report.md", evidence_role="report"),
        _asset("runs/RUN_ARCHIVE/predictions.json", evidence_role="prediction"),
        _asset("runs/RUN_OPEN/report.md", evidence_role="report"),
        _asset("runs/RUN_A_review/predictions.json", evidence_role="prediction"),
        _asset("lost/predictions.json", evidence_role="prediction"),
        _asset("runs/RUN_CONFLICT/report.md", evidence_role="report"),
        _asset("runs/RUN_CONFLICT/manifest.json", evidence_role="manifest"),
        _asset(
            "runs/RUN_UNREADABLE/report.md",
            evidence_role="report",
            access_status=AccessStatus.SCAN_ERROR,
        ),
        _asset("runs/RUN_A/checkpoint.pth", evidence_role="checkpoint"),
        _asset(
            "runs/RUN_A/oversized-manifest.json",
            evidence_role="manifest",
            size_bytes=2 * 1024 * 1024 + 1,
        ),
        _asset("runs/RUN_A/untrusted-report.pdf", evidence_role="report"),
    )
    return root, run_root, assets


def test_explicit_live_binding_wins_without_merging_same_name_or_mtime(evidence_fixture):
    root, run_root, assets = evidence_fixture

    index = index_experiments(
        assets,
        root_paths={ROOT_ID: root},
        process_evidence=(
            ProcessEvidence(
                pid=2718,
                cwd=str(run_root),
                cmdline=f"python runner.py --run-root {run_root}",
            ),
        ),
    )

    same_mtime_candidate = _record_containing(index, "/runs/RUN_A_review/predictions.json")
    orphan = _record_containing(index, "/lost/predictions.json")

    assert index["RUN_A"].experiment_state is ExperimentState.ACTIVE_LIVE
    assert same_mtime_candidate.experiment_id != index["RUN_A"].experiment_id
    assert same_mtime_candidate.experiment_state is ExperimentState.ORPHAN_REVIEW
    assert "LOW_CONFIDENCE_NAME_ONLY" in (same_mtime_candidate.closure_gaps or ())
    assert orphan.experiment_id.startswith("ORPHAN:")
    assert not any(claim.field in {"accuracy", "loss"} for claim in index.claims)
    assert any(
        claim.field == "metrics_reference"
        and claim.source_asset_id == stable_asset_id(Location.LOCAL, ROOT_ID, "runs/RUN_A/scores.json")
        for claim in index.claims
    )


def test_terminal_and_archive_states_require_explicit_closure_without_live_binding(evidence_fixture):
    root, _, assets = evidence_fixture

    index = index_experiments(assets, root_paths={ROOT_ID: root})

    assert index["RUN_A"].experiment_state is ExperimentState.COMPLETE_EVIDENCE
    assert index["RUN_ARCHIVE"].experiment_state is ExperimentState.HISTORICAL_ARCHIVE
    assert index["RUN_OPEN"].experiment_state is ExperimentState.OPEN_INCOMPLETE
    assert "MISSING_EXPECTED_ARTIFACT" in (index["RUN_OPEN"].closure_gaps or ())


def test_live_state_rejects_a_cmdline_that_only_contains_a_longer_similar_root(evidence_fixture):
    root, run_root, assets = evidence_fixture

    index = index_experiments(
        assets,
        root_paths={ROOT_ID: root},
        process_evidence=(
            ProcessEvidence(
                pid=2718,
                cwd=str(run_root),
                cmdline=f"python runner.py --run-root {root / 'runs' / 'RUN_A_review'}",
            ),
        ),
    )

    assert index["RUN_A"].experiment_state is ExperimentState.COMPLETE_EVIDENCE


def test_valid_live_binding_outranks_conflicting_historical_claims(evidence_fixture):
    root, _, assets = evidence_fixture
    conflict_root = root / "runs" / "RUN_CONFLICT"

    index = index_experiments(
        assets,
        root_paths={ROOT_ID: root},
        process_evidence=(
            ProcessEvidence(
                pid=3141,
                cwd=str(conflict_root),
                cmdline=f"python runner.py --run-root {conflict_root}",
            ),
        ),
    )

    assert index["RUN_CONFLICT"].experiment_state is ExperimentState.ACTIVE_LIVE
    assert "CONFLICTING_RUN_ID" in (index["RUN_CONFLICT"].closure_gaps or ())


def test_conflicts_and_unreadable_evidence_stay_separate_for_review(evidence_fixture):
    root, _, assets = evidence_fixture

    index = index_experiments(assets, root_paths={ROOT_ID: root})

    conflict = index["RUN_CONFLICT"]
    unreadable = _record_containing(index, "/runs/RUN_UNREADABLE/report.md")

    assert conflict.experiment_state is ExperimentState.ORPHAN_REVIEW
    assert "CONFLICTING_RUN_ID" in (conflict.closure_gaps or ())
    assert unreadable.experiment_state is ExperimentState.SCAN_ERROR
    assert "UNREADABLE_EVIDENCE" in (unreadable.closure_gaps or ())
    assert all(claim.source_asset_id != stable_asset_id(Location.LOCAL, ROOT_ID, "runs/RUN_A/checkpoint.pth") for claim in index.claims)
    assert all(
        claim.source_asset_id
        != stable_asset_id(Location.LOCAL, ROOT_ID, "runs/RUN_A/oversized-manifest.json")
        for claim in index.claims
    )
    assert all(
        claim.source_asset_id != stable_asset_id(Location.LOCAL, ROOT_ID, "runs/RUN_A/untrusted-report.pdf")
        for claim in index.claims
    )
    assert all(isinstance(claim, EvidenceClaim) for claim in index.claims)


def test_shared_commit_with_one_manifest_does_not_merge_distinct_explicit_runs(tmp_path: Path):
    root = tmp_path / "indexed-assets"
    root.mkdir()
    _write(root, "runs/RUN_COMMIT_A/report.md", "run_id: RUN_COMMIT_A\ngit_commit: shared-commit\nrun_root: runs/RUN_COMMIT_A\n")
    _write(
        root,
        "runs/RUN_COMMIT_A/manifest.json",
        json.dumps({"run_id": "RUN_COMMIT_A", "git_commit": "shared-commit", "run_root": "runs/RUN_COMMIT_A"}),
    )
    _write(root, "runs/RUN_COMMIT_B/report.md", "run_id: RUN_COMMIT_B\ngit_commit: shared-commit\nrun_root: runs/RUN_COMMIT_B\n")

    index = index_experiments(
        (
            _asset("runs/RUN_COMMIT_A/report.md", evidence_role="report"),
            _asset("runs/RUN_COMMIT_A/manifest.json", evidence_role="manifest"),
            _asset("runs/RUN_COMMIT_B/report.md", evidence_role="report"),
        ),
        root_paths={ROOT_ID: root},
    )

    assert index["RUN_COMMIT_A"].experiment_state is ExperimentState.OPEN_INCOMPLETE
    assert index["RUN_COMMIT_B"].experiment_state is ExperimentState.OPEN_INCOMPLETE
    assert "CONFLICTING_RUN_ID" not in (index["RUN_COMMIT_A"].closure_gaps or ())


def test_manifest_terminal_cannot_complete_a_nonterminal_report(tmp_path: Path):
    root = tmp_path / "indexed-assets"
    root.mkdir()
    _write(
        root,
        "runs/RUN_MANIFEST_TERMINAL/report.md",
        "\n".join(
            (
                "run_id: RUN_MANIFEST_TERMINAL",
                "run_root: runs/RUN_MANIFEST_TERMINAL",
                "expected_artifacts:",
                "- predictions.json",
                "",
            )
        ),
    )
    _write(
        root,
        "runs/RUN_MANIFEST_TERMINAL/manifest.json",
        json.dumps(
            {
                "run_id": "RUN_MANIFEST_TERMINAL",
                "run_root": "runs/RUN_MANIFEST_TERMINAL",
                "status": "COMPLETE",
            }
        ),
    )
    _write(root, "runs/RUN_MANIFEST_TERMINAL/predictions.json", '{"opaque_prediction_artifact": true}\n')

    index = index_experiments(
        (
            _asset("runs/RUN_MANIFEST_TERMINAL/report.md", evidence_role="report"),
            _asset("runs/RUN_MANIFEST_TERMINAL/manifest.json", evidence_role="manifest"),
            _asset("runs/RUN_MANIFEST_TERMINAL/predictions.json", evidence_role="prediction"),
        ),
        root_paths={ROOT_ID: root},
    )

    assert index["RUN_MANIFEST_TERMINAL"].experiment_state is ExperimentState.OPEN_INCOMPLETE


def test_reports_without_explicit_run_roots_do_not_merge_only_by_shared_parent(tmp_path: Path):
    root = tmp_path / "indexed-assets"
    root.mkdir()
    _write(root, "reports/shared/report-a.md", "run_id: RUN_PARENT_A\n")
    _write(root, "reports/shared/report-b.md", "run_id: RUN_PARENT_B\n")

    index = index_experiments(
        (
            _asset("reports/shared/report-a.md", evidence_role="report"),
            _asset("reports/shared/report-b.md", evidence_role="report"),
        ),
        root_paths={ROOT_ID: root},
    )

    assert index["RUN_PARENT_A"].experiment_state is ExperimentState.OPEN_INCOMPLETE
    assert index["RUN_PARENT_B"].experiment_state is ExperimentState.OPEN_INCOMPLETE
    assert "CONFLICTING_RUN_ID" not in (index["RUN_PARENT_A"].closure_gaps or ())


def test_explicit_root_relative_expected_artifact_associates_its_indexed_asset(tmp_path: Path):
    root = tmp_path / "indexed-assets"
    root.mkdir()
    _write(
        root,
        "reports/expected-artifact.md",
        "\n".join(
            (
                "run_id: RUN_EXPECTED",
                "status: COMPLETE",
                "expected_artifacts:",
                "- runs/RUN_EXPECTED/predictions.json",
                "",
            )
        ),
    )
    _write(root, "runs/RUN_EXPECTED/predictions.json", '{"opaque_prediction_artifact": true}\n')

    index = index_experiments(
        (
            _asset("reports/expected-artifact.md", evidence_role="report"),
            _asset("runs/RUN_EXPECTED/predictions.json", evidence_role="prediction"),
        ),
        root_paths={ROOT_ID: root},
    )

    expected = index["RUN_EXPECTED"]
    assert expected.experiment_state is ExperimentState.COMPLETE_EVIDENCE
    assert any(path.replace("\\", "/").endswith("/runs/RUN_EXPECTED/predictions.json") for path in (expected.observed_artifacts or ()))


def test_oversized_required_report_is_scan_error_even_with_live_binding(tmp_path: Path):
    root = tmp_path / "indexed-assets"
    root.mkdir()
    run_root = root / "runs" / "RUN_OVERSIZED"
    run_root.mkdir(parents=True)
    _write(root, "runs/RUN_OVERSIZED/report.md", "run_id: RUN_OVERSIZED\n")
    _write(
        root,
        "runs/RUN_OVERSIZED/manifest.json",
        json.dumps({"run_id": "RUN_OVERSIZED", "run_root": "runs/RUN_OVERSIZED"}),
    )

    index = index_experiments(
        (
            _asset(
                "runs/RUN_OVERSIZED/report.md",
                evidence_role="report",
                size_bytes=2 * 1024 * 1024 + 1,
                experiment_id="RUN_OVERSIZED",
            ),
            _asset("runs/RUN_OVERSIZED/manifest.json", evidence_role="manifest"),
        ),
        root_paths={ROOT_ID: root},
        process_evidence=(
            ProcessEvidence(
                pid=1618,
                cwd=str(run_root),
                cmdline=f"python runner.py --run-root {run_root}",
            ),
        ),
    )

    assert index["RUN_OVERSIZED"].experiment_state is ExperimentState.SCAN_ERROR
    assert "UNREADABLE_EVIDENCE" in (index["RUN_OVERSIZED"].closure_gaps or ())
