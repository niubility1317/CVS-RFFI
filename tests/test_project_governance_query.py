from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.project_governance.query_index import QueryStore, build_index, load_latest


ASSET_HEADERS = (
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
)
EXPERIMENT_HEADERS = (
    "experiment_id",
    "run_id",
    "experiment_state",
    "phase",
    "method_or_candidate",
    "report_path",
    "local_artifact_paths",
    "n607_artifact_paths",
    "git_commit",
    "process_evidence",
    "prediction_count",
    "score_count",
    "expected_artifacts",
    "observed_artifacts",
    "closure_gaps",
)
GIT_HEADERS = (
    "asset_id",
    "ownership",
    "repository_root",
    "common_git_dir",
    "branch",
    "head_commit",
    "status_summary",
    "linked_worktrees",
    "error",
)
RETENTION_HEADERS = (
    "asset_id",
    "retention_class",
    "rule_code",
    "reason",
    "evidence_asset_ids",
    "recommended_action",
)
DELETION_HEADERS = (
    "candidate_id",
    "location",
    "absolute_path",
    "asset_kind",
    "size_bytes",
    "reason",
    "evidence",
    "dependencies",
    "recoverability",
    "estimated_space_reclaim",
    "approval_state",
    "approved_scope",
    "execution_state",
)


@dataclass(frozen=True)
class InventoryFixture:
    receipt: Path
    external_root: Path
    database: Path


def _write_csv(path: Path, headers: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _asset_row(*, asset_id: str, location: str, relative_path: str, experiment_id: str) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "scan_id": "PGOV_TEST_001",
        "location": location,
        "root_id": "PROJECT",
        "relative_path": relative_path,
        "display_name": Path(relative_path).name,
        "escaped_name": Path(relative_path).name,
        "asset_kind": "directory",
        "size_bytes": 10,
        "mtime_utc": "2026-08-18T00:00:00Z",
        "access_status": "OK",
        "hash_status": "METADATA_ONLY",
        "sha256": "",
        "experiment_id": experiment_id,
        "git_ownership": "TRACKED_GIT",
        "evidence_role": "RUN_ROOT",
        "retention_class": "REVIEW_REQUIRED",
        "recommended_action": "REVIEW",
        "decision_reason": "fixture",
    }


def write_inventory_fixture(tmp_path: Path) -> InventoryFixture:
    external = tmp_path / "external" / "PGOV_TEST_001"
    external.mkdir(parents=True)
    local = external / "asset_inventory_local.csv"
    n607 = external / "asset_inventory_n607.csv"
    experiments = external / "experiment_index.csv"
    ownership = external / "git_ownership.csv"
    retention = external / "retention_decisions.csv"
    deletion = external / "deletion_candidates.csv"

    _write_csv(local, ASSET_HEADERS, [_asset_row(asset_id="asset-local", location="LOCAL", relative_path="runs/local-a", experiment_id="exp-a")])
    _write_csv(n607, ASSET_HEADERS, [_asset_row(asset_id="asset-remote", location="N607", relative_path="runs/remote-a", experiment_id="exp-a")])
    _write_csv(
        experiments,
        EXPERIMENT_HEADERS,
        [
            {
                "experiment_id": "exp-a",
                "run_id": "RUN_A",
                "experiment_state": "OPEN_INCOMPLETE",
                "phase": "Phase2",
                "method_or_candidate": "fixture-a",
                "report_path": "automation_reports/CV-SincNet/RUN_A/report.md",
                "local_artifact_paths": '["runs/local-a"]',
                "n607_artifact_paths": '["runs/remote-a"]',
                "git_commit": "a" * 40,
                "process_evidence": "[]",
                "prediction_count": 1,
                "score_count": 0,
                "expected_artifacts": '["prediction.json"]',
                "observed_artifacts": '["prediction.json"]',
                "closure_gaps": "[]",
            },
            {
                "experiment_id": "exp-b",
                "run_id": "RUN_B",
                "experiment_state": "COMPLETE_EVIDENCE",
                "phase": "Phase2",
                "method_or_candidate": "fixture-b",
                "report_path": "automation_reports/CV-SincNet/RUN_B/report.md",
                "local_artifact_paths": "[]",
                "n607_artifact_paths": "[]",
                "git_commit": "b" * 40,
                "process_evidence": "[]",
                "prediction_count": 1,
                "score_count": 1,
                "expected_artifacts": "[]",
                "observed_artifacts": "[]",
                "closure_gaps": "[]",
            },
        ],
    )
    _write_csv(
        ownership,
        GIT_HEADERS,
        [
            {
                "asset_id": asset_id,
                "ownership": "TRACKED_GIT",
                "repository_root": "E:/type10-7/github_publish/CVS-RFFI-repo",
                "common_git_dir": "E:/type10-7/github_publish/CVS-RFFI-repo/.git",
                "branch": "main",
                "head_commit": "a" * 40,
                "status_summary": "CLEAN",
                "linked_worktrees": "[]",
                "error": "",
            }
            for asset_id in ("asset-local", "asset-remote")
        ],
    )
    _write_csv(
        retention,
        RETENTION_HEADERS,
        [
            {
                "asset_id": asset_id,
                "retention_class": "REVIEW_REQUIRED",
                "rule_code": "FIXTURE_REVIEW",
                "reason": "fixture",
                "evidence_asset_ids": "[]",
                "recommended_action": "REVIEW",
            }
            for asset_id in ("asset-local", "asset-remote")
        ],
    )
    _write_csv(deletion, DELETION_HEADERS, [])

    sources = (local, n607, experiments, ownership, retention, deletion)
    receipt = tmp_path / "git" / "PGOV_TEST_001" / "scan_receipt.json"
    receipt.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "scan_id": "PGOV_TEST_001",
        "terminal_state": "COMPLETE",
        "source_asset_mutations": 0,
        "moves": 0,
        "overwrites": 0,
        "deletions": 0,
        "counts": {
            "assets": 2,
            "assets_local": 1,
            "assets_n607": 1,
            "experiments": 2,
            "git_ownership": 2,
            "retention_decisions": 2,
            "deletion_candidates": 0,
        },
        "roots": {
            "local": "E:/type10-7",
            "n607": "/home/szu2070436088/2510044040/CV-SincNet",
        },
        "scan_error_counts": {
            "assets": 0,
            "experiments": 1,
            "git_ownership": 0,
            "retention_decisions": 0,
            "deletion_candidates": 0,
        },
        "completed_at_utc": "2026-08-18T00:00:00Z",
        "implementation": {"git_head": "c" * 40},
        "external_files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": "0" * 64}
            for path in sources
        ],
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    return InventoryFixture(receipt=receipt, external_root=external, database=external / "governance.sqlite")


def test_build_index_streams_validated_tables_into_a_new_database(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)

    summary = build_index(
        receipt_path=case.receipt,
        external_root=case.external_root,
        database_path=case.database,
    )

    assert summary.scan_id == "PGOV_TEST_001"
    assert summary.table_counts == {
        "assets": 2,
        "experiments": 2,
        "git_ownership": 2,
        "retention": 2,
        "deletion_candidates": 0,
    }
    with sqlite3.connect(case.database) as connection:
        assert connection.execute(
            "SELECT location, relative_path FROM assets ORDER BY location"
        ).fetchall() == [("LOCAL", "runs/local-a"), ("N607", "runs/remote-a")]
        assert connection.execute("SELECT COUNT(*) FROM deletion_candidates").fetchone() == (0,)


def _receipt_payload(case: InventoryFixture) -> dict[str, object]:
    return json.loads(case.receipt.read_text(encoding="utf-8"))


def _write_receipt(case: InventoryFixture, payload: dict[str, object]) -> None:
    case.receipt.write_text(json.dumps(payload), encoding="utf-8", newline="\n")


def _source_bytes(case: InventoryFixture) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in case.external_root.iterdir()
        if path.suffix == ".csv"
    }


def test_build_index_rejects_a_nonterminal_receipt_without_touching_sources(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    payload = _receipt_payload(case)
    payload["terminal_state"] = "FAILED"
    _write_receipt(case, payload)
    before = _source_bytes(case)

    with pytest.raises(ValueError, match="COMPLETE"):
        build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)

    assert _source_bytes(case) == before
    assert not case.database.exists()


def test_build_index_rejects_an_asset_from_another_scan(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    local = case.external_root / "asset_inventory_local.csv"
    rows = list(csv.DictReader(local.open("r", encoding="utf-8-sig", newline="")))
    rows[0]["scan_id"] = "PGOV_WRONG"
    _write_csv(local, ASSET_HEADERS, rows)
    payload = _receipt_payload(case)
    for entry in payload["external_files"]:
        if Path(entry["path"]).name == local.name:
            entry["bytes"] = local.stat().st_size
    _write_receipt(case, payload)

    with pytest.raises(ValueError, match="scan_id mismatch"):
        build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)

    assert not case.database.exists()


def test_build_index_rejects_a_receipt_csv_outside_the_exact_external_root(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    escaped = tmp_path / "escaped" / "asset_inventory_local.csv"
    escaped.parent.mkdir()
    escaped.write_bytes((case.external_root / escaped.name).read_bytes())
    payload = _receipt_payload(case)
    for entry in payload["external_files"]:
        if Path(entry["path"]).name == escaped.name:
            entry["path"] = str(escaped.resolve())
            entry["bytes"] = escaped.stat().st_size
    _write_receipt(case, payload)

    with pytest.raises(ValueError, match="outside the exact external root"):
        build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)


def test_build_index_rejects_a_wrong_header_or_row_count(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    experiments = case.external_root / "experiment_index.csv"
    experiments.write_text("wrong,header\nvalue,value\n", encoding="utf-8", newline="\n")
    payload = _receipt_payload(case)
    for entry in payload["external_files"]:
        if Path(entry["path"]).name == experiments.name:
            entry["bytes"] = experiments.stat().st_size
    _write_receipt(case, payload)

    with pytest.raises(ValueError, match="unexpected CSV header"):
        build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)


def test_build_index_never_replaces_an_existing_database(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    case.database.write_bytes(b"owned-existing-index")

    with pytest.raises(FileExistsError, match="already exists"):
        build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)

    assert case.database.read_bytes() == b"owned-existing-index"


def _write_latest_pointer(case: InventoryFixture) -> Path:
    pointer = case.receipt.parent.parent / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": "PGOV_TEST_001",
                "receipt_path": str(case.receipt.resolve()),
                "external_root": str(case.external_root.resolve()),
                "sqlite_path": str(case.database.resolve()),
                "created_at_utc": "2026-08-18T00:00:00Z",
                "implementation_git_head": "c" * 40,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    return pointer


@pytest.fixture()
def built_store(tmp_path: Path):
    case = write_inventory_fixture(tmp_path)
    build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)
    with sqlite3.connect(case.database) as connection:
        connection.execute(
            "INSERT INTO git_ownership VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "asset-local",
                "TRACKED_GIT",
                "E:/type10-7/code/second-repo",
                "E:/type10-7/code/second-repo/.git",
                "feature",
                "d" * 40,
                "CLEAN",
                "[]",
                "",
            ),
        )
        connection.commit()
    pointer_path = _write_latest_pointer(case)
    store = QueryStore.open(load_latest(pointer_path))
    try:
        yield store, case, pointer_path
    finally:
        store.close()


def test_latest_pointer_opens_the_matching_database_read_only(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)
    pointer_path = _write_latest_pointer(case)
    before = case.database.stat().st_mtime_ns

    store = QueryStore.open(load_latest(pointer_path))
    assert store.status()["scan_id"] == "PGOV_TEST_001"
    store.close()

    assert case.database.stat().st_mtime_ns == before
    assert not Path(f"{case.database}-wal").exists()
    assert not Path(f"{case.database}-shm").exists()


def test_latest_pointer_rejects_a_scan_id_mismatch(tmp_path: Path) -> None:
    case = write_inventory_fixture(tmp_path)
    build_index(receipt_path=case.receipt, external_root=case.external_root, database_path=case.database)
    pointer_path = _write_latest_pointer(case)
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    payload["scan_id"] = "PGOV_WRONG"
    pointer_path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="scan_id"):
        QueryStore.open(load_latest(pointer_path))


def test_find_accepts_asset_id_and_normalized_absolute_path(built_store) -> None:
    store, _, _ = built_store
    assert store.find_assets("asset-local", limit=20)["items"][0]["relative_path"] == "runs/local-a"
    assert store.find_assets(r"E:\type10-7\runs\local-a", limit=20)["items"][0]["asset_id"] == "asset-local"


def test_find_prefix_is_bounded_and_empty_results_are_explicit(built_store) -> None:
    store, _, _ = built_store
    result = store.find_assets("E:/type10-7/runs", limit=1)
    assert result["count"] == 1
    assert result["truncated"] is False
    assert store.find_assets("E:/type10-7/missing", limit=20) == {
        "count": 0,
        "items": [],
        "query": "E:/type10-7/missing",
        "truncated": False,
    }
    with pytest.raises(ValueError, match="between 1 and 100"):
        store.find_assets("runs", limit=101)


def test_experiment_keeps_one_run_evidence_together(built_store) -> None:
    store, _, _ = built_store
    result = store.experiment("RUN_A")
    assert result["run_id"] == "RUN_A"
    assert result["assets_by_location"] == {"LOCAL": 1, "N607": 1}
    assert result["git_commit"] == "a" * 40


def test_repo_returns_ambiguous_instead_of_guessing(built_store) -> None:
    store, _, _ = built_store
    result = store.repo(r"E:\type10-7\runs\local-a")
    assert result["status"] == "AMBIGUOUS"
    assert [item["repository_root"] for item in result["items"]] == [
        "E:/type10-7/code/second-repo",
        "E:/type10-7/github_publish/CVS-RFFI-repo",
    ]


def test_review_never_promotes_review_rows_to_deletion_candidates(built_store) -> None:
    store, _, _ = built_store
    result = store.review({"retention_class": "REVIEW_REQUIRED"}, limit=20)
    assert result["count"] == 2
    assert result["authorized_deletion_count"] == 0
    assert {item["retention_class"] for item in result["items"]} == {"REVIEW_REQUIRED"}
