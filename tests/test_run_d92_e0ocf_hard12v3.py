from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_d92_e0ocf_hard12v3 as runner

QUERY_ZERO_FIELDS = (
    "query_truth_access", "query_fit_access", "query_update_access", "query_selection_access",
    "query_role_oracle_access", "query_class_quota_access", "query_global_reassignment",
)
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_e0ocf_5arm_hard12v3_v1.json").resolve()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _full_manifest(
    tmp_path: Path, *, method_lock_path: Path = METHOD_LOCK
) -> tuple[Path, str, dict[str, object]]:
    output = tmp_path / "matrix"
    manifest = runner.build_hard12v3_manifest(
        context_path=CONTEXT,
        method_lock_path=method_lock_path,
        output_root=output,
        require_package_files=False,
    )
    for job in manifest["jobs"]:
        for package in job["packages"].values():
            package["expected_seal_sha256"] = "a" * 64
    manifest_path = tmp_path / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, digest, manifest


def _write_prediction_closure(
    root: Path,
    *,
    missing: str | None = None,
    query_violation: bool = False,
    corruption: str | None = None,
) -> None:
    for state in ("before", "after"):
        state_root = root / state
        state_root.mkdir(parents=True, exist_ok=True)
        query_tokens = np.asarray(["q0", "q1", "q2"])
        scenarios = np.asarray(SCENES)
        if state == "after" and corruption == "query_token_drift":
            query_tokens = np.asarray(["q0", "q1", "q9"])
        if state == "after" and corruption == "scenario_order_drift":
            scenarios = scenarios[::-1]
        if missing != f"{state}_prediction":
            arrays = {
                "query_tokens": query_tokens,
                "scenarios": scenarios,
                "predicted_class_handles": np.asarray(["old_0", "old_1", "old_2"]),
            }
            if corruption == "extra_npz_key":
                arrays["extra"] = np.asarray([1, 2, 3])
            np.savez(
                state_root / "prediction_artifact.npz",
                **arrays,
            )
        rows = [
            {
                "scenario": scenario,
                "query_truth_access": query_violation and state == "before",
                "query_fit_access": False,
                "query_update_access": False,
                "query_selection_access": False,
                "query_role_oracle_access": False,
                "query_class_quota_access": False,
                "query_global_reassignment": False,
            }
            for scenario in SCENES
        ]
        if corruption == "duplicate_scene":
            rows[1]["scenario"] = SCENES[0]
        elif corruption == "missing_scene":
            rows.pop()
        if missing != f"{state}_fit_audit":
            _write_json(state_root / "fit_audit.json", rows)
        _write_json(state_root / "resource_audit.json", {"state": state})
        _write_json(
            state_root / "execution_receipt.json",
            {"schema": "cvs.phase2.diag_cosine_exploration_receipt.v1"},
        )
        member_names = (
            "execution_receipt.json",
            "fit_audit.json",
            "prediction_artifact.npz",
            "resource_audit.json",
        )
        if all((state_root / name).is_file() for name in member_names):
            members = [
                {
                    "relative_path": name,
                    "sha256": _sha256(state_root / name),
                    "size_bytes": (state_root / name).stat().st_size,
                }
                for name in member_names
            ]
            commit = {
                "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
                "members": members,
                "artifact_root_sha256": hashlib.sha256(
                    _canonical_bytes(members)
                ).hexdigest(),
                "execution_receipt_sha256": _sha256(
                    state_root / "execution_receipt.json"
                ),
                "prediction_artifact_sha256": _sha256(
                    state_root / "prediction_artifact.npz"
                ),
            }
            if corruption == "stale_commit":
                commit["members"][0]["sha256"] = "0" * 64
            if missing != f"{state}_commit":
                _write_json(state_root / "COMMIT.json", commit)
        if corruption == "corrupt_npz":
            (state_root / "prediction_artifact.npz").write_bytes(b"not-an-npz")
        elif corruption == "empty_commit":
            (state_root / "COMMIT.json").write_bytes(b"")


def _fake_child_run(command: list[str], **_: object) -> SimpleNamespace:
    if "run_d92_e0d_prediction.py" in str(command[1]):
        root = Path(command[command.index("--output-root") + 1])
        _write_prediction_closure(root)
    else:
        path = Path(command[command.index("--output-path") + 1])
        _write_json(path, {"status": "PASS"})
    return SimpleNamespace(returncode=0)


def test_shared_stop_counts_distinct_outers_not_arms(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    output.mkdir()
    fp = "a" * 64
    first = {"job_id": "outer_a__arm_full", "outer_key": "outer_a", "arm_id": "D92_FULL"}
    same = {"job_id": "outer_a__arm_ocf25", "outer_key": "outer_a", "arm_id": "E0_OCF25"}
    second = {"job_id": "outer_b__arm_full", "outer_key": "outer_b", "arm_id": "D92_FULL"}
    assert runner._record_shared_pre_prediction_failure(output, first, fp) is False
    assert runner._record_shared_pre_prediction_failure(output, same, fp) is False
    assert runner._record_shared_pre_prediction_failure(output, second, fp) is True


def test_cli_parser_exposes_prepare_smoke_and_run_shard() -> None:
    parser = runner.parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0


def test_full_matrix_first_smoke_publishes_exact_shared_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", _fake_child_run)
    args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, output_root=str(Path(manifest["output_root"]) / "smoke"), device="cpu", cpu_threads=1)
    receipt = runner.truth_free_smoke(args)
    smoke_root = Path(str(manifest["output_root"])) / "smoke"
    assert (smoke_root / "smoke_receipt.json").is_file()
    assert receipt["matrix_manifest_sha256"] == digest
    assert receipt["selection_sha256"] == runner.CANONICAL_SELECTION_SHA256
    assert receipt["outer_key"] == runner.SMOKE_OUTER_KEY
    assert receipt["job_id"] == f"{runner.SMOKE_OUTER_KEY}__arm_d92_full"
    assert receipt["arm_id"] == "D92_FULL"
    assert receipt["k_shot"] == 1
    assert receipt["truth_open"] is False
    assert all(receipt[field] is False for field in QUERY_ZERO_FIELDS)


def test_full_matrix_run_shard_rejects_absent_or_tampered_smoke_before_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, shard_index=0, shard_count=8, device="cpu", cpu_threads=1)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("child dispatch must not occur")))
    with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="smoke"):
        runner.run_shard(args)
    smoke_root = Path(str(manifest["output_root"])) / "smoke"
    smoke_root.mkdir(parents=True)
    _write_json(smoke_root / "smoke_receipt.json", {"status": "tampered"})
    with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="smoke"):
        runner.run_shard(args)
    assert not (Path(str(manifest["output_root"])) / "events").exists()


def test_full_matrix_run_shard_accepts_valid_shared_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", _fake_child_run)
    smoke_args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, output_root=str(Path(manifest["output_root"]) / "smoke"), device="cpu", cpu_threads=1)
    runner.truth_free_smoke(smoke_args)
    shard_args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, shard_index=0, shard_count=8, device="cpu", cpu_threads=1)
    summary = runner.run_shard(shard_args)
    assert summary["status"] == "PASS"
    expected_ids = [job["job_id"] for job in manifest["jobs"] if job["planned_shard_index"] == 0]
    assert summary["selected_job_count"] == len(expected_ids) == 10
    assert summary["completed_job_ids"] == expected_ids


def test_run_shard_routes_same_missing_commit_on_two_outers_to_shared_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if returncode-zero closure gaps bypassed the shared ledger."""

    manifest_path, digest, manifest = _full_manifest(tmp_path)
    smoke_root = Path(str(manifest["output_root"])) / "smoke" / "diag"
    score_calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if "run_d92_e0d_prediction.py" in str(command[1]):
            root = Path(command[command.index("--output-root") + 1])
            _write_prediction_closure(
                root,
                missing=None if root == smoke_root else "before_commit",
            )
        else:
            score_calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.truth_free_smoke(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            output_root=str(smoke_root.parent),
            device="cpu",
            cpu_threads=1,
        )
    )
    summary = runner.run_shard(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            shard_index=0,
            shard_count=8,
            device="cpu",
            cpu_threads=1,
        )
    )
    stop = json.loads(
        (Path(str(manifest["output_root"])) / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert stop["reason"] == "same_pre_prediction_fingerprint_on_two_distinct_outers"
    assert stop["distinct_outer_count"] == 2
    assert score_calls == []


@pytest.mark.parametrize(
    "corruption",
    (
        "corrupt_npz",
        "extra_npz_key",
        "empty_commit",
        "stale_commit",
        "duplicate_scene",
        "missing_scene",
        "query_token_drift",
        "scenario_order_drift",
    ),
)
def test_run_shard_routes_invalid_prediction_closure_through_shared_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
) -> None:
    """Would fail if rc0 malformed artifacts reached the truth-side scorer."""

    manifest_path, digest, manifest = _full_manifest(tmp_path)
    smoke_root = Path(str(manifest["output_root"])) / "smoke" / "diag"
    score_calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if "run_d92_e0d_prediction.py" in str(command[1]):
            root = Path(command[command.index("--output-root") + 1])
            _write_prediction_closure(
                root,
                corruption=None if root == smoke_root else corruption,
            )
        else:
            score_calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.truth_free_smoke(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            output_root=str(smoke_root.parent),
            device="cpu",
            cpu_threads=1,
        )
    )
    summary = runner.run_shard(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            shard_index=0,
            shard_count=8,
            device="cpu",
            cpu_threads=1,
        )
    )
    stop_path = (
        Path(str(manifest["output_root"]))
        / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"
    )
    assert stop_path.is_file()
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    assert summary["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert stop["reason"] == "same_pre_prediction_fingerprint_on_two_distinct_outers"
    assert stop["distinct_outer_count"] == 2
    assert score_calls == []


def test_run_shard_query_audit_violation_publishes_p0_stop_before_scorer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Would fail if a query-access audit violation reached truth-side scoring."""

    manifest_path, digest, manifest = _full_manifest(tmp_path)
    smoke_root = Path(str(manifest["output_root"])) / "smoke" / "diag"
    prediction_calls: list[list[str]] = []
    score_calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if "run_d92_e0d_prediction.py" in str(command[1]):
            root = Path(command[command.index("--output-root") + 1])
            _write_prediction_closure(root, query_violation=root != smoke_root)
            if root != smoke_root:
                prediction_calls.append(command)
        else:
            score_calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.truth_free_smoke(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            output_root=str(smoke_root.parent),
            device="cpu",
            cpu_threads=1,
        )
    )
    summary = runner.run_shard(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            shard_index=0,
            shard_count=8,
            device="cpu",
            cpu_threads=1,
        )
    )
    stop = json.loads(
        (Path(str(manifest["output_root"])) / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert stop["reason"] == "query_audit_protocol_violation"
    assert stop["distinct_outer_count"] == 1
    assert len(prediction_calls) == 1
    assert score_calls == []


def test_full_manifest_rejects_missing_or_tampered_smoke_outer_key(tmp_path: Path) -> None:
    manifest_path, _, manifest = _full_manifest(tmp_path)
    for value in (None, "rx_7_14__seed_713105__k_1__new_20"):
        payload = dict(manifest)
        if value is None:
            payload.pop("smoke_outer_key", None)
        else:
            payload["smoke_outer_key"] = value
        _write_json(manifest_path, payload)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="smoke_outer_key"):
            runner._load_manifest(manifest_path, digest)


def test_load_manifest_rejects_tampered_canonical_job_role(tmp_path: Path) -> None:
    """Would fail if runner bypassed the shared canonical manifest validator."""

    manifest_path, _, manifest = _full_manifest(tmp_path)
    manifest["jobs"][0]["role"] = "primary"
    _write_json(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="manifest"):
        runner._load_manifest(manifest_path, digest)


def test_load_manifest_rejects_tampered_on_disk_method_lock(tmp_path: Path) -> None:
    """Would fail if runner trusted only the method-lock hash stored in manifest."""

    lock_path = tmp_path / "method_lock.json"
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    _write_json(lock_path, lock)
    manifest_path, digest, _ = _full_manifest(
        tmp_path, method_lock_path=lock_path
    )
    lock["arms"]["E0_OCF50"]["lambda"] = 0.25
    _write_json(lock_path, lock)
    with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="method lock"):
        runner._load_manifest(manifest_path, digest)
