from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from scripts import run_cvs_somph_diag_125_stability as launcher


def _inputs(tmp_path: Path) -> dict[str, Path]:
    cache_root = tmp_path / "cache"
    authority_root = tmp_path / "authority"
    for receiver in launcher.RECEIVERS:
        receiver_leaf = f"rx_{receiver.replace('-', '_')}"
        for seed in launcher.CONFIRMATION_SEEDS:
            cache = cache_root / receiver_leaf / f"seed_{seed}" / "cache_set.json"
            cache.parent.mkdir(parents=True)
            cache.write_text('{"cache":"leo_weak"}\n', encoding="utf-8")
            bundle = (
                authority_root
                / f"authority_bundle_{receiver_leaf}_seed_{seed}"
            )
            bundle.mkdir(parents=True)
            (bundle / "COMMIT.json").write_text(
                json.dumps({"receiver": receiver, "seed": seed}) + "\n",
                encoding="utf-8",
            )
    artifacts = {}
    for name in ("phase1.pth", "runtime.pt", "method.json"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = path
    return {
        "cache_root": cache_root,
        "authority_root": authority_root,
        "checkpoint": artifacts["phase1.pth"],
        "runtime": artifacts["runtime.pt"],
        "method": artifacts["method.json"],
    }


def _manifest(tmp_path: Path) -> dict:
    inputs = _inputs(tmp_path)
    return launcher.build_manifest(
        cache_root=inputs["cache_root"],
        authority_root=inputs["authority_root"],
        phase1_checkpoint=inputs["checkpoint"],
        sealed_runtime=inputs["runtime"],
        method_lock=inputs["method"],
        output_root=tmp_path / "out",
    )


def test_fixed_matrix_has_exact_125_confirmation_pairs_and_k1(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    jobs = manifest["jobs"]
    assert manifest["job_count"] == 125
    assert manifest["row_pair_count"] == 125
    assert manifest["scenario_pair_count"] == 375
    assert manifest["scenario_state_metric_count"] == 750
    assert manifest["claim_scope"] == launcher.CLAIM_SCOPE
    assert manifest["formal_launch_authority"] is False
    assert manifest["locked_shard_count"] == 8
    assert manifest["planned_shard_job_counts"] == [
        16,
        16,
        16,
        16,
        16,
        15,
        15,
        15,
    ]
    assert len({job["job_id"] for job in jobs}) == 125
    assert {job["receiver"] for job in jobs} == set(launcher.RECEIVERS)
    assert {job["seed"] for job in jobs} == set(
        launcher.CONFIRMATION_SEEDS
    )
    assert 713101 not in {job["seed"] for job in jobs}
    expected_slices = set(launcher.SLICES)
    for receiver in launcher.RECEIVERS:
        for seed in launcher.CONFIRMATION_SEEDS:
            actual = {
                (job["k_shot"], job["new_class_count"])
                for job in jobs
                if job["receiver"] == receiver and job["seed"] == seed
            }
            assert actual == expected_slices
    k1 = [job for job in jobs if job["k_shot"] == 1]
    assert len(k1) == 25
    assert {job["new_class_count"] for job in k1} == {20}
    assert all(
        job["support_nesting"]["k1_uses_first_k10_physical_support"]
        for job in k1
    )


def test_manifest_locks_d1_stage2bc_and_phase2_contract(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    assert manifest["candidate"] == "d1_historical_diag_fftrf"
    assert manifest["phase2_contract"] == launcher.PHASE2_CONTRACT
    for job in manifest["jobs"]:
        assert job["candidate"] == launcher.CANDIDATE
        assert job["row_pair"] == {
            "before_registration": "stage2b",
            "after_registration": "stage2c",
        }
        assert job["scenarios"] == list(launcher.SCENARIOS)
        assert job["authority_commit_sha256"] == launcher._sha256(
            Path(job["authority_commit_path"])
        )


def test_job_command_calls_existing_pipeline_with_locked_candidate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    job = manifest["jobs"][0]
    command = launcher._job_command(
        job,
        phase1_checkpoint=manifest["phase1_checkpoint"],
        sealed_runtime=manifest["sealed_runtime"],
        method_lock=manifest["method_lock"],
        device="cuda:3",
    )
    assert command[1] == str(launcher.ROW_PIPELINE)
    assert command[command.index("--candidate") + 1] == launcher.CANDIDATE
    assert command[command.index("--device") + 1] == "cuda:3"
    assert command[command.index("--k-shot") + 1] == str(job["k_shot"])
    assert command[command.index("--new-count") + 1] == str(
        job["new_class_count"]
    )


def test_shards_partition_all_125_jobs_once(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selected = []
    for shard_index in range(launcher.LOCKED_SHARD_COUNT):
        selected.extend(
            job["job_id"]
            for job in manifest["jobs"]
            if job["planned_shard_index"] == shard_index
        )
    assert len(selected) == 125
    assert len(set(selected)) == 125
    assert all(
        launcher._selected(
            job["index"],
            job["planned_shard_index"],
            launcher.LOCKED_SHARD_COUNT,
        )
        for job in manifest["jobs"]
    )


def test_run_rejects_unlocked_shard_count(tmp_path: Path) -> None:
    args = Namespace(
        cache_root="cache",
        authority_root="authority",
        phase1_checkpoint="phase1",
        sealed_runtime="runtime",
        method_lock="method",
        output_root=str(tmp_path / "out"),
        device="cuda:0",
        shard_index=0,
        shard_count=9,
        fail_fast=False,
        manifest_only=True,
    )
    try:
        launcher.run(args)
    except launcher.StabilityLauncherError as exc:
        assert "locked to 8" in str(exc)
    else:
        raise AssertionError("unlocked shard count must fail closed")


def test_technical_failure_is_recorded_and_next_job_continues(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "out"
    jobs = []
    for index in (0, 8):
        jobs.append(
            {
                "index": index,
                "planned_shard_index": 0,
                "job_id": f"job_{index}",
                "receiver": "20-1",
                "seed": 713102,
                "k_shot": 10,
                "new_class_count": 5 + 5 * index,
                "cache_manifest": str(tmp_path / "cache.json"),
                "authority_bundle": str(tmp_path / "authority"),
                "authority_commit_sha256": "a" * 64,
                "output_root": str(output / "jobs" / f"job_{index}"),
            }
        )
    manifest = {
        "schema": launcher.SCHEMA,
        "job_count": 2,
        "locked_shard_count": launcher.LOCKED_SHARD_COUNT,
        "candidate": launcher.CANDIDATE,
        "phase1_checkpoint": str(tmp_path / "phase1.pth"),
        "sealed_runtime": str(tmp_path / "runtime.pt"),
        "method_lock": str(tmp_path / "method.json"),
        "output_root": str(output),
        "jobs": jobs,
    }
    monkeypatch.setattr(launcher, "build_manifest", lambda **_kwargs: manifest)
    returncodes = iter((7, 0))

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    args = Namespace(
        cache_root="cache",
        authority_root="authority",
        phase1_checkpoint="phase1",
        sealed_runtime="runtime",
        method_lock="method",
        output_root=str(output),
        device="cuda:0",
        shard_index=0,
        shard_count=launcher.LOCKED_SHARD_COUNT,
        fail_fast=False,
        manifest_only=False,
    )
    summary = launcher.run(args)
    assert summary["status"] == "PARTIAL_FAILURE"
    assert summary["completed_job_ids"] == ["job_8"]
    assert summary["failures"] == [
        {
            "job_id": "job_0",
            "returncode": 7,
            "error": "row pipeline technical failure",
        }
    ]
    events = [
        json.loads(line)
        for line in Path(summary["events"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [event["event"] for event in events] == [
        "JOB_START",
        "JOB_FAILED",
        "JOB_START",
        "JOB_COMPLETE",
    ]
