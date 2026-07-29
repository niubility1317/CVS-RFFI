from __future__ import annotations

import copy
import hashlib
import json
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cvsrffi.full_ablation_spec import build_phase1_t1_rows
from scripts import run_full_ablation_phase1_t1 as phase1_runner
from scripts.run_full_ablation_phase1_t1 import (
    _Capacity,
    Phase1RunnerError,
    build_phase1_command,
    build_phase1_dispatch_schedule,
    build_phase1_reexport_command,
    is_p0_protocol_failure,
    normalize_exception_fingerprint,
    run_release,
    validate_phase1_row_completion,
    validate_phase1_release_plan,
    validate_phase1_reexport_completion,
    validate_phase1_reuse_manifest,
)


def _plan() -> dict:
    plan = {
        "schema": "cvs.full_ablation.plan.v1",
        "design_id": "cvs_full_ablation_phase1_phase2_20260728",
        "phase": "phase1",
        "stage": "t1",
        "git_commit": "a" * 40,
        "formal_launch_authority": False,
        "seed_registry_sha256": "c" * 64,
        "wisig_pkl_sha256": "f" * 64,
        "python_environment_id": "CVS-RFFI",
        "registered_phase1_train_seeds": [
            7281101,
            7281102,
            7281103,
            7281104,
            7281105,
        ],
        "rows": build_phase1_t1_rows(
            [7281101, 7281102, 7281103, 7281104, 7281105],
            git_commit="a" * 40,
        ),
    }
    return plan


def _seal_for_test(plan: dict) -> None:
    content = {
        key: value
        for key, value in plan.items()
        if key != "sealed_content_sha256"
    }
    plan["sealed_content_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_unsealed_plan_is_dry_run_valid_but_not_launchable() -> None:
    plan = _plan()
    validate_phase1_release_plan(plan, require_launch_authority=False)
    with pytest.raises(Phase1RunnerError, match="launch authority"):
        validate_phase1_release_plan(plan, require_launch_authority=True)


def test_sealed_plan_requires_review_and_local_verified_executors() -> None:
    plan = _plan()
    plan["formal_launch_authority"] = True
    plan["run_id"] = "cvs_full_ablation_phase1_t1_20260728_v1"
    plan["independent_review"] = {"p0_count": 0, "p1_count": 0}
    _seal_for_test(plan)
    with pytest.raises(Phase1RunnerError, match="LOCAL_VERIFIED"):
        validate_phase1_release_plan(plan, require_launch_authority=True)
    for row in plan["rows"]:
        row["executor_status"] = "LOCAL_VERIFIED"
    _seal_for_test(plan)
    validate_phase1_release_plan(plan, require_launch_authority=True)
    plan["python_environment_id"] = "ssr-gpu"
    _seal_for_test(plan)
    with pytest.raises(Phase1RunnerError, match="CVS-RFFI"):
        validate_phase1_release_plan(
            plan,
            require_launch_authority=True,
        )


def test_command_binds_exact_row_identity(tmp_path) -> None:
    row = _plan()["rows"][0]
    command = build_phase1_command(
        row,
        run_id="run-v1",
        python_executable="/envs/CVS-RFFI/bin/python",
        train_script=tmp_path / "train_ssdg.py",
        wisig_pkl=tmp_path / "wisig.pkl",
        output_dir=tmp_path / row["row_key"],
        sealed_plan_sha256="d" * 64,
        seed_registry_sha256="c" * 64,
        wisig_pkl_sha256="f" * 64,
        dataset_receipt_path=str(tmp_path / "dataset_receipt.json"),
        dataset_receipt_sha256="e" * 64,
        environment_receipt_path=str(
            tmp_path / "environment_receipt.json"
        ),
        environment_receipt_sha256="9" * 64,
        python_environment_id="CVS-RFFI",
    )
    joined = " ".join(command)
    assert "--formal_ablation true" in joined
    assert f"--ablation_id {row['ablation_id']}" in joined
    assert f"--seed {row['train_seed']}" in joined
    assert f"--git_commit {row['git_commit']}" in joined
    assert f"--row_key {row['row_key']}" in joined
    assert f"--sealed_plan_sha256 {'d' * 64}" in joined
    assert f"--seed_registry_sha256 {'c' * 64}" in joined
    assert "--python_environment_id CVS-RFFI" in joined
    assert "--device cuda:0" in joined


def test_reexport_command_binds_source_and_current_exporter(tmp_path) -> None:
    row = _plan()["rows"][0]
    command = build_phase1_reexport_command(
        row,
        {
            "source_checkpoint": "/old/checkpoint.pth",
            "source_run_id": "phase1-v3",
        },
        python_executable="/envs/CVS-RFFI/bin/python",
        reexport_script=tmp_path / "reexport.py",
        wisig_pkl=tmp_path / "ManySig.pkl",
        output_dir=tmp_path / row["row_key"],
        exporter_git_commit="a" * 40,
    )
    joined = " ".join(command)
    assert "--checkpoint /old/checkpoint.pth" in joined
    assert f"--row-key {row['row_key']}" in joined
    assert "--source-run-id phase1-v3" in joined
    assert f"--exporter-git-commit {'a' * 40}" in joined


def _write_direct_reuse_fixture(tmp_path, row) -> tuple[dict, dict]:
    source_output = tmp_path / "source" / row["row_key"]
    source_output.mkdir(parents=True)
    source_log = tmp_path / "logs" / f"{row['row_key']}.out"
    source_log.parent.mkdir()
    source_log.write_text("complete\n", encoding="utf-8")
    heldout = {"status": "COMPLETE", "accuracy": 0.5}
    payloads = {
        "phase1_resource_summary.json": {"wall_time_seconds": 1},
        "frozen_phase1_heldout_eval.json": heldout,
        "phase2_zid_prototypes.json": {"schema": "prototype"},
    }
    for name, payload in payloads.items():
        (source_output / name).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    checkpoint = source_output / "best_source_validation_ssdg.pth"
    prototype = source_output / "phase2_zid_prototypes.pt"
    torch.save(
        {"model": {"weight": torch.ones(1)}},
        checkpoint,
    )
    torch.save(
        {"metadata": {"schema": "prototype"}},
        prototype,
    )
    terminal = {
        "status": "COMPLETE",
        "exit_code": 0,
        "heldout_eval": heldout,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
    }
    terminal_path = source_output / "phase1_terminal_status.json"
    terminal_path.write_text(
        json.dumps(terminal),
        encoding="utf-8",
    )
    receipt = {
        "terminal_status": "COMPLETE",
        "exit_code": 0,
        "row_key": row["row_key"],
        "ablation_id": row["ablation_id"],
        "train_seed": row["train_seed"],
        "terminal_manifest_sha256": hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest(),
        "resource_summary_sha256": hashlib.sha256(
            (
                source_output / "phase1_resource_summary.json"
            ).read_bytes()
        ).hexdigest(),
        "selected_checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "prototype_paths": {
            "prototype_path": str(prototype),
            "prototype_json_path": str(
                source_output / "phase2_zid_prototypes.json"
            ),
        },
        "prototype_hashes": {
            "prototype_path": hashlib.sha256(
                prototype.read_bytes()
            ).hexdigest(),
            "prototype_json_path": hashlib.sha256(
                (
                    source_output / "phase2_zid_prototypes.json"
                ).read_bytes()
            ).hexdigest(),
        },
    }
    (
        source_output / "phase1_training_completion_receipt.json"
    ).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    payloads["phase1_terminal_status.json"] = terminal
    payloads["phase1_training_completion_receipt.json"] = receipt
    entry = {
        "row_key": row["row_key"],
        "mode": "direct_reuse",
        "source_run_id": "phase1-v3",
        "source_output_dir": str(source_output),
        "source_log_path": str(source_log),
    }
    return entry, payloads


def test_reuse_manifest_accepts_complete_direct_and_reexport_rows(
    tmp_path,
) -> None:
    plan = _plan()
    direct_entry, _ = _write_direct_reuse_fixture(
        tmp_path,
        plan["rows"][0],
    )
    checkpoint = tmp_path / "b0.pth"
    checkpoint.write_bytes(b"checkpoint")
    reexport_entry = {
        "row_key": plan["rows"][1]["row_key"],
        "mode": "reexport_only",
        "source_run_id": "phase1-v3",
        "source_checkpoint": str(checkpoint),
    }
    result = validate_phase1_reuse_manifest(
        {
            "schema": "cvs.full_ablation.phase1_reuse.v1",
            "rows": [direct_entry, reexport_entry],
        },
        plan,
        check_artifacts=True,
    )
    assert set(result) == {
        plan["rows"][0]["row_key"],
        plan["rows"][1]["row_key"],
    }


def test_reuse_dispatch_rebalances_twenty_tasks_across_all_slots() -> None:
    plan = _plan()
    direct_rows = plan["rows"][:10]
    reuse_entries = {
        str(row["row_key"]): {
            "row_key": str(row["row_key"]),
            "mode": "direct_reuse",
        }
        for row in direct_rows
    }
    reuse_entries[str(plan["rows"][10]["row_key"])] = {
        "row_key": str(plan["rows"][10]["row_key"]),
        "mode": "reexport_only",
    }
    schedule = build_phase1_dispatch_schedule(plan, reuse_entries)
    assert len(schedule) == 20
    assert len({(gpu, slot) for _row, gpu, slot in schedule}) == 16
    assert {gpu for _row, gpu, _slot in schedule} == set(range(8))


def test_reuse_manifest_rejects_incomplete_direct_row(tmp_path) -> None:
    plan = _plan()
    entry, _ = _write_direct_reuse_fixture(
        tmp_path,
        plan["rows"][0],
    )
    (
        Path(entry["source_output_dir"])
        / "phase2_zid_prototypes.pt"
    ).write_bytes(b"")
    with pytest.raises(
        Phase1RunnerError,
        match="absent or empty artifacts",
    ):
        validate_phase1_reuse_manifest(
            {
                "schema": "cvs.full_ablation.phase1_reuse.v1",
                "rows": [entry],
            },
            plan,
            check_artifacts=True,
        )


@pytest.mark.parametrize(
    "fault",
    (
        "corrupt_checkpoint",
        "corrupt_prototype",
        "terminal_exit",
        "empty_resource",
        "heldout_mismatch",
    ),
)
def test_direct_reuse_rejects_damaged_or_unbound_artifacts(
    tmp_path,
    fault,
) -> None:
    plan = _plan()
    row = plan["rows"][0]
    entry, _ = _write_direct_reuse_fixture(tmp_path, row)
    output = Path(entry["source_output_dir"])
    terminal_path = output / "phase1_terminal_status.json"
    receipt_path = output / "phase1_training_completion_receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if fault == "corrupt_checkpoint":
        checkpoint = output / "best_source_validation_ssdg.pth"
        checkpoint.write_bytes(b"not-a-checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        terminal["selected_checkpoint_sha256"] = digest
        receipt["selected_checkpoint_sha256"] = digest
    elif fault == "corrupt_prototype":
        prototype = output / "phase2_zid_prototypes.pt"
        prototype.write_bytes(b"not-a-prototype")
        receipt["prototype_hashes"]["prototype_path"] = (
            hashlib.sha256(prototype.read_bytes()).hexdigest()
        )
    elif fault == "terminal_exit":
        terminal["exit_code"] = 7
    elif fault == "empty_resource":
        resource = output / "phase1_resource_summary.json"
        resource.write_text("{}", encoding="utf-8")
        receipt["resource_summary_sha256"] = hashlib.sha256(
            resource.read_bytes()
        ).hexdigest()
    elif fault == "heldout_mismatch":
        terminal["heldout_eval"]["accuracy"] = 0.9
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    receipt["terminal_manifest_sha256"] = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(Phase1RunnerError):
        validate_phase1_reuse_manifest(
            {
                "schema": "cvs.full_ablation.phase1_reuse.v1",
                "rows": [entry],
            },
            plan,
            check_artifacts=True,
        )


def test_reexport_completion_rejects_corrupt_artifact(tmp_path) -> None:
    row = _plan()["rows"][0]
    output = tmp_path / row["row_key"]
    output.mkdir()
    prototype = output / "phase2_zid_prototypes.pt"
    prototype_json = output / "phase2_zid_prototypes.json"
    torch.save({"metadata": {"schema": "prototype"}}, prototype)
    prototype_json.write_text(
        json.dumps({"schema": "prototype"}),
        encoding="utf-8",
    )
    source_checkpoint = tmp_path / "source_checkpoint.pth"
    torch.save(
        {"model": {"weight": torch.ones(1)}},
        source_checkpoint,
    )
    entry = {
        "source_run_id": "phase1-v3",
        "source_checkpoint": str(source_checkpoint),
    }
    receipt = {
        "schema": "cvs.phase1.prototype_reexport_receipt.v1",
        "status": "COMPLETE",
        "exit_code": 0,
        "row_key": row["row_key"],
        "source_run_id": "phase1-v3",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": hashlib.sha256(
            source_checkpoint.read_bytes()
        ).hexdigest(),
        "exporter_git_commit": "a" * 40,
        "prototype_paths": {
            "prototype_path": str(prototype),
            "prototype_json_path": str(prototype_json),
        },
        "prototype_hashes": {
            "prototype_path": hashlib.sha256(
                prototype.read_bytes()
            ).hexdigest(),
            "prototype_json_path": hashlib.sha256(
                prototype_json.read_bytes()
            ).hexdigest(),
        },
    }
    (output / "phase1_reexport_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    validate_phase1_reexport_completion(
        entry=entry,
        row=row,
        output_dir=output,
        return_code=0,
        exporter_git_commit="a" * 40,
    )
    prototype.write_bytes(b"damaged")
    with pytest.raises(
        phase1_runner.Phase1ProtocolError,
        match="artifact drift",
    ):
        validate_phase1_reexport_completion(
            entry=entry,
            row=row,
            output_dir=output,
            return_code=0,
            exporter_git_commit="a" * 40,
        )


@pytest.mark.parametrize(
    "fault",
    (
        "cross_row_paths",
        "corrupt_prototype",
        "empty_prototype_json",
        "wrong_checkpoint_hash",
    ),
)
def test_reexport_rejects_unbound_or_damaged_artifacts(
    tmp_path,
    fault,
) -> None:
    row = _plan()["rows"][0]
    output = tmp_path / row["row_key"]
    output.mkdir()
    prototype = output / "phase2_zid_prototypes.pt"
    prototype_json = output / "phase2_zid_prototypes.json"
    torch.save({"metadata": {"schema": "prototype"}}, prototype)
    prototype_json.write_text(
        json.dumps({"schema": "prototype"}),
        encoding="utf-8",
    )
    source_checkpoint = tmp_path / "source_checkpoint.pth"
    torch.save(
        {"model": {"weight": torch.ones(1)}},
        source_checkpoint,
    )
    entry = {
        "source_run_id": "phase1-v3",
        "source_checkpoint": str(source_checkpoint),
    }
    receipt = {
        "schema": "cvs.phase1.prototype_reexport_receipt.v1",
        "status": "COMPLETE",
        "exit_code": 0,
        "row_key": row["row_key"],
        "source_run_id": "phase1-v3",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": hashlib.sha256(
            source_checkpoint.read_bytes()
        ).hexdigest(),
        "exporter_git_commit": "a" * 40,
        "prototype_paths": {
            "prototype_path": str(prototype),
            "prototype_json_path": str(prototype_json),
        },
        "prototype_hashes": {
            "prototype_path": hashlib.sha256(
                prototype.read_bytes()
            ).hexdigest(),
            "prototype_json_path": hashlib.sha256(
                prototype_json.read_bytes()
            ).hexdigest(),
        },
    }
    if fault == "cross_row_paths":
        other = tmp_path / "other-row"
        other.mkdir()
        other_pt = other / prototype.name
        other_json = other / prototype_json.name
        torch.save({"metadata": {"schema": "prototype"}}, other_pt)
        other_json.write_text(
            json.dumps({"schema": "prototype"}),
            encoding="utf-8",
        )
        receipt["prototype_paths"] = {
            "prototype_path": str(other_pt),
            "prototype_json_path": str(other_json),
        }
        receipt["prototype_hashes"] = {
            "prototype_path": hashlib.sha256(
                other_pt.read_bytes()
            ).hexdigest(),
            "prototype_json_path": hashlib.sha256(
                other_json.read_bytes()
            ).hexdigest(),
        }
    elif fault == "corrupt_prototype":
        prototype.write_bytes(b"not-a-prototype")
        receipt["prototype_hashes"]["prototype_path"] = (
            hashlib.sha256(prototype.read_bytes()).hexdigest()
        )
    elif fault == "empty_prototype_json":
        prototype_json.write_text("{}", encoding="utf-8")
        receipt["prototype_hashes"]["prototype_json_path"] = (
            hashlib.sha256(prototype_json.read_bytes()).hexdigest()
        )
    elif fault == "wrong_checkpoint_hash":
        receipt["source_checkpoint_sha256"] = "0" * 64
    (output / "phase1_reexport_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(phase1_runner.Phase1ProtocolError):
        validate_phase1_reexport_completion(
            entry=entry,
            row=row,
            output_dir=output,
            return_code=0,
            exporter_git_commit="a" * 40,
        )


def test_plan_rejects_row_commit_or_arm_drift() -> None:
    plan = _plan()
    drift = copy.deepcopy(plan)
    drift["rows"][0]["git_commit"] = "b" * 40
    with pytest.raises(Phase1RunnerError, match="Git commit"):
        validate_phase1_release_plan(drift, require_launch_authority=False)
    drift = copy.deepcopy(plan)
    drift["rows"][0]["ablation_id"] = "P1-UNKNOWN"
    with pytest.raises(Phase1RunnerError, match="arm set"):
        validate_phase1_release_plan(drift, require_launch_authority=False)


def test_plan_rejects_noncanonical_row_identity_or_slot() -> None:
    plan = _plan()
    drift = copy.deepcopy(plan)
    drift["rows"][0]["row_key"] = "forged"
    with pytest.raises(
        Phase1RunnerError,
        match="exact registered 6x5",
    ):
        validate_phase1_release_plan(
            drift,
            require_launch_authority=False,
        )
    drift = copy.deepcopy(plan)
    drift["rows"][0]["worker"] = {"gpu": 0, "slot": 1}
    with pytest.raises(
        Phase1RunnerError,
        match="canonical row drift",
    ):
        validate_phase1_release_plan(
            drift,
            require_launch_authority=False,
        )


def test_exception_fingerprint_ignores_paths_addresses_and_numbers() -> None:
    first = "RuntimeError at C:\\run\\row1.py:123 address 0xABC value 9"
    second = "RuntimeError at D:\\other\\row2.py:987 address 0xDEF value 42"
    assert normalize_exception_fingerprint(first) == normalize_exception_fingerprint(
        second
    )


def test_p0_classifier_stops_on_formal_protocol_drift() -> None:
    assert is_p0_protocol_failure(
        "Phase1AbLATIONConfigError: formal Phase1 dataset receipt drift",
        None,
    )
    assert not is_p0_protocol_failure(
        "RuntimeError: transient worker allocation failure",
        RuntimeError("transient worker allocation failure"),
    )


def test_capacity_counts_external_and_owned_processes(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeProcess:
        pid = 22001

        def poll(self):
            return None

    launches: list[list[str]] = []

    def fake_popen(command, **kwargs):
        launches.append(list(command))
        return FakeProcess()

    monkeypatch.setattr(
        phase1_runner,
        "_gpu_process_pids",
        lambda: {gpu: ({11001} if gpu == 0 else set()) for gpu in range(8)},
    )
    monkeypatch.setattr(phase1_runner.subprocess, "Popen", fake_popen)
    capacity = _Capacity(0.0)
    process = capacity.launch(
        0,
        ["python", "train.py"],
        cwd=tmp_path,
        env={},
        stdout=None,
        stop_event=threading.Event(),
    )
    assert process.pid == 22001
    assert launches == [["python", "train.py"]]
    assert set(capacity.owned[0]) == {22001}

    class OneWaitStop:
        def __init__(self):
            self.stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _seconds):
            self.stopped = True

    with pytest.raises(Phase1RunnerError, match="dispatch stopped"):
        capacity.launch(
            0,
            ["python", "second.py"],
            cwd=tmp_path,
            env={},
            stdout=None,
            stop_event=OneWaitStop(),
        )
    assert len(launches) == 1


def test_terminate_owned_escalates_exact_process_group(monkeypatch) -> None:
    class FakeProcess:
        pid = 33001

        def poll(self):
            return None

    monkeypatch.setattr(phase1_runner.signal, "SIGKILL", 9, raising=False)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        phase1_runner.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    capacity = _Capacity(0.0)
    capacity.owned[3][33001] = FakeProcess()
    capacity.terminate_owned(grace_seconds=0.0)
    assert signals == [
        (33001, signal.SIGTERM),
        (33001, 9),
    ]


def test_execute_rejects_unreviewed_train_script(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        phase1_runner,
        "validate_phase1_release_plan",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        phase1_runner,
        "verify_release_checkout",
        lambda *_args, **_kwargs: None,
    )
    args = SimpleNamespace(
        repo_root=str(tmp_path),
        train_script=str(tmp_path / "unreviewed.py"),
    )
    with pytest.raises(Phase1RunnerError, match="reviewed train_ssdg.py"):
        run_release(args, {})


def test_execute_rejects_unreviewed_reuse_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        phase1_runner,
        "validate_phase1_release_plan",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        phase1_runner,
        "verify_release_checkout",
        lambda *_args, **_kwargs: None,
    )
    plan = {
        "release_files": {
            (
                "configs/full_ablation_20260728/"
                "phase1_t1_reuse_v5.json"
            ): "a" * 64,
        }
    }
    args = SimpleNamespace(
        repo_root=str(tmp_path),
        train_script=str(tmp_path / "code" / "SSDG" / "train_ssdg.py"),
        reuse_manifest=str(tmp_path / "unreviewed.json"),
    )
    with pytest.raises(
        Phase1RunnerError,
        match="reviewed reuse manifest",
    ):
        run_release(args, plan)


def test_completion_receipt_binds_row_plan_split_and_terminal(tmp_path) -> None:
    plan = _plan()
    plan["formal_launch_authority"] = True
    plan["run_id"] = "phase1-v1"
    plan["independent_review"] = {"p0_count": 0, "p1_count": 0}
    for row in plan["rows"]:
        row["executor_status"] = "LOCAL_VERIFIED"
        row["config_hash"] = "d" * 64
    _seal_for_test(plan)
    row = plan["rows"][0]
    output = tmp_path / row["row_key"]
    output.mkdir()
    split_payload = {
        "schema": "cvs.phase1.source_split_receipt.v1",
        "seed": row["train_seed"],
        "split_mode": "tx_rx_day_1_7_2",
        "source_days": ["d0", "d1"],
        "target_days": ["d2", "d3"],
        "source_receivers": [f"r{i}" for i in range(7)],
        "target_receivers": [f"r{i}" for i in range(7, 12)],
        "source_target_receiver_overlap_count": 0,
        "labeled_indices_sha256": "1" * 64,
        "unlabeled_indices_sha256": "2" * 64,
        "source_validation_indices_sha256": "3" * 64,
        "labeled_size": 1,
        "unlabeled_size": 1,
        "source_validation_size": 1,
        "wisig_pkl_sha256": plan["wisig_pkl_sha256"],
    }
    split_receipt = {
        **split_payload,
        "split_manifest_sha256": hashlib.sha256(
            json.dumps(
                split_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }
    dataset_receipt_path = tmp_path / "dataset_receipt.json"
    environment_receipt_path = tmp_path / "environment_receipt.json"
    dataset_receipt = {
        "schema": "cvs.phase1.dataset_receipt.v1",
        "wisig_pkl_sha256": plan["wisig_pkl_sha256"],
    }
    environment_receipt = {
        "schema": "cvs.phase1.python_environment_receipt.v1",
        "environment_id": "CVS-RFFI",
    }
    dataset_receipt_path.write_text(
        json.dumps(dataset_receipt),
        encoding="utf-8",
    )
    environment_receipt_path.write_text(
        json.dumps(environment_receipt),
        encoding="utf-8",
    )
    terminal = {
        "status": "COMPLETE",
        "exit_code": 0,
        "source_split_receipt": split_receipt,
        "dataset_receipt": dataset_receipt,
        "environment_receipt": environment_receipt,
    }
    terminal_path = output / "phase1_terminal_status.json"
    checkpoint_path = output / "best_source_validation_ssdg.pth"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint_path)
    checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    terminal.update(
        {
            "selected_checkpoint": str(checkpoint_path),
            "selected_checkpoint_sha256": checkpoint_hash,
        }
    )
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    resource_path = output / "phase1_resource_summary.json"
    resource_path.write_text(
        json.dumps({"wall_time_seconds": 1.0}),
        encoding="utf-8",
    )
    heldout_path = output / "frozen_phase1_heldout_eval.json"
    heldout_payload = {"status": "COMPLETE", "strict_udu_acc": 0.5}
    heldout_path.write_text(
        json.dumps(heldout_payload),
        encoding="utf-8",
    )
    terminal["heldout_eval"] = heldout_payload
    terminal["heldout_eval_path"] = str(heldout_path)
    terminal["heldout_eval_sha256"] = hashlib.sha256(
        heldout_path.read_bytes()
    ).hexdigest()
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    prototype_path = output / "phase2_zid_prototypes.pt"
    prototype_json_path = output / "phase2_zid_prototypes.json"
    torch.save({"metadata": {"schema": "prototype"}}, prototype_path)
    prototype_json_path.write_text(
        json.dumps({"schema": "prototype"}),
        encoding="utf-8",
    )
    dataset_receipt_hash = hashlib.sha256(
        dataset_receipt_path.read_bytes()
    ).hexdigest()
    environment_receipt_hash = hashlib.sha256(
        environment_receipt_path.read_bytes()
    ).hexdigest()
    receipt = {
        "run_id": plan["run_id"],
        "row_key": row["row_key"],
        "ablation_id": row["ablation_id"],
        "train_seed": row["train_seed"],
        "git_commit": plan["git_commit"],
        "sealed_plan_sha256": plan["sealed_content_sha256"],
        "seed_registry_sha256": plan["seed_registry_sha256"],
        "wisig_pkl_sha256": plan["wisig_pkl_sha256"],
        "dataset_receipt_sha256": dataset_receipt_hash,
        "environment_receipt_sha256": environment_receipt_hash,
        "dataset_receipt": dataset_receipt,
        "environment_receipt": environment_receipt,
        "resolved_config_hash": row["config_hash"],
        "method_config_hash": row["method_config_hash"],
        "terminal_manifest_sha256": hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest(),
        "resource_summary_sha256": hashlib.sha256(
            resource_path.read_bytes()
        ).hexdigest(),
        "source_split_receipt": split_receipt,
        "selected_checkpoint_sha256": checkpoint_hash,
        "prototype_paths": {
            "prototype_path": str(prototype_path),
            "prototype_json_path": str(prototype_json_path),
        },
        "prototype_hashes": {
            "prototype_path": hashlib.sha256(
                prototype_path.read_bytes()
            ).hexdigest(),
            "prototype_json_path": hashlib.sha256(
                prototype_json_path.read_bytes()
            ).hexdigest(),
        },
        "heldout_eval_path": str(heldout_path),
        "heldout_eval_sha256": hashlib.sha256(
            heldout_path.read_bytes()
        ).hexdigest(),
        "terminal_status": "COMPLETE",
        "exit_code": 0,
    }
    (output / "phase1_training_completion_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    assert (
        validate_phase1_row_completion(
            row=row,
            plan=plan,
            output_dir=output,
            return_code=0,
            dataset_receipt_sha256=dataset_receipt_hash,
            environment_receipt_sha256=environment_receipt_hash,
            dataset_receipt_path=dataset_receipt_path,
            environment_receipt_path=environment_receipt_path,
        )["row_key"]
        == row["row_key"]
    )
    receipt["resource_summary_sha256"] = "0" * 64
    (output / "phase1_training_completion_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(
        Phase1RunnerError,
        match="resource-summary hash drift",
    ):
        validate_phase1_row_completion(
            row=row,
            plan=plan,
            output_dir=output,
            return_code=0,
            dataset_receipt_sha256=dataset_receipt_hash,
            environment_receipt_sha256=environment_receipt_hash,
            dataset_receipt_path=dataset_receipt_path,
            environment_receipt_path=environment_receipt_path,
        )
    receipt["resource_summary_sha256"] = hashlib.sha256(
        resource_path.read_bytes()
    ).hexdigest()
    heldout_path.write_text("{}", encoding="utf-8")
    (output / "phase1_training_completion_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(
        Phase1RunnerError,
        match="heldout-eval artifact hash drift",
    ):
        validate_phase1_row_completion(
            row=row,
            plan=plan,
            output_dir=output,
            return_code=0,
            dataset_receipt_sha256=dataset_receipt_hash,
            environment_receipt_sha256=environment_receipt_hash,
            dataset_receipt_path=dataset_receipt_path,
            environment_receipt_path=environment_receipt_path,
        )


def test_completion_rejects_cross_row_prototype_paths(tmp_path) -> None:
    plan = _plan()
    plan["run_id"] = "phase1-v1"
    row = plan["rows"][0]
    output = tmp_path / row["row_key"]
    other = tmp_path / "other-row"
    output.mkdir()
    other.mkdir()
    external_pt = other / "phase2_zid_prototypes.pt"
    external_json = other / "phase2_zid_prototypes.json"
    torch.save({"metadata": {"schema": "prototype"}}, external_pt)
    external_json.write_text(
        json.dumps({"schema": "prototype"}),
        encoding="utf-8",
    )
    receipt = {
        "terminal_status": "COMPLETE",
        "prototype_paths": {
            "prototype_path": str(external_pt),
            "prototype_json_path": str(external_json),
        },
        "prototype_hashes": {
            "prototype_path": hashlib.sha256(
                external_pt.read_bytes()
            ).hexdigest(),
            "prototype_json_path": hashlib.sha256(
                external_json.read_bytes()
            ).hexdigest(),
        },
    }
    (output / "phase1_terminal_status.json").write_text(
        json.dumps({"status": "COMPLETE"}),
        encoding="utf-8",
    )
    (output / "phase1_training_completion_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(Phase1RunnerError):
        validate_phase1_row_completion(
            row=row,
            plan=plan,
            output_dir=output,
            return_code=0,
        )


def test_p0_disabled_terminal_is_immediate_protocol_failure(tmp_path) -> None:
    plan = _plan()
    plan["run_id"] = "phase1-v1"
    row = plan["rows"][0]
    output = tmp_path / row["row_key"]
    output.mkdir()
    (output / "phase1_terminal_status.json").write_text(
        json.dumps({"status": "NON_PROMOTABLE_P0_DISABLED"}),
        encoding="utf-8",
    )
    (output / "phase1_training_completion_receipt.json").write_text(
        json.dumps(
            {"terminal_status": "NON_PROMOTABLE_P0_DISABLED"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        phase1_runner.Phase1ProtocolError,
        match="NON_PROMOTABLE_P0_DISABLED",
    ):
        validate_phase1_row_completion(
            row=row,
            plan=plan,
            output_dir=output,
            return_code=8,
        )
