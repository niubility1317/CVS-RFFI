from __future__ import annotations

import importlib.util
import json
import threading
from collections import Counter
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_phase1_pairbicad_cv2_screen24_20260901.py"
)


def _load_launcher():
    if not SCRIPT_PATH.is_file():
        pytest.fail("CV2 E200 repair launcher is not implemented yet")
    spec = importlib.util.spec_from_file_location("phase1_pairbicad_cv2_e200", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_build_plan_is_exactly_the_frozen_24_row_matrix() -> None:
    launcher = _load_launcher()

    rows = launcher.build_plan()

    expected_candidates = {
        *(f"CV2-B{i}" for i in range(4)),
        *(f"CV2-D{i}" for i in range(4)),
        *(f"CV2-T{i}" for i in range(4)),
    }
    assert len(rows) == 24
    assert {row.candidate_id for row in rows} == expected_candidates
    assert {(row.fold, row.seed) for row in rows} == {
        (fold, 392002) for fold in (1, 8)
    }
    assert len({row.row_id for row in rows}) == 24
    assert Counter(row.gpu_id for row in rows) == Counter({gpu: 3 for gpu in range(8)})
    assert {row.fold: row.source_receivers for row in rows} == {
        1: (3, 4, 6, 8),
        8: (1, 3, 4, 6),
    }
    assert all(row.seed == 392002 for row in rows)
    assert all(row.train_days == (1, 2, 3) for row in rows)
    assert all(row.source_only for row in rows)
    assert all(row.epochs == 200 for row in rows)
    assert all(row.termination_mode == "epochs" for row in rows)


def test_plan_rows_carry_preparsed_frozen_configuration_and_runtime_contracts() -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan()

    for row in rows:
        assert row.method_lock["candidate_id"] == row.candidate_id
        assert row.method_lock["frozen"] is True
        assert row.method_lock["dynamic_alias"] is False
        assert row.method_lock["source_only"] is True
        assert row.configuration["candidate_id"] == row.candidate_id
        assert row.configuration["phase1_method"] == "bicad_xr"
        assert row.configuration["epochs"] == 200
        assert row.configuration["termination_mode"] == "epochs"
        assert "optimizer_updates" not in row.configuration
        assert row.method_lock["runtime_contracts"]["termination"] == {
            "mode": "epochs",
            "epochs": 200,
        }

    baseline_rows = [
        row
        for row in rows
        if row.candidate_id in {"CV2-B3", "CV2-D0", "CV2-T0"}
    ]
    assert len(baseline_rows) == 6
    assert all(
        row.method_lock["configuration_role"] == "static_branch_baseline"
        for row in baseline_rows
    )


def test_plan_declares_final_checkpoint_and_all_four_evaluation_artifacts() -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan()

    expected = {
        "final_checkpoint": "bicad_xr_final.pth",
        "clean": "evaluations/clean.json",
        "leo_clear_weak": "evaluations/leo_clear_weak.json",
        "leo_low_elev_weak": "evaluations/leo_low_elev_weak.json",
        "leo_rain_weak": "evaluations/leo_rain_weak.json",
    }
    assert all(row.expected_artifacts == expected for row in rows)


def test_plan_uses_configuration_resolved_before_build_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    expected = [
        (row.candidate_id, row.configuration, row.method_lock)
        for row in launcher.build_plan()
    ]

    monkeypatch.setattr(
        launcher,
        "candidate_config",
        lambda *_args, **_kwargs: pytest.fail("runtime config resolution"),
    )
    monkeypatch.setattr(
        launcher,
        "method_lock_payload",
        lambda *_args, **_kwargs: pytest.fail("runtime lock resolution"),
    )

    actual = [
        (row.candidate_id, row.configuration, row.method_lock)
        for row in launcher.build_plan()
    ]
    assert actual == expected


def test_build_train_command_is_source_only_and_epoch_terminated() -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    roots = launcher.LauncherRoots(
        Path("repo"),
        Path("python"),
        Path("runs") / "cv2-screen24",
        Path("dataset") / "ManySig.pkl",
    )

    command = launcher.build_train_command(row, roots, run_id="cv2-screen24")
    options = {token for token in command if token.startswith("--")}

    assert _option_value(command, "--phase1_method") == "bicad_xr"
    assert _option_value(command, "--candidate_id") == row.candidate_id
    assert _option_value(command, "--seed") == "392002"
    assert _option_value(command, "--wisig_train_days") == "1,2,3"
    assert _option_value(command, "--wisig_train_rxs") == "3,4,6,8"
    assert _option_value(command, "--wisig_test_days") == ""
    assert _option_value(command, "--wisig_test_rxs") == ""
    assert _option_value(command, "--phase1_source_only_eval") == "true"
    assert _option_value(command, "--epochs") == "200"
    assert "--bicad_optimizer_updates" not in options
    assert "--bicad_loro_eval_interval_updates" not in options
    assert _option_value(command, "--bicad_loro_receiver") == str(row.heldout_receiver)
    assert _option_value(command, "--device") == "cuda:0"
    row_key = json.loads(_option_value(command, "--row_key"))
    assert row_key == {
        "candidate_id": row.candidate_id,
        "epochs": 200,
        "fold": row.fold,
        "gpu_id": row.gpu_id,
        "row_id": row.row_id,
        "seed": row.seed,
        "termination_mode": "epochs",
    }
    assert "--candidate_alias" not in options
    assert not any(
        any(token in option.lower() for token in ("target", "phase2", "support", "query", "truth"))
        for option in options
    )


def test_worker_command_is_detached_and_refers_to_one_static_row() -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[7]
    roots = launcher.LauncherRoots(
        Path("repo"),
        Path("python"),
        Path("runs") / "cv2-screen24",
        Path("dataset") / "ManySig.pkl",
    )

    command = launcher.build_worker_command(row, roots, run_id="cv2-screen24")

    assert command[0:2] == ["python", "-u"]
    assert "--worker-row-id" in command
    assert _option_value(command, "--worker-row-id") == row.row_id
    assert _option_value(command, "--run-id") == "cv2-screen24"
    assert _option_value(command, "--run-root") == str(roots.run_root)


def test_dry_run_emits_one_json_plan_without_creating_or_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = _load_launcher()

    monkeypatch.setattr(
        launcher,
        "dispatch_detached_workers",
        lambda *_args, **_kwargs: pytest.fail("dry-run launched dispatcher"),
    )
    exit_code = launcher.main(
        [
            "--dry-run",
            "--run-id",
            "cv2-screen24-dry-run",
            "--output-root",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["row_count"] == 24
    assert payload["seed"] == 392002
    assert payload["max_active_per_gpu"] == 2
    assert len(payload["rows"]) == 24
    assert all(row["epochs"] == 200 for row in payload["rows"])
    assert all(row["termination_mode"] == "epochs" for row in payload["rows"])
    assert all("optimizer_updates" not in row for row in payload["rows"])
    assert not list(tmp_path.iterdir())


def test_plan_queued_rows_uses_rows_assigned_to_each_gpu_not_total_capacity() -> None:
    launcher = _load_launcher()
    rows = [row._replace(gpu_id=0) for row in launcher.build_plan()[:3]]

    payload = launcher.build_plan_payload(
        rows,
        gpu_capacities={gpu: 2 for gpu in range(8)},
    )

    assert payload["queued_rows"] == 1


def test_parser_has_no_matrix_or_forbidden_data_role_overrides() -> None:
    launcher = _load_launcher()
    destinations = {action.dest.lower() for action in launcher.build_parser()._actions}

    assert "candidates" not in destinations
    assert "folds" not in destinations
    assert "seed" not in destinations
    assert not any(
        any(token in destination for token in ("target", "phase2", "support", "query", "truth"))
        for destination in destinations
    )


def test_run_layout_and_plan_are_non_overwriting(tmp_path: Path) -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan()[:2]

    run_root = launcher.reserve_run_layout(tmp_path, "cv2-screen24", rows)
    assert run_root.is_dir()
    assert all((run_root / row.row_id).is_dir() for row in rows)
    with pytest.raises(FileExistsError):
        launcher.reserve_run_layout(tmp_path, "cv2-screen24", rows)

    launcher.write_plan_json(run_root, rows, run_id="cv2-screen24")
    with pytest.raises(FileExistsError):
        launcher.write_plan_json(run_root, rows, run_id="cv2-screen24")


def test_dispatcher_queues_third_row_per_gpu_and_never_exceeds_two_active(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan()
    roots = launcher.LauncherRoots(
        Path("repo"),
        Path("python"),
        tmp_path / "cv2-screen24",
        Path("dataset") / "ManySig.pkl",
    )
    lock = threading.Lock()
    active: Counter[int] = Counter()
    peak: Counter[int] = Counter()

    class FakeProcess:
        def __init__(self, gpu_id: int) -> None:
            self.gpu_id = gpu_id
            self.poll_count = 0

        def poll(self) -> int | None:
            with lock:
                self.poll_count += 1
                if self.poll_count < 2:
                    return None
                active[self.gpu_id] -= 1
                return 0

    def worker_launcher(row: object, _roots: object, *, run_id: str) -> FakeProcess:
        assert run_id == "cv2-screen24"
        gpu_id = int(row.gpu_id)
        with lock:
            active[gpu_id] += 1
            peak[gpu_id] = max(peak[gpu_id], active[gpu_id])
        return FakeProcess(gpu_id)

    statuses = launcher.dispatch_detached_workers(
        rows,
        roots,
        run_id="cv2-screen24",
        worker_launcher=worker_launcher,
        poll_interval=0,
    )

    assert set(statuses) == {row.row_id for row in rows}
    assert set(statuses.values()) == {"ARTIFACTS_COMPLETE"}
    assert max(peak.values()) <= 2
    assert all(value == 0 for value in active.values())


def test_dispatcher_honors_preflight_reduced_gpu_capacity(tmp_path: Path) -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan()
    roots = launcher.LauncherRoots(
        Path("repo"),
        Path("python"),
        tmp_path / "cv2-screen24",
        Path("dataset") / "ManySig.pkl",
    )
    active: Counter[int] = Counter()
    peak: Counter[int] = Counter()

    class FakeProcess:
        def __init__(self, gpu_id: int) -> None:
            self.gpu_id = gpu_id
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            if self.poll_count < 2:
                return None
            active[self.gpu_id] -= 1
            return 0

    def worker_launcher(row: object, _roots: object, *, run_id: str) -> FakeProcess:
        active[row.gpu_id] += 1
        peak[row.gpu_id] = max(peak[row.gpu_id], active[row.gpu_id])
        return FakeProcess(row.gpu_id)

    capacities = {gpu_id: 2 for gpu_id in launcher.GPU_IDS}
    capacities[0] = 1
    statuses = launcher.dispatch_detached_workers(
        rows,
        roots,
        run_id="cv2-screen24",
        worker_launcher=worker_launcher,
        poll_interval=0,
        gpu_capacities=capacities,
    )

    assert set(statuses.values()) == {"ARTIFACTS_COMPLETE"}
    assert peak[0] == 1
    assert all(peak[gpu_id] <= capacities[gpu_id] for gpu_id in launcher.GPU_IDS)


def test_gpu_capacity_parser_requires_all_gpus_and_caps_each_at_two() -> None:
    launcher = _load_launcher()
    parsed = launcher._parse_gpu_capacities("0:1,1:2,2:2,3:2,4:2,5:2,6:2,7:2")

    assert parsed[0] == 1
    assert sum(parsed.values()) == 15
    with pytest.raises(ValueError):
        launcher._parse_gpu_capacities("0:1,1:2")
    with pytest.raises(ValueError):
        launcher._parse_gpu_capacities("0:3,1:2,2:2,3:2,4:2,5:2,6:2,7:2")


def test_real_mode_writes_static_plan_before_dispatch_without_child_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    observed: dict[str, object] = {}

    def fake_dispatch(
        rows: object,
        roots: object,
        *,
        run_id: str,
        gpu_capacities: object,
    ) -> dict[str, str]:
        observed["row_count"] = len(rows)
        observed["run_root"] = roots.run_root
        observed["plan_exists"] = (roots.run_root / "plan.json").is_file()
        observed["run_id"] = run_id
        observed["gpu_capacities"] = gpu_capacities
        return {row.row_id: "ARTIFACTS_COMPLETE" for row in rows}

    monkeypatch.setattr(launcher, "dispatch_detached_workers", fake_dispatch)
    exit_code = launcher.main(
        [
            "--run-id",
            "cv2-screen24-real-mode-test",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert observed == {
        "row_count": 24,
        "run_root": tmp_path / "cv2-screen24-real-mode-test",
        "plan_exists": True,
        "run_id": "cv2-screen24-real-mode-test",
        "gpu_capacities": {gpu_id: 2 for gpu_id in range(8)},
    }


def test_worker_runs_strict_final_evaluation_before_marking_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    run_root = tmp_path / "cv2-screen24"
    row_root = run_root / row.row_id
    row_root.mkdir(parents=True)
    checkpoint = row_root / "bicad_xr_final.pth"
    roots = launcher.LauncherRoots(
        Path("repo"),
        Path("python"),
        run_root,
        Path("dataset") / "ManySig.pkl",
    )

    class Completed:
        returncode = 0

    def fake_run(*_args: object, **_kwargs: object) -> Completed:
        checkpoint.write_bytes(b"checkpoint")
        return Completed()

    calls: list[Path] = []

    def fake_evaluate(checkpoint_path: Path, **_kwargs: object) -> dict[str, object]:
        calls.append(Path(checkpoint_path))
        return {
            "complete": True,
            "status": "ARTIFACTS_COMPLETE",
            "missing": [],
            "reconstruction": {"missing": [], "unexpected": [], "shape_mismatch": []},
            "evaluations": {scenario: {} for scenario in launcher.FORMAL_SCENARIOS},
        }

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(launcher, "_cv2_runtime_expectation", lambda *_args: {})
    class Context:
        build_model = object()
        restore_trainer_runtime = object()

        def evaluate(self, *_args: object) -> dict[str, object]:
            return {}

    monkeypatch.setattr(launcher, "_build_final_evaluation_context", lambda *_args: Context())
    monkeypatch.setattr(launcher, "evaluate_final_checkpoint", fake_evaluate, raising=False)
    monkeypatch.setattr(
        launcher,
        "_validate_worker_artifacts",
        lambda _row, _root, _checkpoint, evaluation, _expectation: evaluation,
        raising=False,
    )

    status = launcher.run_training_worker(row, roots, run_id="cv2-screen24")

    assert status == "ARTIFACTS_COMPLETE"
    assert calls == [checkpoint]
    assert (row_root / "ARTIFACTS_COMPLETE.json").is_file()
    assert not (row_root / "TECHNICAL_FAILURE.json").exists()


def test_e200_runtime_expectation_binds_200_epochs_and_terminal_update(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    selection = {
        "planned_updates": 10400,
        "stop_update": 10400,
        "stopped_early": False,
        "final_epoch": 200,
        "source_only": True,
        **{name: False for name in launcher.SOURCE_ONLY_ACCESS_FLAGS},
    }
    (tmp_path / "source_loro_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    (tmp_path / "metrics_epoch.jsonl").write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in range(1, 201)),
        encoding="utf-8",
    )

    expectation = launcher._cv2_runtime_expectation(tmp_path, row)

    assert expectation == {
        "candidate_id": row.candidate_id,
        "fold": row.fold,
        "seed": 392002,
        "optimizer_updates": 10400,
        "planned_optimizer_updates": 10400,
        "source_receivers": row.source_receivers,
        "train_days": (1, 2, 3),
    }


def test_final_evaluation_context_restores_with_frozen_candidate_budget() -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    repo_root = Path(__file__).resolve().parents[3]
    roots = launcher.LauncherRoots(
        repo_root,
        Path("python"),
        Path("runs") / "cv2-screen24",
        Path("dataset") / "ManySig.pkl",
    )

    context = launcher._build_final_evaluation_context(
        row,
        roots,
        launcher.build_train_command(row, roots, run_id="cv2-screen24"),
    )

    assert context.row.candidate_id == row.candidate_id
    assert context.row.optimizer_updates == 5000
    assert context.row.source_receivers == row.source_receivers
