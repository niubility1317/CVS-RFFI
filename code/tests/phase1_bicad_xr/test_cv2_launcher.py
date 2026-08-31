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
    / "launch_phase1_pairbicad_cv2_screen24_20260831.py"
)


def _load_launcher():
    if not SCRIPT_PATH.is_file():
        pytest.fail("CV2 launcher is not implemented yet")
    spec = importlib.util.spec_from_file_location("phase1_pairbicad_cv2_screen24", SCRIPT_PATH)
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


def test_plan_rows_carry_preparsed_frozen_configuration_without_aliases() -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan()

    for row in rows:
        assert row.method_lock["candidate_id"] == row.candidate_id
        assert row.method_lock["frozen"] is True
        assert row.method_lock["dynamic_alias"] is False
        assert row.method_lock["source_only"] is True
        assert row.configuration["candidate_id"] == row.candidate_id
        assert row.configuration["phase1_method"] == "bicad_xr"


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


def test_plan_uses_configuration_resolved_before_build_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    expected = [
        (row.candidate_id, row.configuration, row.method_lock)
        for row in launcher.build_plan()
    ]

    monkeypatch.setattr(launcher, "candidate_config", lambda *_args, **_kwargs: pytest.fail("runtime config resolution"))
    monkeypatch.setattr(launcher, "method_lock_payload", lambda *_args, **_kwargs: pytest.fail("runtime lock resolution"))

    actual = [
        (row.candidate_id, row.configuration, row.method_lock)
        for row in launcher.build_plan()
    ]
    assert actual == expected


def test_build_train_command_is_source_only_and_static() -> None:
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
    assert _option_value(command, "--bicad_optimizer_updates") == str(row.optimizer_updates)
    assert _option_value(command, "--device") == "cuda:0"
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
    assert not list(tmp_path.iterdir())


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
    assert set(statuses.values()) == {"RUNNING"}
    assert max(peak.values()) <= 2
    assert all(value == 0 for value in active.values())


def test_real_mode_writes_static_plan_before_dispatch_without_child_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    observed: dict[str, object] = {}

    def fake_dispatch(rows: object, roots: object, *, run_id: str) -> dict[str, str]:
        observed["row_count"] = len(rows)
        observed["run_root"] = roots.run_root
        observed["plan_exists"] = (roots.run_root / "plan.json").is_file()
        observed["run_id"] = run_id
        return {row.row_id: "RUNNING" for row in rows}

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
    }
