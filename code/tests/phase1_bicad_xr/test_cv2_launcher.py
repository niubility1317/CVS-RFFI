from __future__ import annotations

import importlib.util
import json
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvsrffi.phase1_bicad_xr.metrics import (
    BiCADXRMetricStore,
    FORMAL_EVAL_SCENARIOS,
)


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
    assert _option_value(command, "--bicad_loro_receiver") == str(row.heldout_receiver)
    assert _option_value(command, "--bicad_loro_eval_interval_updates") == "500"
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


def _write_dynamic_cv2_selection(
    row_root: Path,
    *,
    stop_update: int = 2_000,
    status: str = "SCIENTIFICALLY_CONVERGED",
) -> None:
    plan = {
        "unlabeled_physical_count": 16_000,
        "source_receiver_count": 4,
        "unlabeled_per_four_updates": 120,
        "u_cycle_updates": 534,
        "eval_interval_updates": 500,
        "min_activation_updates": 1_600,
        "safety_updates": 6_400,
    }
    scientific = status == "SCIENTIFICALLY_CONVERGED"
    selection = {
        "planned_updates": 6_400,
        "stop_update": stop_update,
        "interval": 500,
        "stopped_early": True,
        "cv2_coverage_plan": plan,
        "cv2_terminal": {
            "status": status,
            "scientifically_converged": scientific,
            "artifacts_allowed": True,
        },
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    (row_root / "source_loro_selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )
    curve = {
        "update": stop_update,
        "cv2_decision": {"status": status},
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    (row_root / "source_loro_curve.jsonl").write_text(
        json.dumps(curve) + "\n",
        encoding="utf-8",
    )


def test_dynamic_cv2_runtime_expectation_binds_actual_terminal_update(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    row = next(
        item
        for item in launcher.build_plan()
        if item.configuration["coverage_convergence"]
    )
    _write_dynamic_cv2_selection(tmp_path)

    expectation = launcher._cv2_runtime_expectation(tmp_path, row)

    assert expectation["optimizer_updates"] == 2_000
    assert expectation["planned_optimizer_updates"] == 6_400


def test_dynamic_cv2_runtime_expectation_rejects_inconsistent_coverage_plan(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    row = next(
        item
        for item in launcher.build_plan()
        if item.configuration["coverage_convergence"]
    )
    _write_dynamic_cv2_selection(tmp_path)
    selection_path = tmp_path / "source_loro_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["cv2_coverage_plan"]["safety_updates"] = 6_000
    selection["planned_updates"] = 6_000
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent"):
        launcher._cv2_runtime_expectation(tmp_path, row)


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
    }


def _write_cv2_runtime_artifacts(row_root: Path, row: object, checkpoint_name: str) -> None:
    runtime = {
        "phase1_method": "bicad_xr",
        "candidate_id": row.candidate_id,
        "fold": row.fold,
        "seed": row.seed,
        "optimizer_update": row.optimizer_updates,
        "total_updates": row.optimizer_updates,
        "source_receivers": list(row.source_receivers),
        "train_days": list(row.train_days),
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    (row_root / "checkpoint_runtime.json").write_text(
        json.dumps(
            {
                "checkpoint_path": checkpoint_name,
                "runtime": runtime,
                "reconstruction": {
                    "missing": [],
                    "unexpected": [],
                    "shape_mismatch": [],
                },
                "strict_reconstruction": True,
                "trainer_runtime_strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            }
        ),
        encoding="utf-8",
    )
    (row_root / "diagnostics.json").write_text(
        json.dumps(BiCADXRMetricStore().snapshot()),
        encoding="utf-8",
    )


def _install_fake_strict_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    launcher: object,
    row: object,
    row_root: Path,
    calls: list[str],
    *,
    bad_scene: str | None = None,
    bad_field: str | None = None,
) -> None:
    class FakeContext:
        def build_model(self, _payload: object) -> object:
            return object()

        def restore_trainer_runtime(self, _model: object, _payload: object) -> None:
            return None

        def evaluate(self, _model: object, scenario: str) -> dict[str, object]:
            calls.append(scenario)
            result: dict[str, object] = {
                "accuracy": 0.75,
                "floor_accuracy": 0.50,
                "per_class_accuracy": {"0": 0.75, "1": 0.50},
                "log": f"{scenario} complete\n",
            }
            if scenario == bad_scene and bad_field == "finite":
                result["accuracy"] = float("nan")
            if scenario == bad_scene and bad_field == "source_only":
                result["target_access"] = True
            return result

    context = FakeContext()
    monkeypatch.setattr(
        launcher,
        "_build_final_evaluation_context",
        lambda *_args, **_kwargs: context,
        raising=False,
    )

    def fake_strict_entrypoint(
        checkpoint: Path,
        *,
        expected_runtime: object,
        output_dir: Path,
        model_builder: object,
        trainer_runtime_restorer: object,
        evaluator: object,
    ) -> dict[str, object]:
        assert checkpoint.name == "bicad_xr_final.pth"
        assert expected_runtime["candidate_id"] == row.candidate_id
        assert callable(model_builder)
        assert callable(trainer_runtime_restorer)
        assert callable(evaluator)
        _write_cv2_runtime_artifacts(output_dir, row, checkpoint.name)
        for scenario in FORMAL_EVAL_SCENARIOS:
            metrics = evaluator(object(), scenario)
            if not isinstance(metrics, dict):
                raise TypeError("fake evaluator callback must return a mapping")
            log_path = output_dir / "evaluations" / f"{scenario}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(str(metrics.get("log", "complete\n")), encoding="utf-8")
        return {"complete": True, "status": "ARTIFACTS_COMPLETE"}

    monkeypatch.setattr(
        launcher,
        "evaluate_final_checkpoint",
        fake_strict_entrypoint,
        raising=False,
    )


def test_worker_closes_four_source_only_evaluations_before_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    (row_root / "bicad_xr_final.pth").write_bytes(b"checkpoint")
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2], Path("python"), tmp_path, tmp_path / "ManySig.pkl"
    )
    calls: list[str] = []
    _install_fake_strict_evaluator(monkeypatch, launcher, row, row_root, calls)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    status = launcher.run_training_worker(row, roots, run_id="cv2-run")

    assert status == "ARTIFACTS_COMPLETE"
    assert calls == list(FORMAL_EVAL_SCENARIOS)
    assert (row_root / "ARTIFACTS_COMPLETE.json").is_file()
    worker_status = json.loads(
        (row_root / "worker_status.json").read_text(encoding="utf-8")
    )
    assert worker_status["status"] == "ARTIFACTS_COMPLETE"
    for scenario in FORMAL_EVAL_SCENARIOS:
        path = row_root / "evaluations" / f"{scenario}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["checkpoint"] == "bicad_xr_final.pth"
        assert payload["source_only"] is True
        assert all(payload[name] is False for name in (
            "target_access",
            "phase2_access",
            "support_access",
            "query_access",
            "truth_access",
        ))
        assert payload["accuracy"] == pytest.approx(0.75)
        assert not list(path.parent.glob("*.tmp-*"))


@pytest.mark.parametrize("bad_field", ["finite", "source_only"])
def test_worker_preserves_partial_artifacts_and_stops_on_eval_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_field: str,
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan()[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    (row_root / "bicad_xr_final.pth").write_bytes(b"checkpoint")
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2], Path("python"), tmp_path, tmp_path / "ManySig.pkl"
    )
    calls: list[str] = []
    _install_fake_strict_evaluator(
        monkeypatch,
        launcher,
        row,
        row_root,
        calls,
        bad_scene="leo_clear_weak",
        bad_field=bad_field,
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    status = launcher.run_training_worker(row, roots, run_id="cv2-run")

    assert status == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert not (row_root / "ARTIFACTS_COMPLETE.json").exists()
    worker_status = json.loads(
        (row_root / "worker_status.json").read_text(encoding="utf-8")
    )
    assert worker_status["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert (row_root / "TECHNICAL_FAILURE.json").is_file()
    assert (row_root / "bicad_xr_final.pth").is_file()
    assert calls[:2] == ["clean", "leo_clear_weak"]
