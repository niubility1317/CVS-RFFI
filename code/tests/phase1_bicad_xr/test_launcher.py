from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvsrffi.phase1_bicad_xr.metrics import (
    FORMAL_EVAL_SCENARIOS,
    BiCADXRMetricStore,
    validate_artifact_closure,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_phase1_bicad_xr_matrix_20260830.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location("phase1_bicad_xr_launcher", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_plan_is_24_source_only_rows() -> None:
    launcher = _load_launcher()

    rows = launcher.build_plan(stage="quick")

    assert len(rows) == 24
    assert {row.candidate_id for row in rows} == {
        "D0",
        "D5",
        "E1",
        "ADV3B02-BiCAD-XDC-V1",
    }
    assert {(row.fold, row.seed) for row in rows} == {
        (fold, seed)
        for fold in (1, 8)
        for seed in (392001, 392002, 392003)
    }
    assert all(row.optimizer_updates == 5000 for row in rows)
    assert all(row.source_only and row.target_access is False for row in rows)
    assert all(
        not any(
            (
                row.phase2_access,
                row.support_access,
                row.query_access,
                row.truth_access,
            )
        )
        for row in rows
    )
    assert len({row.row_id for row in rows}) == 24


def test_build_plan_supports_all_d0_through_f3_candidates_and_confirm_folds() -> None:
    launcher = _load_launcher()
    candidates = tuple(
        [f"D{i}" for i in range(7)]
        + [f"E{i}" for i in range(5)]
        + [f"F{i}" for i in range(4)]
    )

    rows = launcher.build_plan(stage="confirm", candidates=candidates)

    assert {row.candidate_id for row in rows} == set(candidates)
    assert {row.fold for row in rows} == {1, 2, 3, 4, 5}
    assert {row.seed for row in rows} == {392001, 392002, 392003}


def test_gpu_packing_is_exactly_three_rows_on_each_of_eight_gpus() -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="quick")

    packed = launcher.pack_rows(
        rows,
        gpu_ids=tuple(range(8)),
        max_jobs_per_gpu=3,
    )

    assert Counter(row.gpu_id for row in packed) == Counter({gpu: 3 for gpu in range(8)})


def test_max_jobs_per_gpu_defaults_to_three_and_rejects_more() -> None:
    launcher = _load_launcher()

    args = launcher.parse_args(["--stage", "quick", "--dry-run", "--run-id", "r1"])
    assert args.max_jobs_per_gpu == 3
    with pytest.raises(SystemExit):
        launcher.parse_args(
            [
                "--stage",
                "quick",
                "--dry-run",
                "--run-id",
                "r2",
                "--max-jobs-per-gpu",
                "4",
            ]
        )


def test_safe_preflight_reduces_slots_without_touching_unrelated_processes() -> None:
    launcher = _load_launcher()
    inventory = [
        {
            "gpu_id": 0,
            "free_memory_mb": 9000,
            "processes": [{"pid": 101, "used_memory_mb": 7000, "command": "unrelated"}],
        },
        {"gpu_id": 1, "free_memory_mb": 25000, "processes": []},
    ]

    before = json.loads(json.dumps(inventory))
    slots = launcher.safe_gpu_slots(
        inventory,
        max_jobs_per_gpu=3,
        estimated_row_memory_mb=8000,
        reserve_memory_mb=2000,
    )

    assert slots == {0: 0, 1: 2}
    assert inventory == before


def test_run_and_row_directories_are_never_overwritten(tmp_path: Path) -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="quick")[:2]

    run_root = launcher.reserve_run_layout(tmp_path, "immutable_run", rows)
    assert run_root.is_dir()
    assert all((run_root / row.row_id).is_dir() for row in rows)
    with pytest.raises(FileExistsError):
        launcher.reserve_run_layout(tmp_path, "immutable_run", rows)


def test_dry_run_prints_24_packed_rows_without_creating_or_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = _load_launcher()

    def forbidden_launch(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not create a training process")

    monkeypatch.setattr(launcher, "launch_row_process", forbidden_launch)
    exit_code = launcher.main(
        [
            "--stage",
            "quick",
            "--dry-run",
            "--run-id",
            "phase1_bicad_xr_dryrun_20260830",
            "--output-root",
            str(tmp_path),
            "--gpu-ids",
            "0,1,2,3,4,5,6,7",
        ]
    )

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert exit_code == 0
    assert len(payloads) == 24
    assert Counter(payload["gpu_id"] for payload in payloads) == Counter(
        {gpu: 3 for gpu in range(8)}
    )
    assert not (tmp_path / "phase1_bicad_xr_dryrun_20260830").exists()
    forbidden = ("target", "phase2", "support", "query", "truth")
    assert not any(
        any(token in key.lower() for token in forbidden)
        for payload in payloads
        for key in payload
    )


def test_launcher_cli_exposes_no_target_or_phase2_inputs() -> None:
    launcher = _load_launcher()
    destinations = {action.dest.lower() for action in launcher.build_parser()._actions}

    forbidden = ("target", "phase2", "support", "query", "truth")
    assert not any(any(token in name for token in forbidden) for name in destinations)


def test_train_command_is_consumed_by_real_ssdg_parser_and_records_row_runtime(
    tmp_path: Path,
) -> None:
    from SSDG import train_ssdg
    from cvsrffi.phase1_bicad_xr.config import candidate_config

    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick")[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        tmp_path,
        tmp_path / "ManySig.pkl",
    )

    command = launcher.build_train_command(row, roots, run_id="formal-run")
    parsed = train_ssdg.parse(command[3:])
    row_record = json.loads(parsed.row_key)

    assert parsed.phase1_method == "bicad_xr"
    assert parsed.candidate_id == row.candidate_id
    assert parsed.seed == row.seed
    assert parsed.wisig_train_rxs == ",".join(map(str, row.source_receivers))
    assert parsed.wisig_train_days == ",".join(map(str, row.train_days))
    assert parsed.run_id == f"formal-run-{row.row_id}"
    assert Path(parsed.output_dir) == row_root
    assert parsed.batch_size == 96
    assert parsed.use_tx_rx_balanced_sampler is True
    assert parsed.balanced_sampler_tx_per_batch == 6
    assert parsed.balanced_sampler_domain_per_batch == 4
    assert parsed.balanced_sampler_samples_per_cell == 4
    assert parsed.balanced_sampler_replacement is False
    assert row_record == {
        "fold": row.fold,
        "optimizer_updates": 5000,
        "row_id": row.row_id,
    }
    assert candidate_config(parsed.candidate_id).optimizer_updates == 5000


def _write_strict_worker_artifacts(row_root: Path, row: object) -> None:
    checkpoint = row_root / "bicad_xr_final.pth"
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
                "checkpoint_path": checkpoint.name,
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
        json.dumps(BiCADXRMetricStore().snapshot()), encoding="utf-8"
    )
    evaluations = row_root / "evaluations"
    evaluations.mkdir()
    for scenario in FORMAL_EVAL_SCENARIOS:
        (evaluations / f"{scenario}.log").write_text("complete\n", encoding="utf-8")
        (evaluations / f"{scenario}.json").write_text(
            json.dumps(
                {
                    "scenario": scenario,
                    "checkpoint": checkpoint.name,
                    "checkpoint_load_strict": True,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "shape_mismatches": [],
                    "accuracy": 1.0,
                    "floor_accuracy": 1.0,
                    "per_class_accuracy": {"0": 1.0},
                }
            ),
            encoding="utf-8",
        )


def test_worker_exit_zero_without_final_evaluations_is_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick")[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    (row_root / "bicad_xr_final.pth").write_bytes(b"checkpoint")
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2], Path("python"), tmp_path, tmp_path / "ManySig.pkl"
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        launcher,
        "evaluate_final_checkpoint",
        lambda *args, **kwargs: {
            "complete": False,
            "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
            "missing": list(FORMAL_EVAL_SCENARIOS),
        },
        raising=False,
    )

    status = launcher.launch_row_process(row, roots, run_id="formal-run")

    assert status == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert not (row_root / "ARTIFACTS_COMPLETE.json").exists()
    failure = json.loads((row_root / "TECHNICAL_FAILURE.json").read_text(encoding="utf-8"))
    assert failure["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"


def test_worker_marks_complete_only_after_strict_four_scene_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick")[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    (row_root / "bicad_xr_final.pth").write_bytes(b"checkpoint")
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2], Path("python"), tmp_path, tmp_path / "ManySig.pkl"
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    def strict_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        _write_strict_worker_artifacts(row_root, row)
        return validate_artifact_closure(row_root)

    monkeypatch.setattr(
        launcher, "evaluate_final_checkpoint", strict_evaluate, raising=False
    )

    status = launcher.launch_row_process(row, roots, run_id="formal-run")

    assert status == "ARTIFACTS_COMPLETE"
    marker = json.loads(
        (row_root / "ARTIFACTS_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert marker["complete"] is True
    assert set(marker["evaluations"]) == set(FORMAL_EVAL_SCENARIOS)
