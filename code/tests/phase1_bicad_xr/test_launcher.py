from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


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
