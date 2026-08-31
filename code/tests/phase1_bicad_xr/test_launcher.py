from __future__ import annotations

import importlib.util
import json
import threading
import time
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


def test_pairbicad_plan_is_exactly_the_frozen_30_row_source_only_matrix() -> None:
    launcher = _load_launcher()

    rows = launcher.build_plan(stage="pairbicad")

    assert len(rows) == 30
    assert {(row.candidate_id, row.fold, row.seed) for row in rows} == {
        (candidate, fold, seed)
        for candidate in ("P0", "P1", "P2", "P3", "P4")
        for fold in (1, 8)
        for seed in (392001, 392002, 392003)
    }
    assert len({row.row_id for row in rows}) == 30
    assert all(row.optimizer_updates == 4000 for row in rows)
    assert all(row.train_days == (1, 2, 3) for row in rows)
    assert all(row.source_only for row in rows)
    assert all(
        not any(
            (
                row.target_access,
                row.phase2_access,
                row.support_access,
                row.query_access,
                row.truth_access,
            )
        )
        for row in rows
    )
    assert {
        row.fold: row.source_receivers
        for row in rows
    } == {
        1: (3, 4, 6, 8),
        8: (1, 3, 4, 6),
    }


def test_pairbicad_queue_assigns_all_rows_without_reinterpreting_legacy_pack_capacity() -> None:
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="pairbicad")

    queued = launcher.queue_rows(rows, gpu_ids=tuple(range(8)))

    assert len(queued) == 30
    assert Counter(row.gpu_id for row in queued) == Counter(
        {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 3, 7: 3}
    )
    with pytest.raises(ValueError, match="safe capacity"):
        launcher.pack_rows(rows, gpu_ids=tuple(range(8)), max_jobs_per_gpu=2)


def test_pairbicad_run_plan_queues_rows_and_never_exceeds_two_active_per_gpu() -> None:
    launcher = _load_launcher()
    rows = launcher.queue_rows(
        launcher.build_plan(stage="pairbicad"), gpu_ids=tuple(range(8))
    )
    lock = threading.Lock()
    active: Counter[int] = Counter()
    peak: Counter[int] = Counter()

    def runner(row: object) -> str:
        gpu_id = int(row.gpu_id)
        with lock:
            active[gpu_id] += 1
            peak[gpu_id] = max(peak[gpu_id], active[gpu_id])
        time.sleep(0.01)
        with lock:
            active[gpu_id] -= 1
        return "ok"

    statuses = launcher.run_plan(rows, runner, max_active_per_gpu=2)

    assert set(statuses) == {row.row_id for row in rows}
    assert set(statuses.values()) == {"ok"}
    assert max(peak.values()) <= 2


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
    assert parsed.sample_rate_hz == pytest.approx(25e6)
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


def test_pairbicad_train_command_uses_physical_batch_48_and_u4000() -> None:
    from SSDG import train_ssdg

    launcher = _load_launcher()
    row = launcher.build_plan(stage="pairbicad")[0]
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        Path("/tmp/pairbicad-run"),
        Path("/tmp/ManySig.pkl"),
    )

    command = launcher.build_train_command(row, roots, run_id="pairbicad-run")
    parsed = train_ssdg.parse(command[3:])
    row_record = json.loads(parsed.row_key)

    assert parsed.candidate_id == "P0"
    assert parsed.batch_size == 48
    assert row_record["optimizer_updates"] == 4000
    assert parsed.wisig_train_days == "1,2,3"
    assert parsed.wisig_train_rxs == "3,4,6,8"


def test_pairbicad_reconstruction_preserves_candidate_protocol() -> None:
    launcher = _load_launcher()

    config = launcher.reconstruction_config("P4")

    assert config.batch_size == 48
    assert config.optimizer_updates == 4000
    assert config.concat_sat_start_epoch == 1
    assert config.satellite_supervision_mode == "ce_only_plus_pair_selfsup"
    assert config.strict_pair_concat is True
    assert config.lambda_sat_cls_start == pytest.approx(0.5)
    assert config.lambda_sat_cls_end == pytest.approx(1.0)


def test_pairbicad_formal_restore_reconstructs_training_domain_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cvsrffi.phase1_bicad_xr.trainer as trainer_module

    launcher = _load_launcher()
    row = next(
        row
        for row in launcher.build_plan(stage="pairbicad")
        if row.candidate_id == "P4" and row.fold == 1 and row.seed == 392002
    )
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        Path("/tmp/pairbicad-run"),
        Path("/tmp/ManySig.pkl"),
    )
    context = launcher._FormalEvaluationContext(row, roots, ())
    context.ssdg = SimpleNamespace(_BiCADXRConcatForward=lambda model: model)
    context.device = "cpu"
    captured: dict[str, object] = {}

    class CapturingTrainer:
        def __init__(self, model: object, config: object, **dimensions: object) -> None:
            captured["model"] = model
            captured["config"] = config
            captured["dimensions"] = dimensions

        def to(self, device: object) -> "CapturingTrainer":
            captured["device"] = device
            return self

        def load_checkpoint_runtime(
            self, runtime: object, *, strict: bool
        ) -> None:
            captured["runtime"] = runtime
            captured["strict"] = strict

    monkeypatch.setattr(trainer_module, "BiCADXRTrainer", CapturingTrainer)
    runtime = {
        "num_receivers": 99,
        "num_days": 99,
        "num_channels": 99,
    }

    context.restore_trainer_runtime(object(), {"bicad_xr_runtime": runtime})

    assert captured["dimensions"] == {
        "num_receivers": len(row.source_receivers),
        "num_days": len(row.train_days),
        "num_channels": 2,
    }
    assert captured["runtime"] is runtime
    assert captured["strict"] is True


@pytest.mark.parametrize(
    ("stage", "candidate", "updates"),
    [
        ("pairbicad_convergence", "P2", 9000),
        ("pairbicad_final", "P4", 6500),
    ],
)
def test_formal_restore_uses_planned_row_budget_for_new_pairbicad_stages(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    candidate: str,
    updates: int,
) -> None:
    import cvsrffi.phase1_bicad_xr.trainer as trainer_module

    launcher = _load_launcher()
    rows = launcher.build_plan(
        stage=stage,
        candidates=(candidate, "P3") if stage == "pairbicad_convergence" else (candidate,),
        folds=(1, 8) if stage == "pairbicad_convergence" else (1, 2, 3, 4, 5),
        seeds=(392001, 392002, 392003),
        optimizer_updates=updates,
    )
    row = rows[0]
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        Path("/tmp/pairbicad-run"),
        Path("/tmp/ManySig.pkl"),
    )
    context = launcher._FormalEvaluationContext(row, roots, ())
    context.ssdg = SimpleNamespace(_BiCADXRConcatForward=lambda model: model)
    context.device = "cpu"
    captured: dict[str, object] = {}

    class CapturingTrainer:
        def __init__(self, model: object, config: object, **dimensions: object) -> None:
            captured["config"] = config
            captured["dimensions"] = dimensions

        def to(self, device: object) -> "CapturingTrainer":
            return self

        def load_checkpoint_runtime(self, runtime: object, *, strict: bool) -> None:
            captured["runtime"] = runtime
            captured["strict"] = strict

    monkeypatch.setattr(trainer_module, "BiCADXRTrainer", CapturingTrainer)
    context.restore_trainer_runtime(
        object(),
        {"bicad_xr_runtime": {"candidate_config": {"optimizer_updates": updates}}},
    )

    assert captured["config"].optimizer_updates == updates
    assert captured["strict"] is True


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


def test_pairbicad_dry_run_prints_all_30_rows_without_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = _load_launcher()

    def forbidden_launch(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not create a training process")

    monkeypatch.setattr(launcher, "launch_row_process", forbidden_launch)
    exit_code = launcher.main(
        [
            "--stage",
            "pairbicad",
            "--dry-run",
            "--run-id",
            "phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1",
            "--output-root",
            str(tmp_path),
            "--gpu-ids",
            "0,1,2,3,4,5,6,7",
            "--max-jobs-per-gpu",
            "2",
        ]
    )

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert exit_code == 0
    assert len(payloads) == 30
    assert Counter(payload["gpu_id"] for payload in payloads) == Counter(
        {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 3, 7: 3}
    )
    assert all(payload["optimizer_updates"] == 4000 for payload in payloads)
    assert all(payload["source_receivers"] for payload in payloads)
    assert all(payload["train_days"] == [1, 2, 3] for payload in payloads)
    assert not (tmp_path / "phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1").exists()
    forbidden = ("target", "phase2", "support", "query", "truth")
    assert not any(
        any(token in key.lower() for token in forbidden)
        for payload in payloads
        for key in payload
    )


def test_pairbicad_convergence_plan_is_exactly_two_candidates_by_two_folds_and_three_seeds() -> None:
    launcher = _load_launcher()

    rows = launcher.build_plan(
        stage="pairbicad_convergence",
        candidates=("P2", "P4"),
        folds=(1, 8),
        seeds=(392001, 392002, 392003),
        optimizer_updates=9000,
    )

    assert len(rows) == 12
    assert {row.candidate_id for row in rows} == {"P2", "P4"}
    assert {row.fold for row in rows} == {1, 8}
    assert {row.seed for row in rows} == {392001, 392002, 392003}
    assert all(row.optimizer_updates == 9000 for row in rows)
    assert all(row.stage == "pairbicad_convergence" for row in rows)
    assert len({row.row_id for row in rows}) == 12


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidates": ("P2",)}, "exactly 2"),
        ({"candidates": ("P2", "D0")}, "PairBiCAD"),
        ({"candidates": ("P2", "P2")}, "unique"),
        ({"folds": (1, 2)}, "folds"),
        ({"seeds": (392001, 392002)}, "seeds"),
        ({"optimizer_updates": 8500}, "optimizer_updates"),
    ],
)
def test_pairbicad_convergence_rejects_any_frozen_matrix_mismatch(
    kwargs: dict[str, object], message: str
) -> None:
    launcher = _load_launcher()
    base = {
        "stage": "pairbicad_convergence",
        "candidates": ("P2", "P4"),
        "folds": (1, 8),
        "seeds": (392001, 392002, 392003),
        "optimizer_updates": 9000,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        launcher.build_plan(**base)


def test_pairbicad_final_plan_is_exactly_one_candidate_by_five_folds_and_three_seeds() -> None:
    launcher = _load_launcher()

    rows = launcher.build_plan(
        stage="pairbicad_final",
        candidates=("P4",),
        folds=(1, 2, 3, 4, 5),
        seeds=(392001, 392002, 392003),
        optimizer_updates=6500,
    )

    assert len(rows) == 15
    assert {row.candidate_id for row in rows} == {"P4"}
    assert {row.fold for row in rows} == {1, 2, 3, 4, 5}
    assert {row.seed for row in rows} == {392001, 392002, 392003}
    assert all(row.optimizer_updates == 6500 for row in rows)
    assert all(row.stage == "pairbicad_final" for row in rows)


@pytest.mark.parametrize("updates", [3999, 4250, 9500])
def test_pairbicad_final_rejects_non_frozen_budget(updates: int) -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="4000.*9000|500"):
        launcher.build_plan(
            stage="pairbicad_final",
            candidates=("P4",),
            folds=(1, 2, 3, 4, 5),
            seeds=(392001, 392002, 392003),
            optimizer_updates=updates,
        )


def test_pairbicad_stage_parser_defaults_new_stages_to_two_jobs_per_gpu() -> None:
    launcher = _load_launcher()

    convergence = launcher.parse_args(["--stage", "pairbicad_convergence", "--dry-run"])
    final = launcher.parse_args(["--stage", "pairbicad_final", "--dry-run"])

    assert convergence.max_jobs_per_gpu == 2
    assert convergence.optimizer_updates == 9000
    assert final.max_jobs_per_gpu == 2
    assert final.optimizer_updates is None


def test_pairbicad_loro_receiver_mapping_is_source_only_and_disjoint() -> None:
    launcher = _load_launcher()
    expected = {1: 1, 2: 3, 3: 4, 4: 6, 5: 8, 8: 8}

    rows = launcher.build_plan(
        stage="pairbicad_convergence",
        candidates=("P2", "P4"),
        folds=(1, 8),
        seeds=(392001, 392002, 392003),
        optimizer_updates=9000,
    )
    for row in rows:
        heldout = launcher.LORO_HELDOUT_RECEIVER[row.fold]
        assert heldout == expected[row.fold]
        assert heldout in launcher.SOURCE_RECEIVERS
        assert heldout not in row.source_receivers


def _command_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_pairbicad_convergence_command_has_all_five_source_loro_controls() -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(
        stage="pairbicad_convergence",
        candidates=("P2", "P4"),
        folds=(1, 8),
        seeds=(392001, 392002, 392003),
        optimizer_updates=9000,
    )[0]
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        Path("/tmp/pairbicad-convergence-run"),
        Path("/tmp/ManySig.pkl"),
    )

    command = launcher.build_train_command(row, roots, run_id="pairbicad-convergence")

    assert _command_value(command, "--bicad_optimizer_updates") == "9000"
    assert _command_value(command, "--bicad_loro_receiver") == "1"
    assert _command_value(command, "--bicad_loro_eval_interval_updates") == "500"
    assert _command_value(command, "--bicad_loro_min_updates") == "4000"
    assert _command_value(command, "--bicad_loro_patience") == "5"


def test_pairbicad_final_command_disables_online_source_loro_evaluation() -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(
        stage="pairbicad_final",
        candidates=("P4",),
        folds=(1, 2, 3, 4, 5),
        seeds=(392001, 392002, 392003),
        optimizer_updates=6500,
    )[0]
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        Path("/tmp/pairbicad-final-run"),
        Path("/tmp/ManySig.pkl"),
    )

    command = launcher.build_train_command(row, roots, run_id="pairbicad-final")
    options = {token for token in command if token.startswith("--")}

    assert _command_value(command, "--bicad_optimizer_updates") == "6500"
    assert _command_value(command, "--bicad_loro_eval_interval_updates") == "0"
    assert "--bicad_loro_receiver" not in options
    assert "--bicad_loro_min_updates" not in options
    assert "--bicad_loro_patience" not in options


def _write_source_loro_fixture(row_root: Path, row: object, *, stop_update: int) -> None:
    (row_root / "source_loro").mkdir(parents=True, exist_ok=True)
    (row_root / "source_loro" / f"checkpoint_u{stop_update}.pth").write_bytes(b"best")
    (row_root / "source_loro_curve.jsonl").write_text(
        json.dumps(
            {
                "update": stop_update,
                "planned_updates": row.optimizer_updates,
                "source_only": True,
                "target_access": False,
                "phase2_access": False,
                "support_access": False,
                "query_access": False,
                "truth_access": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (row_root / "source_loro_selection.json").write_text(
        json.dumps(
            {
                "planned_updates": row.optimizer_updates,
                "stop_update": stop_update,
                "best_update": stop_update,
                "patience": 5,
                "interval": 500,
                "source_only": True,
                "target_access": False,
                "phase2_access": False,
                "support_access": False,
                "query_access": False,
                "truth_access": False,
            }
        ),
        encoding="utf-8",
    )


def test_pairbicad_convergence_runtime_expectation_uses_selection_stop_update(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(
        stage="pairbicad_convergence",
        candidates=("P2", "P4"),
        folds=(1, 8),
        seeds=(392001, 392002, 392003),
        optimizer_updates=9000,
    )[0]
    _write_source_loro_fixture(tmp_path, row, stop_update=6500)

    expectation = launcher._convergence_runtime_expectation(tmp_path, row)

    assert expectation["optimizer_updates"] == 6500
    assert expectation["planned_optimizer_updates"] == 9000


@pytest.mark.parametrize(
    "fixture_change",
    [
        lambda root: None,
        lambda root: (root / "source_loro_selection.json").write_text(
            json.dumps({"planned_updates": 9000, "stop_update": 6500}), encoding="utf-8"
        ),
        lambda root: (root / "source_loro_selection.json").write_text(
            json.dumps(
                {
                    "planned_updates": 9000,
                    "stop_update": 6500,
                    "best_update": 7000,
                    "patience": 5,
                    "interval": 500,
                    "source_only": True,
                }
            ),
            encoding="utf-8",
        ),
    ],
)
def test_pairbicad_convergence_missing_or_incorrect_selection_is_technical_failure(
    tmp_path: Path, fixture_change: object
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(
        stage="pairbicad_convergence",
        candidates=("P2", "P4"),
        folds=(1, 8),
        seeds=(392001, 392002, 392003),
        optimizer_updates=9000,
    )[0]
    (tmp_path / "source_loro_curve.jsonl").write_text("{}\n", encoding="utf-8")
    fixture_change(tmp_path)

    with pytest.raises((FileNotFoundError, ValueError)):
        launcher._convergence_runtime_expectation(tmp_path, row)


def test_pairbicad_convergence_missing_selection_marks_row_as_technical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    row = launcher.build_plan(
        stage="pairbicad_convergence",
        candidates=("P2", "P4"),
        folds=(1, 8),
        seeds=(392001, 392002, 392003),
        optimizer_updates=9000,
    )[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    roots = launcher.LauncherRoots(
        SCRIPT_PATH.resolve().parents[2],
        Path("python"),
        tmp_path,
        tmp_path / "ManySig.pkl",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    status = launcher.launch_row_process(row, roots, run_id="pairbicad-convergence")

    assert status == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    failure = json.loads((row_root / "TECHNICAL_FAILURE.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "SOURCE_LORO_CLOSURE_FAILED"
