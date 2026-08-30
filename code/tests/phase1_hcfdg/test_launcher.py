from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "launch_phase1_hcfdg_matrix_20260830.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("phase1_hcfdg_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_quick_plan_is_exact_source_loro_matrix():
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="quick", folds=(1, 8))

    assert len(rows) == 36
    assert {row.candidate_id for row in rows} == {f"A{i}" for i in range(6)}
    assert {row.seed for row in rows} == {392001, 392002, 392003}
    assert {row.heldout_receiver for row in rows} == {1, 8}
    assert all(row.optimizer_updates == 4000 for row in rows)
    assert all(row.heldout_receiver not in row.source_receivers for row in rows)
    assert all(set(row.source_receivers) | {row.heldout_receiver} == {1, 3, 4, 6, 8} for row in rows)
    assert all(row.train_days == (1, 2, 3) for row in rows)


def test_deep_plan_selects_only_the_explicit_seed_and_keeps_v2_budget():
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="deep", folds=(1, 8), gpus=(0, 1), seeds=(392002,))

    assert len(rows) == 8
    assert len({row.row_id for row in rows}) == 8
    assert {row.candidate_id for row in rows} == {"A6", "A7", "A8", "A9"}
    assert {row.heldout_receiver for row in rows} == {1, 8}
    assert {row.seed for row in rows} == {392002}
    assert all(row.optimizer_updates == 6300 for row in rows)
    assert all(row.seed == 392002 for row in rows)
    assert all(row.heldout_receiver not in row.source_receivers for row in rows)


def test_plan_rejects_a_seed_outside_the_frozen_selection_set():
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="seeds"):
        launcher.build_plan(stage="deep", folds=(1, 8), seeds=(392004,))


def test_default_run_id_does_not_reuse_the_a0_a5_matrix_name():
    launcher = _load_launcher()

    assert "a0a5" not in launcher.RUN_ID_DEFAULT.lower()


def test_a9_content_keys_are_finite_scale_invariant_and_structure_aware():
    launcher = _load_launcher()
    generator = torch.Generator().manual_seed(7)
    signal = torch.randn(64, generator=generator)
    iq = torch.stack((signal, signal.roll(1)), dim=0).unsqueeze(0)
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(9))
    batch = torch.cat((iq, iq[:, :, permutation]), dim=0)

    keys = launcher._content_keys_from_iq(batch)
    scaled_keys = launcher._content_keys_from_iq(batch * 7.5)

    assert keys.shape[0] == 2
    assert keys.shape[1] >= 16
    assert torch.isfinite(keys).all()
    torch.testing.assert_close(keys, scaled_keys, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(keys[0], keys[1])


def test_smoke_budget_reaches_stage3_without_changing_formal_v2_budget():
    launcher = _load_launcher()
    from cvsrffi.phase1_hcfdg.trainer import stage_for_update

    budget = launcher.HCFDG_SMOKE_STAGE_BUDGET
    stages = [stage_for_update(update, budget) for update in range(1, budget.total_updates + 1)]
    smoke_config = launcher._resolve_worker_config("A9", smoke=True)
    formal_config = launcher._resolve_worker_config("A9", smoke=False)

    assert stages == ["stage0", "stage1", "stage2", "stage3"]
    assert budget.total_updates < 6300
    assert smoke_config.optimizer_updates == budget.total_updates
    assert formal_config.optimizer_updates == 6300
    assert smoke_config.counterfactual_mode != "off"
    assert smoke_config.use_hdro is True
    assert smoke_config.use_content_conditioning is True


def test_train_command_binds_row_and_never_mentions_phase2_or_target(tmp_path):
    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick", folds=(1, 8))[0]
    roots = launcher.LauncherRoots(
        code_root=tmp_path / "release",
        python=Path("python"),
        run_root=tmp_path / "run",
        wisig_pkl=tmp_path / "wisig.pkl",
    )

    command = launcher.build_train_command(row, roots)
    joined = " ".join(command).lower()

    assert _value_after(command, "--candidate-id") == row.candidate_id
    assert _value_after(command, "--heldout-rx") == str(row.heldout_receiver)
    assert _value_after(command, "--source-rxs") == ",".join(map(str, row.source_receivers))
    assert _value_after(command, "--train-days") == "1,2,3"
    assert _value_after(command, "--optimizer-updates") == "4000"
    assert _value_after(command, "--seed") == str(row.seed)
    assert not any(token in joined for token in ("phase2", "target", "query", "truth"))


def test_train_command_does_not_reject_a_legal_worktree_name_containing_phase2(tmp_path):
    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick", folds=(1, 8))[0]
    roots = launcher.LauncherRoots(
        code_root=tmp_path / "phase2-canonical-union-maxq",
        python=Path("python"),
        run_root=tmp_path / "run",
        wisig_pkl=tmp_path / "ManySig.pkl",
    )

    command = launcher.build_train_command(row, roots)

    assert "phase2-canonical-union-maxq" in " ".join(command)


def test_numpy_boolean_masks_are_converted_with_explicit_torch_dtype():
    launcher = _load_launcher()
    values = np.asarray([True, False, True], dtype=np.bool_)

    result = launcher._bool_tensor(values)

    assert result.dtype == torch.bool
    assert result.tolist() == [True, False, True]


def test_output_root_is_immutable_and_unknown_stages_fail_closed(tmp_path):
    launcher = _load_launcher()
    root = tmp_path / "fresh"
    assert launcher.validate_output_root(root) == root.resolve()
    root.mkdir()
    with pytest.raises(FileExistsError):
        launcher.validate_output_root(root)
    with pytest.raises(ValueError, match="stage"):
        launcher.build_plan(stage="unknown", folds=(1, 8))
    with pytest.raises(ValueError, match="v2_passed"):
        launcher.build_plan(stage="residual", folds=(1, 8), v2_passed=False)


def test_plan_json_is_written_once_and_final_status_waits_for_all_rows(tmp_path):
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="quick", folds=(1, 8))[:2]
    root = tmp_path / "run"
    launcher.write_plan_json(root, rows, run_id="test-run")
    with pytest.raises(FileExistsError):
        launcher.write_plan_json(root, rows, run_id="test-run")

    statuses = {rows[0].row_id: "ARTIFACTS_COMPLETE", rows[1].row_id: "RUNNING"}
    with pytest.raises(ValueError, match="terminal"):
        launcher.write_final_status(root, rows, statuses)
    statuses[rows[1].row_id] = "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    path = launcher.write_final_status(root, rows, statuses)
    assert json.loads(path.read_text(encoding="utf-8"))["row_count"] == 2


def test_artifact_closure_requires_strict_checkpoint_and_all_four_scenarios(tmp_path):
    launcher = _load_launcher()
    row_root = tmp_path / "row"
    row_root.mkdir()
    (row_root / "final_hcfdg.pt").write_bytes(b"checkpoint")
    for scenario in launcher.FINAL_SCENARIOS:
        (row_root / f"eval_{scenario}.json").write_text(
            json.dumps({
                "scenario": scenario,
                "checkpoint_load_strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            }),
            encoding="utf-8",
        )
        (row_root / f"eval_{scenario}.log").write_text("ok\n", encoding="utf-8")

    assert launcher.validate_artifact_closure(row_root)["status"] == "ARTIFACTS_COMPLETE"
    (row_root / "eval_leo_rain_weak.json").unlink()
    with pytest.raises(ValueError, match="leo_rain_weak"):
        launcher.validate_artifact_closure(row_root)


def test_scheduler_never_exceeds_two_active_rows_per_gpu():
    launcher = _load_launcher()
    rows = launcher.build_plan(stage="quick", folds=(1, 8), gpus=(0, 1))[:12]
    active = {0: 0, 1: 0}
    peak = {0: 0, 1: 0}
    lock = threading.Lock()

    def fake_run(row):
        with lock:
            active[row.gpu] += 1
            peak[row.gpu] = max(peak[row.gpu], active[row.gpu])
        time.sleep(0.01)
        with lock:
            active[row.gpu] -= 1
        return "ARTIFACTS_COMPLETE"

    statuses = launcher.run_plan(rows, fake_run, max_active_per_gpu=2)

    assert set(statuses.values()) == {"ARTIFACTS_COMPLETE"}
    assert max(peak.values()) <= 2


def test_evaluate_final_checkpoint_requires_strict_reconstruction(tmp_path):
    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick", folds=(1, 8))[0]
    row_root = tmp_path / row.row_id
    row_root.mkdir()
    checkpoint = row_root / "final_hcfdg.pt"
    checkpoint.write_bytes(b"checkpoint")

    def bad_reconstruct(_path):
        return object(), {"missing_keys": ["head.weight"], "unexpected_keys": [], "shape_mismatches": []}

    with pytest.raises(ValueError, match="strict"):
        launcher.evaluate_final_checkpoint(
            row,
            row_root,
            reconstruct_fn=bad_reconstruct,
            evaluate_fn=lambda *_args, **_kwargs: {"accuracy": 1.0},
        )


def test_worker_source_arguments_are_accepted_by_real_ssdg_parser(tmp_path):
    launcher = _load_launcher()
    code_root = SCRIPT.resolve().parents[2]
    ssdg = launcher._load_ssdg_module(code_root)
    args = launcher.build_arg_parser().parse_args(
        [
            "--worker-row",
            "--candidate-id", "A2",
            "--heldout-rx", "8",
            "--source-rxs", "1,3,4,6",
            "--train-days", "1,2,3",
            "--optimizer-updates", "4000",
            "--seed", "392001",
            "--wisig-pkl", str(tmp_path / "wisig.pkl"),
            "--row-root", str(tmp_path / "row"),
        ]
    )

    parsed = launcher._source_runtime_args(ssdg, args, "dual")

    assert parsed.phase1_source_only_eval is True
    assert parsed.use_sat_consistency is False
    assert parsed.wisig_train_rxs == "1,3,4,6"
    assert parsed.wisig_train_days == "1,2,3"


def test_smoke_mode_is_explicit_and_never_changes_formal_commands(tmp_path):
    launcher = _load_launcher()
    row = launcher.build_plan(stage="quick", folds=(1, 8))[0]
    roots = launcher.LauncherRoots(tmp_path, Path("python"), tmp_path / "run", tmp_path / "wisig.pkl")
    assert "--smoke" not in launcher.build_train_command(row, roots)
    args = launcher.build_arg_parser().parse_args(["--worker-row", "--smoke"])
    assert args.smoke is True


def test_formal_dispatcher_writes_plan_and_terminal_status(tmp_path, monkeypatch):
    launcher = _load_launcher()
    source = tmp_path / "ManySig.pkl"
    source.write_bytes(b"source")
    run_root = tmp_path / "formal-run"
    observed = []

    def fake_run_row(row, roots):
        observed.append((row.row_id, roots.run_root))
        row_root = roots.run_root / row.row_id
        row_root.mkdir(parents=True)
        (row_root / "final_hcfdg.pt").write_bytes(b"checkpoint")
        for scenario in launcher.FINAL_SCENARIOS:
            payload = {
                "scenario": scenario,
                "checkpoint_load_strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            }
            (row_root / f"eval_{scenario}.json").write_text(json.dumps(payload), encoding="utf-8")
            (row_root / f"eval_{scenario}.log").write_text("ok\n", encoding="utf-8")
        return "ARTIFACTS_COMPLETE"

    monkeypatch.setattr(launcher, "run_row", fake_run_row)
    result = launcher.main(
        [
            "--formal",
            "--run-id", "formal-test",
            "--stage", "quick",
            "--folds", "1,8",
            "--gpus", "0,1",
            "--run-root", str(run_root),
            "--wisig-pkl", str(source),
        ]
    )

    assert result == 0
    assert len(observed) == 36
    assert json.loads((run_root / "plan.json").read_text(encoding="utf-8"))["row_count"] == 36
    final = json.loads((run_root / "final_status.json").read_text(encoding="utf-8"))
    assert set(final["statuses"].values()) == {"ARTIFACTS_COMPLETE"}


def test_formal_deep_seed_selection_writes_an_eight_row_plan(tmp_path, monkeypatch):
    launcher = _load_launcher()
    source = tmp_path / "ManySig.pkl"
    source.write_bytes(b"source")
    run_root = tmp_path / "formal-deep-run"
    observed = []

    def fake_run_row(row, roots):
        observed.append(row)
        row_root = roots.run_root / row.row_id
        row_root.mkdir(parents=True)
        (row_root / "final_hcfdg.pt").write_bytes(b"checkpoint")
        for scenario in launcher.FINAL_SCENARIOS:
            payload = {
                "scenario": scenario,
                "checkpoint_load_strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            }
            (row_root / f"eval_{scenario}.json").write_text(json.dumps(payload), encoding="utf-8")
            (row_root / f"eval_{scenario}.log").write_text("ok\n", encoding="utf-8")
        return "ARTIFACTS_COMPLETE"

    monkeypatch.setattr(launcher, "run_row", fake_run_row)
    result = launcher.main(
        [
            "--formal",
            "--run-id", "formal-deep-seed392002",
            "--stage", "deep",
            "--seeds", "392002",
            "--folds", "1,8",
            "--gpus", "0,1",
            "--run-root", str(run_root),
            "--wisig-pkl", str(source),
        ]
    )

    plan = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    assert result == 0
    assert len(observed) == 8
    assert plan["row_count"] == 8
    assert plan["seeds"] == [392002]
    assert all(row["candidate_id"] in {"A6", "A7", "A8", "A9"} for row in plan["rows"])
    assert all(row["seed"] == 392002 for row in plan["rows"])
    assert all(row["optimizer_updates"] == 6300 for row in plan["rows"])


def test_formal_deep_requires_explicit_run_id_and_rejects_short_budget(tmp_path, monkeypatch):
    launcher = _load_launcher()
    source = tmp_path / "ManySig.pkl"
    source.write_bytes(b"source")

    def fake_run_plan(rows, row_runner, *, max_active_per_gpu):
        return {row.row_id: "ARTIFACTS_COMPLETE" for row in rows}

    monkeypatch.setattr(launcher, "run_plan", fake_run_plan)
    with pytest.raises(ValueError, match="run-id"):
        launcher.main(
            [
                "--formal",
                "--stage", "deep",
                "--seeds", "392002",
                "--run-root", str(tmp_path / "missing-run-id"),
                "--wisig-pkl", str(source),
            ]
        )

    with pytest.raises(ValueError, match="6300"):
        launcher.main(
            [
                "--formal",
                "--run-id", "formal-deep-short-budget",
                "--stage", "deep",
                "--seeds", "392002",
                "--optimizer-updates", "1",
                "--run-root", str(tmp_path / "short-budget"),
                "--wisig-pkl", str(source),
            ]
        )


def test_formal_dispatcher_refuses_existing_root_and_missing_source(tmp_path):
    launcher = _load_launcher()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        launcher.main(
            ["--formal", "--run-root", str(existing), "--wisig-pkl", str(tmp_path / "missing.pkl")]
        )
    with pytest.raises(FileNotFoundError):
        launcher.main(
            ["--formal", "--run-root", str(tmp_path / "fresh"), "--wisig-pkl", str(tmp_path / "missing.pkl")]
        )


def test_formal_dispatcher_stops_queued_rows_after_two_matching_failures(tmp_path, monkeypatch):
    launcher = _load_launcher()
    source = tmp_path / "ManySig.pkl"
    source.write_bytes(b"source")
    started = []

    def systemic_failure(row, _roots):
        started.append(row.row_id)
        raise RuntimeError("deterministic pre-prediction failure")

    monkeypatch.setattr(launcher, "run_row", systemic_failure)
    result = launcher.main(
        [
            "--formal",
            "--folds", "1,8",
            "--gpus", "0,1",
            "--run-root", str(tmp_path / "formal-run"),
            "--wisig-pkl", str(source),
        ]
    )

    assert result == 2
    assert 2 <= len(started) < 36
    final = json.loads((tmp_path / "formal-run" / "final_status.json").read_text(encoding="utf-8"))
    assert set(final["statuses"].values()) == {"STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"}
