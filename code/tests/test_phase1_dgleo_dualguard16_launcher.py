from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_dgleo_dualguard16_20260712.py"


def _module():
    spec = importlib.util.spec_from_file_location("dualguard16", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dualguard16_matrix_has_eight_mechanism_cells_two_paired_seeds_and_two_per_gpu():
    module = _module()
    rows = module.build_matrix()

    assert len(rows) == 16
    assert Counter(row["gpu"] for row in rows) == Counter({gpu: 2 for gpu in range(8)})
    assert set(Counter(row["cell"] for row in rows).values()) == {2}
    assert {row["seed"] for row in rows} == set(module.PAIRED_SEEDS)
    assert all(row["source_only"] for row in rows)
    assert all(row["checkpoint_selection"] == "final_only" for row in rows)


def test_dualguard16_command_uses_bounded_geometry_dg_guard_and_disjoint_satellite_protocol():
    module = _module()
    row = next(row for row in module.build_matrix() if row["cell"] == "C7_FULL_JOINT")
    command = module.build_command(
        row,
        root=Path("/srv/CV-SincNet"),
        python=Path("/env/python"),
        run_id="unit_dualguard16",
        wisig_pkl=Path("/srv/CV-SincNet/Dataset_WigSig/ManySig.pkl"),
        teacher_ckpt=Path("/srv/CV-SincNet/teacher.pth"),
    )
    joined = " ".join(str(value) for value in command)

    assert "--epochs 120" in joined
    assert "--checkpoint_selection final_only" in joined
    assert "--tail_safety_absolute_violation_drives_state false" in joined
    assert "--tail_safety_training_stop_enabled false" in joined
    assert "--tail_safety_reference_requires_absolute_safe false" in joined
    assert "--source_episode_multiview_normalize true" in joined
    assert "--source_episode_local_radius_floor_deg 4.0" in joined
    assert "--source_episode_local_density_cap 1.5" in joined
    assert "--source_episode_structural_start_epoch 40" in joined
    assert "--os_eff_max_budget 0.22" in joined
    assert "--os_budget_min_closed_scale 1.0" in joined
    assert "--max_grad_norm 5.0" in joined
    assert "--source_val_dg_health_guard true" in joined
    assert "--direct_metric_zid_p95_target_deg 62.0" in joined
    assert "--direct_metric_zid_p99_target_deg 80.0" in joined
    assert "--u_direct_idle_blocks_promotion true" in joined
    assert "--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in joined
    assert "--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,geo_clear,mixed_orbit" in joined
    assert "--use_concat_sat_channel_aug" in joined
    assert "--lambda_sat_cls 1.0" in joined
    assert "--lambda_sat_cons 0.1" in joined
    assert "--lambda_u_domain 0.24" in joined
    assert "--lambda_u_adv 0.12" in joined
    assert "--lambda_u_sat_cons 0.42" in joined


def test_dualguard16_hard_wall_clock_limit_cannot_exceed_ten_hours():
    module = _module()
    parser = module.build_parser()
    assert parser.parse_args([]).wall_hours == 10.0
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_terminate_process_groups(active)" in source
    assert "time.monotonic() >= deadline" in source
    assert "WALL_CLOCK_TIMEOUT" in source


def test_dualguard16_scheduler_returns_nonzero_when_any_child_is_not_complete():
    module = _module()
    outcome, exit_code = module._scheduler_outcome(
        [
            {"candidate_id": "ok", "returncode": 0, "status": "COMPLETE"},
            {"candidate_id": "tail", "returncode": 4, "status": "STOPPED_TAIL"},
        ],
        expected_count=2,
        timed_out=False,
    )

    assert exit_code != 0
    assert outcome["status"] == "CHILD_FAILURE"
    assert outcome["candidate_status_counts"] == {"COMPLETE": 1, "STOPPED_TAIL": 1}
    assert outcome["non_complete_candidates"] == ["tail"]


def test_dualguard16_scheduler_returns_zero_only_when_every_child_is_complete():
    module = _module()
    outcome, exit_code = module._scheduler_outcome(
        [
            {"candidate_id": "a", "returncode": 0, "status": "COMPLETE"},
            {"candidate_id": "b", "returncode": 0, "status": "COMPLETE"},
        ],
        expected_count=2,
        timed_out=False,
    )

    assert exit_code == 0
    assert outcome["status"] == "COMPLETE"
    assert outcome["missing_terminal_count"] == 0
    assert outcome["non_complete_candidates"] == []


def test_dualguard16_scheduler_reports_missing_terminal_and_timeout_distinctly():
    module = _module()
    missing, missing_exit = module._scheduler_outcome(
        [{"candidate_id": "a", "returncode": 0, "status": "COMPLETE"}],
        expected_count=2,
        timed_out=False,
    )
    timed_out, timeout_exit = module._scheduler_outcome(
        [{"candidate_id": "a", "returncode": 0, "status": "COMPLETE"}],
        expected_count=2,
        timed_out=True,
    )

    assert missing_exit != 0
    assert missing["status"] == "SCHEDULER_INCOMPLETE"
    assert missing["missing_terminal_count"] == 1
    assert timeout_exit == 124
    assert timed_out["status"] == "WALL_CLOCK_TIMEOUT"


def test_gpu_capacity_allows_one_unrelated_process_when_memory_is_sufficient(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.p1base, "_pmon_pids", lambda: {0: {101}})
    monkeypatch.setattr(module, "_gpu_free_memory_mib", lambda: {0: 18000})
    monkeypatch.setattr(module, "_pid_command_tokens", lambda pid: ["python", "other_train.py", "--run_id", "other_run"])

    snapshot = module.gpu_launch_snapshot(
        run_id="target_run",
        gpus=[0],
        max_total_compute_per_gpu=2,
        min_free_memory_mib=10000,
        allow_unrelated_compute=True,
    )

    assert snapshot["blocked"] == {}
    assert snapshot["gpus"]["0"]["unrelated_pids"] == [101]
    assert snapshot["gpus"]["0"]["compute_experiment_count"] == 1


def test_gpu_capacity_groups_multiple_cuda_pids_from_one_experiment(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.p1base, "_pmon_pids", lambda: {0: {301, 302, 303}})
    monkeypatch.setattr(module, "_gpu_free_memory_mib", lambda: {0: 18000})
    monkeypatch.setattr(
        module,
        "_pid_command_tokens",
        lambda pid: [
            "python",
            "other_train.py",
            "--run_id",
            "other_run",
            "--candidate_id",
            "other_candidate",
        ],
    )

    snapshot = module.gpu_launch_snapshot(
        run_id="target_run",
        gpus=[0],
        max_total_compute_per_gpu=2,
        min_free_memory_mib=10000,
        allow_unrelated_compute=True,
    )

    assert snapshot["blocked"] == {}
    assert snapshot["gpus"]["0"]["compute_experiment_count"] == 1
    assert snapshot["gpus"]["0"]["compute_experiments"][0]["pids"] == [301, 302, 303]


def test_gpu_capacity_blocks_a_third_experiment_even_when_memory_is_sufficient(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.p1base, "_pmon_pids", lambda: {0: {401, 402}})
    monkeypatch.setattr(module, "_gpu_free_memory_mib", lambda: {0: 18000})
    monkeypatch.setattr(
        module,
        "_pid_command_tokens",
        lambda pid: ["python", "other_train.py", "--run_id", f"other_run_{pid}"],
    )
    monkeypatch.setattr(module, "_process_group_id", lambda pid: pid)

    snapshot = module.gpu_launch_snapshot(
        run_id="target_run",
        gpus=[0],
        max_total_compute_per_gpu=2,
        min_free_memory_mib=10000,
        allow_unrelated_compute=True,
    )

    assert snapshot["gpus"]["0"]["compute_experiment_count"] == 2
    assert snapshot["blocked"]["0"] == ["compute_experiment_capacity"]


def test_gpu_capacity_blocks_duplicate_target_and_low_memory(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.p1base, "_pmon_pids", lambda: {0: {202}})
    monkeypatch.setattr(module, "_gpu_free_memory_mib", lambda: {0: 8000})
    monkeypatch.setattr(module, "_pid_command_tokens", lambda pid: ["python", "train.py", "--run_id", "target_run"])

    snapshot = module.gpu_launch_snapshot(
        run_id="target_run",
        gpus=[0],
        max_total_compute_per_gpu=2,
        min_free_memory_mib=10000,
        allow_unrelated_compute=True,
    )

    assert snapshot["gpus"]["0"]["target_pids"] == [202]
    assert snapshot["blocked"]["0"] == ["target_run_already_active", "insufficient_free_memory"]
