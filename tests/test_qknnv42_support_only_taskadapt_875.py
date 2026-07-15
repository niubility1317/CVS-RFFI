from __future__ import annotations

import json

import pytest

from paper_reproduction.scripts.run_cvs_qknnv42_support_only_taskadapt_875 import (
    EPOCHS,
    K_GRID,
    RECEIVERS,
    SEEDS,
    build_tasks,
    validate_base_config,
)


def test_matrix_is_exactly_125_baselines_plus_750_task_specific_adapters(tmp_path) -> None:
    tasks = build_tasks(
        adapter_root=tmp_path / "adapters",
        output_root=tmp_path / "results",
        log_root=tmp_path / "logs",
    )
    assert len(tasks) == 875
    assert sum(task.epochs == 0 for task in tasks) == 125
    assert sum(task.epochs > 0 for task in tasks) == 750
    keys = {(task.arm, task.receiver, task.seed, task.k_shot) for task in tasks}
    assert len(keys) == 875
    assert {task.epochs for task in tasks if task.epochs} == set(EPOCHS)
    assert {task.k_shot for task in tasks} == set(K_GRID)
    assert {task.receiver for task in tasks} == set(RECEIVERS)
    assert {task.seed for task in tasks} == set(SEEDS)
    assert all(f"_e_{task.epochs}" in task.adapter_run_dir for task in tasks if task.epochs)


def test_base_config_guard_rejects_dense_query_or_wrong_new_count() -> None:
    config = {
        "method": "cvs_qknnv42",
        "stage": "Stage2-C",
        "target_old_tx_labels": ["o1"],
        "target_new_tx_labels": ["n1", "n2"],
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_labelprop_mode": "disabled",
        "non_deployment_oracle_diagnostic": False,
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "qknnv42_aux_feature_key": "fft_logmag_features",
        "qknnv42_aux_feature_dim": 96,
        "qknnv42_expected_tta_view_count": 1,
        "support_pool_max_k": 20,
        "feature_npz_by_scenario": {
            "leo_clear_weak": "a.npz",
            "leo_low_elev_weak": "b.npz",
            "leo_rain_weak": "c.npz",
        },
    }
    validate_base_config(config, new_count=2)
    bad = json.loads(json.dumps(config))
    bad["qknnv42_labelprop_mode"] = "dense_transductive"
    with pytest.raises(ValueError, match="qknnv42_labelprop_mode"):
        validate_base_config(bad, new_count=2)
    with pytest.raises(ValueError, match="new_count"):
        validate_base_config(config, new_count=20)
