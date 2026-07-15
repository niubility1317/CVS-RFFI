from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_reproduction.cvs_aligned.cvs_method_runner import _attach_post_adapter_resources
from paper_reproduction.scripts.run_cvs_qknnv42_idnorm_tta5_1000 import (
    EPOCHS,
    K_GRID,
    RECEIVERS,
    SEEDS,
    build_tasks,
    validate_base_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "paper_reproduction"
    / "configs"
    / "cvs_qknnv42_idnorm_tta5_1000_stage2c_20260715_n607.json"
)


def test_matrix_is_125_baseline_plus_875_independent_adapters(tmp_path: Path) -> None:
    tasks = build_tasks(
        adapter_root=tmp_path / "adapters",
        output_root=tmp_path / "results",
        log_root=tmp_path / "logs",
    )
    assert len(tasks) == 1000
    assert sum(task.epochs == 0 for task in tasks) == 125
    assert sum(task.epochs > 0 for task in tasks) == 875
    assert {task.epochs for task in tasks if task.epochs} == set(EPOCHS)
    for epoch in EPOCHS:
        arm = [task for task in tasks if task.epochs == epoch]
        assert len(arm) == len(RECEIVERS) * len(SEEDS) * len(K_GRID) == 125
        assert len({task.adapter_run_dir for task in arm}) == 125


def test_base_config_is_nonoracle_singleview_leo_only() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_base_config(config)
    assert config["clean_sample_access"] is False
    assert config["phase2_sample_view_policy"] == "leo_weak_only_no_clean_access"
    assert config["qknnv42_expected_tta_view_count"] == 1
    assert config["qknnv42_labelprop_mode"] == "disabled"
    assert config["non_deployment_oracle_diagnostic"] is False


def _cache_manifest() -> dict:
    return {
        "payload_source": "cvs_stage2c_support_only_id_norm_late_feature_tta5_v1",
        "adapter": {
            "method": "support_only_id_norm_late_feature_tta5_v1",
            "support_only": True,
            "query_update_forbidden": True,
            "query_labels_used_for_training": False,
            "old_new_role_used_by_optimizer": False,
            "class_quota_used_at_inference": False,
            "query_view_count": 5,
            "epochs": 60,
            "resource_tier": "non_extreme_light_large_adapter_diagnostic",
            "diagnostic_only": True,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "clean_sample_access": False,
            "adapter_state_format": "fp16_delta_from_strict_checkpoint",
            "resources": {
                "trainable_parameters": 289_685,
                "original_checkpoint_trainable_parameters": 289_685,
                "original_checkpoint_gradient_updates": 60,
                "adapter_state_bytes_fp16": 579_370,
                "adapter_macs_per_query": 0,
                "deployment_added_macs_per_query_after_merge": 0,
                "backbone_forward_count_per_query": 5,
                "full_model_finetune": False,
            },
        },
    }


def test_runner_accepts_only_exact_large_adapter_provenance() -> None:
    info = {
        "estimated_macs_per_query": 11,
        "estimated_head_macs": 22,
        "persistent_state_bytes": 33,
    }
    _attach_post_adapter_resources(info, _cache_manifest(), support_count=8, query_count=160)
    assert info["post_feature_adapter_parameter_count"] == 289_685
    assert info["post_feature_adapter_state_bytes"] == 579_370
    assert info["post_feature_adapter_macs_per_sample"] == 0


@pytest.mark.parametrize("field,value", [
    ("query_view_count", 1),
    ("clean_sample_access", True),
    ("diagnostic_only", False),
])
def test_runner_rejects_protocol_drift(field: str, value: object) -> None:
    manifest = _cache_manifest()
    manifest["adapter"][field] = value
    info = {
        "estimated_macs_per_query": 0,
        "estimated_head_macs": 0,
        "persistent_state_bytes": 0,
    }
    with pytest.raises(ValueError):
        _attach_post_adapter_resources(info, manifest, support_count=8, query_count=160)
