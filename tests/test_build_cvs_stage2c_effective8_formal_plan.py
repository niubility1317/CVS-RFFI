from __future__ import annotations

import json
from pathlib import Path

from paper_reproduction.scripts.build_cvs_stage2c_effective8_formal_plan import (
    POLICY,
    generate_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    REPO_ROOT
    / "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json"
)


def test_generated_plan_has_exact_locked_matrix_and_no_legacy_phase2_inputs(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "generated"
    manifest = generate_plan(
        PLAN,
        out_dir=out_dir,
        runtime_project_root="/srv/CV-SincNet",
    )
    assert manifest["expected_counts"] == {
        "source_cache_sets": 2,
        "target_cache_sets": 25,
        "benchmark_invocations": 300,
        "formal_scenario_rows": 900,
        "collection_invocations": 1,
        "summary_invocations": 1,
    }
    assert len(manifest["commands"]["source_cache_build"]) == 2
    assert len(manifest["commands"]["target_cache_build"]) == 25
    assert len(manifest["commands"]["benchmark"]) == 300
    assert len(manifest["commands"]["collect"]) == 1
    assert len(manifest["commands"]["summarize"]) == 1

    configs = sorted((out_dir / "stage2_configs").rglob("*.json"))
    assert len(configs) == 300
    for path in configs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["phase2_sample_view_policy"] == POLICY
        assert payload["clean_sample_access"] is False
        assert payload["clean_derived_signal_access"] is False
        assert payload["target_channel_view"] == "leo_weak_only"
        assert payload["old_new_role_oracle_used"] is False
        assert payload["class_quota_used"] is False
        assert payload["query_fit_used"] is False
        assert "raw_iq_input_len" not in payload
        assert "feature_npz_by_scenario" not in payload
        assert payload["leo_weak_cache_set_manifest"].endswith("cache_set.json")
        assert not Path(payload["direct_adv3b02_class_mapping_source"]).is_absolute()


def test_training_command_uses_sealed_cache_and_same_leo_reference_names(
    tmp_path: Path,
) -> None:
    manifest = generate_plan(
        PLAN,
        out_dir=tmp_path / "generated",
        runtime_project_root="/srv/CV-SincNet",
    )
    command = manifest["commands"]["train"][0]
    assert "--source_leo_weak_cache_set_manifest" in command
    assert "--leo_reference_identity_weight" in command
    assert "--leo_reference_margin_weight" in command
    assert "--wisig_pkl" not in command
    assert "--clean_identity_weight" not in command
    assert "--clean_feature_margin_weight" not in command
    assert command[command.index("--source_tx_ids") + 1] == (
        "14-10,14-7,20-15,20-19,6-15,8-20"
    )
    assert command[command.index("--source_rxs") + 1] == (
        "1-1,1-19,14-7,18-2,19-2,2-1"
    )
    validation_command = manifest["commands"]["source_validation"][0]
    assert validation_command[validation_command.index("--source_train_rxs") + 1] == (
        "1-1,1-19,14-7,18-2,19-2,2-1"
    )
    assert validation_command[validation_command.index("--source_val_rxs") + 1] == (
        "2-19"
    )
