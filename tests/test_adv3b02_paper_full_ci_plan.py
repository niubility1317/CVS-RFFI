import argparse
import json
from pathlib import Path

from paper_reproduction.scripts.build_adv3b02_paper_full_ci_plan import build
from paper_reproduction.scripts.run_adv3b02_paper_full_ci_plan import _load_plan


def test_paper_full_plan_has_complete_matrix_and_locked_methods(tmp_path):
    artifacts = {}
    for name in ("checkpoint", "candidate", "adapter", "head", "tta"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    split = {
        "target_old_tx_labels": ["o0", "o1", "o2", "o3", "o4", "o5"],
        "nested_target_new_tx_labels": {
            "2": ["n0", "n1"],
            "5": [f"n{i}" for i in range(5)],
            "10": [f"n{i}" for i in range(10)],
            "20": [f"n{i}" for i in range(20)],
        },
        "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "confirmation_seeds": [713101, 713102, 713103, 713104, 713105],
        "k_values": [1, 5, 10, 20],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    output = tmp_path / "plan.json"
    plan = build(
        argparse.Namespace(
            experiment_id="paper_full_test",
            run_root=str(tmp_path / "run"),
            target_cache_root=str(tmp_path / "cache"),
            class_split=split_path,
            base_checkpoint=str(artifacts["checkpoint"]),
            candidate_lock=str(artifacts["candidate"]),
            adapter=str(artifacts["adapter"]),
            head_artifact=str(artifacts["head"]),
            tta_policy=str(artifacts["tta"]),
            smoke_receipt=None,
            output=output,
        )
    )
    assert plan["counts"] == {"packages": 100, "cells": 800, "scenario_rows": 2400}
    assert plan["backbone_uniformly_frozen"] is False
    assert plan["base_source_reference_access_allowed"] is True
    assert plan["new_class_counts"] == [2, 5, 10, 20]
    assert len(plan["smoke_cell_ids"]) == 4
    assert set(plan["methods"]) == {"csil_paper_full", "mopc_hr_paper_full"}
    assert _load_plan(output)["authority_state"].endswith("SMOKE_REQUIRED")


def test_predictor_source_has_no_truth_or_channel_resampling_surface():
    source = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction/scripts/run_adv3b02_paper_full_ci_truth_free_predictor.py"
    ).read_text(encoding="utf-8")
    assert "query_y" not in source
    assert "query_truth" not in source
    assert "apply_leo" not in source
    assert "satellite_channel" not in source
    assert "query_rows_used_for_training\": 0" in source
    assert "query_members_opened_before_model_lock\": False" in source


def test_comparison_bundle_relaxes_only_set_level_protocol_and_keeps_leo_check():
    source = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction/scripts/build_adv3b02_paper_full_ci_bundle.py"
    ).read_text(encoding="utf-8")
    assert "load_verified_leo_weak_cache(" in source
    assert "new_class_leo_iq_verified" in source
    assert "load_verified_leo_weak_cache_set =" in source
    assert "stage2_main_method_protocol_exempt_new_class_leo_required" in source
