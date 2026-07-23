import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.leo_weak_cache import (
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    post_channel_iq_sha256,
    sha256_file,
)
from paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle import (
    _comparison_reference_arrays,
    load_comparison_inner_leo_cache,
    load_comparison_leo_cache_set,
    load_verified_comparison_stage2_predictor_bundle,
)
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
    assert "load_comparison_inner_leo_cache(" in source
    assert "new_class_leo_iq_verified" in source
    assert "load_verified_leo_weak_cache_set =" in source
    assert "load_verified_stage2_predictor_bundle = (" in source
    assert "_assert_scenario_alignment = _comparison_reference_arrays" in source
    assert "_assert_scenario_physical_independence = lambda" in source
    assert "stage2_main_method_protocol_exempt_new_class_leo_required" in source


def _write_legacy_comparison_cache(path: Path, scenario: str) -> None:
    iq = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    sample_ids = np.asarray([f"legacy-sample-{index}" for index in range(4)])
    roles = np.asarray(["target_old", "target_old", "target_new", "target_new"])
    seeds = np.asarray([11, 12, 13, 14], dtype=np.int64)
    channel_hash = canonical_json_sha256({"scenario": scenario})
    iq_hashes = [post_channel_iq_sha256(row) for row in iq]
    overlays = [
        overlay_id(
            sample_id=sample_ids[index],
            scenario=scenario,
            satellite_seed=int(seeds[index]),
            channel_config_sha256=channel_hash,
            iq_sha256=iq_hashes[index],
        )
        for index in range(4)
    ]
    manifest = {
        "schema": "cvs_leo_weak_iq_cache_v1",
        "artifact_stage": "phase1_offline_prechannel_export",
        "contains_post_channel_iq_only": True,
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "target_channel_scenarios": [scenario],
        "scenario": scenario,
        "iq_array_key": "leo_weak_iq",
        "output_roles": ["target_old", "target_new"],
        "row_count": 4,
        "channel_config_sha256": channel_hash,
        "physical_sample_ids_sha256": ids_sha256(sample_ids.tolist()),
        "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
        "overlay_ids_sha256": ids_sha256(overlays),
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
    }
    np.savez_compressed(
        path,
        leo_weak_iq=iq,
        raw_labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
        domain_labels=np.zeros(4, dtype=np.int64),
        tx_ids=np.asarray(["a", "b", "c", "d"]),
        rx_ids=np.asarray(["r"] * 4),
        day_ids=np.asarray(["d"] * 4),
        eq_ids=np.asarray(["e"] * 4),
        sig_ids=np.asarray(["s"] * 4),
        dataset_role=roles,
        channel_views=np.asarray(["rx_base"] * 4),
        sat_scenarios=np.asarray([scenario] * 4),
        satellite_seeds=seeds,
        overlay_applied=np.ones(4, dtype=bool),
        sample_ids=sample_ids,
        post_channel_iq_sha256=np.asarray(iq_hashes),
        overlay_ids=np.asarray(overlays),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def test_comparison_inner_loader_verifies_legacy_leo_without_source_indices(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    arrays, manifest, audit = load_comparison_inner_leo_cache(
        path,
        expected_scenario="leo_clear_weak",
        allowed_roles={"target_old", "target_new"},
    )
    assert "source_dataset_sha256" not in arrays
    assert "source_record_indices" not in arrays
    assert manifest["schema"] == "cvs_leo_weak_iq_cache_v1"
    assert audit["new_class_leo_iq_verified"] is True
    assert audit["exact_legacy_member_set_verified"] is True


def test_comparison_inner_loader_rejects_post_channel_iq_tamper(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["leo_weak_iq"] = payload["leo_weak_iq"].copy()
    payload["leo_weak_iq"][2, 0, 0] += 1.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="IQ digest mismatch"):
        load_comparison_inner_leo_cache(
            path,
            expected_scenario="leo_clear_weak",
            allowed_roles={"target_old", "target_new"},
        )


def test_comparison_set_loader_verifies_outer_hash_and_preserves_ids(tmp_path):
    scenario_paths = {}
    scenario_hashes = {}
    for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        cache = tmp_path / f"{scenario}.npz"
        _write_legacy_comparison_cache(cache, scenario)
        scenario_paths[scenario] = cache.name
        scenario_hashes[scenario] = sha256_file(cache)
    manifest_path = tmp_path / "set.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cvs_leo_weak_iq_cache_set_v1",
                "cache_npz_by_scenario": scenario_paths,
                "cache_sha256_by_scenario": scenario_hashes,
            }
        ),
        encoding="utf-8",
    )
    arrays, _manifest, audit = load_comparison_leo_cache_set(
        manifest_path,
        expected_scope="stage2_registered",
        allowed_roles={"target_old", "target_new"},
    )
    assert audit["status"] == "PASS_COMPARISON_SCOPE"
    assert {
        str(arrays[scenario]["sample_ids"][0]) for scenario in arrays
    } == {"legacy-sample-0"}
    assert all(
        audit["scenario_audits"][scenario][
            "verified_sample_ids_preserved_for_scenario_alignment"
        ]
        for scenario in arrays
    )
    assert _comparison_reference_arrays(arrays) is arrays["leo_clear_weak"]


def test_comparison_final_bundle_validator_keeps_strict_per_scenario_checks(
    monkeypatch,
):
    calls = []

    def fake_strict(
        package_root,
        *,
        detached_seal_path,
        expected_seal_sha256,
        scenario=None,
    ):
        calls.append(scenario)
        manifest = {"package": str(package_root), "version": 1}
        audit = {
            "seal": {"sha256": expected_seal_sha256},
            "sample_level_post_channel_iq_sha256_status": "PASS",
        }
        return (
            {scenario: {"support": np.asarray([scenario])}},
            {scenario: {"query": np.asarray([scenario])}},
            manifest,
            audit,
        )

    monkeypatch.setattr(
        "paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle."
        "_strict_stage2_bundle_loader",
        fake_strict,
    )
    support, query, manifest, audit = (
        load_verified_comparison_stage2_predictor_bundle(
            "bundle",
            detached_seal_path="seal.json",
            expected_seal_sha256="a" * 64,
        )
    )
    assert calls == ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
    assert tuple(support) == tuple(calls)
    assert tuple(query) == tuple(calls)
    assert manifest["version"] == 1
    assert audit["sample_level_post_channel_iq_sha256_status"] == "PASS"
    assert (
        audit["cross_scenario_physical_sample_token_disjointness"]
        == "EXEMPT_EXTERNAL_COMPARISON_BASELINE"
    )
