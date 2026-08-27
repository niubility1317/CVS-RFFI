from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi.sf_tapft_slim_matrix import build_row_config, validate_slim_matrix
from cvsrffi.target_only_progressive_adapt import SFTAPFTConfig


def _matrix() -> dict:
    return {
        "schema": "cvs.sf_tapft.slim_matrix.v1",
        "run_id": "slim-test",
        "shared_config": {
            "method": "sf_tapft_v1",
            "permission": "DIAGNOSTIC_NON_FORMAL",
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "capsule",
            "split_id": "split",
            "checkpoint_path": "/checkpoint",
            "support_path": "/support",
            "phase1_bundle": {"package_root": "/bundle"},
        },
        "base_sf_tapft": {
            "trainability_profile": "p1_head_norm",
            "norm_scope": "all",
            "norm_affine": "weight_bias",
            "phase_steps": [4500, 0, 0],
            "scheduler_reference_steps": 0,
        },
        "rows": [
            {
                "row_id": "S02",
                "candidate_id": "T3_ONLY",
                "gpu": 1,
                "overrides": {"norm_scope": "t3"},
            }
        ],
    }


def test_matrix_row_merges_only_declared_sf_tapft_overrides() -> None:
    matrix = validate_slim_matrix(_matrix())
    config, gpu = build_row_config(matrix, "S02")
    assert gpu == 1
    assert config["candidate_id"] == "T3_ONLY"
    assert config["sf_tapft"] == {
        "trainability_profile": "p1_head_norm",
        "norm_scope": "t3",
        "norm_affine": "weight_bias",
        "phase_steps": [4500, 0, 0],
        "scheduler_reference_steps": 0,
    }
    assert "rows" not in config


def test_matrix_rejects_duplicate_rows_and_unknown_overrides() -> None:
    duplicate = _matrix()
    duplicate["rows"].append(dict(duplicate["rows"][0]))
    with pytest.raises(ValueError, match="row_id"):
        validate_slim_matrix(duplicate)

    unknown = _matrix()
    unknown["rows"][0]["overrides"] = {"query_path": "/forbidden"}
    with pytest.raises(ValueError, match="override"):
        validate_slim_matrix(unknown)


def test_s15plus_release_matrix_resolves_all_report_candidates() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827.json"
    )
    matrix = validate_slim_matrix(json.loads(source.read_text(encoding="utf-8")))
    assert [row["row_id"] for row in matrix["rows"]] == ["F1", "F2", "F3", "Q2A", "Q2B"]
    for row_id in ("F1", "F2", "F3", "Q2A", "Q2B"):
        config, _ = build_row_config(matrix, row_id)
        normalized = dict(config["sf_tapft"])
        normalized["phase_steps"] = tuple(normalized["phase_steps"])
        normalized["validation_steps"] = tuple(normalized["validation_steps"])
        normalized["norm_rules"] = tuple(tuple(rule) for rule in normalized["norm_rules"])
        parsed = SFTAPFTConfig(**normalized)
        if row_id.startswith("Q2"):
            assert parsed.phase_steps == (4500, 0, 0)
            assert parsed.scheduler_reference_steps == 4500
            assert parsed.validation_steps == ()
        else:
            assert parsed.phase_steps == (300, 0, 0)
            assert parsed.scheduler_reference_steps == 300
            assert parsed.validation_steps[-1] == 300
