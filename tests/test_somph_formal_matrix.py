from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cvsrffi.somph_predictor_bundle import (
    ADV3B02_CHECKPOINT_SHA256 as PREDICTOR_ADV3B02_CHECKPOINT_SHA256,
)
from cvsrffi.somph_formal_matrix import (
    ADV3B02_CHECKPOINT_SHA256,
    CONFIRMATION_SEEDS,
    FORMAL_RECEIVERS,
    NEW_TX_IDS,
    SomphFormalMatrixError,
    build_formal_matrix,
    validate_formal_matrix,
)


def test_lightweight_matrix_checkpoint_lock_matches_predictor() -> None:
    assert ADV3B02_CHECKPOINT_SHA256 == PREDICTOR_ADV3B02_CHECKPOINT_SHA256


def _resign(payload: dict) -> None:
    from cvsrffi.somph_formal_matrix import _sha256_json

    for row in payload["rows"]:
        unsigned = dict(row)
        unsigned.pop("structural_row_sha256", None)
        row["structural_row_sha256"] = _sha256_json(unsigned)
    unsigned_payload = dict(payload)
    unsigned_payload.pop("matrix_sha256", None)
    payload["matrix_sha256"] = _sha256_json(unsigned_payload)


def _write_controller_policy(
    path: Path, *, controller_root: Path, forbidden_roots: list[Path]
) -> str:
    payload = {
        "schema": "cvs.phase2.somph_offline_controller_policy.v1",
        "allowed_offline_controller_root": str(controller_root.resolve()),
        "forbidden_phase2_roots": [
            str(value.resolve()) for value in forbidden_roots
        ],
        "formal_launch_authority": False,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_formal_matrix_has_exact_development_and_confirmation_coverage():
    payload = build_formal_matrix()
    assert payload["stage2b_row_count"] == 120
    assert payload["stage2c_structural_pair_cell_count"] == 360
    assert payload["stage2c_state_row_count"] == 720
    assert payload["confirmation_stage2b_row_count"] == 100
    assert payload["confirmation_stage2c_structural_pair_cell_count"] == 300
    assert payload["data_bound_stage2c_pair_count"] == 0
    assert len(payload["rows"]) == 840
    assert payload["receivers"] == list(FORMAL_RECEIVERS)
    assert payload["confirmation_seeds"] == list(CONFIRMATION_SEEDS)
    assert 713106 in payload["confirmation_seeds"]


def test_stage2c_before_after_registries_are_exact_nested_prefixes():
    payload = build_formal_matrix()
    selected = {
        (row["registration_state"], row["new_class_count"]): row
        for row in payload["rows"]
        if row["stage"] == "stage2c"
        and row["receiver"] == "20-1"
        and row["seed"] == 713101
        and row["k_shot"] == 10
    }
    for new_count in (5, 10, 20):
        before = selected[("before", new_count)]
        after = selected[("after", new_count)]
        assert before["registered_tx_ids"] == payload["old_tx_ids"]
        assert after["registered_tx_ids"] == payload["old_tx_ids"] + list(
            NEW_TX_IDS[:new_count]
        )


def test_matrix_rejects_missing_confirmation_row():
    payload = build_formal_matrix()
    payload["rows"].pop()
    with pytest.raises(SomphFormalMatrixError, match="coverage mismatch"):
        validate_formal_matrix(payload)


def test_matrix_rejects_clean_or_role_oracle_permission():
    for field in (
        "clean_sample_access",
        "phase2_query_role_oracle_access",
        "phase2_query_class_quota_access",
    ):
        payload = build_formal_matrix()
        payload["rows"][0][field] = True
        with pytest.raises(SomphFormalMatrixError, match="Phase2 contract drift"):
            validate_formal_matrix(payload)


def test_matrix_rejects_nonnested_after_registry():
    payload = build_formal_matrix()
    row = next(
        item
        for item in payload["rows"]
        if item["stage"] == "stage2c"
        and item["registration_state"] == "after"
        and item["new_class_count"] == 20
    )
    row["registered_tx_ids"][-1] = "unregistered-tx"
    with pytest.raises(SomphFormalMatrixError, match="nested/exact"):
        validate_formal_matrix(payload)


def test_matrix_digest_detects_controller_mutation():
    payload = build_formal_matrix()
    mutated = copy.deepcopy(payload)
    mutated["old_tx_ids"][0] = "changed"
    with pytest.raises(SomphFormalMatrixError):
        validate_formal_matrix(mutated)


def test_matrix_rejects_resigned_authority_alias_and_semantic_drift():
    for mutate in (
        lambda payload: payload.update({"launch_authority": True}),
        lambda payload: payload["rows"][0].update({"protocol_status": "PASS"}),
        lambda payload: payload.update({"stage2b_row_count": 999}),
        lambda payload: payload["rows"][0].update(
            {"seed_role": "independent_confirmation"}
        ),
        lambda payload: payload["rows"][0].update(
            {"support_selection_rule": "arbitrary"}
        ),
        lambda payload: payload["rows"][0].update(
            {"distinct_leo_weak_overlay_per_scenario": False}
        ),
        lambda payload: payload["rows"][0].update({"k_shot": True}),
    ):
        payload = build_formal_matrix()
        mutate(payload)
        _resign(payload)
        with pytest.raises(SomphFormalMatrixError):
            validate_formal_matrix(payload)


def test_stage2c_before_after_share_stable_pair_id_but_are_unbound():
    payload = build_formal_matrix()
    rows = [
        row
        for row in payload["rows"]
        if row["stage"] == "stage2c"
        and row["receiver"] == "20-1"
        and row["seed"] == 713101
        and row["k_shot"] == 10
        and row["new_class_count"] == 20
    ]
    assert {row["registration_state"] for row in rows} == {"before", "after"}
    assert len({row["pair_id"] for row in rows}) == 1
    assert {row["data_binding_status"] for row in rows} == {
        "UNBOUND_REQUIREMENT_TEMPLATE"
    }
    assert all(row["row_manifest_sha256"] is None for row in rows)


def test_only_locked_development_cell_can_select_hyperparameters():
    payload = build_formal_matrix()
    selectable = [row for row in payload["rows"] if row["selection_eligible"]]
    assert selectable
    assert {
        (row["receiver"], row["seed"], row["k_shot"]) for row in selectable
    } == {("20-1", 713101, 10)}
    assert all(not row["confirmation_aggregate_eligible"] for row in selectable)
    assert all(
        row["confirmation_aggregate_eligible"]
        for row in payload["rows"]
        if row["seed"] in CONFIRMATION_SEEDS
    )


def test_controller_policy_loader_enforces_locked_root_and_forbidden_overlap(
    tmp_path: Path,
):
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "build_cvs_somph_formal_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("somph_matrix_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller_root = tmp_path / "offline_controller"
    phase2_root = tmp_path / "predictor_package"
    controller_root.mkdir()
    phase2_root.mkdir()
    policy = tmp_path / "controller_policy.json"
    policy_sha = _write_controller_policy(
        policy,
        controller_root=controller_root,
        forbidden_roots=[phase2_root],
    )
    loaded_root, loaded_forbidden = module._load_controller_policy(
        policy, expected_sha256=policy_sha
    )
    assert loaded_root == controller_root.resolve()
    assert loaded_forbidden == [phase2_root.resolve()]


def test_cli_rejects_unregistered_policy_id(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "build_cvs_somph_formal_matrix.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(tmp_path / "formal_matrix.json"),
            "--controller-policy-id",
            "caller-forged-policy",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not (tmp_path / "formal_matrix.json").exists()


def test_payload_policy_mutation_does_not_poison_future_builds():
    first = build_formal_matrix()
    first["success_criteria"]["k10_target_old_overall_accuracy_min"] = 0.123
    first["resource_limits"]["adapter_parameters_max"] = 999_999
    first["development_lock"]["receiver"] = "8-8"
    second = build_formal_matrix()
    assert second["success_criteria"]["k10_target_old_overall_accuracy_min"] == 0.92
    assert second["resource_limits"]["adapter_parameters_max"] == 50_000
    assert second["development_lock"]["receiver"] == "20-1"
