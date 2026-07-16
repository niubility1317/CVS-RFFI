from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_sparse_pairwise_fisher_guard import (
    fit_before_after_locked,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d14_support_only_pairwise_fisher_guard.py"
)
SPEC = importlib.util.spec_from_file_location("run_d14_support_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _artifact(rows: np.ndarray, seed: int = 5):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(len(rows), 2, 8)).astype(np.float32)
    parents = [
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for value in iq
    ]
    cursor = 0

    def extract(_: np.ndarray) -> np.ndarray:
        nonlocal cursor
        result = rows[cursor : cursor + 1]
        cursor += 1
        return result

    return _build_runtime_authorized_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"sid_{seed}_{index}" for index in range(len(iq))],
        parent_received_iq_sha256=parents,
        sealed_runtime_sha256="a" * 64,
        feature_code_sha256="b" * 64,
        sealed_phase1_checkpoint_sha256="c" * 64,
        extract_single_received_iq=extract,
        operator_id="base",
        view_seed=0,
    )


def _small_state(candidate_index: int = 1):
    rng = np.random.default_rng(44)
    old_classes = ("old0", "old1")
    new_classes = ("new0",)
    k = 3
    old_rows = rng.normal(size=(len(old_classes) * k, 12)).astype(np.float32)
    new_rows = rng.normal(size=(len(new_classes) * k, 12)).astype(np.float32)
    old_labels = np.repeat(np.asarray(old_classes), k)
    new_labels = np.repeat(np.asarray(new_classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(old_classes))
    joint_ranks = np.concatenate(
        [ranks, np.tile(np.arange(k, dtype=np.int64), len(new_classes))]
    )
    return fit_before_after_locked(
        _artifact(old_rows),
        old_labels,
        ranks,
        _artifact(np.concatenate([old_rows, new_rows])),
        np.concatenate([old_labels, new_labels]),
        joint_ranks,
        k_shot=k,
        hyperparameters=runner._candidates()[candidate_index],
    ).after_state


def test_candidate_grid_is_base_only_unified_and_has_true_zero() -> None:
    candidates = runner._candidates()
    assert len(candidates) == 4
    assert all(value.operator_id == "base" for value in candidates)
    zero = [value for value in candidates if value.force_zero]
    assert len(zero) == 1
    assert zero[0].candidate_id == "d14_z0_true_zero_base"
    assert zero[0].gamma_old == 0.0
    assert zero[0].gamma_new == 0.0
    assert zero[0].max_old_edges == 0
    assert all(
        value.select_band_old == 0.20
        for value in candidates
        if not value.force_zero
    )


def test_unified_lock_is_deterministic_and_records_independent_select_band() -> None:
    first = runner._candidate_lock(runner._candidates())
    second = runner._candidate_lock(runner._candidates())
    assert first == second
    assert len(first["lock_sha256"]) == 64
    assert "all_three_scenarios" in first["selection_scope"]
    assert first["operator_scope"].startswith("base_only_mvp")
    assert all(
        "band_select_old" in row and "band_old" in row
        for row in first["candidates"]
    )


def test_actual_state_serialization_roundtrip_and_resource_audit(tmp_path: Path) -> None:
    audit = runner._write_state(tmp_path, stem="state", state=_small_state())
    assert audit["serialized_state_under_256kib"] is True
    assert audit["content_verified_after_write"] is True
    assert audit["state_rebuilt_and_prediction_bitwise_verified"] is True
    assert len(audit["npz_sha256"]) == 64
    assert len(audit["metadata_sha256"]) == 64
    assert (tmp_path / "state.npz").exists()
    assert (tmp_path / "state.json").exists()


def test_all_positive_fail_selects_and_commits_true_zero(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": value.candidate_id,
            "force_zero": value.force_zero,
            "all_scenario_gate_pass": False,
        }
        for value in runner._candidates()
    ]
    selected, promotion = runner._select_candidate(rows, runner._candidates())
    assert selected == "d14_z0_true_zero_base"
    assert promotion is False
    state = _small_state(candidate_index=0)
    assert len(state.old_edge_pairs) == 0
    assert not np.any(state.new_rivals >= 0)
    binding = runner._write_state(tmp_path, stem="state_leo_clear_weak_after_k10", state=state)
    commit = {
        "status": "SUPPORT_ONLY_D14_DIAGNOSTIC_NOT_SELECTED_NO_QUERY_OPEN",
        "promotion_ready_for_single_query_candidate": False,
        "support_candidate_gate_pass_before_authority": False,
        "formal_launch_authority": False,
        "query_opened": False,
        "selected_candidate_id": selected,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
        "combined_feature_code_sha256": state.feature_code_sha256,
        "state_sha256": {"leo_clear_weak:after:k10": binding},
    }
    commit_path = tmp_path / "COMMIT.json"
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    commit_sha = hashlib.sha256(commit_path.read_bytes()).hexdigest()
    rebuilt = runner.load_committed_state(
        tmp_path,
        state_key="leo_clear_weak:after:k10",
        expected_commit_sha256=commit_sha,
        require_formal_promotion=False,
    )
    assert rebuilt.hyperparameters.force_zero is True
    assert rebuilt.hyperparameters.gamma_old == 0.0
    assert rebuilt.hyperparameters.gamma_new == 0.0
    assert len(rebuilt.old_edge_pairs) == 0
    assert not np.any(rebuilt.new_rivals >= 0)


def test_committed_state_rejects_diagnostic_or_candidate_mismatch(
    tmp_path: Path,
) -> None:
    state = _small_state(candidate_index=1)
    binding = runner._write_state(
        tmp_path, stem="state_leo_clear_weak_after_k10", state=state
    )
    commit = {
        "status": "SUPPORT_ONLY_D14_DIAGNOSTIC_SELECTED_NO_PROMOTION_NO_QUERY_OPEN",
        "promotion_ready_for_single_query_candidate": False,
        "support_candidate_gate_pass_before_authority": True,
        "formal_launch_authority": False,
        "query_opened": False,
        "selected_candidate_id": "d14_z0_true_zero_base",
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
        "combined_feature_code_sha256": state.feature_code_sha256,
        "state_sha256": {"leo_clear_weak:after:k10": binding},
    }
    commit_path = tmp_path / "COMMIT.json"
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    commit_sha = hashlib.sha256(commit_path.read_bytes()).hexdigest()
    try:
        runner.load_committed_state(
            tmp_path,
            state_key="leo_clear_weak:after:k10",
            expected_commit_sha256=commit_sha,
        )
    except runner.D14RunnerError as exc:
        assert "formal promotion authority" in str(exc)
    else:
        raise AssertionError("diagnostic COMMIT loaded as formal deployment state")
    try:
        runner.load_committed_state(
            tmp_path,
            state_key="leo_clear_weak:after:k10",
            expected_commit_sha256=commit_sha,
            require_formal_promotion=False,
        )
    except runner.D14RunnerError as exc:
        assert "metadata binding drift" in str(exc)
    else:
        raise AssertionError("candidate-mismatched diagnostic state loaded")


def test_preopen_authority_conflict_forces_diagnostic() -> None:
    bad = {
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "control_state": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
    }
    passed, audit = runner._authority_pass(
        bad,
        bad,
        authority_evidence=None,
        expected_authority_evidence_sha256=None,
        before_package_root_sha256="a" * 64,
        after_package_root_sha256="b" * 64,
        before_seal_sha256="c" * 64,
        after_seal_sha256="d" * 64,
    )
    assert passed is False
    assert audit["preopen_formal_authority_pass"] is False


def test_runner_source_has_no_query_or_scorer_package_loader() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_verified_somph_predictor_bundle" in source
    assert "query_root" not in source
    assert "scorer_root" not in source
    assert "_select_artifact" not in source
    assert "_ARTIFACT_TOKEN" not in source
    assert "operator_id=\"base\"" in source
    assert "--before-seal-sha256" in source
    assert "--after-seal-sha256" in source
    assert "expected_seal_sha256=_sha256_file(seal)" not in source
    assert "development_select only" in source
