from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_sprtdr import fit_before_after_locked


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d17_support_only_sprtdr.py"
)
SPEC = importlib.util.spec_from_file_location("run_d17_support_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _metric(values: dict[str, float]) -> dict:
    return {
        "overall_accuracy": sum(values.values()) / len(values),
        "min_class_accuracy": min(values.values()),
        "per_class_accuracy": values,
    }


def _fold(index: int = 0) -> dict:
    before = _metric({"old0": 0.4, "old1": 0.9})
    base_before = _metric({"old0": 0.4, "old1": 0.9})
    after = _metric({"old0": 0.5, "old1": 0.9})
    base_after = _metric({"old0": 0.4, "old1": 0.9})
    new = _metric({"new0": 0.5, "new1": 0.9})
    base_new = _metric({"new0": 0.4, "new1": 0.9})
    return {
        "fold": index,
        "before_old": before,
        "base_before_old": base_before,
        "after_old": after,
        "base_after_old": base_after,
        "after_new": new,
        "base_after_new": base_new,
        "joint": _metric(
            {"old0": 0.5, "old1": 0.9, "new0": 0.5, "new1": 0.9}
        ),
        "base_joint": _metric(
            {"old0": 0.4, "old1": 0.9, "new0": 0.4, "new1": 0.9}
        ),
        "H_old_new": 0.7,
        "base_H_old_new": 0.65,
        "old_forgetting": -0.05,
        "candidate_vs_z0_per_class_non_degraded": {
            "before_old": {"old0": True, "old1": True},
            "after_old": {"old0": True, "old1": True},
            "after_new": {"new0": True, "new1": True},
        },
        "old_score_bitwise_locked": True,
        "old_pairs": [[0, 1]],
        "new_rivals": [[0, -1], [-1, -1]],
        "floor_handles": {"old": ["old0"], "new": ["new0"]},
    }


def test_candidate_grid_and_k1_canonical_true_zero() -> None:
    k1 = runner._candidates(1)
    assert len(k1) == 1
    assert k1[0].candidate_id == "d17_z0_true_zero_base"
    assert k1[0].force_zero is True
    assert k1[0].rank == 0
    for k_shot in (5, 10):
        values = runner._candidates(k_shot)
        assert [value.margin_band for value in values] == [0.0, 0.02, 0.04]
        assert [value.candidate_id for value in values] == [
            "d17_z0_true_zero_base",
            "d17_sprtdr_mb002",
            "d17_sprtdr_mb004",
        ]
        lock = runner._candidate_lock(values, k_shot=k_shot)
        assert lock == runner._candidate_lock(values, k_shot=k_shot)


def _rows(k_shot: int, prefix: str = "x") -> dict[str, np.ndarray]:
    labels = np.repeat(np.asarray(["old0", "old1"]), k_shot)
    ranks = np.tile(np.arange(k_shot, dtype=np.int64), 2)
    return {
        "labels": labels,
        "ranks": ranks,
        "tokens": np.asarray(
            [f"{prefix}_physical_{index}" for index in range(len(labels))]
        ),
        "hashes": np.asarray(
            [f"{index + (100 if prefix != 'x' else 0):064x}" for index in range(len(labels))]
        ),
    }


def test_exact_k_means_unique_independent_physical_observations() -> None:
    for k_shot in (1, 5, 10):
        result = runner._validate_exact_k_rows(
            _rows(k_shot), k_shot=k_shot, scenario="leo_clear_weak"
        )
        assert result["exact_k_pass"] is True
        assert result["physical_support_count"] == 2 * k_shot
    duplicate = _rows(5)
    duplicate["tokens"][1] = duplicate["tokens"][0]
    try:
        runner._validate_exact_k_rows(
            duplicate, k_shot=5, scenario="leo_clear_weak"
        )
    except runner.D17RunnerError:
        pass
    else:
        raise AssertionError("duplicate physical observation increased K")


def test_gate_is_per_fold_per_class_and_requires_both_floor_roles_strict() -> None:
    result = {"folds": [_fold(index) for index in range(5)]}
    passed = runner._scenario_gate(result, force_zero=False)
    assert passed["all_folds_gate_pass"] is True
    assert passed["old_floor_strict_gain_in_every_fold"] is True
    assert passed["new_floor_strict_gain_in_every_fold"] is True

    no_new_floor_gain = copy.deepcopy(result)
    no_new_floor_gain["folds"][3]["after_new"]["per_class_accuracy"][
        "new0"
    ] = 0.4
    failed = runner._scenario_gate(no_new_floor_gain, force_zero=False)
    assert failed["all_folds_gate_pass"] is False
    assert failed["new_floor_strict_gain_in_every_fold"] is False

    one_class_degraded = copy.deepcopy(result)
    one_class_degraded["folds"][2][
        "candidate_vs_z0_per_class_non_degraded"
    ]["after_new"]["new1"] = False
    assert (
        runner._scenario_gate(one_class_degraded, force_zero=False)[
            "all_folds_gate_pass"
        ]
        is False
    )


def test_old_score_lock_does_not_mask_old_decision_forgetting() -> None:
    result = {"folds": [_fold(index) for index in range(5)]}
    forgotten = copy.deepcopy(result)
    fold = forgotten["folds"][4]
    fold["after_old"]["per_class_accuracy"]["old1"] = 0.8
    fold["old_forgetting"] = 0.05
    assert fold["old_score_bitwise_locked"] is True
    gate = runner._scenario_gate(forgotten, force_zero=False)
    assert gate["all_folds_gate_pass"] is False
    assert gate["folds"][4]["old_decision_forgetting_pass"] is False
    assert gate["folds"][4]["old_score_columns_bitwise_locked"] is True


def test_all_positive_fail_selects_canonical_true_zero() -> None:
    candidates = runner._candidates(10)
    rows = [
        {
            "candidate_id": value.candidate_id,
            "force_zero": value.force_zero,
            "all_scenario_all_fold_gate_pass": False,
            "margin_band": value.margin_band,
            "worst_old_floor": 0.0,
            "worst_new_floor": 0.0,
            "worst_old_decision_forgetting": 0.0,
            "mean_H_old_new": 0.0,
            "mean_joint_accuracy": 0.0,
        }
        for value in candidates
    ]
    selected, passed = runner._select_candidate(rows, candidates)
    assert selected == "d17_z0_true_zero_base"
    assert passed is False


def test_cross_scenario_physical_and_parent_hash_disjointness() -> None:
    rows = {
        scenario: _rows(5, prefix=f"s{index}")
        for index, scenario in enumerate(runner.FORMAL_LEO_WEAK_SCENARIOS)
    }
    for index, scenario in enumerate(runner.FORMAL_LEO_WEAK_SCENARIOS):
        rows[scenario]["hashes"] = np.asarray(
            [f"{index * 100 + value:064x}" for value in range(10)]
        )
    assert runner._cross_scenario_disjointness(rows)[
        "all_pairwise_disjoint"
    ] is True
    rows[runner.FORMAL_LEO_WEAK_SCENARIOS[1]]["hashes"][0] = rows[
        runner.FORMAL_LEO_WEAK_SCENARIOS[0]
    ]["hashes"][0]
    try:
        runner._cross_scenario_disjointness(rows)
    except runner.D17RunnerError as exc:
        assert "reused across LEO scenarios" in str(exc)
    else:
        raise AssertionError("cross-scenario received-IQ reuse was accepted")


def test_runner_is_sealed_support_only_without_query_or_formal_authority_cli() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from run_d14_support_only_pairwise_fisher_guard import" in source
    assert "_load_enrollment" in source
    assert "_payload_rows" in source
    assert "_build_feature_artifact" in source
    assert "--before-seal-sha256" in source
    assert "--after-seal-sha256" in source
    assert "--query" not in source
    assert "--truth" not in source
    assert "--scorer" not in source
    assert '"formal_launch_authority": False' in source
    assert "development_select only" in source
    assert 'output / "training_log.jsonl"' in source
    assert '"after_state_audit"' in source
    assert '"sprtdr_head_upper_bound_macs"' in source
    assert '"peak_python_tracemalloc_bytes"' in source


def _artifact(features: np.ndarray):
    iq = np.stack(
        [
            np.full((2, 12), float(index + 1), dtype=np.float32)
            for index in range(len(features))
        ]
    )
    hashes = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    ]
    cursor = 0

    def extract_single_received_iq(_):
        nonlocal cursor
        value = features[cursor : cursor + 1]
        cursor += 1
        return value

    return _build_runtime_authorized_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"pid_{index}" for index in range(len(features))],
        parent_received_iq_sha256=hashes,
        sealed_runtime_sha256="a" * 64,
        feature_code_sha256="b" * 64,
        sealed_phase1_checkpoint_sha256="c" * 64,
        extract_single_received_iq=extract_single_received_iq,
        operator_id="base",
        view_seed=0,
    )


def _k1_after_state():
    before_features = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    after_features = np.concatenate(
        [before_features, np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)]
    )
    fitted = fit_before_after_locked(
        _artifact(before_features),
        np.asarray(["oa", "ob"]),
        np.asarray([0, 0], dtype=np.int64),
        _artifact(after_features),
        np.asarray(["oa", "ob", "nx"]),
        np.asarray([0, 0, 0], dtype=np.int64),
        k_shot=1,
        hyperparameters=runner._candidates(1)[0],
    )
    return fitted.after_state


def test_state_npz_metadata_commit_roundtrip_is_bitwise_and_no_overwrite(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "selected_state"
    audit = runner._write_state_roundtrip(
        state_dir, state=_k1_after_state()
    )
    assert {value.name for value in state_dir.iterdir()} == {
        "state.npz",
        "metadata.json",
        "COMMIT",
    }
    assert audit["serialization"] == "canonical_npz_no_pickle"
    assert audit["semantic_state_validation_verified"] is True
    assert audit["state_sha_roundtrip_verified"] is True
    assert audit["fixed_probe_score_bitwise_verified"] is True
    assert audit["serialized_state_total_bytes"] == sum(
        (state_dir / name).stat().st_size
        for name in ("state.npz", "metadata.json", "COMMIT")
    )
    try:
        runner._load_state(
            state_dir, expected_commit_sha256="0" * 64
        )
    except runner.D17RunnerError as exc:
        assert "COMMIT SHA mismatch" in str(exc)
    else:
        raise AssertionError("incorrect external COMMIT SHA was accepted")
    try:
        runner._write_state_roundtrip(state_dir, state=_k1_after_state())
    except runner.D17RunnerError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("sealed state output was overwritten")


def test_state_serialization_never_uses_pickle() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "np.savez_compressed" in source
    assert "allow_pickle=False" in source
    assert "import pickle" not in source
