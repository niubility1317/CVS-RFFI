from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d62_crossfitted_fisher_row_splice.py"
SPEC = importlib.util.spec_from_file_location("probe_d62_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d62 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d62)


def test_partitions_are_class_balanced_exact_once() -> None:
    labels = np.repeat(np.arange(4), 5)
    folds = d62._partitions(labels, 4, 5)
    assert len(folds) == 5
    assert all(np.array_equal(labels[fold], np.arange(4)) for fold in folds)
    assert sorted(int(value) for fold in folds for value in fold) == list(range(20))


def test_partitions_reject_unequal_k() -> None:
    with pytest.raises(d62.D62ProbeError, match="symmetric"):
        d62._partitions(np.asarray([0, 0, 1, 1, 1]), 2, 2)


def _perfect_scores(k: int = 4, classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    truth = np.tile(np.arange(classes), (k, 1))
    scores = np.zeros((k, classes, classes), dtype=np.float64)
    for fold in range(k):
        scores[fold, np.arange(classes), np.arange(classes)] = 2.0
    return scores, truth


def test_gate_rejects_when_no_strict_improvement() -> None:
    base, truth = _perfect_scores()
    gate = d62._pareto_gate(base, base.copy(), truth)
    assert not np.any(gate["initial_accept"])
    assert gate["exact_fallback"] is True
    assert gate["status"] == "no_row_accepted_exact_d46_fallback"


def test_gate_accepts_coordinate_with_tp_gain_and_no_fp_cost() -> None:
    base, truth = _perfect_scores()
    base[0, 0] = [0.0, 1.0, 0.0]
    residual = base.copy()
    residual[0, 0, 0] = 2.0
    gate = d62._pareto_gate(base, residual, truth)
    assert gate["final_accept"].tolist() == [True, False, False]
    assert gate["base_positive"][0] == 3
    assert gate["coordinate_positive"][0] == 4


def test_gate_rejects_tp_gain_that_adds_false_positive() -> None:
    base, truth = _perfect_scores()
    base[0, 0] = [0.0, 1.0, 0.0]
    residual = base.copy()
    residual[0, 0, 0] = 2.0
    residual[0, 1, 0] = 3.0
    gate = d62._pareto_gate(base, residual, truth)
    assert gate["initial_accept"][0] is np.False_ or not gate["initial_accept"][0]


def test_joint_interaction_can_trigger_atomic_fallback() -> None:
    # A deterministic search over small score perturbations finds a case where
    # each row is individually Pareto-safe but their simultaneous replacement
    # creates a harmful argmax interaction.
    rng = np.random.default_rng(19)
    truth = np.tile(np.arange(3), (4, 1))
    found = None
    for _ in range(10000):
        base = rng.normal(size=(4, 3, 3))
        residual = base + rng.normal(scale=1.0, size=base.shape)
        gate = d62._pareto_gate(base, residual, truth)
        if np.any(gate["initial_accept"]) and not gate["atomic_safe"]:
            found = gate
            break
    assert found is not None
    assert not np.any(found["final_accept"])
    assert found["status"] == "joint_gate_atomic_exact_d46_fallback"


def test_class_permutation_equivariance() -> None:
    rng = np.random.default_rng(23)
    truth = np.tile(np.arange(4), (5, 1))
    base = rng.normal(size=(5, 4, 4))
    residual = base + rng.normal(scale=0.4, size=base.shape)
    first = d62._pareto_gate(base, residual, truth)
    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_truth = permutation[truth]
    permuted_base = base[:, inverse][:, :, inverse]
    permuted_residual = residual[:, inverse][:, :, inverse]
    second = d62._pareto_gate(permuted_base, permuted_residual, permuted_truth[:, inverse])
    np.testing.assert_array_equal(second["final_accept"], first["final_accept"][inverse])


def test_formula_has_no_role_or_tunable_weight() -> None:
    lowered = d62.FORMULA.lower()
    assert "tp1_c" in lowered and "fp1_c" in lowered
    assert "alpha" not in lowered and "role" not in lowered


def _confirmation_runner_stub():
    def registered_handles(manifest):
        return tuple(manifest["registered"])

    def original_guard(_before, _after):
        raise AssertionError("development guard must be replaced")

    return SimpleNamespace(
        _require_d42_development_cell=original_guard,
        legacy=SimpleNamespace(_registered_handles=registered_handles),
        D42_DEVELOPMENT_RECEIVER="20-1",
        D42_DEVELOPMENT_NEW_CLASS_COUNT=5,
        D25RunnerError=RuntimeError,
    )


def _confirmation_cell(seed=713102):
    before = {
        "receiver": "20-1",
        "seed": seed,
        "k_shot": 10,
        "registered": ("old0", "old1"),
    }
    after = {
        **before,
        "registered": (*before["registered"], "n0", "n1", "n2", "n3", "n4"),
    }
    return before, after


def test_confirmation_guard_accepts_only_exact_preregistered_cell() -> None:
    runner = _confirmation_runner_stub()
    original = d62._install_confirmation_cell_guard(runner, 713102)
    assert original is not None
    runner._require_d42_development_cell(*_confirmation_cell())
    with pytest.raises(RuntimeError, match="D62 confirmation cell"):
        runner._require_d42_development_cell(*_confirmation_cell(713103))


def test_confirmation_guard_rejects_unregistered_seed_before_install() -> None:
    runner = _confirmation_runner_stub()
    with pytest.raises(d62.D62ProbeError, match="not preregistered"):
        d62._install_confirmation_cell_guard(runner, 713101)
    assert d62._install_confirmation_cell_guard(runner, None) is None


def test_confirmation_probe_explicitly_audits_centering_roundoff() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ALLOW_FP32_CENTERING_ARGMAX_DRIFT = True" in source
    assert '"fp32_centering_argmax_drift_allowed"' in source
    assert '"fp32_centering_audit"' in source
    assert '"non_equivalent_fit_count"' in source


def test_built_fit_records_all_outer_and_inner_components() -> None:
    def base_fit(rows, labels, class_count, k_shot):
        values = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        means = np.stack(
            [values[y == index].mean(axis=0) for index in range(class_count)]
        )
        return (
            means.astype(np.float32),
            (-0.5 * np.sum(means**2, axis=1)).astype(np.float32),
            {
                "unit_covariance_fallback": True,
                "covariance_equation_residual_max": 0.0,
            },
        )

    stub = SimpleNamespace(
        _fit_equal_prior_lda=base_fit,
        FEATURE_DIM=9,
        ENERGY_EPSILON=1.0e-12,
        BLOCK_SLICES=(slice(0, 3), slice(3, 6), slice(6, 9)),
    )
    rng = np.random.default_rng(31)
    means = rng.normal(size=(3, 9))
    rows = np.concatenate(
        [means[index] + rng.normal(scale=0.3, size=(3, 9)) for index in range(3)]
    )
    labels = np.repeat(np.arange(3), 3)
    fit, records = d62.build_d62_fit(stub)
    coefficient, intercept, audit = fit(rows, labels, 3, 3)
    assert coefficient.shape == (3, 9) and intercept.shape == (3,)
    assert np.isfinite(coefficient).all() and np.isfinite(intercept).all()
    assert audit["d62_actual_k"] == 3
    assert len(records) == 2 * (3 + 1)
    assert {record["component"] for record in records} == {"full", "block3"}
