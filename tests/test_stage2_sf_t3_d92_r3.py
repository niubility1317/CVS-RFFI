from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


def _candidate_module():
    return importlib.import_module("cvsrffi.stage2_sf_t3_d92_r3")


def _support_fixture(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260828)
    identity = rng.normal(0.0, 0.02, (len(labels), 160)).astype(np.float32)
    fft = rng.normal(0.0, 0.02, (len(labels), 96)).astype(np.float32)
    for column, class_id in enumerate(dict.fromkeys(labels.tolist())):
        mask = labels == class_id
        identity[mask, column] += 4.0
        fft[mask, column] += 2.0
    return identity, fft


def test_r3_dual_delta_persists_only_t3_norm_without_target_head() -> None:
    module = _candidate_module()
    anchor = {
        "model.t3.norm.weight": torch.tensor([1.0, 2.0]),
        "model.t3.norm.bias": torch.tensor([0.0, 1.0]),
        "head.weight": torch.tensor([100.0]),
    }
    first = {
        "model.t3.norm.weight": torch.tensor([3.0, 4.0]),
        "model.t3.norm.bias": torch.tensor([2.0, 3.0]),
        "head.weight": torch.tensor([200.0]),
    }
    second = {
        "model.t3.norm.weight": torch.tensor([1.0, 6.0]),
        "model.t3.norm.bias": torch.tensor([-2.0, 5.0]),
        "head.weight": torch.tensor([-300.0]),
    }

    delta, audit = module.aggregate_r3_t3_norm_delta(anchor, (first, second))

    assert set(delta) == {
        "model.t3.norm.weight",
        "model.t3.norm.bias",
    }
    assert torch.equal(delta["model.t3.norm.weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(delta["model.t3.norm.bias"], torch.tensor([0.0, 3.0]))
    assert audit["dual_delta_count"] == 2
    assert audit["temporary_target_head_persisted"] is False
    assert audit["deployment_delta_parameter_names"] == [
        "model.t3.norm.bias",
        "model.t3.norm.weight",
    ]


def test_r3_dual_delta_rejects_a_non_dual_ensemble() -> None:
    module = _candidate_module()
    anchor = {
        "model.t3.norm.weight": torch.ones(2),
        "model.t3.norm.bias": torch.zeros(2),
    }

    with pytest.raises(ValueError, match="exactly two"):
        module.aggregate_r3_t3_norm_delta(anchor, (anchor,))


def test_d92_in_loop_crossfit_is_support_only_and_reports_robust_risk() -> None:
    module = _candidate_module()
    class_ids = (101, 205, 309, 402, 518, 623, 711, 824)
    labels = np.repeat(np.asarray(class_ids, dtype=np.int64), 4)
    identity, fft = _support_fixture(labels)

    result = module.crossfit_d92_support_risk(
        identity,
        fft,
        labels,
        class_ids=class_ids,
        old_class_ids=class_ids[:6],
        folds=2,
        seed=713101,
        device="cpu",
    )

    assert len(result.folds) == 2
    assert np.isfinite(result.total)
    assert result.macro_nll >= 0.0
    assert result.class_tail_nll >= result.macro_nll
    assert 0.0 <= result.class_floor_error <= 1.0
    assert result.old_new_balance >= 0.0
    covered = []
    for row in result.folds:
        assert set(row.train_indices).isdisjoint(row.heldout_indices)
        covered.extend(row.heldout_indices)
        assert row.fit_support_rows == len(row.train_indices)
        assert row.heldout_support_rows == len(row.heldout_indices)
    assert sorted(covered) == list(range(len(labels)))
    assert result.audit["support_only"] is True
    assert result.audit["query_rows_used"] == 0
    assert result.audit["query_truth_opened"] is False
    assert result.audit["d92_method_lock"] == "D92-E0-NORF32"
    assert result.audit["rf32_used"] is False


def test_d92_in_loop_risk_is_invariant_to_class_label_permutation() -> None:
    module = _candidate_module()
    class_ids = (10, 20, 30, 40, 50, 60, 70, 80)
    labels = np.repeat(np.asarray(class_ids, dtype=np.int64), 4)
    identity, fft = _support_fixture(labels)
    permutation = {10: 800, 20: 300, 30: 700, 40: 100, 50: 600, 60: 200, 70: 500, 80: 400}
    permuted_labels = np.asarray([permutation[int(value)] for value in labels], dtype=np.int64)
    permuted_registry = tuple(permutation[value] for value in class_ids)

    original = module.crossfit_d92_support_risk(
        identity,
        fft,
        labels,
        class_ids=class_ids,
        old_class_ids=class_ids[:6],
        folds=2,
        seed=392002,
        device="cpu",
    )
    permuted = module.crossfit_d92_support_risk(
        identity,
        fft,
        permuted_labels,
        class_ids=permuted_registry,
        old_class_ids=permuted_registry[:6],
        folds=2,
        seed=392002,
        device="cpu",
    )

    assert permuted.total == pytest.approx(original.total, abs=1.0e-7)
    assert permuted.macro_nll == pytest.approx(original.macro_nll, abs=1.0e-7)
    assert permuted.class_tail_nll == pytest.approx(original.class_tail_nll, abs=1.0e-7)
    assert permuted.class_floor_error == pytest.approx(original.class_floor_error, abs=1.0e-7)
    assert permuted.old_new_balance == pytest.approx(original.old_new_balance, abs=1.0e-7)


def test_candidate_spec_exposes_runner_contract_and_protocol_audit() -> None:
    module = _candidate_module()

    spec = module.build_candidate_spec()

    assert spec["candidate_id"] == "R3_T3NORM_D92_INLOOP"
    assert spec["runner_entrypoints"] == {
        "aggregate_delta": "aggregate_r3_t3_norm_delta",
        "support_risk": "crossfit_d92_support_risk",
    }
    assert spec["adaptation"]["ensemble"] == "R3_DUAL_DELTA"
    assert spec["adaptation"]["persistent_parameter_names"] == [
        "model.t3.norm.bias",
        "model.t3.norm.weight",
    ]
    assert spec["adaptation"]["temporary_target_head_persisted"] is False
    assert spec["registration"]["method_lock"] == "D92-E0-NORF32"
    assert spec["registration"]["rf32_used"] is False
    assert spec["risk"]["support_only"] is True
    assert spec["risk"]["label_permutation_invariant"] is True
    assert spec["protocol_audit"]["query_rows_used"] == 0
    assert spec["protocol_audit"]["query_truth_opened"] is False
    assert spec["protocol_audit"]["query_role_opened"] is False
