from __future__ import annotations

import numpy as np
import torch

from cvsrffi.stage2_sf_erbt_four_state import fit_erbt_registration_pair
from cvsrffi.stage2_sf_r5d92 import (
    R5D92FoldMetrics,
    choose_r5d92_candidate,
    complementary_leave_pair_splits,
    covariance_block_audit,
    evaluate_d92_crossfit_from_fold_features,
    run_r5d92_support_selection,
)
from cvsrffi.target_only_progressive_adapt import SFTAPFTConfig, TargetOnlyAdaptationDataset
from test_sf_tapft_pace import _GeneralCacheableBackbone


def test_complementary_leave_pair_splits_hold_each_k10_row_once() -> None:
    labels = np.repeat(np.arange(6, dtype=np.int64), 10)

    splits = complementary_leave_pair_splits(labels)

    assert len(splits) == 5
    held_counts = np.zeros(len(labels), dtype=np.int64)
    for split in splits:
        assert len(split.fit_indices) == 48
        assert len(split.heldout_indices) == 12
        assert np.bincount(labels[list(split.fit_indices)], minlength=6).tolist() == [8] * 6
        assert np.bincount(labels[list(split.heldout_indices)], minlength=6).tolist() == [2] * 6
        assert set(split.fit_indices).isdisjoint(split.heldout_indices)
        held_counts[list(split.heldout_indices)] += 1
    assert held_counts.tolist() == [1] * 60


def _metrics(
    candidate_id: str,
    *,
    old_pre: float,
    old_post: float,
    new_acc: float,
    fold_h: tuple[float, ...],
    min_old: float = 0.5,
    min_new: float = 0.5,
    positive_definite: bool = True,
) -> R5D92FoldMetrics:
    return R5D92FoldMetrics(
        candidate_id=candidate_id,
        old_pre=old_pre,
        old_post=old_post,
        new_acc=new_acc,
        harmonic_old_new=(2.0 * old_post * new_acc / (old_post + new_acc)),
        forgetting=old_pre - old_post,
        min_old=min_old,
        min_new=min_new,
        old_to_new=0.1,
        old_to_wrong_old=0.1,
        fold_h=fold_h,
        covariance_positive_definite=positive_definite,
        covariance_condition_number=10.0,
        identity_covariance_trace=1.0,
        fft_covariance_trace=2.0,
    )


def test_candidate_selection_applies_hard_constraints_then_lcb_h() -> None:
    baseline = _metrics(
        "J1_R0_D92", old_pre=0.80, old_post=0.70, new_acc=0.60, fold_h=(0.62,) * 5
    )
    infeasible = _metrics(
        "R5_BAD_FORGETTING",
        old_pre=0.86,
        old_post=0.71,
        new_acc=0.70,
        fold_h=(0.70,) * 5,
    )
    safer = _metrics(
        "R5_SAFE",
        old_pre=0.82,
        old_post=0.73,
        new_acc=0.66,
        fold_h=(0.64, 0.65, 0.66, 0.67, 0.68),
    )
    higher_mean_lower_lcb = _metrics(
        "R5_UNSTABLE",
        old_pre=0.82,
        old_post=0.74,
        new_acc=0.68,
        fold_h=(0.50, 0.70, 0.70, 0.70, 0.70),
    )

    decision = choose_r5d92_candidate(
        baseline,
        [infeasible, safer, higher_mean_lower_lcb],
        epsilon_pre=0.01,
        epsilon_old=0.02,
        epsilon_new=0.02,
    )

    assert decision.selected_candidate_id == "R5_SAFE"
    assert decision.fallback_used is False
    assert decision.feasible_candidate_ids == ("R5_SAFE", "R5_UNSTABLE")
    assert "R5_BAD_FORGETTING" in decision.rejections


def test_candidate_selection_falls_back_when_covariance_is_not_positive_definite() -> None:
    baseline = _metrics(
        "J1_R0_D92", old_pre=0.80, old_post=0.70, new_acc=0.60, fold_h=(0.62,) * 5
    )
    invalid = _metrics(
        "R5_NON_PD",
        old_pre=0.82,
        old_post=0.72,
        new_acc=0.68,
        fold_h=(0.68,) * 5,
        positive_definite=False,
    )

    decision = choose_r5d92_candidate(baseline, [invalid])

    assert decision.selected_candidate_id == "J1_R0_D92"
    assert decision.fallback_used is True
    assert decision.feasible_candidate_ids == ()


def test_covariance_block_audit_reports_identity_fft_traces_and_condition() -> None:
    covariance = np.diag(np.concatenate([np.full(160, 2.0), np.full(96, 0.5)]))

    audit = covariance_block_audit(covariance, block_dims=(160, 96))

    assert audit["positive_definite"] is True
    assert audit["condition_number"] == 4.0
    assert audit["block_traces"] == [320.0, 48.0]


def test_d92_registration_pair_accepts_crossfit_k8_without_support_duplication() -> None:
    rng = np.random.default_rng(20260828)
    labels = np.repeat(np.arange(11, dtype=np.int64), 8)
    identity = rng.normal(0.0, 0.01, (88, 160)).astype(np.float32)
    fft = rng.normal(0.0, 0.01, (88, 96)).astype(np.float32)
    for class_id in range(11):
        mask = labels == class_id
        identity[mask, class_id] += 4.0
        fft[mask, class_id] += 2.0

    reg0, reg1, audit = fit_erbt_registration_pair(
        identity[:48],
        fft[:48],
        labels[:48],
        identity,
        fft,
        labels,
        old_class_ids=tuple(range(6)),
        registered_class_ids=tuple(range(11)),
        seed=713101,
        device="cpu",
    )

    assert audit["k_shot"] == 8
    assert reg0.audit["support_rows"] == 48
    assert reg1.audit["support_rows"] == 88
    assert reg1.audit["k_shot"] == 8
    assert len(reg1.audit["d92_covariance_block_traces"]) == 2
    assert np.array_equal(reg1.predict(identity, fft), labels)


def test_d92_crossfit_metrics_use_heldout_old_and_new_support() -> None:
    rng = np.random.default_rng(55)
    labels = np.repeat(np.arange(11, dtype=np.int64), 10)
    identity = rng.normal(0.0, 0.01, (110, 160)).astype(np.float32)
    received_iq = rng.normal(0.0, 0.01, (110, 2, 256)).astype(np.float32)
    for class_id in range(11):
        identity[labels == class_id, class_id] += 5.0
        received_iq[labels == class_id, 0, class_id] += 3.0
    splits = complementary_leave_pair_splits(labels)

    metrics = evaluate_d92_crossfit_from_fold_features(
        "R5_SYNTHETIC",
        tuple(identity.copy() for _ in range(5)),
        received_iq,
        labels,
        splits=splits,
        old_class_count=6,
        seed=713101,
        device="cpu",
    )

    assert metrics.old_pre == 1.0
    assert metrics.old_post == 1.0
    assert metrics.new_acc == 1.0
    assert metrics.harmonic_old_new == 1.0
    assert metrics.forgetting == 0.0
    assert metrics.fold_h == (1.0,) * 5
    assert metrics.covariance_positive_definite is True
    assert metrics.identity_covariance_trace > 0.0
    assert metrics.fft_covariance_trace > 0.0


def test_r5d92_support_selection_runs_top2_and_returns_one_polished_model() -> None:
    torch.manual_seed(61)
    old = TargetOnlyAdaptationDataset(
        received_iq=torch.randn(60, 4, 8),
        labels=torch.tensor([class_id for class_id in range(6) for _ in range(10)]),
        physical_ids=tuple(f"old-{index}" for index in range(60)),
        groups=tuple(f"old-{index}" for index in range(60)),
    )
    labels = np.repeat(np.arange(11, dtype=np.int64), 10)
    registered_iq = np.random.default_rng(101).normal(
        0.0, 0.01, (110, 2, 256)
    ).astype(np.float32)
    for class_id in range(11):
        registered_iq[labels == class_id, 0, class_id] = 5.0

    def identity_extractor(_model, rows, _device):
        result = np.random.default_rng(99).normal(0.0, 0.01, (len(rows), 160)).astype(np.float32)
        class_ids = np.argmax(rows[:, 0, :11], axis=1)
        result[np.arange(len(rows)), class_ids] += 5.0
        return result

    model = _GeneralCacheableBackbone()
    model.cls_head.head = torch.nn.Linear(4, 6, bias=False)
    selection = run_r5d92_support_selection(
        model,
        old,
        registered_iq,
        labels,
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(2, 1, 1),
            scheduler_reference_steps=4,
            cache_storage_dtype="float32",
            suffix_compute_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
            seed=61,
        ),
        steps=(2, 3, 4),
        polish_steps=1,
        seed=713101,
        device="cpu",
        identity_extractor=identity_extractor,
    )

    assert selection.pool.trajectory_fit_count == 5
    assert len(selection.metrics) == 3
    assert selection.baseline_candidate_id == "J1_R0_D92"
    assert selection.result.audit.total_steps == 1
    assert selection.result.audit.query_opened is False
