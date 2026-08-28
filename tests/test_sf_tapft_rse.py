from __future__ import annotations

import copy

import pytest
import torch

import cvsrffi.target_only_progressive_adapt as adapt_module
from cvsrffi.target_only_progressive_adapt import (
    SFTAPFTConfig,
    average_trainable_states,
    balanced_rse_subsets,
    fit_sf_tapft_inplace,
    fit_sf_tapft_rse_strength_selection,
    fit_sf_tapft_rse_delta_ensemble,
    interpolate_trainable_state,
    phase_rotate_iq,
    robust_support_risk,
    select_rse_strength,
)
from test_sf_tapft_pace import _GeneralCacheableBackbone, _dataset


def test_phase_rotation_preserves_amplitude_and_physical_sample_count() -> None:
    iq = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, -1.0], [1.0, 2.0]]])
    rotated = phase_rotate_iq(iq, radians=0.05)

    assert rotated.shape == iq.shape
    assert torch.allclose(rotated.square().sum(dim=1), iq.square().sum(dim=1), atol=1.0e-6)
    assert not torch.equal(rotated, iq)


def test_rse_config_rejects_non_e0_or_partial_view_controls() -> None:
    with pytest.raises(ValueError, match="RSE requires exactly t3"):
        SFTAPFTConfig(
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"), ("t2", "weight")),
            rse_snapshot_steps=(2, 4),
        )
    with pytest.raises(ValueError, match="rse_view"):
        SFTAPFTConfig(
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            rse_view_weight=0.05,
            rse_view_phase_radians=0.0,
        )


def test_robust_support_risk_penalizes_class_tail_and_margin_regression() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    frozen = torch.tensor([[3.0, 0.0], [2.5, 0.0], [0.0, 2.8], [0.0, 2.4]])
    stable = frozen.clone()
    regressed = torch.tensor([[1.0, 0.9], [0.8, 0.7], [0.8, 0.9], [0.7, 0.8]])

    stable_risk = robust_support_risk(stable, labels, frozen_logits=frozen)
    regressed_risk = robust_support_risk(regressed, labels, frozen_logits=frozen)

    assert stable_risk.margin_regression == pytest.approx(0.0)
    assert regressed_risk.margin_regression > 0.0
    assert regressed_risk.total > stable_risk.total


def test_dual_view_fit_caches_two_views_but_counts_six_physical_samples() -> None:
    torch.manual_seed(113)
    result = fit_sf_tapft_inplace(
        _GeneralCacheableBackbone(),
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(2, 1, 1),
            scheduler_reference_steps=4,
            rse_view_weight=0.05,
            rse_view_phase_radians=0.05,
            cache_storage_dtype="float32",
            suffix_compute_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
        ),
        checkpoint_selection_mode="final_step",
    )

    assert result.audit.training_sample_count == 6
    assert result.audit.effective_view_count == 2
    assert result.audit.prefix_cache_build_forward_steps == 2
    assert result.audit.prefix_cache_tensor_bytes > 0
    assert len(result.audit.view_consistency_losses) == 4
    assert set(result.audit.updated_parameter_names) == {"t3.norm.bias", "t3.norm.weight"}


def test_rse_snapshots_are_retained_in_one_training_trajectory() -> None:
    torch.manual_seed(127)
    result = fit_sf_tapft_inplace(
        _GeneralCacheableBackbone(),
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(2, 1, 1),
            scheduler_reference_steps=4,
            rse_snapshot_steps=(2, 3, 4),
            cache_storage_dtype="float32",
            suffix_compute_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
        ),
        checkpoint_selection_mode="final_step",
    )

    assert tuple(result.retained_trainable_snapshots) == (2, 3, 4)
    for state in result.retained_trainable_snapshots.values():
        assert set(state) == {
            "model.t3.norm.bias",
            "model.t3.norm.weight",
            "head.weight",
        }
        assert all(not value.requires_grad and value.device.type == "cpu" for value in state.values())


def test_rse_snapshot_retention_does_not_change_final_model() -> None:
    config = SFTAPFTConfig(
        adapter_rank=2,
        trainability_profile="p1_head_norm",
        norm_rules=(("t3", "weight_bias"),),
        phase_steps=(2, 1, 1),
        scheduler_reference_steps=4,
        cache_storage_dtype="float32",
        suffix_compute_dtype="float32",
        mixed_precision=False,
        checkpoint_average_top_k=1,
        seed=131,
    )
    torch.manual_seed(131)
    base = _GeneralCacheableBackbone()
    left = fit_sf_tapft_inplace(
        copy.deepcopy(base), _dataset(), config, checkpoint_selection_mode="final_step"
    )
    right = fit_sf_tapft_inplace(
        copy.deepcopy(base),
        _dataset(),
        SFTAPFTConfig(**{**config.__dict__, "rse_snapshot_steps": (2, 4)}),
        checkpoint_selection_mode="final_step",
    )

    assert all(
        torch.equal(left.model.state_dict()[name], right.model.state_dict()[name])
        for name in left.model.state_dict()
    )
    assert all(
        torch.equal(left.head.state_dict()[name], right.head.state_dict()[name])
        for name in left.head.state_dict()
    )


def test_interpolation_and_delta_average_stay_on_the_registered_state() -> None:
    anchor = {"model.t3.norm.weight": torch.tensor([1.0, 2.0]), "head.weight": torch.tensor([3.0])}
    first = {"model.t3.norm.weight": torch.tensor([3.0, 4.0]), "head.weight": torch.tensor([5.0])}
    second = {"model.t3.norm.weight": torch.tensor([1.0, 6.0]), "head.weight": torch.tensor([1.0])}

    half = interpolate_trainable_state(anchor, first, alpha=0.5)
    averaged = average_trainable_states(anchor, (first, second))

    assert torch.equal(half["model.t3.norm.weight"], torch.tensor([2.0, 3.0]))
    assert torch.equal(averaged["model.t3.norm.weight"], torch.tensor([2.0, 5.0]))
    assert torch.equal(averaged["head.weight"], torch.tensor([3.0]))
    with pytest.raises(ValueError, match="aligned keys"):
        average_trainable_states(anchor, ({"head.weight": torch.tensor([1.0])},))


def test_balanced_rse_subsets_are_deterministic_and_keep_eight_per_class() -> None:
    labels = torch.tensor([class_id for class_id in range(3) for _ in range(10)])
    first = balanced_rse_subsets(labels, per_class=8, count=2, seed=149)
    second = balanced_rse_subsets(labels, per_class=8, count=2, seed=149)

    assert first == second
    assert len(first) == 2
    assert first[0] != first[1]
    for subset in first:
        selected = labels[torch.tensor(subset)]
        assert torch.bincount(selected, minlength=3).tolist() == [8, 8, 8]


def test_strength_selector_uses_mean_crossfit_risk_and_falls_back_to_alpha_zero() -> None:
    winning = select_rse_strength(
        {
            (250, 0.0): (1.0, 1.2),
            (250, 0.5): (0.8, 0.9),
            (520, 1.0): (0.7, 0.8),
        }
    )
    fallback = select_rse_strength(
        {
            (250, 0.0): (0.5, 0.5),
            (250, 0.5): (0.5, 0.5),
            (520, 1.0): (0.6, 0.6),
        }
    )

    assert winning == (520, 1.0)
    assert fallback == (250, 0.0)


def test_strength_selection_runs_one_trajectory_per_fold_and_returns_one_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(157)
    monkeypatch.setattr(adapt_module, "select_rse_strength", lambda _risks: (2, 0.5))
    selection = fit_sf_tapft_rse_strength_selection(
        _GeneralCacheableBackbone(),
        _dataset(),
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
            seed=157,
        ),
        steps=(2, 4),
        alphas=(0.0, 0.5, 1.0),
        repeats=1,
        folds=2,
    )

    assert selection.crossfit_fit_count == 2
    assert selection.crossfit_validation_forward_steps == 2
    assert selection.crossfit_validation_suffix_forward_steps == 14
    assert len(selection.fold_rows) == 12
    assert selection.selected_step == 2
    assert selection.selected_alpha == 0.5
    assert selection.result.audit.total_steps == 4
    assert selection.result.audit.selected_checkpoint_steps == (2,)
    assert selection.result.audit.query_opened is False
    assert set(selection.result.base_parameter_anchors) == {
        "model.t3.norm.bias",
        "model.t3.norm.weight",
        "head.weight",
    }


def test_delta_ensemble_uses_common_anchor_and_returns_full_support_polish() -> None:
    torch.manual_seed(163)
    ensemble = fit_sf_tapft_rse_delta_ensemble(
        _GeneralCacheableBackbone(),
        _dataset(),
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
            seed=163,
        ),
        ensemble_count=2,
        per_class=1,
        polish_steps=1,
    )

    assert ensemble.subset_fit_count == 2
    assert ensemble.polish_steps == 1
    assert len(ensemble.subset_indices) == 2
    assert ensemble.result.audit.training_sample_count == 6
    assert ensemble.result.audit.query_opened is False
    assert set(ensemble.result.audit.updated_parameter_names) == {
        "t3.norm.bias",
        "t3.norm.weight",
    }
    assert ensemble.result.base_parameter_anchors.keys() == ensemble.common_anchor.keys()
