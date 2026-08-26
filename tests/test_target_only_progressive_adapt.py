from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

import cvsrffi.target_only_progressive_adapt as tapft
from cvsrffi.target_only_progressive_adapt import (
    CheckpointAverager,
    FoldMetrics,
    GroupedTargetCVSelector,
    L2SPRegularizer,
    ProgressiveTrainabilityPolicy,
    SFTAPFTConfig,
    TargetOnlyAdaptationDataset,
    TargetPrototypeHead,
    TrainableDeltaAverager,
    ensure_time_adapter,
    fit_sf_tapft,
    leave_one_out_prototype_logits,
    select_sf_tapft_by_grouped_cv,
)


class _ToyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(self.conv(value)))


class _ToyHeadCarrier(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.head = nn.Linear(weight.size(1), weight.size(0), bias=False)
        with torch.no_grad():
            self.head.weight.copy_(weight)


class _ToyCapacityBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Linear(dim, dim, bias=False)
        self.pw = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)


class _ToyCapacityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_dim = 4
        self.meta_adapter_time = nn.Identity()
        self.time_fuse = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))
        self.t1 = _ToyCapacityBlock(4)
        self.t2 = _ToyCapacityBlock(4)
        self.t3 = _ToyCapacityBlock(4)
        self.fuse = nn.Sequential(nn.Linear(4, 4, bias=False), nn.ReLU())
        self.cls_head = _ToyHeadCarrier(torch.eye(4)[:2])
        self.cls_head.id_proj = nn.Linear(4, 4, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_dim = 4
        self.meta_adapter_time = nn.Identity()
        self.t3 = _ToyBlock(4)
        self.freq_branch = nn.Linear(4, 4, bias=False)
        self.cls_head = _ToyHeadCarrier(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor, y=None, return_aux: bool = False):
        del y
        time = self.meta_adapter_time(self.t3(x))
        # The frequency branch participates in the graph but should never be
        # trainable under SF-TAPFT V1.
        joint = time + 0.05 * self.freq_branch(x).detach()
        logits = self.cls_head.head(joint)
        if return_aux:
            return {"feat_joint": joint, "logits": logits}
        return logits


class _ToyDualModel(nn.Module):
    def __init__(self, base: _ToyModel) -> None:
        super().__init__()
        self.id_backbone = base
        self.dom_backbone = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor, y_tx=None, return_aux: bool = False):
        aux = self.id_backbone(x, y=y_tx, return_aux=True)
        if return_aux:
            return {"z_id": aux["feat_joint"], "tx_logits": aux["logits"]}
        return aux["logits"]


class _ToySincBufferDriftModel(_ToyModel):
    """Models an eval-time floating Sinc state that must never be averaged."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("sinc_reference", torch.full((4096,), 16_777_216.0))

    def forward(self, x: torch.Tensor, y=None, return_aux: bool = False):
        with torch.no_grad():
            self.sinc_reference.add_(2.0)
        return super().forward(x, y=y, return_aux=return_aux)


def _dataset() -> TargetOnlyAdaptationDataset:
    return TargetOnlyAdaptationDataset(
        received_iq=torch.tensor(
            [
                [2.0, 0.0, 0.2, 0.0],
                [1.7, 0.1, 0.0, 0.2],
                [1.9, -0.1, 0.1, 0.0],
                [0.0, 2.0, 0.0, 0.2],
                [0.1, 1.8, 0.2, 0.0],
                [-0.1, 1.9, 0.0, 0.1],
            ]
        ),
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
        physical_ids=("a0", "a1", "a2", "b0", "b1", "b2"),
        groups=("g0", "g1", "g2", "g0", "g1", "g2"),
    )


def test_dataset_rejects_eval_role_and_duplicate_physical_samples() -> None:
    with pytest.raises(ValueError, match="target_train"):
        TargetOnlyAdaptationDataset(
            received_iq=torch.ones(2, 4),
            labels=torch.tensor([0, 1]),
            physical_ids=("p0", "p1"),
            groups=("g0", "g1"),
            role="target_eval",
        )
    with pytest.raises(ValueError, match="unique"):
        TargetOnlyAdaptationDataset(
            received_iq=torch.ones(2, 4),
            labels=torch.tensor([0, 1]),
            physical_ids=("p0", "p0"),
            groups=("g0", "g1"),
        )


def test_target_prototype_head_imprints_source_and_target_directions() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    head = TargetPrototypeHead.from_source_and_target(
        source_weights=source,
        target_prototypes=target,
        source_class_ids=(10, 11),
        target_class_ids=(10, 12),
        rho=0.5,
        scale=8.0,
    )
    expected_old = torch.tensor([2 ** -0.5, 2 ** -0.5])
    assert head.class_ids == (10, 11, 12)
    assert torch.allclose(head.weight[0], expected_old, atol=1e-6)
    assert torch.allclose(head.weight[1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(head.weight[2], torch.tensor([1.0, 0.0]))
    logits = head(torch.tensor([[1.0, 0.0]]))
    assert logits.shape == (1, 3)


def test_leave_one_out_prototype_excludes_current_sample_and_uses_k1_fallback() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1])
    fallback = torch.tensor([[1.0, 1.0], [-1.0, 0.0]])
    logits = leave_one_out_prototype_logits(
        embeddings,
        labels,
        class_count=2,
        fallback_weights=fallback,
        scale=1.0,
    )
    # Sample 0 must see sample 1, not the [0.5, 0.5] self-inclusive mean.
    assert torch.allclose(logits[0], torch.tensor([0.0, -1.0]), atol=1e-6)
    # Class 1 has K=1, so its own class anchor comes from the fallback weight.
    assert torch.allclose(logits[2], torch.tensor([-2 ** -0.5, 1.0]), atol=1e-6)


def test_l2sp_is_zero_at_snapshot_and_positive_after_drift() -> None:
    layer = nn.Linear(3, 2)
    regularizer = L2SPRegularizer.from_named_parameters(layer.named_parameters())
    assert regularizer(layer.named_parameters()).item() == pytest.approx(0.0)
    with torch.no_grad():
        layer.weight.add_(1.0)
    assert regularizer(layer.named_parameters()).item() == pytest.approx(1.0)


def test_progressive_policy_never_unfreezes_frequency_branch() -> None:
    model = _ToyModel()
    ensure_time_adapter(model, rank=2)
    policy = ProgressiveTrainabilityPolicy()
    phase_a = policy.apply(model, "A")
    assert phase_a == ("t3.norm.bias", "t3.norm.weight")
    phase_b = policy.apply(model, "B")
    assert set(phase_a).issubset(phase_b)
    assert any(name.startswith("meta_adapter_time.") for name in phase_b)
    phase_c = policy.apply(model, "C")
    assert any(name.startswith("t3.conv.") for name in phase_c)
    assert not any(name.startswith("freq_branch.") for name in phase_c)


def test_capacity_profiles_expose_exact_nested_p0_to_p4_parameter_sets() -> None:
    model = _ToyCapacityModel()
    ensure_time_adapter(model, rank=2)

    p0 = set(ProgressiveTrainabilityPolicy("p0_head_only").parameter_names(model, "C"))
    p1 = set(ProgressiveTrainabilityPolicy("p1_head_norm").parameter_names(model, "C"))
    p2 = set(ProgressiveTrainabilityPolicy("p2_time_adapter").parameter_names(model, "C"))
    p3 = set(ProgressiveTrainabilityPolicy("p3_full_t3").parameter_names(model, "C"))
    p4 = set(ProgressiveTrainabilityPolicy("p4_time_fusion").parameter_names(model, "C"))

    assert p0 == set()
    assert p0 < p1 < p2 < p3 < p4
    assert "t2.pw.weight" not in p3
    assert {
        "t2.pw.weight",
        "time_fuse.0.weight",
        "fuse.0.weight",
        "cls_head.id_proj.weight",
    }.issubset(p4)


def test_head_only_profile_runs_without_requiring_model_parameter_groups() -> None:
    model = _ToyModel()
    before = copy.deepcopy(model.state_dict())
    result = fit_sf_tapft(
        model,
        _dataset(),
        SFTAPFTConfig(
            trainability_profile="p0_head_only",
            phase_steps=(2, 0, 0),
            checkpoint_average_top_k=1,
            adapter_rank=2,
            mixed_precision=False,
        ),
    )
    assert result.audit.total_steps == 2
    assert all(not names for names in result.audit.trainable_names_by_phase.values())
    assert result.audit.nonpermitted_changed_names == ()
    assert all(torch.equal(result.model.state_dict()[name], value) for name, value in before.items())


def test_config_rejects_unknown_trainability_profile() -> None:
    with pytest.raises(ValueError, match="trainability_profile"):
        SFTAPFTConfig(trainability_profile="p9_not_real")


def test_grouped_selector_keeps_groups_disjoint_and_uses_hierarchical_ties() -> None:
    selector = GroupedTargetCVSelector(folds=3, seed=7)
    splits = selector.split(labels=_dataset().labels, groups=_dataset().groups)
    assert len(splits) == 3
    for train_indices, val_indices in splits:
        train_groups = {_dataset().groups[i] for i in train_indices}
        val_groups = {_dataset().groups[i] for i in val_indices}
        assert train_groups.isdisjoint(val_groups)

    frozen = FoldMetrics(
        balanced_accuracy=0.75,
        nll=0.60,
        true_class_margin=0.20,
        fold_variance=0.01,
        source_distance=0.0,
        non_degrading_fold_fraction=1.0,
    )
    adapted = FoldMetrics(
        balanced_accuracy=0.75,
        nll=0.50,
        true_class_margin=0.25,
        fold_variance=0.02,
        source_distance=0.1,
        non_degrading_fold_fraction=1.0,
    )
    assert selector.choose(frozen=frozen, adapted=adapted) == "adapted"
    unsafe = FoldMetrics(
        balanced_accuracy=0.80,
        nll=0.45,
        true_class_margin=0.30,
        fold_variance=0.02,
        source_distance=0.1,
        non_degrading_fold_fraction=0.4,
    )
    assert selector.choose(frozen=frozen, adapted=unsafe) == "zero_adapt"
    accuracy_up_but_nll_worse = FoldMetrics(
        balanced_accuracy=0.80,
        nll=0.70,
        true_class_margin=0.30,
        fold_variance=0.01,
        source_distance=0.1,
        non_degrading_fold_fraction=1.0,
    )
    assert selector.choose(frozen=frozen, adapted=accuracy_up_but_nll_worse) == "zero_adapt"


def test_selector_uses_stratified_fallback_when_real_groups_are_absent() -> None:
    dataset = _dataset()
    selector = GroupedTargetCVSelector(folds=3, seed=31)
    splits = selector.split(labels=dataset.labels, groups=None)
    assert len(splits) == 3
    for train_indices, validation_indices in splits:
        assert set(train_indices).isdisjoint(validation_indices)
        assert set(dataset.labels[list(validation_indices)].tolist()) == {0, 1}


def test_checkpoint_averager_uses_top_scores_and_rejects_incompatible_states() -> None:
    states = [
        ({"w": torch.tensor([1.0, 3.0])}, 0.4),
        ({"w": torch.tensor([3.0, 5.0])}, 0.9),
        ({"w": torch.tensor([5.0, 7.0])}, 0.8),
    ]
    averaged = CheckpointAverager(top_k=2).average(states)
    assert torch.equal(averaged["w"], torch.tensor([4.0, 6.0]))
    with pytest.raises(ValueError, match="keys"):
        CheckpointAverager(top_k=2).average(
            [({"w": torch.ones(1)}, 1.0), ({"x": torch.ones(1)}, 0.5)]
        )


def test_trainable_delta_averager_restores_frozen_large_float_and_averages_permitted_delta() -> None:
    anchor = {
        "trainable": torch.tensor([1.0, 3.0]),
        "sinc_reference": torch.full((4096,), 16_777_216.0),
    }
    states = [
        (
            {
                "trainable": torch.tensor([3.0, 7.0]),
                "sinc_reference": torch.full((4096,), 16_777_218.0),
            },
            0.9,
        ),
        (
            {
                "trainable": torch.tensor([5.0, 9.0]),
                "sinc_reference": torch.full((4096,), 16_777_220.0),
            },
            0.8,
        ),
        (
            {
                "trainable": torch.tensor([9.0, 17.0]),
                "sinc_reference": torch.full((4096,), 16_777_222.0),
            },
            0.2,
        ),
    ]

    averaged = TrainableDeltaAverager(top_k=2).average(
        states,
        anchor_state=anchor,
        permitted_names={"trainable"},
    )

    assert torch.equal(averaged["trainable"], torch.tensor([4.0, 8.0]))
    assert torch.equal(averaged["sinc_reference"], anchor["sinc_reference"])


def test_trainable_delta_averager_rejects_changed_nonfloating_permitted_state() -> None:
    with pytest.raises(ValueError, match="non-floating permitted"):
        TrainableDeltaAverager(top_k=1).average(
            [({"step": torch.tensor(2, dtype=torch.int64)}, 1.0)],
            anchor_state={"step": torch.tensor(1, dtype=torch.int64)},
            permitted_names={"step"},
        )


def test_fit_sf_tapft_is_reproducible_updates_allowed_scope_and_freezes_result() -> None:
    base = _ToyModel()
    before = copy.deepcopy(base.state_dict())
    config = SFTAPFTConfig(
        phase_steps=(1, 1, 1),
        warmup_ratio=0.0,
        checkpoint_average_top_k=1,
        adapter_rank=2,
        lambda_proto=0.5,
        lambda_l2sp=1.0e-4,
        selective_kd_weight=0.0,
        seed=13,
    )
    first = fit_sf_tapft(copy.deepcopy(base), _dataset(), config)
    second = fit_sf_tapft(copy.deepcopy(base), _dataset(), config)

    assert first.audit.method == "sf_tapft_v1"
    assert first.audit.permission == "DIAGNOSTIC_NON_FORMAL"
    assert first.audit.total_steps == 3
    assert first.audit.source_loader_opened is False
    assert first.audit.target_eval_opened is False
    assert first.audit.query_opened is False
    assert torch.equal(first.model.freq_branch.weight, before["freq_branch.weight"])
    assert not torch.equal(first.model.t3.conv.weight, before["t3.conv.weight"])
    assert all(not parameter.requires_grad for parameter in first.model.parameters())
    assert all(not parameter.requires_grad for parameter in first.head.parameters())
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])
    assert torch.equal(first.head.weight, second.head.weight)


def test_fit_supports_torch21_cuda_amp_scaler_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(torch.amp, "GradScaler")
    result = fit_sf_tapft(
        _ToyModel(),
        _dataset(),
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            mixed_precision=False,
            seed=41,
        ),
    )
    assert result.audit.total_steps == 3


def test_top_checkpoint_average_requires_and_uses_disjoint_inner_validation() -> None:
    dataset = _dataset()
    train_indices = torch.tensor([0, 1, 3, 4])
    validation_indices = torch.tensor([2, 5])
    inner_train = TargetOnlyAdaptationDataset(
        received_iq=dataset.received_iq[train_indices],
        labels=dataset.labels[train_indices],
        physical_ids=tuple(dataset.physical_ids[index] for index in train_indices.tolist()),
        groups=None,
    )
    inner_validation = TargetOnlyAdaptationDataset(
        received_iq=dataset.received_iq[validation_indices],
        labels=dataset.labels[validation_indices],
        physical_ids=tuple(dataset.physical_ids[index] for index in validation_indices.tolist()),
        groups=None,
    )
    config = SFTAPFTConfig(
        phase_steps=(1, 1, 1),
        warmup_ratio=0.0,
        checkpoint_average_top_k=2,
        adapter_rank=2,
        seed=37,
    )
    with pytest.raises(ValueError, match="inner validation"):
        fit_sf_tapft(_ToyModel(), inner_train, config)
    result = fit_sf_tapft(
        _ToyModel(),
        inner_train,
        config,
        checkpoint_validation=inner_validation,
    )
    assert result.audit.checkpoint_selection_role == "target_inner_validation"
    with pytest.raises(ValueError, match="disjoint"):
        fit_sf_tapft(
            _ToyModel(),
            inner_train,
            config,
            checkpoint_validation=inner_train,
        )


def test_top_checkpoint_average_restores_nonpermitted_sinc_buffer_to_post_adapter_anchor() -> None:
    dataset = _dataset()
    train_indices = torch.tensor([0, 1, 3, 4])
    validation_indices = torch.tensor([2, 5])
    inner_train = TargetOnlyAdaptationDataset(
        received_iq=dataset.received_iq[train_indices],
        labels=dataset.labels[train_indices],
        physical_ids=tuple(dataset.physical_ids[index] for index in train_indices.tolist()),
        groups=None,
    )
    inner_validation = TargetOnlyAdaptationDataset(
        received_iq=dataset.received_iq[validation_indices],
        labels=dataset.labels[validation_indices],
        physical_ids=tuple(dataset.physical_ids[index] for index in validation_indices.tolist()),
        groups=None,
    )
    result = fit_sf_tapft(
        _ToySincBufferDriftModel(),
        inner_train,
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=3,
            adapter_rank=2,
            mixed_precision=False,
            seed=43,
        ),
        checkpoint_validation=inner_validation,
    )

    # The first target-only forward builds target prototypes, then the adapter
    # anchor is captured. Subsequent checkpoint snapshots must not alter this
    # non-permitted Sinc-like floating buffer through averaging.
    assert torch.equal(result.model.sinc_reference, torch.full((4096,), 16_777_218.0))
    assert result.audit.nonpermitted_changed_names == ()
    permitted_model_names = set(result.audit.trainable_names_by_phase["C"])
    changed_model_names = {
        name.removeprefix("model.")
        for name in result.audit.permitted_changed_names
        if name.startswith("model.")
    }
    assert changed_model_names.issubset(permitted_model_names)


def test_fit_sf_tapft_binds_dual_identity_backbone_without_updating_domain_backbone() -> None:
    base = _ToyDualModel(_ToyModel())
    domain_before = base.dom_backbone.weight.detach().clone()
    result = fit_sf_tapft(
        base,
        _dataset(),
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            seed=17,
        ),
    )
    assert isinstance(result.model.id_backbone.meta_adapter_time, nn.Module)
    assert torch.equal(result.model.dom_backbone.weight, domain_before)
    assert any(name.startswith("id_backbone.t3.") for name in result.audit.updated_parameter_names)


def test_grouped_cv_selection_uses_only_target_train_folds_and_returns_oof_metrics() -> None:
    selection = select_sf_tapft_by_grouped_cv(
        _ToyModel(),
        _dataset(),
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            seed=29,
        ),
        folds=3,
    )
    assert selection.selected in {"adapted", "zero_adapt"}
    assert len(selection.fold_rows) == 3
    assert selection.frozen_metrics.source_distance == 0.0
    assert selection.adapted_metrics.source_distance >= 0.0
    assert all(row.train_groups.isdisjoint(row.validation_groups) for row in selection.fold_rows)
    assert all(row.query_opened is False for row in selection.fold_rows)
    # Removing per-stage collection would hide which fold-best phase lengths
    # produced the one conservative schedule.
    assert all(tuple(stage.phase for stage in row.stage_validation_rows) == ("A", "B", "C") for row in selection.fold_rows)
    assert selection.selected_phase_steps == (1, 1, 1)
    if selection.selected == "adapted":
        assert selection.adapted_result is not None
    else:
        assert selection.adapted_result is None


def test_stage_metrics_match_hand_derived_classwise_values_and_flip_directions() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    frozen_logits = torch.tensor(
        [[2.0, 0.0], [-0.5, 0.5], [0.1, 0.6], [-0.2, 0.8]]
    )
    adapted_logits = torch.tensor(
        [[2.0, 0.0], [0.5, -0.5], [0.7, 0.2], [-0.2, 0.8]]
    )

    # Swapping per-class and per-row reductions, or reversing either flip
    # direction, changes at least one of these independently derived literals.
    metrics = tapft._stage_validation_metrics(
        adapted_logits,
        frozen_logits,
        labels,
        registered_class_indices=(0, 1),
        permitted_parameter_distance=1.25,
    )

    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.macro_f1 == pytest.approx(0.7333333333333334)
    assert metrics.class_floor == pytest.approx(0.5)
    assert metrics.nll == pytest.approx(0.4318820834159851)
    assert metrics.per_class_recall == pytest.approx((1.0, 0.5))
    assert metrics.per_class_margin == pytest.approx((1.5, 0.25))
    assert metrics.positive_flips == 1
    assert metrics.negative_flips == 1
    assert metrics.permitted_parameter_distance == pytest.approx(1.25)


def test_stage_metrics_include_registered_class_absent_from_validation_with_zero_recall_and_margin() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    adapted_logits = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    frozen_logits = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Deriving the universe from validation labels would drop registered class
    # 2, omit its two false positives and overstate BA, floor and macro-F1.
    metrics = tapft._stage_validation_metrics(
        adapted_logits,
        frozen_logits,
        labels,
        registered_class_indices=(0, 1, 2),
        permitted_parameter_distance=0.0,
    )

    assert metrics.per_class_recall == pytest.approx((0.5, 0.5, 0.0))
    assert metrics.per_class_margin == pytest.approx((0.5, 0.5, 0.0))
    assert metrics.balanced_accuracy == pytest.approx(1.0 / 3.0)
    assert metrics.class_floor == 0.0
    assert metrics.macro_f1 == pytest.approx(4.0 / 9.0)


def test_stage_metric_order_prefers_class_floor_before_lower_nll() -> None:
    higher_floor = tapft.StageValidationMetrics(
        balanced_accuracy=0.75,
        macro_f1=0.60,
        class_floor=0.50,
        nll=1.50,
        per_class_recall=(1.0, 0.5),
        per_class_margin=(0.1, 0.1),
        positive_flips=0,
        negative_flips=0,
        permitted_parameter_distance=2.0,
    )
    lower_nll = tapft.StageValidationMetrics(
        balanced_accuracy=0.75,
        macro_f1=0.90,
        class_floor=0.25,
        nll=0.10,
        per_class_recall=(1.0, 0.25),
        per_class_margin=(2.0, 2.0),
        positive_flips=0,
        negative_flips=0,
        permitted_parameter_distance=0.1,
    )

    # Moving -NLL before floor would select the lower-floor checkpoint.
    assert max((higher_floor, lower_nll), key=tapft._stage_metric_order_key) is higher_floor


def test_lower_median_uses_conservative_even_fold_value() -> None:
    # Using the ordinary even-count median would yield non-observed phase lengths.
    fold_phase_lengths = (
        [400, 450, 500, 500],
        [1000, 1200, 1100, 1300],
        [300, 400, 500, 500],
    )
    assert tuple(tapft._lower_median(values) for values in fold_phase_lengths) == (
        450,
        1100,
        400,
    )


def test_zero_step_phases_emit_no_rows_and_select_zero_lengths() -> None:
    selection = select_sf_tapft_by_grouped_cv(
        _ToyModel(),
        _dataset(),
        SFTAPFTConfig(
            phase_steps=(0, 1, 0),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            seed=31,
        ),
        folds=3,
    )

    # Synthesizing metrics for configured zero-step phases would falsely make
    # A or C eligible for the unified refit schedule.
    assert all(tuple(stage.phase for stage in row.stage_validation_rows) == ("B",) for row in selection.fold_rows)
    assert selection.selected_phase_steps == (0, 1, 0)


def test_fit_rejects_unknown_checkpoint_selection_mode() -> None:
    with pytest.raises(ValueError, match="checkpoint_selection_mode"):
        fit_sf_tapft(
            _ToyModel(),
            _dataset(),
            SFTAPFTConfig(
                phase_steps=(1, 1, 1),
                warmup_ratio=0.0,
                checkpoint_average_top_k=1,
                adapter_rank=2,
                seed=47,
            ),
            checkpoint_selection_mode="earliest",
        )
    with pytest.raises(ValueError, match="checkpoint_selection_mode"):
        fit_sf_tapft(
            _ToyModel(),
            _dataset(),
            SFTAPFTConfig(
                phase_steps=(1, 1, 1),
                warmup_ratio=0.0,
                checkpoint_average_top_k=1,
                adapter_rank=2,
                seed=47,
            ),
            checkpoint_selection_mode=[],
        )


def test_final_step_checkpoint_ignores_an_earlier_better_train_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Removing the explicit final-step branch would let step 1 win this
    # deliberately descending score fixture and return the wrong state.
    monkeypatch.setattr(
        tapft,
        "_target_train_snapshot_score",
        lambda _loss, step: 2.0 if step == 1 else 1.0,
        raising=False,
    )
    config = SFTAPFTConfig(
        phase_steps=(2, 0, 0),
        warmup_ratio=0.0,
        checkpoint_average_top_k=1,
        adapter_rank=2,
        mixed_precision=False,
        seed=53,
    )

    best = fit_sf_tapft(
        _ToyModel(),
        _dataset(),
        config,
        checkpoint_selection_mode="best",
    )
    final = fit_sf_tapft(
        _ToyModel(),
        _dataset(),
        config,
        checkpoint_selection_mode="final_step",
    )

    assert best.audit.selected_checkpoint_steps == (1,)
    assert final.audit.selected_checkpoint_steps == (2,)
    assert final.audit.checkpoint_selection_role == "fixed_final_step"
    assert any(
        not torch.equal(value, final.model.state_dict()[name])
        for name, value in best.model.state_dict().items()
    ) or not torch.equal(best.head.weight, final.head.weight)


def test_opt_in_grouped_selection_refits_adapted_candidate_on_all_support_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold_results = []
    real_validation_logits = tapft._adapted_validation_logits

    def record_fold_result(result, dataset):
        fold_results.append(result)
        return real_validation_logits(result, dataset)

    monkeypatch.setattr(tapft, "_adapted_validation_logits", record_fold_result)
    monkeypatch.setattr(
        tapft.GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "adapted"),
    )
    dataset = _dataset()
    selection = select_sf_tapft_by_grouped_cv(
        _ToyModel(),
        dataset,
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            mixed_precision=False,
            seed=59,
        ),
        folds=3,
        full_support_refit=True,
    )

    assert len(fold_results) == 3
    assert selection.full_support_result is not None
    assert selection.adapted_result is selection.full_support_result
    assert selection.final_training_sample_count == len(dataset.physical_ids)
    assert selection.fold0_as_final is False
    assert selection.full_support_result.audit.training_sample_count == len(dataset.physical_ids)
    assert selection.full_support_result.audit.phase_steps == selection.selected_phase_steps
    assert selection.full_support_result.audit.checkpoint_selection_role == "fixed_final_step"
    assert selection.full_support_result.audit.selected_checkpoint_steps == (
        sum(selection.selected_phase_steps),
    )
    assert all(
        result.audit.training_sample_count < len(dataset.physical_ids)
        for result in fold_results
    )
    assert all(selection.full_support_result is not result for result in fold_results)


def test_opt_in_zero_adapt_does_not_refit_full_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tapft.GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "zero_adapt"),
    )
    selection = select_sf_tapft_by_grouped_cv(
        _ToyModel(),
        _dataset(),
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            mixed_precision=False,
            seed=61,
        ),
        folds=3,
        full_support_refit=True,
    )

    assert selection.selected == "zero_adapt"
    assert selection.adapted_result is None
    assert selection.full_support_result is None
    assert selection.final_training_sample_count == 0
    assert selection.fold0_as_final is False


def test_non_opt_in_grouped_selection_preserves_fold0_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold_results = []
    real_validation_logits = tapft._adapted_validation_logits

    def record_fold_result(result, dataset):
        fold_results.append(result)
        return real_validation_logits(result, dataset)

    monkeypatch.setattr(tapft, "_adapted_validation_logits", record_fold_result)
    monkeypatch.setattr(
        tapft.GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "adapted"),
    )
    selection = select_sf_tapft_by_grouped_cv(
        _ToyModel(),
        _dataset(),
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            mixed_precision=False,
            seed=67,
        ),
        folds=3,
    )

    assert len(fold_results) == 3
    assert selection.adapted_result is fold_results[0]
    assert selection.full_support_result is None
    assert selection.fold0_as_final is True
