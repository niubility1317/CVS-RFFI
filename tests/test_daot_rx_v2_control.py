from __future__ import annotations

import pytest
import torch

from cvsrffi.branch_invariance import BranchInvariancePolicy, branch_invariance_loss
from cvsrffi.daot_gradient_control import PersistentConflictProjector
from cvsrffi.daot_source_selection import (
    allocate_structured_batch,
    build_receiver_holdout_folds,
    source_only_selection_score,
)
from cvsrffi.orbit_teacher import TensorTemporalOrbitMemory, adv3b02_daot_rx_v2_schedule
from cvsrffi.selective_nuisance_subspace import SelectiveNuisanceSubspace
from SSDG.train_ssdg import (
    _backward_with_daot_persistent_projection,
    _daot_memory_step,
    _validate_daot_config,
    build_arg_parser,
)


def test_branch_policy_does_not_apply_one_null_budget_to_every_branch() -> None:
    policy = BranchInvariancePolicy.default()

    assert policy.budget("time", "sto") < policy.budget("time", "clock_skew")
    assert policy.budget("frequency", "rx_filter") < policy.budget("frequency", "tx_spectral_asymmetry")
    assert policy.budget("dac", "rx_iq_residual") < policy.budget("dac", "tx_iq")
    assert policy.budget("pa", "agc") < policy.budget("pa", "pa_memory")


def test_branch_invariance_loss_penalizes_only_excess_over_branch_budget() -> None:
    sensitivities = {
        ("time", "sto"): torch.tensor([0.2]),
        ("time", "clock_skew"): torch.tensor([0.2]),
    }
    result = branch_invariance_loss(sensitivities, policy=BranchInvariancePolicy.default())

    assert float(result["loss"]) > 0.0
    assert result["violations"][("time", "sto")].item() > 0.0
    assert result["violations"][("time", "clock_skew")].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("epoch", "active"),
    [
        (5, {"base"}),
        (15, {"base", "orbit_feature"}),
        (30, {"base", "orbit_feature", "soft", "rx"}),
        (50, {"base", "orbit_feature", "soft", "rx", "tangent"}),
        (65, {"base", "orbit_feature", "soft", "rx", "tangent", "route"}),
        (75, {"base", "orbit_feature", "soft", "rx", "tangent", "route", "tail"}),
    ],
)
def test_rx_v2_schedule_activates_auxiliaries_independently(epoch: int, active: set[str]) -> None:
    state = adv3b02_daot_rx_v2_schedule(epoch, total_epochs=200)
    observed = {name for name, scale in state.scales.items() if scale > 0.0}

    assert observed == active


def test_conflict_projector_waits_for_persistent_conflict_then_removes_it() -> None:
    controller = PersistentConflictProjector(window=2, threshold=-0.1)
    base = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-1.0, 1.0])

    first, first_info = controller.project("route", auxiliary=auxiliary, base=base)
    second, second_info = controller.project("route", auxiliary=auxiliary, base=base)

    assert torch.equal(first, auxiliary)
    assert first_info["projected"] is False
    assert second_info["projected"] is True
    assert float(torch.dot(second, base)) >= -1e-6


def test_tensor_temporal_memory_uses_reliability_momentum_and_ttl() -> None:
    memory = TensorTemporalOrbitMemory(feature_dim=2, capacity=4, base_momentum=0.8, ttl=2)
    memory.update(
        keys=torch.tensor([10]),
        features=torch.tensor([[1.0, 0.0]]),
        reliability=torch.tensor([1.0]),
        scenario_bin=torch.tensor([1]),
        receiver_bin=torch.tensor([3]),
        step=1,
    )
    values, found, metadata = memory.lookup(torch.tensor([10]), step=2)
    _, expired, _ = memory.lookup(torch.tensor([10]), step=4)

    assert found.tolist() == [True]
    assert torch.allclose(values.norm(dim=1), torch.ones(1))
    assert metadata["scenario_bin"].tolist() == [1]
    assert expired.tolist() == [False]


def test_tensor_temporal_memory_keeps_key_across_adjacent_epochs() -> None:
    memory = TensorTemporalOrbitMemory(feature_dim=2, capacity=4, base_momentum=0.8, ttl=64)
    memory.update(
        keys=torch.tensor([10]),
        features=torch.tensor([[1.0, 0.0]]),
        reliability=torch.tensor([1.0]),
        scenario_bin=torch.tensor([1]),
        receiver_bin=torch.tensor([3]),
        step=7,
    )

    _, found, _ = memory.lookup(torch.tensor([10]), step=8)

    assert found.tolist() == [True]
    assert _daot_memory_step(epoch=8, batch_idx=999) == 8


def test_selective_subspace_is_identity_when_disabled_and_suppresses_nuisance_when_enabled() -> None:
    nuisance = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    fingerprint = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.0, -1.0], [-0.1, -0.9]])
    feature = torch.tensor([[1.0, 1.0]])
    disabled = SelectiveNuisanceSubspace(feature_dim=2, max_rank=1, weight=0.0)
    enabled = SelectiveNuisanceSubspace(feature_dim=2, max_rank=1, weight=1.0)

    assert torch.equal(disabled.project(feature), feature)
    assert enabled.update(nuisance, fingerprint) is True
    projected = enabled.project(feature)
    assert abs(float(projected[0, 0])) < abs(float(feature[0, 0]))
    assert abs(float(projected[0, 1])) > 0.5


def test_source_only_selection_builds_five_receiver_folds_and_penalizes_cost() -> None:
    folds = build_receiver_holdout_folds([1, 3, 4, 6, 8])
    fast = source_only_selection_score(
        cvar20=0.70,
        receiver_floor=0.65,
        clean_accuracy=0.82,
        leo_weak_mean=0.73,
        receiver_probe=0.40,
        relative_cost=1.0,
    )
    slow = source_only_selection_score(
        cvar20=0.70,
        receiver_floor=0.65,
        clean_accuracy=0.82,
        leo_weak_mean=0.73,
        receiver_probe=0.40,
        relative_cost=2.0,
    )

    assert len(folds) == 5
    assert {fold["holdout_receiver"] for fold in folds} == {1, 3, 4, 6, 8}
    assert all(len(fold["train_receivers"]) == 4 for fold in folds)
    assert fast > slow


def test_structured_batch_allocation_keeps_three_receivers_and_report_ratios() -> None:
    allocation = allocate_structured_batch(batch_size=100, receiver_count=5)

    assert allocation == {
        "cross_rx_labeled": 30,
        "balanced_unlabeled": 45,
        "hard_group": 15,
        "base_remainder": 10,
        "receiver_count": 5,
    }
    with pytest.raises(ValueError, match="at least three receivers"):
        allocate_structured_batch(batch_size=100, receiver_count=2)


def test_training_gradient_controller_projects_only_after_persistent_identity_conflict() -> None:
    class IdentityScaler:
        def get_scale(self):
            return 1.0

        def scale(self, value):
            return value

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_backbone = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.id_backbone.weight)

    model = ToyModel()
    controller = PersistentConflictProjector(window=2, threshold=-0.1)

    first = _backward_with_daot_persistent_projection(
        model,
        IdentityScaler(),
        base_loss=(model.id_backbone.weight - 1.0).square().sum(),
        auxiliary_groups={"route": model.id_backbone.weight.sum()},
        controller=controller,
    )
    first_grad = model.id_backbone.weight.grad.clone()
    second = _backward_with_daot_persistent_projection(
        model,
        IdentityScaler(),
        base_loss=(model.id_backbone.weight - 1.0).square().sum(),
        auxiliary_groups={"route": model.id_backbone.weight.sum()},
        controller=controller,
    )

    assert first["route"]["projected"] is False
    assert second["route"]["projected"] is True
    assert float(first_grad) == pytest.approx(-1.0)
    assert float(model.id_backbone.weight.grad) == pytest.approx(-2.0)


def test_training_gradient_controller_preserves_external_optimizer_head_gradient() -> None:
    class IdentityScaler:
        def get_scale(self):
            return 1.0

        def scale(self, value):
            return value

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_backbone = torch.nn.Linear(1, 1, bias=False)

    model = ToyModel()
    external_head = torch.nn.Linear(1, 1, bias=False)
    base_loss = model.id_backbone.weight.square().sum() + external_head.weight.square().sum()
    _backward_with_daot_persistent_projection(
        model,
        IdentityScaler(),
        base_loss=base_loss,
        auxiliary_groups={"route": model.id_backbone.weight.sum()},
        controller=PersistentConflictProjector(window=2, threshold=-0.1),
        optimizer_parameters=[*model.parameters(), *external_head.parameters()],
    )

    assert external_head.weight.grad is not None
    assert torch.isfinite(external_head.weight.grad).all()


def test_rx_v2_cli_override_has_priority_over_profile_defaults() -> None:
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "smoke-only",
            "--use_adv3b02_daot_stn_rx_v2",
            "true",
            "--daot_lambda_route",
            "0",
            "--daot_lambda_rx",
            "0",
        ]
    )

    _validate_daot_config(args)

    assert args.daot_lambda_route == pytest.approx(0.0)
    assert args.daot_lambda_rx == pytest.approx(0.0)
    assert args.daot_lambda_tail == pytest.approx(0.10)


def test_rx_v2_profile_disables_unregistered_legacy_losses() -> None:
    args = build_arg_parser().parse_args(
        ["--output_dir", "smoke-only", "--use_adv3b02_daot_stn_rx_v2", "true"]
    )

    _validate_daot_config(args)

    assert args.daot_lambda_nuisance == pytest.approx(0.0)
    assert args.daot_lambda_fingerprint == pytest.approx(0.0)
