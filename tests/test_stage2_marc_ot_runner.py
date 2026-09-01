from __future__ import annotations

import inspect
from collections import OrderedDict

import pytest
import torch
from torch import nn

from cvsrffi.stage2_marc_ot_runner import (
    MARCOT_PROGRESSIVE_STAGES,
    MARCOTRunnerConfig,
    combine_blockwise_gradients,
    predict_registered_logits,
    resolve_block_learning_rates,
    select_support_safe_state,
    train_marc_ot_arm,
)


class _IdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t1 = nn.Linear(2, 2, bias=False)
        self.t2 = nn.Linear(2, 2, bias=False)
        self.t3 = nn.Linear(2, 2, bias=False)
        self.f1 = nn.Linear(2, 2, bias=False)
        self.f2 = nn.Linear(2, 2, bias=False)
        self.f3 = nn.Linear(2, 2, bias=False)
        self.time_projection = nn.Linear(2, 2, bias=False)
        self.frequency_projection = nn.Linear(2, 2, bias=False)
        self.fusion = nn.Linear(2, 2, bias=False)
        self.identity_mapping = nn.Linear(2, 2, bias=False)
        self.norm = nn.LayerNorm(2)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _IdentityBackbone()
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))

    def forward(self, values: torch.Tensor, return_aux: bool = True):
        features = values.float()
        logits = torch.stack((features[:, 0], features[:, 1]), dim=1)
        return {"tx_logits": logits, "z_id": features}


def _state(seed: int) -> OrderedDict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return OrderedDict(
        weight=torch.randn(2, 2),
        counter=torch.tensor(17, dtype=torch.int64),
    )


def test_training_surface_has_no_query_argument() -> None:
    names = tuple(inspect.signature(train_marc_ot_arm).parameters)
    assert all("query" not in name.lower() for name in names)


def test_r8_combines_primary_with_projected_calibration_gradient() -> None:
    combined = combine_blockwise_gradients(
        ("id_backbone.t1.weight",),
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-2.0, 1.0]),),
        ratio_cap=10.0,
    )
    assert torch.allclose(combined[0], torch.tensor([1.0, 1.0]))


def test_task_conditioned_block_learning_rates_are_bounded_and_routed() -> None:
    config = MARCOTRunnerConfig(
        stage_steps=(1, 1, 1, 1),
        learning_rate_min=1.0e-5,
        learning_rate_max=3.0e-4,
    )
    resolved = resolve_block_learning_rates(
        ("id_backbone.t1.weight", "id_backbone.t2.weight", "id_backbone.norm.weight"),
        {"t1": 2.0e-5, "t2": 4.0e-5},
        config=config,
    )
    assert resolved == (2.0e-5, 4.0e-5, 3.0e-4)
    with pytest.raises(ValueError, match="bounds"):
        resolve_block_learning_rates(
            ("id_backbone.t1.weight",),
            {"t1": 1.0e-2},
            config=config,
        )


def test_all_unsafe_interpolations_restore_base_state_duals_and_integer_buffer() -> None:
    base = _state(1)
    candidate = _state(2)
    base_duals = {"class_duals": torch.tensor([0.25, 0.5])}
    candidate_duals = {"class_duals": torch.tensor([8.0, 9.0])}

    selected = select_support_safe_state(
        base,
        candidate,
        base_duals=base_duals,
        candidate_duals=candidate_duals,
        evaluator=lambda _state, _duals: {"safe": False, "oof_ba": 0.0},
        grid=(1.0, 0.5, 0.0),
        trainable_parameter_names=("weight",),
    )

    assert selected.selected_alpha == 0.0
    assert selected.query_rows_used == 0
    for name, value in base.items():
        assert torch.equal(selected.state[name], value)
        assert selected.state[name].dtype == value.dtype
    assert torch.equal(selected.duals["class_duals"], base_duals["class_duals"])


def test_progressive_runner_uses_fixed_stage_order_and_refreezes() -> None:
    model = _TinyModel()
    observed: list[tuple[str, tuple[str, ...]]] = []

    def stage_update(current, stage, trainable_names, _steps, duals):
        observed.append((stage, tuple(trainable_names)))
        candidate = {name: value.detach().clone() for name, value in current.state_dict().items()}
        for name in trainable_names:
            candidate[name] = candidate[name] + 0.01
        return candidate, {name: value + 0.01 for name, value in duals.items()}

    audit = train_marc_ot_arm(
        model,
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([0, 1]),
        ("s0", "s1"),
        arm="R8",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
        initial_duals={"class_duals": torch.zeros(2)},
        stage_update=stage_update,
        support_evaluator=lambda _state, _duals: {"safe": True, "oof_ba": 1.0},
    )

    assert tuple(stage for stage, _ in observed) == MARCOT_PROGRESSIVE_STAGES
    assert all(names for _, names in observed)
    assert audit.query_rows_used == 0
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_runner_refreezes_after_stage_exception() -> None:
    model = _TinyModel()

    with pytest.raises(RuntimeError, match="boom"):
        train_marc_ot_arm(
            model,
            torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
            torch.tensor([0, 0, 1, 1]),
            ("s0", "s1", "s2", "s3"),
            arm="R8",
            config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
            stage_update=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_prediction_requires_frozen_eval_and_is_batch_and_order_invariant() -> None:
    model = _TinyModel()
    query = torch.tensor([[3.0, 1.0], [0.5, 4.0], [2.0, 1.0]])
    tokens = ("q2", "q0", "q1")

    with pytest.raises(ValueError, match="frozen eval"):
        predict_registered_logits(
            model,
            query,
            query_tokens=tokens,
            class_registry=("old0", "old1"),
            batch_size=2,
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    first = predict_registered_logits(
        model,
        query,
        query_tokens=tokens,
        class_registry=("old0", "old1"),
        batch_size=1,
    )
    permutation = torch.tensor([1, 2, 0])
    reordered = predict_registered_logits(
        model,
        query[permutation],
        query_tokens=tuple(tokens[index] for index in permutation.tolist()),
        class_registry=("old0", "old1"),
        batch_size=3,
    )
    first_by_token = dict(zip(first["query_tokens"].tolist(), first["predictions"].tolist()))
    reordered_by_token = dict(
        zip(reordered["query_tokens"].tolist(), reordered["predictions"].tolist())
    )
    assert first_by_token == reordered_by_token == {"q2": 0, "q0": 1, "q1": 0}
