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
import cvsrffi.stage2_marc_ot_runner as runner_subject


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


class _DifferentiableTinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _IdentityBackbone()

    def forward(self, values: torch.Tensor, return_aux: bool = True):
        features = self.id_backbone.norm(values.float())
        return {"tx_logits": features, "z_id": features}


def _test_calibration_transform(
    model: nn.Module,
    values: torch.Tensor,
    _labels: torch.Tensor,
    _tokens: tuple[str, ...],
    _fit_scope: str,
) -> torch.Tensor:
    return model(values, return_aux=True)["z_id"]


def _state(seed: int) -> OrderedDict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return OrderedDict(
        weight=torch.randn(2, 2),
        counter=torch.tensor(17, dtype=torch.int64),
    )


def test_training_surface_has_no_query_argument() -> None:
    names = tuple(inspect.signature(train_marc_ot_arm).parameters)
    assert all("query" not in name.lower() for name in names)


@pytest.mark.parametrize("arm", ("R2", "R4", "R6", "R8"))
def test_canonical_feature_arms_reject_missing_transform(arm: str) -> None:
    with pytest.raises(ValueError, match="calibration_feature_transform"):
        train_marc_ot_arm(
            _TinyModel(),
            torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
            torch.tensor([0, 0, 1, 1]),
            ("a0", "a1", "b0", "b1"),
            arm=arm,
            config=MARCOTRunnerConfig(stage_steps=(0, 0, 0, 0), fold_count=2),
        )


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


def test_t1_f1_learning_rate_has_explicit_lower_floor() -> None:
    config = MARCOTRunnerConfig(
        learning_rate_min=1.0e-5,
        learning_rate_t1_f1_min=3.0e-6,
    )
    assert resolve_block_learning_rates(
        ("id_backbone.t1.weight",), {"t1": 3.0e-6}, config=config
    ) == (3.0e-6,)
    with pytest.raises(ValueError, match="outside frozen bounds"):
        resolve_block_learning_rates(
            ("id_backbone.t2.weight",), {"t2": 3.0e-6}, config=config
        )


def test_production_selection_uses_exact_d92_identity160_fft96(monkeypatch) -> None:
    import cvsrffi.stage2_binova_d92 as d92_module
    import cvsrffi.stage2_binova_features as feature_module

    observed: dict[str, object] = {}

    class Model(nn.Module):
        def forward(self, values, return_aux=True):
            identity = values.reshape(len(values), -1)[:, :160]
            return {"tx_logits": identity[:, :6], "z_id": identity}

    class Fit:
        def score(self, identity, fft):
            observed["score_geometry"] = (identity.shape, fft.shape)
            return torch.eye(6).numpy()

    def fake_fit(identity, fft, labels, **kwargs):
        observed["fit_geometry"] = (identity.shape, fft.shape, tuple(kwargs["class_ids"]))
        return Fit()

    monkeypatch.setattr(d92_module, "exact_d92_fit", fake_fit)
    monkeypatch.setattr(
        feature_module,
        "make_fft96",
        lambda iq: torch.zeros(len(iq), 96).numpy(),
    )
    model = Model()
    state = model.state_dict()
    fit_iq = torch.randn(6, 2, 256)
    validation_iq = torch.randn(6, 2, 256)
    labels = torch.arange(6)
    result = runner_subject._default_fold_metrics(
        model,
        state,
        fit_iq,
        labels,
        validation_iq,
        labels,
        selection_mode="EXACT_D92_OLD_ONLY",
        seed=713102,
    )
    assert result["safe"] is True
    assert observed["fit_geometry"] == ((6, 160), (6, 96), tuple(range(6)))
    assert observed["score_geometry"] == ((6, 160), (6, 96))


@pytest.mark.parametrize(
    ("config", "expected_decay", "expected_clip"),
    (
        (MARCOTRunnerConfig(), 0.0, None),
        (
            MARCOTRunnerConfig(
                optimizer_weight_decay=1.0e-4,
                gradient_clip_norm=5.0,
                learning_rate_t1_f1_min=3.0e-6,
            ),
            1.0e-4,
            5.0,
        ),
    ),
)
def test_optimizer_v1_legacy_and_v2_explicit_semantics(
    monkeypatch, config, expected_decay, expected_clip
) -> None:
    observed: dict[str, object] = {"clips": []}
    real_adamw = torch.optim.AdamW

    def adamw(*args, **kwargs):
        observed["weight_decay"] = kwargs["weight_decay"]
        return real_adamw(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", adamw)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda _parameters, max_norm: observed["clips"].append(max_norm),
    )
    model = _DifferentiableTinyModel()
    base = {name: value.detach().clone() for name, value in model.state_dict().items()}
    runner_subject._default_stage_update(
        model,
        "norm_fusion_projection",
        ("id_backbone.norm.weight",),
        1,
        {},
        values=torch.tensor([[2.0, 0.1], [1.5, 0.2], [0.1, 2.0], [0.2, 1.5]]),
        labels=torch.tensor([0, 0, 1, 1]),
        tokens=("a0", "a1", "b0", "b1"),
        arm="R1",
        config=config,
        bank_task_features=None,
        calibration_feature_transform=None,
        fit_scope="full_support",
        block_learning_rates=None,
        original_base=base,
    )
    assert observed["weight_decay"] == expected_decay
    assert observed["clips"] == ([] if expected_clip is None else [expected_clip])


def test_default_stage_transform_receives_fit_iq_model_and_keeps_gradient_path() -> None:
    model = _DifferentiableTinyModel()
    values = torch.tensor([[2.0, 0.1], [1.5, 0.2], [0.1, 2.0], [0.2, 1.5]])
    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("a0", "a1", "b0", "b1")
    observed: list[tuple[int, tuple[str, ...], bool, str]] = []

    def transform(current, fit_iq, fit_labels, fit_tokens, fit_scope):
        assert current is model
        logits, features = runner_subject._forward_identity(current, fit_iq)
        del logits, fit_labels
        observed.append(
            (fit_iq.data_ptr(), tuple(fit_tokens), features.requires_grad, fit_scope)
        )
        return features

    runner_subject._default_stage_update(
        model,
        "norm_fusion_projection",
        ("id_backbone.norm.weight", "id_backbone.norm.bias"),
        1,
        {"class_duals": torch.zeros(2)},
        values=values,
        labels=labels,
        tokens=tokens,
        arm="R2",
        config=MARCOTRunnerConfig(stage_steps=(1, 0, 0, 0), fold_count=2),
        bank_task_features=None,
        calibration_feature_transform=transform,
        fit_scope="crossfit",
        block_learning_rates=None,
        original_base={name: value.detach().clone() for name, value in model.state_dict().items()},
    )

    assert observed == [(values.data_ptr(), tokens, True, "crossfit")]


def test_production_r1_r2_single_step_contains_nonzero_supcon_gradient_increment() -> None:
    """Removing SupCon from the real R2 stage makes the weight-scaled gradient delta zero."""
    from cvsrffi.stage2_marc_ot import supervised_contrastive_support_loss

    class StageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_backbone = _IdentityBackbone()

        def forward(self, values: torch.Tensor, return_aux: bool = True):
            del return_aux
            features = self.id_backbone.t1(values.float())
            return {"tx_logits": features, "z_id": features}

    torch.manual_seed(17)
    reference = StageModel()
    initial = {
        name: value.detach().clone() for name, value in reference.state_dict().items()
    }
    values = torch.tensor(
        [[2.0, 0.1], [0.1, 2.0], [1.8, 0.2], [0.2, 1.8]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("a0", "a1", "b0", "b1")
    trainable = ("id_backbone.t1.weight",)

    def production_step(arm: str, weight: float):
        model = StageModel()
        model.load_state_dict(initial, strict=True)
        config = MARCOTRunnerConfig(
            stage_steps=(1, 0, 0, 0),
            fold_count=2,
            learning_rate_min=9.0e-4,
            learning_rate_max=1.0e-3,
            supcon_weight=weight,
        )
        state, _duals = runner_subject._default_stage_update(
            model,
            "norm_fusion_projection",
            trainable,
            1,
            {"class_duals": torch.zeros(2)},
            values=values,
            labels=labels,
            tokens=tokens,
            arm=arm,
            config=config,
            bank_task_features=None,
            calibration_feature_transform=(
                _test_calibration_transform if arm == "R2" else None
            ),
            fit_scope="crossfit",
            block_learning_rates=None,
            original_base=initial,
        )
        gradients = torch.cat(
            [dict(model.named_parameters())[name].grad.detach().reshape(-1) for name in trainable]
        )
        updates = torch.cat(
            [(state[name] - initial[name]).detach().reshape(-1) for name in trainable]
        )
        return gradients, updates

    r1_gradient, r1_update = production_step("R1", 0.1)
    r1_high_gradient, _r1_high_update = production_step("R1", 0.2)
    r2_low_gradient, r2_low_update = production_step("R2", 0.1)
    r2_high_gradient, r2_high_update = production_step("R2", 0.2)

    expected_model = StageModel()
    expected_model.load_state_dict(initial, strict=True)
    expected_features = expected_model(values, return_aux=True)["z_id"]
    expected_supcon = supervised_contrastive_support_loss(
        expected_features,
        labels,
        temperature=0.07,
    ).loss
    expected_increment = torch.cat(
        [
            value.detach().reshape(-1)
            for value in torch.autograd.grad(
                0.1 * expected_supcon,
                tuple(dict(expected_model.named_parameters())[name] for name in trainable),
            )
        ]
    )

    assert torch.count_nonzero(expected_increment)
    assert torch.allclose(
        r2_high_gradient - r2_low_gradient,
        expected_increment,
        atol=1.0e-5,
        rtol=1.0e-4,
    )
    assert torch.equal(r1_gradient, r1_high_gradient)
    assert torch.count_nonzero(r2_high_gradient - r2_low_gradient)
    assert torch.count_nonzero(r1_update)
    assert torch.count_nonzero(r2_low_update)
    assert torch.count_nonzero(r2_high_update)


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

    def stage_update(current, stage, trainable_names, _steps, duals, *_fit):
        if _fit[-1] == "full_support":
            observed.append((stage, tuple(trainable_names)))
        candidate = {name: value.detach().clone() for name, value in current.state_dict().items()}
        for name in trainable_names:
            candidate[name] = candidate[name] + 0.01
        return candidate, {name: value + 0.01 for name, value in duals.items()}

    audit = train_marc_ot_arm(
        model,
        torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        torch.tensor([0, 0, 1, 1]),
        ("s0", "s1", "s2", "s3"),
        arm="R8",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
        initial_duals={"class_duals": torch.zeros(2)},
        calibration_feature_transform=_test_calibration_transform,
        stage_update=stage_update,
        support_evaluator=lambda *_args: {"safe": True, "oof_ba": 1.0},
    )

    assert tuple(stage for stage, _ in observed) == MARCOT_PROGRESSIVE_STAGES
    assert all(names for _, names in observed)
    assert audit.query_rows_used == 0
    assert audit.held_out_support_evidence is True
    assert audit.support_cv_evidence == {
        "schema": "cvs.phase2.marc_ot.support_cv.v1",
        "source": "TRUE_HELD_OUT_CROSSFIT",
        "fold_count": 2,
        "baseline_balanced_accuracy": 1.0,
        "selected_balanced_accuracy": 1.0,
        "balanced_accuracy_delta_pp": 0.0,
        "baseline_class_floor": 1.0,
        "selected_class_floor": 1.0,
        "class_floor_delta_pp": 0.0,
    }
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_global_alpha_preserves_early_acceptance_when_later_stages_reject() -> None:
    model = _TinyModel()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def stage_update(current, stage, trainable_names, _steps, duals, *_fit):
        candidate = {name: value.detach().clone() for name, value in current.state_dict().items()}
        for name in trainable_names:
            candidate[name] = candidate[name] + 0.1
        return candidate, {name: value + 0.1 for name, value in duals.items()}

    def evaluator(state, _duals, *_fold):
        changed = [
            name
            for name, value in state.items()
            if value.is_floating_point() and not torch.equal(value, original[name])
        ]
        stage_one = all(
            torch.allclose(state[name], original[name] + 0.1) for name in changed
        )
        return {"safe": bool(changed) and stage_one}

    audit = train_marc_ot_arm(
        model,
        torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        torch.tensor([0, 0, 1, 1]),
        ("a0", "a1", "b0", "b1"),
        arm="R8",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
        calibration_feature_transform=_test_calibration_transform,
        stage_update=stage_update,
        support_evaluator=evaluator,
    )

    assert audit.stage_selected_alphas == (1.0, 0.0, 0.0, 0.0)
    assert audit.selected_alpha == 1.0
    assert torch.equal(model.counter, original["counter"])


def test_adapter_crossfit_never_optimizes_validation_tokens() -> None:
    model = _TinyModel()
    all_tokens = {"a0", "a1", "b0", "b1"}
    crossfit_updates: list[set[str]] = []
    evaluated_folds: list[tuple[set[str], set[str]]] = []
    learning_rate_scopes: list[tuple[str, set[str]]] = []

    def stage_update(
        current,
        _stage,
        trainable_names,
        _steps,
        duals,
        _fit_iq,
        _fit_labels,
        fit_tokens,
        fit_scope,
    ):
        if fit_scope == "crossfit":
            crossfit_updates.append(set(fit_tokens))
        candidate = {name: value.detach().clone() for name, value in current.state_dict().items()}
        for name in trainable_names:
            candidate[name] = candidate[name] + 0.01
        return candidate, {name: value.detach().clone() for name, value in duals.items()}

    def evaluator(
        _state,
        _duals,
        _fit_iq,
        _fit_labels,
        fit_tokens,
        _validation_iq,
        _validation_labels,
        validation_tokens,
    ):
        evaluated_folds.append((set(fit_tokens), set(validation_tokens)))
        return {"safe": True, "oof_ba": 1.0, "oof_floor": 1.0}

    def learning_rate_factory(_fit_iq, _fit_labels, fit_tokens, fit_scope):
        learning_rate_scopes.append((fit_scope, set(fit_tokens)))
        return {}

    train_marc_ot_arm(
        model,
        torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        torch.tensor([0, 0, 1, 1]),
        ("a0", "a1", "b0", "b1"),
        arm="R8",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
        calibration_feature_transform=_test_calibration_transform,
        block_learning_rate_factory=learning_rate_factory,
        stage_update=stage_update,
        support_evaluator=evaluator,
    )

    assert crossfit_updates
    assert evaluated_folds
    assert all(fit and validation for fit, validation in evaluated_folds)
    assert all(fit.isdisjoint(validation) for fit, validation in evaluated_folds)
    assert all(fit | validation == all_tokens for fit, validation in evaluated_folds)
    assert all(updated != all_tokens for updated in crossfit_updates)
    assert any(scope == "crossfit" for scope, _tokens in learning_rate_scopes)
    assert any(scope == "full_support" for scope, _tokens in learning_rate_scopes)
    assert all(
        tokens != all_tokens
        for scope, tokens in learning_rate_scopes
        if scope == "crossfit"
    )
    assert all(
        tokens == all_tokens
        for scope, tokens in learning_rate_scopes
        if scope == "full_support"
    )


def test_initial_bank_candidate_is_fit_per_fold_before_full_support_refit() -> None:
    model = _TinyModel()
    all_tokens = {"a0", "a1", "b0", "b1"}
    factory_calls: list[tuple[str, set[str]]] = []

    def initial_state_factory(_fit_iq, _fit_labels, fit_tokens, fit_scope):
        factory_calls.append((fit_scope, set(fit_tokens)))
        candidate = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        for name, value in candidate.items():
            if value.is_floating_point():
                candidate[name] = value + 0.05
        return candidate

    def stage_update(current, _stage, _names, _steps, duals, *_fit):
        return (
            {name: value.detach().clone() for name, value in current.state_dict().items()},
            {name: value.detach().clone() for name, value in duals.items()},
        )

    train_marc_ot_arm(
        model,
        torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        torch.tensor([0, 0, 1, 1]),
        ("a0", "a1", "b0", "b1"),
        arm="R4",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
        calibration_feature_transform=_test_calibration_transform,
        initial_state_factory=initial_state_factory,
        stage_update=stage_update,
        support_evaluator=lambda *_args: {"safe": True},
    )

    crossfit_tokens = [tokens for scope, tokens in factory_calls if scope == "crossfit"]
    full_tokens = [tokens for scope, tokens in factory_calls if scope == "full_support"]
    assert crossfit_tokens and all(tokens != all_tokens for tokens in crossfit_tokens)
    assert full_tokens == [all_tokens]


def test_all_unsafe_stages_restore_immutable_original_model_duals_and_buffer() -> None:
    model = _TinyModel()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_duals = {"class_duals": torch.tensor([0.25, 0.5])}
    initial_scopes: list[str] = []

    def initial_state_factory(_fit_iq, _fit_labels, _fit_tokens, fit_scope):
        initial_scopes.append(fit_scope)
        candidate = {name: value.detach().clone() for name, value in model.state_dict().items()}
        for name, value in candidate.items():
            if value.is_floating_point():
                candidate[name] = value + 3.0
        candidate["counter"] = torch.tensor(123, dtype=torch.int64)
        return candidate

    def stage_update(current, _stage, trainable_names, _steps, duals, *_fit):
        candidate = {name: value.detach().clone() for name, value in current.state_dict().items()}
        for name in trainable_names:
            candidate[name] = candidate[name] + 2.0
        candidate["counter"] = torch.tensor(999, dtype=torch.int64)
        return candidate, {name: value + 4.0 for name, value in duals.items()}

    audit = train_marc_ot_arm(
        model,
        torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        torch.tensor([0, 0, 1, 1]),
        ("a0", "a1", "b0", "b1"),
        arm="R8",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1), fold_count=2),
        calibration_feature_transform=_test_calibration_transform,
        initial_state_factory=initial_state_factory,
        initial_duals=original_duals,
        stage_update=stage_update,
        support_evaluator=lambda *_args: {"safe": False},
    )

    assert audit.selected_alpha == 0.0
    assert audit.initial_selected_alpha == 0.0
    assert audit.stage_selected_alphas == (0.0, 0.0, 0.0, 0.0)
    assert "full_support" not in initial_scopes
    for name, value in original.items():
        assert torch.equal(model.state_dict()[name], value)
    assert audit.final_duals["class_duals"] == (0.25, 0.5)
    assert torch.equal(model.counter, original["counter"])


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
            calibration_feature_transform=_test_calibration_transform,
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
