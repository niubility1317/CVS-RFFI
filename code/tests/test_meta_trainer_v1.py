from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.meta_adapter import ResidualMetaAdapter  # noqa: E402
from cvsrffi.meta_episodes import EpisodeKind, MetaEpisode, MetaSampleRef  # noqa: E402
from cvsrffi.meta_trainer import (  # noqa: E402
    AdaptationCurve,
    MetaEpisodeBatch,
    MetaTrainerConfig,
    SourceCheckpointCandidate,
    SourceHoldoutDelta,
    build_phase1b_optimizer,
    build_phase1c_optimizer,
    evaluate_adaptation_curve,
    optimizer_parameter_names,
    run_meta_train_step,
    select_source_checkpoint,
)
from model import build_model  # noqa: E402


class TinyMetaModel(nn.Module):
    def __init__(self, class_count: int = 3) -> None:
        super().__init__()
        self.t_proj = nn.Linear(3, 4)
        self.f_proj = nn.Linear(3, 4)
        self.fuse = nn.Sequential(nn.Linear(8, 4))
        self.meta_adapter_time = ResidualMetaAdapter(4, rank=2)
        self.meta_adapter_freq = ResidualMetaAdapter(4, rank=2)
        self.meta_adapter_fusion = ResidualMetaAdapter(4, rank=2)
        self.dropout = nn.Dropout(p=0.2)
        self.cls_head = nn.Linear(4, class_count)
        self.register_buffer("frozen_counter", torch.tensor(3.0))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        return_aux: bool = True,
    ) -> dict[str, torch.Tensor]:
        del y
        t = self.meta_adapter_time(self.t_proj(x))
        f = self.meta_adapter_freq(self.f_proj(x))
        z = self.dropout(self.meta_adapter_fusion(self.fuse(torch.cat((t, f), dim=1))))
        logits = self.cls_head(z)
        if not return_aux:
            return {"logits": logits}
        return {"logits": logits, "feat_cls": z}


def _model() -> TinyMetaModel:
    torch.manual_seed(7)
    return TinyMetaModel()


def _config(**kwargs) -> MetaTrainerConfig:
    kwargs.setdefault("source_receiver_ids", (0,))
    return MetaTrainerConfig(**kwargs)


def _episode(role: str = "L_s", *, rx_i: int | str = 0) -> MetaEpisode:
    support = tuple(
        MetaSampleRef(
            dataset_index=i,
            tx_i=i,
            rx_i=rx_i,
            day_i=0,
            eq_i=0,
            capture_block_i=0,
            physical_sample_id=f"support-{i}",
            role=role,
            view="clean",
        )
        for i in range(2)
    )
    query_adapt = (
        MetaSampleRef(
            dataset_index=10,
            tx_i=0,
            rx_i=rx_i,
            day_i=1,
            eq_i=0,
            capture_block_i=1,
            physical_sample_id="query-0",
            role=role,
            view="clean",
        ),
        MetaSampleRef(
            dataset_index=11,
            tx_i=1,
            rx_i=rx_i,
            day_i=1,
            eq_i=0,
            capture_block_i=1,
            physical_sample_id="query-1",
            role=role,
            view="leo_clear_weak",
        ),
    )
    query_guard = (
        MetaSampleRef(
            dataset_index=12,
            tx_i=2,
            rx_i=rx_i,
            day_i=1,
            eq_i=0,
            capture_block_i=1,
            physical_sample_id="query-2",
            role=role,
            view="leo_rain_weak",
        ),
    )
    return MetaEpisode(
        kind=EpisodeKind.CLEAN_TO_LEO,
        support=support,
        query_adapt=query_adapt,
        query_guard=query_guard,
        adapt_class_ids=frozenset({0, 1}),
        guard_class_ids=frozenset({2}),
        k_shot=1,
        seed=17,
    )


def _batch(role: str = "L_s", *, rx_i: int | str = 0) -> MetaEpisodeBatch:
    episode = _episode(role, rx_i=rx_i)
    return MetaEpisodeBatch(
        episode=episode,
        support_x=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]),
        support_y=torch.tensor([0, 1], dtype=torch.long),
        query_x=torch.tensor(
            [[0.8, 0.1, 0.2], [0.2, 0.9, -0.1], [-0.4, 0.2, 0.8]],
        ),
        query_y=torch.tensor([0, 1, 2], dtype=torch.long),
        adapt_mask=torch.tensor([True, True, False]),
        guard_mask=torch.tensor([False, False, True]),
        frozen_prototypes=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ),
    )


def _parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
    }


def _candidate(
    candidate_id: str,
    clean_delta_pp: float,
    guard_floor_delta_pp: float,
    worst_a3_delta_pp: float,
    parameter_count: int,
    latency_ms: float,
) -> SourceCheckpointCandidate:
    return SourceCheckpointCandidate(
        candidate_id=candidate_id,
        clean_delta_pp=clean_delta_pp,
        guard_floor_delta_pp=guard_floor_delta_pp,
        worst_a3_delta_pp=worst_a3_delta_pp,
        parameter_count=parameter_count,
        latency_ms=latency_ms,
        source_holdouts=(
            SourceHoldoutDelta(
                holdout_id=f"{candidate_id}-holdout",
                a0=0.0,
                a3=worst_a3_delta_pp / 100.0,
            ),
        ),
    )


def test_phase1b_optimizer_contains_exact_adapter_and_log_step_size_parameters():
    model = _model()
    optimizer = build_phase1b_optimizer(model, _config())
    names = set(optimizer_parameter_names(model, optimizer))
    expected = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("meta_adapter_")
        and (
            name.endswith(("down.weight", "down.bias", "up.weight", "up.bias", "gate"))
            or name.endswith("log_step_size")
        )
    }
    assert names == expected
    assert not any(name.startswith(("t_proj", "f_proj", "fuse", "cls_head")) for name in names)
    assert all(
        parameter.requires_grad == (name in expected)
        for name, parameter in model.named_parameters()
    )


def test_phase1c_optimizer_adds_only_real_backbone_projection_parameters_at_ratio_lr():
    model = _model()
    config = _config(adapter_outer_lr=2.0e-3)
    optimizer = build_phase1c_optimizer(model, config)
    names = set(optimizer_parameter_names(model, optimizer))
    phase1b_names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("meta_adapter_")
        and (
            name.endswith(("down.weight", "down.bias", "up.weight", "up.bias", "gate"))
            or name.endswith("log_step_size")
        )
    }
    added = names - phase1b_names
    assert added == {
        "t_proj.weight",
        "t_proj.bias",
        "f_proj.weight",
        "f_proj.bias",
        "fuse.0.weight",
        "fuse.0.bias",
    }
    assert not any(name.startswith("cls_head") for name in names)
    assert all(
        group["lr"] in {config.adapter_outer_lr, config.adapter_outer_lr * 0.05}
        for group in optimizer.param_groups
    )
    backbone_names = set()
    for group in optimizer.param_groups:
        if group["lr"] == pytest.approx(config.adapter_outer_lr * 0.05):
            backbone_names.update(optimizer_parameter_names(model, optimizer, group=group))
    assert backbone_names == added


def test_meta_batch_defaults_to_four_and_inner_loop_to_three_steps():
    config = _config()
    assert config.meta_batch_size == 4
    assert config.inner_steps == 3


def test_run_meta_train_step_averages_four_independent_episodes_and_logs_finite_terms():
    model = _model()
    optimizer = build_phase1b_optimizer(model, _config())
    before = _parameter_snapshot(model)
    result = run_meta_train_step(model, [_batch() for _ in range(4)], optimizer, _config())
    assert torch.isfinite(result.loss)
    assert len(result.episode_logs) == 4
    assert all(log["episode_kind"] == EpisodeKind.CLEAN_TO_LEO.value for log in result.episode_logs)
    assert all(log["k_shot"] == 1 and log["inner_steps"] == 3 for log in result.episode_logs)
    for log in result.episode_logs:
        for key in ("loss_adapt", "loss_guard", "loss_floor"):
            assert isinstance(log[key], float) and torch.isfinite(torch.tensor(log[key]))
        assert log["grad_cos_support_query"] is None or torch.isfinite(
            torch.tensor(log["grad_cos_support_query"])
        )
    changed = [
        name
        for name, value in model.named_parameters()
        if not torch.equal(value.detach(), before[name])
    ]
    assert changed
    assert set(changed).issubset(set(result.optimizer_parameter_names))


def test_run_meta_train_step_rejects_validation_roles_and_wrong_meta_batch_size():
    model = _model()
    optimizer = build_phase1b_optimizer(model, _config())
    with pytest.raises(ValueError, match="L_s"):
        run_meta_train_step(model, [_batch("V_cal") for _ in range(4)], optimizer, _config())
    with pytest.raises(ValueError, match="meta batch"):
        run_meta_train_step(model, [_batch()], optimizer, _config())


def test_outer_step_does_not_change_head_backbone_or_buffers():
    model = _model()
    optimizer = build_phase1b_optimizer(model, _config())
    before = _parameter_snapshot(model)
    result = run_meta_train_step(model, [_batch() for _ in range(4)], optimizer, _config())
    whitelist = set(result.optimizer_parameter_names)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if name not in whitelist:
            assert torch.equal(value.detach(), before[name]), name


def test_meta_episode_batch_rejects_physical_id_overlap_and_train_revalidates():
    base = _batch()
    duplicate_support = replace(
        base.episode.support[1],
        physical_sample_id=base.episode.support[0].physical_sample_id,
    )
    invalid = replace(base.episode, support=(base.episode.support[0], duplicate_support))
    with pytest.raises(ValueError, match="physical_sample_id"):
        replace(base, episode=invalid)

    bypass = _batch()
    object.__setattr__(bypass, "episode", invalid)
    model = _model()
    optimizer = build_phase1b_optimizer(model, _config())
    with pytest.raises(ValueError, match="physical_sample_id"):
        run_meta_train_step(model, [bypass] * 4, optimizer, _config())


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("support_x", torch.tensor([1.0, 0.0, 0.5])),
        ("support_y", torch.tensor([[0, 1]], dtype=torch.long)),
        ("query_x", torch.tensor([[0.8, 0.1, 0.2], [0.2, 0.9, -0.1]])),
        ("query_y", torch.tensor([0, 1], dtype=torch.long)),
        ("adapt_mask", torch.tensor([[True, True, False]])),
        ("guard_mask", torch.tensor([True, False])),
    ),
)
def test_train_entry_revalidates_mutated_meta_episode_batch_before_forward(field, replacement):
    batch = _batch()
    object.__setattr__(batch, field, replacement)
    model = _model()
    optimizer = build_phase1b_optimizer(model, _config())
    forward_calls = []
    hook = model.register_forward_pre_hook(lambda *_args: forward_calls.append(True))
    try:
        with pytest.raises(ValueError, match="MetaEpisodeBatch integrity"):
            run_meta_train_step(model, [batch] * 4, optimizer, _config())
    finally:
        hook.remove()
    assert forward_calls == []


def test_source_receiver_allowlist_is_explicit_and_rejects_unknown_ids():
    with pytest.raises((TypeError, ValueError), match="source_receiver_ids"):
        MetaTrainerConfig()
    config = _config()
    model = _model()
    optimizer = build_phase1b_optimizer(model, config)
    with pytest.raises(ValueError, match="source_receiver_ids|999"):
        run_meta_train_step(model, [_batch(rx_i=999)] * 4, optimizer, config)
    with pytest.raises(ValueError, match="source_receiver_ids|999"):
        evaluate_adaptation_curve(model, [_batch("V_cal", rx_i=999)], config)


def test_evaluate_adaptation_curve_has_fixed_steps_and_preserves_model_state_and_grads():
    model = _model()
    model.train()
    before = _parameter_snapshot(model)
    curve = evaluate_adaptation_curve(model, [_batch("V_cal"), _batch("V_select")], _config())
    assert isinstance(curve, AdaptationCurve)
    assert curve.steps == (0, 1, 3, 5, 10)
    assert len(curve.rows) == 10
    assert {row.step for row in curve.rows} == {0, 1, 3, 5, 10}
    assert all(row.role in {"V_cal", "V_select"} for row in curve.rows)
    assert all(row.mean_accuracy is not None for row in curve.rows)
    assert all(torch.isfinite(torch.tensor(row.mean_accuracy)) for row in curve.rows)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training
    after = _parameter_snapshot(model)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_evaluate_adaptation_curve_preserves_callers_existing_gradients():
    model = _model()
    expected = {}
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))
        expected[id(parameter)] = parameter.grad.detach().clone()
    evaluate_adaptation_curve(model, [_batch("V_cal")], _config())
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, expected[id(parameter)])


def test_evaluate_adaptation_curve_temporarily_uses_eval_mode_and_restores_nested_flags():
    model = _model()
    model.train()
    model.dropout.eval()
    before = {id(module): module.training for module in model.modules()}
    observed = []
    handle = model.dropout.register_forward_hook(lambda module, args, output: observed.append(module.training))
    try:
        evaluate_adaptation_curve(model, [_batch("V_cal")], _config())
    finally:
        handle.remove()
    assert observed and not any(observed)
    assert {id(module): module.training for module in model.modules()} == before


def test_evaluate_adaptation_curve_rejects_training_role_and_target_receiver_marker():
    model = _model()
    with pytest.raises(ValueError, match="V_cal|V_select"):
        evaluate_adaptation_curve(model, [_batch("L_s")], _config())
    with pytest.raises(ValueError, match="target"):
        evaluate_adaptation_curve(model, [_batch("V_cal", rx_i="target_receiver")], _config())


def test_source_selection_filters_floor_and_zero_step_then_uses_deterministic_ties():
    candidates = [
        _candidate("bad_clean", -0.6, 0.0, 9.0, 1, 1.0),
        _candidate("bad_guard", 0.0, -0.1, 9.0, 1, 1.0),
        _candidate("valid_large", -0.2, 0.0, 1.1, 20, 2.0),
        _candidate("valid_small", -0.2, 0.0, 1.1, 10, 3.0),
    ]
    assert select_source_checkpoint(candidates).candidate_id == "valid_small"


def test_source_selection_uses_latency_then_candidate_id_and_fails_when_empty():
    tied = [
        _candidate("zeta", 0.0, 0.0, 2.0, 10, 4.0),
        _candidate("alpha", 0.0, 0.0, 2.0, 10, 4.0),
        _candidate("beta", 0.0, 0.0, 2.0, 10, 3.0),
    ]
    assert select_source_checkpoint(tied).candidate_id == "beta"
    with pytest.raises(ValueError, match="eligible|candidate"):
        select_source_checkpoint(
            [_candidate("bad", -0.6, -0.1, 3.0, 1, 1.0)]
        )
    with pytest.raises(ValueError, match="empty|candidate"):
        select_source_checkpoint([])


def test_source_checkpoint_candidate_rejects_target_or_query_fields():
    with pytest.raises(TypeError):
        SourceCheckpointCandidate(
            candidate_id="bad",
            clean_delta_pp=0.0,
            guard_floor_delta_pp=0.0,
            worst_a3_delta_pp=1.0,
            source_holdouts=(SourceHoldoutDelta("bad-holdout", 0.0, 0.01),),
            target_accuracy=0.9,  # type: ignore[call-arg]
        )


def test_source_checkpoint_rejects_claimed_worst_delta_that_disagrees_with_curve():
    with pytest.raises(ValueError, match="derived|worst"):
        SourceCheckpointCandidate(
            candidate_id="claimed",
            clean_delta_pp=0.0,
            guard_floor_delta_pp=0.0,
            worst_a3_delta_pp=99.0,
            parameter_count=1,
            latency_ms=1.0,
            source_holdouts=(SourceHoldoutDelta("h", 0.90, 0.0),),
        )


def test_run_meta_train_step_rolls_back_optimizer_state_after_partial_step_failure():
    class PartialFailOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {"lr": 1.0})

        @torch.no_grad()
        def step(self, closure=None):
            del closure
            parameter = self.param_groups[0]["params"][0]
            parameter.add_(1.0)
            self.state[parameter]["partial_marker"] = torch.tensor(1.0)
            raise RuntimeError("synthetic partial optimizer failure")

    config = _config()
    model = _model()
    baseline_optimizer = build_phase1b_optimizer(model, config)
    params = [parameter for group in baseline_optimizer.param_groups for parameter in group["params"]]
    optimizer = PartialFailOptimizer(params)
    before = _parameter_snapshot(model)
    with pytest.raises(RuntimeError, match="partial optimizer failure"):
        run_meta_train_step(model, [_batch()] * 4, optimizer, config)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        assert torch.equal(value.detach(), before[name]), name
    assert optimizer.state_dict()["state"] == {}


def test_real_cvsincnet_phase1_parameter_groups_are_18_and_24_names():
    config = _config(adapter_outer_lr=2.0e-3)
    model_b = build_model(
        dataset="wisig",
        input_len=256,
        model_variant="base",
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    )
    optimizer_b = build_phase1b_optimizer(model_b, config)
    names_b = set(optimizer_parameter_names(model_b, optimizer_b))
    assert len(names_b) == 18
    model_c = build_model(
        dataset="wisig",
        input_len=256,
        model_variant="base",
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    )
    optimizer_c = build_phase1c_optimizer(model_c, config)
    names_c = set(optimizer_parameter_names(model_c, optimizer_c))
    assert len(names_c) == 24
    assert names_c - names_b == {
        "t_proj.weight",
        "t_proj.bias",
        "f_proj.weight",
        "f_proj.bias",
        "fuse.0.weight",
        "fuse.0.bias",
    }


def test_meta_episode_batch_requires_nonempty_y_guard_partition():
    batch = _batch()
    episode = replace(batch.episode, query_guard=(), guard_class_ids=frozenset())
    with pytest.raises(ValueError, match="Y_guard|query_guard"):
        MetaEpisodeBatch(
            episode=episode,
            support_x=batch.support_x,
            support_y=batch.support_y,
            query_x=batch.query_x[:2],
            query_y=batch.query_y[:2],
            adapt_mask=torch.tensor([True, True]),
            guard_mask=torch.tensor([False, False]),
            frozen_prototypes=batch.frozen_prototypes,
        )


def test_fomaml_fixed_lr_optimizer_freezes_meta_sgd_step_sizes():
    model = _model()
    config = replace(_config(), learn_step_sizes=False)
    optimizer = build_phase1b_optimizer(model, config)
    names = optimizer_parameter_names(model, optimizer)
    assert names
    assert all(not name.endswith("log_step_size") for name in names)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith("log_step_size")
    )
