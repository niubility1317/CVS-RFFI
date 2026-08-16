"""Behavioral checks for frozen MIRAGE causal arms and training losses."""

from __future__ import annotations

import dataclasses
import importlib

import pytest
import torch
import torch.nn.functional as functional

from cvsrffi.phase1_mirage.head import OpenHeadOutput
from cvsrffi.phase1_mirage.model import MIRAGEConfig
from cvsrffi.phase1_mirage.proxy import build_proxy_episode


def _config_api():
    """Import Task 6 configuration lazily so RED proves it is absent."""

    try:
        module = importlib.import_module("cvsrffi.phase1_mirage.config")
    except ModuleNotFoundError as error:
        if error.name == "cvsrffi.phase1_mirage.config":
            pytest.fail("missing frozen MIRAGE arm configuration module")
        raise
    return module.ArmConfig, module.arm_config, module.arm_diff


def _loss_api():
    """Import Task 6 loss API lazily so RED proves it is absent."""

    try:
        module = importlib.import_module("cvsrffi.phase1_mirage.losses")
    except ModuleNotFoundError as error:
        if error.name == "cvsrffi.phase1_mirage.losses":
            pytest.fail("missing MIRAGE causal loss module")
        raise
    return (
        module.BoundaryMixupBatch,
        module.build_boundary_mixup,
        module.compute_arm_losses,
        module.group_cvar,
        module.pseudo_accept_mask,
        module.resolve_group_ids,
    )


def _boundary_mixup_loss_api():
    module = importlib.import_module("cvsrffi.phase1_mirage.losses")
    try:
        return module.boundary_mixup_loss
    except AttributeError:
        pytest.fail("missing public boundary mixup loss")


def _episode_and_labels():
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    episode = build_proxy_episode(labels, split_role="train_l", seed=17, episode_index=0)
    return labels, episode


def _open_output(batch_size: int, class_count: int, *, unknown_logit: torch.Tensor | None = None) -> OpenHeadOutput:
    if unknown_logit is None:
        unknown_logit = torch.zeros(batch_size, requires_grad=True)
    return OpenHeadOutput(
        class_scores=torch.zeros(batch_size, class_count, requires_grad=True),
        class_distances=torch.ones(batch_size, class_count, requires_grad=True),
        radius_margins=torch.tensor(
            [[-0.2, -0.1, 0.3]] * batch_size,
            dtype=torch.float32,
            requires_grad=True,
        ),
        energy=torch.linspace(-0.2, 0.3, batch_size, requires_grad=True),
        unknown_risk=torch.sigmoid(unknown_logit),
    )


def _loss_inputs(*, include_group: bool = False, unknown_logit: torch.Tensor | None = None):
    """Build hand-checkable source-only inputs for a complete arm invocation."""

    _, build_boundary_mixup, _, _, _, _ = _loss_api()
    labels, episode = _episode_and_labels()
    batch_size = labels.numel()
    class_count = 3
    supervised_logits = torch.tensor(
        [[3.0, 0.1, -1.0], [2.5, 0.2, -1.0], [0.1, 3.0, -1.0], [0.2, 2.7, -1.0], [-1.0, 0.2, 3.0], [-1.0, 0.1, 2.8]],
        requires_grad=True,
    )
    pseudo_student_logits = torch.tensor(
        [[3.0, 0.1, -1.0], [0.1, 3.0, -1.0], [-1.0, 0.1, 3.0]],
        requires_grad=True,
    )
    pseudo_teacher_logits = torch.tensor(
        [[5.0, -1.0, -2.0], [-1.0, 5.0, -2.0], [-2.0, -1.0, 5.0]],
        requires_grad=True,
    )
    boundary_embeddings = functional.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.0, 0.0, 1.0, 0.0], [0.2, 0.0, 0.8, 0.0]],
            requires_grad=True,
        ),
        dim=1,
    ).detach().requires_grad_()
    registered_row_mask = torch.zeros(batch_size, dtype=torch.bool)
    registered_row_mask[episode.registered_rows] = True
    mixup = build_boundary_mixup(
        boundary_embeddings,
        labels,
        registered_row_mask,
        lambdas=0.5,
    )
    mixup_output = _open_output(mixup.mixed_embeddings.shape[0], class_count)
    result = {
        "supervised_logits": supervised_logits,
        "supervised_labels": labels,
        "pseudo_student_logits": pseudo_student_logits,
        "pseudo_teacher_logits": pseudo_teacher_logits,
        "pseudo_inside_radius": torch.tensor([True, True, True]),
        "pseudo_radius_margins": torch.tensor([-0.4, -0.2, -0.1]),
        "masked_prediction": torch.randn(3, 4, requires_grad=True),
        "masked_target": torch.randn(3, 4, requires_grad=True),
        "cross_receiver_features": torch.randn(6, 4, requires_grad=True),
        "cross_receiver_ids": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
        "proxy_episode": episode,
        "open_output": _open_output(batch_size, class_count, unknown_logit=unknown_logit),
        "boundary_embeddings": boundary_embeddings,
        "boundary_mixup_batch": mixup,
        "boundary_mixup_output": mixup_output,
    }
    if include_group:
        result["group_losses"] = torch.tensor(
            [0.1, 0.2, 0.4, 0.6, 1.1, 1.3], requires_grad=True
        )
        result["receiver_ids"] = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
        result["day_ids"] = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.int64)
        result["scene_ids"] = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    return result


def test_all_arms_share_budget_and_change_only_declared_mechanisms():
    """Catch a causal arm that gains hidden capacity, budget, or mechanism changes."""

    ArmConfig, arm_config, arm_diff = _config_api()
    configs = [arm_config(name) for name in ("B0", "A", "B", "C")]

    assert all(isinstance(config, ArmConfig) for config in configs)
    assert {config.encoder for config in configs} == {configs[0].encoder}
    assert {config.epochs for config in configs} == {200}
    assert {
        (
            config.optimizer,
            config.learning_rate,
            config.weight_decay,
            config.batch_size,
            config.steps_per_epoch,
        )
        for config in configs
    } == {
        (
            configs[0].optimizer,
            configs[0].learning_rate,
            configs[0].weight_decay,
            configs[0].batch_size,
            configs[0].steps_per_epoch,
        )
    }
    assert arm_diff("A", "B0") == {"masked_latent", "cross_receiver", "prototype_pseudo"}
    assert arm_diff("B", "A") == {"proxy_open_loss", "radius_energy", "boundary_mixup"}
    assert arm_diff("C", "B") == {"group_cvar"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        configs[0].epochs = 1


@pytest.mark.parametrize(
    "encoder",
    (
        MIRAGEConfig(token_dim=256),
        MIRAGEConfig(transformer_layers=3),
    ),
)
def test_arm_config_rejects_custom_encoder_capacity(encoder):
    """Catch an arm-specific token or layer configuration hidden behind a frozen dataclass."""

    ArmConfig, arm_config, _ = _config_api()

    with pytest.raises(ValueError, match="encoder"):
        ArmConfig(arm_id="B0", encoder=encoder, mechanisms=arm_config("B0").mechanisms)


def test_pseudo_label_requires_all_four_conditions_and_supports_broadcasting():
    """Catch acceptance if any confidence, margin, view, or radius gate is omitted."""

    _, _, _, _, pseudo_accept_mask, _ = _loss_api()

    mask = pseudo_accept_mask(top1=0.96, margin=0.21, views_agree=True, inside_radius=True)
    assert mask.shape == torch.Size([])
    assert bool(mask.item())
    broadcast = pseudo_accept_mask(
        top1=torch.tensor([0.96, 0.95, 0.99]),
        margin=0.21,
        views_agree=torch.tensor([True, False, True]),
        inside_radius=torch.tensor([True, True, False]),
    )
    assert torch.equal(broadcast, torch.tensor([True, False, False]))
    assert not bool(pseudo_accept_mask(0.96, 0.19, True, True).item())
    assert not bool(pseudo_accept_mask(0.94, 0.21, True, True).item())
    assert not bool(pseudo_accept_mask(0.96, 0.21, False, True).item())
    assert not bool(pseudo_accept_mask(0.96, 0.21, True, False).item())


@pytest.mark.parametrize(
    ("top1", "margin", "views_agree", "inside_radius", "message"),
    (
        (torch.tensor([float("nan")]), torch.tensor([0.3]), torch.tensor([True]), torch.tensor([True]), "finite"),
        (torch.tensor([0.9, 0.9]), torch.tensor([0.3, 0.3, 0.3]), torch.tensor([True]), torch.tensor([True]), "broadcast"),
        (torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.bool), torch.tensor([], dtype=torch.bool), "non-empty"),
    ),
)
def test_pseudo_gate_rejects_nonfinite_incompatible_or_empty_inputs(top1, margin, views_agree, inside_radius, message):
    """Catch a fail-open pseudo path for malformed teacher evidence."""

    _, _, _, _, pseudo_accept_mask, _ = _loss_api()

    with pytest.raises(ValueError, match=message):
        pseudo_accept_mask(top1, margin, views_agree, inside_radius)


def test_empty_accepted_pseudo_loss_is_graph_connected_scalar_zero():
    """Catch a detached zero when every unlabeled sample fails the four-way gate."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    inputs["pseudo_inside_radius"] = torch.zeros(3, dtype=torch.bool)

    losses = compute_arm_losses("B0", **inputs)

    assert losses["ema_pseudo"].shape == torch.Size([])
    assert losses["ema_pseudo"].item() == 0.0
    losses["ema_pseudo"].backward()
    assert inputs["pseudo_student_logits"].grad is not None
    assert torch.equal(inputs["pseudo_student_logits"].grad, torch.zeros_like(inputs["pseudo_student_logits"]))


def test_proxy_rows_never_enter_registered_ce_in_the_same_episode():
    """Catch proxy labels/logits contributing to registered supervised CE in B/C."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    episode = inputs["proxy_episode"]
    first = compute_arm_losses("B", **inputs)
    expected = functional.cross_entropy(
        inputs["supervised_logits"][episode.registered_rows],
        inputs["supervised_labels"][episode.registered_rows],
    )
    assert torch.allclose(first["registered_ce"], expected)

    mutated = dict(inputs)
    changed_logits = inputs["supervised_logits"].detach().clone().requires_grad_(True)
    changed_logits.data[episode.proxy_rows] = torch.tensor([[-100.0, 100.0, -100.0]]).repeat(
        episode.proxy_rows.numel(), 1
    )
    mutated["supervised_logits"] = changed_logits
    second = compute_arm_losses("B", **mutated)
    assert torch.allclose(second["registered_ce"], first["registered_ce"])

    first["registered_ce"].backward()
    assert torch.equal(
        inputs["supervised_logits"].grad[episode.proxy_rows],
        torch.zeros_like(inputs["supervised_logits"].grad[episode.proxy_rows]),
    )


def test_b_rejects_proxy_episode_that_leaks_a_proxy_label_into_registered_rows():
    """Catch a forged partition that leaves one proxy-class row in registered CE."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    episode = inputs["proxy_episode"]
    forged = dataclasses.replace(
        episode,
        registered_rows=torch.cat((episode.registered_rows, episode.proxy_rows[:1])),
        proxy_rows=episode.proxy_rows[1:],
    )
    inputs["proxy_episode"] = forged

    with pytest.raises(ValueError, match="proxy_rows"):
        compute_arm_losses("B", **inputs)


def test_b_rejects_proxy_episode_with_a_noncanonical_registered_class_mask():
    """Catch a proxy episode that masks an additional registered class row."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    episode = inputs["proxy_episode"]
    forged_mask = episode.registered_class_mask.clone()
    extra_masked_class = next(index for index in range(forged_mask.numel()) if index != episode.proxy_class)
    forged_mask[extra_masked_class] = False
    inputs["proxy_episode"] = dataclasses.replace(episode, registered_class_mask=forged_mask)

    with pytest.raises(ValueError, match="registered_class_mask"):
        compute_arm_losses("B", **inputs)


def test_b_rejects_proxy_episode_with_reordered_but_equivalent_rows():
    """Catch accepting row sets that differ from the deterministic episode receipt order."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    episode = inputs["proxy_episode"]
    inputs["proxy_episode"] = dataclasses.replace(
        episode,
        registered_rows=episode.registered_rows.flip(0),
    )

    with pytest.raises(ValueError, match="registered_rows"):
        compute_arm_losses("B", **inputs)


def test_boundary_mixup_uses_only_different_registered_classes_and_normalizes():
    """Catch same-class, out-of-range, or unnormalized B/C boundary mixes."""

    BoundaryMixupBatch, build_boundary_mixup, _, _, _, _ = _loss_api()
    embeddings = functional.normalize(torch.randn(6, 5, requires_grad=True), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    registered = torch.tensor([True, True, True, True, False, False])
    mixup = build_boundary_mixup(embeddings, labels, registered, lambdas=0.35)

    assert isinstance(mixup, BoundaryMixupBatch)
    assert mixup.left_indices.numel() > 0
    assert torch.all(labels[mixup.left_indices] != labels[mixup.right_indices])
    assert torch.all(registered[mixup.left_indices])
    assert torch.all(registered[mixup.right_indices])
    assert torch.all((mixup.lambdas >= 0.35) & (mixup.lambdas <= 0.65))
    assert torch.allclose(
        mixup.mixed_embeddings.norm(dim=1), torch.ones(mixup.mixed_embeddings.shape[0]), atol=1e-5
    )
    with pytest.raises(ValueError, match="lambdas"):
        build_boundary_mixup(embeddings, labels, registered, lambdas=0.7)


def test_boundary_mixup_without_a_legal_pair_returns_a_graph_connected_zero():
    """Catch an invalid mixup fallback or a detached no-pair loss."""

    _, build_boundary_mixup, _, _, _, _ = _loss_api()
    boundary_mixup_loss = _boundary_mixup_loss_api()
    inputs = _loss_inputs()
    labels = torch.zeros(6, dtype=torch.int64)
    mixup = build_boundary_mixup(
        inputs["boundary_embeddings"],
        labels,
        torch.ones(6, dtype=torch.bool),
        lambdas=0.5,
    )
    assert mixup.mixed_embeddings.shape == (0, 4)
    zero = boundary_mixup_loss(mixup, None, registered_class_mask=torch.tensor([True]))

    assert zero.shape == torch.Size([])
    assert zero.item() == 0.0
    zero.backward()
    assert inputs["boundary_embeddings"].grad is not None


def test_b_rejects_a_forged_same_class_boundary_mixup_batch():
    """Catch bypassing B's strict different-registered-class mixup construction."""

    BoundaryMixupBatch, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    episode = inputs["proxy_episode"]
    registered_rows = episode.registered_rows
    registered_labels = inputs["supervised_labels"][registered_rows]
    same_class = registered_rows[registered_labels == registered_labels[0]]
    assert same_class.numel() == 2
    left, right = same_class[0:1], same_class[1:2]
    forged = BoundaryMixupBatch(
        mixed_embeddings=functional.normalize(
            inputs["boundary_embeddings"][left] + inputs["boundary_embeddings"][right],
            dim=1,
        ),
        left_indices=left,
        right_indices=right,
        lambdas=torch.tensor([0.5]),
    )
    inputs["boundary_mixup_batch"] = forged
    inputs["boundary_mixup_output"] = _open_output(1, 3)

    with pytest.raises(ValueError, match="boundary mixup"):
        compute_arm_losses("B", **inputs)


def test_b_rejects_an_explicit_empty_mixup_batch_even_if_internal_build_would_be_empty():
    """Catch callers using an explicit empty object to disable B/C boundary mixup."""

    BoundaryMixupBatch, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    inputs["boundary_embeddings"] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    empty_embeddings = inputs["boundary_embeddings"][:0] * 0.0
    inputs["boundary_mixup_batch"] = BoundaryMixupBatch(
        mixed_embeddings=empty_embeddings,
        left_indices=torch.empty(0, dtype=torch.int64),
        right_indices=torch.empty(0, dtype=torch.int64),
        lambdas=torch.empty(0),
    )
    inputs["boundary_mixup_output"] = None

    with pytest.raises(ValueError, match="boundary mixup"):
        compute_arm_losses("B", **inputs)


def test_b_none_mixup_builds_a_graph_connected_zero_for_opposing_registered_embeddings():
    """Catch rejecting an internal no-norm mixup instead of returning a safe zero."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    opposing_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    inputs["boundary_embeddings"] = opposing_embeddings
    inputs["boundary_mixup_batch"] = None
    inputs["boundary_mixup_output"] = None

    losses = compute_arm_losses("B", **inputs)

    assert losses["boundary_mixup"].shape == torch.Size([])
    assert losses["boundary_mixup"].item() == 0.0
    losses["boundary_mixup"].backward()
    assert opposing_embeddings.grad is not None
    assert torch.equal(opposing_embeddings.grad, torch.zeros_like(opposing_embeddings))


def test_group_resolution_uses_only_fixed_sample_count_hierarchy():
    """Catch loss- or class-dependent fallback instead of the fixed receiver hierarchy."""

    _, _, _, _, _, resolve_group_ids = _loss_api()
    receiver = torch.tensor([0] * 15 + [0] + [1] * 8 + [1] * 8 + [2] * 5, dtype=torch.int64)
    day = torch.tensor([0] * 15 + [1] + [0] * 8 + [1] * 8 + [0] * 5, dtype=torch.int64)
    scene = torch.tensor([0] * 16 + [0] * 8 + [1] * 8 + [0] * 5, dtype=torch.int64)

    groups = resolve_group_ids(receiver, day, scene, min_group_size=16)

    assert torch.equal(groups[:16], torch.full((16,), groups[0]))
    assert torch.equal(groups[16:32], torch.full((16,), groups[16]))
    assert groups[0] != groups[16]
    assert torch.equal(groups[32:], torch.full((5,), groups[32]))
    assert groups[32] not in {groups[0].item(), groups[16].item()}


def test_group_cvar_selects_the_worst_thirty_percent_of_group_means():
    """Catch CVaR over rows, the best groups, or an incorrect ceiling tail count."""

    _, _, _, group_cvar, _, _ = _loss_api()
    losses = torch.tensor([1.0, 3.0, 2.0, 8.0, 10.0], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)

    result = group_cvar(losses, groups, tail_fraction=0.30)
    two_group_tail = group_cvar(losses, groups, tail_fraction=0.34)

    assert torch.allclose(result, torch.tensor(10.0))
    assert torch.allclose(two_group_tail, torch.tensor(7.5))
    result.backward()
    assert torch.equal(losses.grad, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


@pytest.mark.parametrize("tail_fraction", (0.0, -0.1, 1.01, float("nan")))
def test_group_cvar_rejects_invalid_tail_fractions(tail_fraction):
    """Catch a silently clamped or malformed worst-group tail configuration."""

    _, _, _, group_cvar, _, _ = _loss_api()
    with pytest.raises(ValueError, match="tail_fraction"):
        group_cvar(torch.tensor([1.0]), torch.tensor([0]), tail_fraction=tail_fraction)


def test_c_rejects_external_preparsed_group_ids():
    """Catch a caller bypassing C's frozen source-field fallback resolution."""

    _, _, compute_arm_losses, _, _, resolve_group_ids = _loss_api()
    inputs = _loss_inputs(include_group=True)
    inputs["group_ids"] = resolve_group_ids(
        inputs["receiver_ids"],
        inputs["day_ids"],
        inputs["scene_ids"],
        min_group_size=16,
    )

    with pytest.raises(ValueError, match="group_ids"):
        compute_arm_losses("C", **inputs)


def test_c_resolves_raw_group_sources_at_its_frozen_settings():
    """Catch C accepting caller-selected groups or non-frozen fallback controls."""

    _, _, compute_arm_losses, group_cvar, _, resolve_group_ids = _loss_api()
    inputs = _loss_inputs(include_group=True)

    result = compute_arm_losses("C", **inputs)
    expected_groups = resolve_group_ids(
        inputs["receiver_ids"],
        inputs["day_ids"],
        inputs["scene_ids"],
        min_group_size=16,
    )
    expected = group_cvar(inputs["group_losses"], expected_groups, tail_fraction=0.30)

    assert torch.allclose(result["group_cvar"], expected)


@pytest.mark.parametrize("missing", ("receiver_ids", "day_ids", "scene_ids"))
def test_c_requires_all_raw_group_source_fields(missing):
    """Catch C falling back to unproven pre-resolved group assignments."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs(include_group=True)
    inputs.pop(missing)

    with pytest.raises(ValueError, match=missing):
        compute_arm_losses("C", **inputs)


@pytest.mark.parametrize("arm", ("B0", "A", "B"))
def test_non_c_arms_ignore_group_source_fields(arm):
    """Catch B0/A/B reading C-only group metadata or its graph."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs(include_group=True)
    inputs["receiver_ids"] = object()
    inputs["day_ids"] = object()
    inputs["scene_ids"] = object()
    inputs["group_ids"] = object()

    result = compute_arm_losses(arm, **inputs)

    assert "group_cvar" not in result
    assert torch.isfinite(result["total"])


def test_proxy_open_and_group_cvar_gradients_exist_only_in_declared_arms():
    """Catch B/C rejection or C-only CVaR gradients leaking into earlier arms."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    for arm in ("B0", "A", "B", "C"):
        unknown_logit = torch.zeros(6, requires_grad=True)
        inputs = _loss_inputs(include_group=True, unknown_logit=unknown_logit)
        losses = compute_arm_losses(arm, **inputs)
        assert set(losses).issuperset({"registered_ce", "ema_pseudo", "weak_strong_consistency", "total"})
        assert all(value.shape == torch.Size([]) and bool(torch.isfinite(value)) for value in losses.values())
        losses["total"].backward()

        if arm in {"B", "C"}:
            assert unknown_logit.grad is not None
            assert bool((unknown_logit.grad.abs() > 0).any())
            assert {"proxy_bce", "radius_energy", "boundary_mixup"}.issubset(losses)
        else:
            assert unknown_logit.grad is None
            assert "proxy_bce" not in losses
        if arm == "C":
            assert inputs["group_losses"].grad is not None
            assert bool((inputs["group_losses"].grad.abs() > 0).any())
            assert "group_cvar" in losses
        else:
            assert inputs["group_losses"].grad is None
            assert "group_cvar" not in losses


def test_b_and_c_require_their_proxy_episode_but_b0_does_not():
    """Catch a proxy rejection arm that can run without source-role episode evidence."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()
    inputs = _loss_inputs()
    inputs["proxy_episode"] = None

    b0 = compute_arm_losses("B0", **inputs)
    assert torch.isfinite(b0["total"])
    with pytest.raises(ValueError, match="proxy_episode"):
        compute_arm_losses("B", **inputs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"tail_fraction": 0.50}, "tail_fraction"),
        ({"min_group_size": 8}, "min_group_size"),
    ),
)
def test_c_rejects_nonfrozen_group_cvar_overrides(overrides, message):
    """Catch a formal C loss that lets callers tune CVaR tail or fallback support."""

    _, _, compute_arm_losses, _, _, _ = _loss_api()

    with pytest.raises(ValueError, match=message):
        compute_arm_losses("C", **_loss_inputs(include_group=True), **overrides)
