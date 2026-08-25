from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_objectives import (  # noqa: E402
    FROZEN_PROTOTYPE_CLASS_FLOOR_OBJECTIVE,
    LossBreakdown,
    MetaObjectiveConfig,
    outer_objective,
    support_adaptation_loss,
    support_objective,
)
from cvsrffi.meta_adapter import iter_inner_adapter_parameters  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _outputs(
    logits: torch.Tensor,
    embedding: torch.Tensor | None = None,
    *,
    key: str = "feat_cls",
) -> dict[str, torch.Tensor]:
    if embedding is None:
        embedding = logits[:, :2]
    return {"logits": logits, key: embedding}


def _fixture(
    *,
    classes: int = 4,
    rows: int = 6,
    dim: int = 3,
    requires_grad: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    pre_logits = torch.randn(rows, classes, requires_grad=requires_grad)
    post_logits = (pre_logits.detach() + 0.15 * torch.randn(rows, classes)).requires_grad_(
        requires_grad
    )
    pre_embedding = torch.randn(rows, dim, requires_grad=requires_grad)
    post_embedding = (pre_embedding.detach() + 0.1 * torch.randn(rows, dim)).requires_grad_(
        requires_grad
    )
    pre = _outputs(pre_logits, pre_embedding)
    post = _outputs(post_logits, post_embedding)
    labels = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    prototypes = F.normalize(torch.randn(classes, dim), dim=1)
    return pre, post, labels, prototypes


def _masks(rows: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    assert rows == 6
    return (
        torch.tensor([True, True, False, False, False, False]),
        torch.tensor([False, False, True, True, False, False]),
    )


def test_guard_rows_never_enter_adapt_ce_and_counts_are_exact():
    pre, post, labels, prototypes = _fixture()
    adapt_mask, guard_mask = _masks()

    result = outer_objective(
        pre,
        post,
        labels,
        adapt_mask,
        guard_mask,
        prototypes,
        MetaObjectiveConfig(
            lambda_floor=0.0,
            lambda_topology=0.0,
            lambda_zero_step=0.0,
        ),
    )

    expected = F.cross_entropy(post["logits"][adapt_mask], labels[adapt_mask])
    assert result.adapt_count == 2
    assert result.guard_count == 2
    torch.testing.assert_close(result.adapt, expected)

    changed_guard = copy.deepcopy(post)
    changed_guard["logits"] = changed_guard["logits"].detach().clone()
    changed_guard["logits"][guard_mask] = 100.0
    changed = outer_objective(
        pre,
        changed_guard,
        labels,
        adapt_mask,
        guard_mask,
        prototypes,
        MetaObjectiveConfig(
            lambda_floor=0.0,
            lambda_guard=0.0,
            lambda_topology=0.0,
            lambda_zero_step=0.0,
        ),
    )
    torch.testing.assert_close(changed.adapt, result.adapt)


def test_outer_rejects_overlapping_or_mismatched_masks():
    pre, post, labels, prototypes = _fixture()
    with pytest.raises(ValueError, match="overlap"):
        outer_objective(
            pre,
            post,
            labels,
            torch.tensor([True, False, False, False, False, False]),
            torch.tensor([True, False, False, False, False, False]),
            prototypes,
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="length"):
        outer_objective(
            pre,
            post,
            labels,
            torch.ones(5, dtype=torch.bool),
            torch.zeros(6, dtype=torch.bool),
            prototypes,
            MetaObjectiveConfig(),
        )


def test_objectives_reject_label_or_output_shape_errors():
    pre, post, labels, prototypes = _fixture()
    adapt_mask, guard_mask = _masks()
    with pytest.raises(ValueError, match="range"):
        outer_objective(
            pre,
            post,
            torch.tensor([0, 1, 2, 4, 0, 1]),
            adapt_mask,
            guard_mask,
            prototypes,
            MetaObjectiveConfig(),
        )
    malformed = dict(post)
    malformed["feat_cls"] = torch.randn(5, 3)
    with pytest.raises(ValueError, match="batch"):
        outer_objective(
            pre,
            malformed,
            labels,
            adapt_mask,
            guard_mask,
            prototypes,
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="labels"):
        support_objective(
            post,
            labels[:-1],
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )


@pytest.mark.parametrize(
    ("adapt_mask", "guard_mask"),
    [
        (torch.zeros(6, dtype=torch.bool), torch.tensor([False, False, True, True, False, False])),
        (torch.tensor([True, True, False, False, False, False]), torch.zeros(6, dtype=torch.bool)),
        (torch.zeros(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool)),
    ],
)
def test_empty_masks_and_single_class_are_finite_same_device_dtype(adapt_mask, guard_mask):
    pre, post, _, prototypes = _fixture()
    labels = torch.zeros(6, dtype=torch.long)
    result = outer_objective(
        pre,
        post,
        labels,
        adapt_mask,
        guard_mask,
        prototypes,
        MetaObjectiveConfig(),
    )
    assert isinstance(result, LossBreakdown)
    assert result.total.device == post["logits"].device
    assert result.total.dtype == post["logits"].dtype
    for value in (
        result.total,
        result.adapt,
        result.guard,
        result.floor,
        result.topology,
        result.zero_step,
    ):
        assert value.ndim == 0
        assert torch.isfinite(value)
    assert result.topology.item() == 0.0


def test_floor_uses_per_class_mean_not_sample_count():
    logits = torch.tensor(
        [[2.0, 0.0], [1.5, 0.0]],
        requires_grad=True,
    )
    pre = _outputs(logits.detach().clone().requires_grad_(), torch.eye(2))
    post = _outputs(logits, torch.eye(2))
    prototypes = torch.eye(2)
    masks = torch.ones(2, dtype=torch.bool)
    config = MetaObjectiveConfig(
        lambda_adapt=0.0,
        lambda_guard=0.0,
        lambda_topology=0.0,
        lambda_zero_step=0.0,
        floor_tau=0.4,
    )
    base = outer_objective(pre, post, torch.tensor([0, 1]), masks, ~masks, prototypes, config)

    duplicated_logits = torch.cat([logits.detach()[0:1], logits.detach()[1:2].repeat(10, 1)]).requires_grad_()
    duplicated_embedding = torch.eye(2)[torch.tensor([0] + [1] * 10)]
    duplicated = outer_objective(
        _outputs(duplicated_logits.detach().clone().requires_grad_(), duplicated_embedding),
        _outputs(duplicated_logits, duplicated_embedding),
        torch.tensor([0] + [1] * 10),
        torch.ones(11, dtype=torch.bool),
        torch.zeros(11, dtype=torch.bool),
        prototypes,
        config,
    )
    torch.testing.assert_close(base.floor, duplicated.floor)


def test_floor_guard_and_total_are_invariant_to_joint_class_permutation():
    pre, post, labels, prototypes = _fixture()
    adapt_mask, guard_mask = _masks()
    config = MetaObjectiveConfig()
    original = outer_objective(pre, post, labels, adapt_mask, guard_mask, prototypes, config)

    permutation = torch.tensor([3, 1, 0, 2])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())

    def permute(outputs):
        return {
            "logits": outputs["logits"][:, permutation],
            "feat_cls": outputs["feat_cls"],
        }

    permuted = outer_objective(
        permute(pre),
        permute(post),
        inverse[labels],
        adapt_mask,
        guard_mask,
        prototypes[permutation],
        config,
    )
    for name in ("adapt", "guard", "floor", "topology", "zero_step", "total"):
        torch.testing.assert_close(getattr(original, name), getattr(permuted, name))


def test_topology_compares_pairwise_cosine_structure_and_single_class_is_zero():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    masks = torch.ones(4, dtype=torch.bool)
    pre = _outputs(
        torch.zeros(4, 2, requires_grad=True),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    post = _outputs(
        torch.zeros(4, 2, requires_grad=True),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    )
    result = outer_objective(
        pre,
        post,
        labels,
        masks,
        torch.zeros(4, dtype=torch.bool),
        torch.eye(2),
        MetaObjectiveConfig(
            lambda_adapt=0.0,
            lambda_guard=0.0,
            lambda_floor=0.0,
            lambda_zero_step=0.0,
        ),
    )
    assert result.topology.item() > 0.0

    single = outer_objective(
        pre,
        post,
        torch.zeros(4, dtype=torch.long),
        masks,
        torch.zeros(4, dtype=torch.bool),
        torch.eye(2),
        MetaObjectiveConfig(
            lambda_adapt=0.0,
            lambda_guard=0.0,
            lambda_floor=0.0,
            lambda_zero_step=0.0,
        ),
    )
    assert single.topology.item() == 0.0


def test_zero_step_is_pre_update_query_ce():
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    pre_logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]], requires_grad=True)
    post_logits = -pre_logits.detach()
    pre = _outputs(pre_logits, torch.randn(3, 2, requires_grad=True))
    post = _outputs(post_logits.requires_grad_(), torch.randn(3, 2, requires_grad=True))
    masks = torch.ones(3, dtype=torch.bool)
    result = outer_objective(
        pre,
        post,
        labels,
        masks,
        torch.zeros(3, dtype=torch.bool),
        torch.eye(2),
        MetaObjectiveConfig(
            lambda_adapt=0.0,
            lambda_guard=0.0,
            lambda_floor=0.0,
            lambda_topology=0.0,
            lambda_zero_step=1.0,
        ),
    )
    torch.testing.assert_close(result.zero_step, F.cross_entropy(pre_logits, labels))


def test_support_uses_fixed_head_prototype_anchor_and_adapter_l2sp_only():
    logits = torch.randn(4, 3, requires_grad=True)
    embedding = torch.randn(4, 2, requires_grad=True)
    outputs = _outputs(logits, embedding)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    prototypes = torch.randn(3, 2, requires_grad=True)
    initial = {
        "meta_adapter_time.down.weight": torch.zeros(2, 2, requires_grad=True),
        "meta_adapter_time.gate": torch.zeros((), requires_grad=True),
    }
    current = {
        key: value.detach().clone().requires_grad_() for key, value in initial.items()
    }
    initial_snapshot = {key: value.detach().clone() for key, value in initial.items()}
    result = support_objective(
        outputs,
        labels,
        prototypes,
        initial,
        current,
        MetaObjectiveConfig(lambda_view_consistency=0.0),
    )
    assert result.guard_count == 0
    assert result.adapt_count == 0
    assert result.guard.item() == 0.0
    assert result.floor.item() == 0.0
    assert result.topology.item() == 0.0
    assert result.zero_step.item() == 0.0

    result.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
    assert all(value.grad is not None for value in current.values())
    assert prototypes.grad is None
    assert all(value.grad is None for value in initial.values())
    for key, value in initial.items():
        torch.testing.assert_close(value, initial_snapshot[key])


def test_class_floor_support_loss_is_scale_matched_and_emphasizes_weak_class():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    equal_logits = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]],
        requires_grad=True,
    )
    baseline_equal = F.cross_entropy(equal_logits, labels)
    robust_equal = support_adaptation_loss(
        equal_logits,
        labels,
        adaptation_objective=FROZEN_PROTOTYPE_CLASS_FLOOR_OBJECTIVE,
    )
    torch.testing.assert_close(robust_equal, baseline_equal)

    unequal_logits = torch.tensor(
        [[4.0, 0.0], [4.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        requires_grad=True,
    )
    baseline = F.cross_entropy(unequal_logits, labels)
    robust = support_adaptation_loss(
        unequal_logits,
        labels,
        adaptation_objective=FROZEN_PROTOTYPE_CLASS_FLOOR_OBJECTIVE,
    )
    assert robust > baseline
    robust.backward()
    easy_grad = unequal_logits.grad[:2].abs().sum()
    weak_grad = unequal_logits.grad[2:].abs().sum()
    assert weak_grad > easy_grad


def test_support_rejects_adapter_key_mismatch_and_non_adapter_state():
    outputs = _outputs(torch.randn(2, 2, requires_grad=True), torch.randn(2, 2, requires_grad=True))
    labels = torch.tensor([0, 1])
    prototypes = torch.eye(2)
    with pytest.raises(ValueError, match="same keys"):
        support_objective(
            outputs,
            labels,
            prototypes,
            {"meta_adapter_time.gate": torch.zeros(())},
            {"meta_adapter_freq.gate": torch.zeros(())},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="log_step_size"):
        support_objective(
            outputs,
            labels,
            prototypes,
            {"meta_adapter_time.log_step_size": torch.zeros(())},
            {"meta_adapter_time.log_step_size": torch.ones(())},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="adapter"):
        support_objective(
            outputs,
            labels,
            prototypes,
            {"cls_head.weight": torch.zeros(2, 2)},
            {"cls_head.weight": torch.ones(2, 2)},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="adapter"):
        support_objective(
            outputs,
            labels,
            prototypes,
            {"meta_adapter_time.norm.weight": torch.zeros(2, 2)},
            {"meta_adapter_time.norm.weight": torch.ones(2, 2)},
            MetaObjectiveConfig(),
        )


def test_output_contract_supports_adv3b02_aliases_but_rejects_missing_keys():
    logits = torch.randn(2, 3, requires_grad=True)
    embedding = torch.randn(2, 4, requires_grad=True)
    outputs = {"tx_logits": logits, "z_id": embedding}
    result = support_objective(
        outputs,
        torch.tensor([0, 1]),
        torch.randn(3, 4),
        {},
        {},
        MetaObjectiveConfig(),
    )
    assert torch.isfinite(result.total)
    with pytest.raises(ValueError, match="logits"):
        support_objective(
            {"feat_cls": embedding},
            torch.tensor([0, 1]),
            torch.randn(3, 4),
            {},
            {},
            MetaObjectiveConfig(),
        )


def test_output_alias_conflicts_are_rejected_instead_of_taking_first_key():
    logits = torch.randn(2, 3, requires_grad=True)
    embedding = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 1])
    prototypes = torch.randn(3, 4)

    with pytest.raises(ValueError, match="ambiguous logits"):
        support_objective(
            {"logits": logits, "tx_logits": logits + 0.1, "feat_cls": embedding},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="logits.*shape"):
        support_objective(
            {"logits": logits, "tx_logits": torch.randn(3, 3), "feat_cls": embedding},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="ambiguous embedding"):
        support_objective(
            {"logits": logits, "feat_cls": embedding, "z_id": embedding + 0.1},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="embedding.*shape"):
        support_objective(
            {"logits": logits, "feat_cls": embedding, "z_id": torch.randn(3, 4)},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )

    same = support_objective(
        {"logits": logits, "tx_logits": logits, "feat_cls": embedding, "z_id": embedding},
        labels,
        prototypes,
        {},
        {},
        MetaObjectiveConfig(),
    )
    assert torch.isfinite(same.total)


@pytest.mark.parametrize(
    "key",
    [
        "encoder.down.weight",
        "unrelated.gate",
        "meta_adapter_fake.gate",
        "meta_adapter_time.norm.weight",
        "meta_adapter_time.log_step_size",
    ],
)
def test_l2sp_rejects_non_v1_adapter_full_names(key):
    outputs = _outputs(torch.randn(2, 2, requires_grad=True), torch.randn(2, 2, requires_grad=True))
    labels = torch.tensor([0, 1])
    prototypes = torch.eye(2)
    with pytest.raises(ValueError, match="adapter|log_step_size"):
        support_objective(
            outputs,
            labels,
            prototypes,
            {key: torch.zeros(2, 2)},
            {key: torch.ones(2, 2)},
            MetaObjectiveConfig(),
        )


@pytest.mark.parametrize("fallback", ["feat_joint", "base"])
def test_feat_cls_does_not_silently_ignore_conflicting_semantic_fallback(fallback):
    logits = torch.randn(2, 3, requires_grad=True)
    embedding = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 1])
    prototypes = torch.randn(3, 4)
    with pytest.raises(ValueError, match="embedding.*semantic"):
        support_objective(
            {"logits": logits, "feat_cls": embedding, fallback: embedding + 0.1},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="embedding.*semantic"):
        support_objective(
            {"logits": logits, "feat_cls": embedding, fallback: torch.randn(3, 4)},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )
    with pytest.raises(ValueError, match="embedding.*semantic"):
        support_objective(
            {"logits": logits, "feat_cls": embedding, fallback: embedding.double()},
            labels,
            prototypes,
            {},
            {},
            MetaObjectiveConfig(),
        )


def test_l2sp_accepts_real_dual_model_inner_snapshot_names():
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        dataset="wisig",
        input_len=64,
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    ).eval()
    outputs = model(torch.randn(2, 2, 64), return_aux=True)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in iter_inner_adapter_parameters(model)
    }
    current = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in iter_inner_adapter_parameters(model)
    }
    result = support_objective(
        outputs,
        torch.tensor([0, 1]),
        torch.randn(3, outputs["z_id"].shape[1]),
        initial,
        current,
        MetaObjectiveConfig(),
    )
    assert torch.isfinite(result.total)
    result.total.backward()
    assert all(parameter.grad is not None for parameter in current.values())


@pytest.mark.parametrize(
    "key",
    [
        "wrapper.id_backbone.meta_adapter_time.gate",
        "id_backbone.meta_adapter_fake.gate",
        "id_backbone.meta_adapter_time.norm.weight",
        "id_backbone.meta_adapter_time.log_step_size",
    ],
)
def test_l2sp_rejects_pseudo_nested_or_invalid_dual_names(key):
    outputs = _outputs(torch.randn(2, 2, requires_grad=True), torch.randn(2, 2, requires_grad=True))
    with pytest.raises(ValueError, match="adapter|log_step_size"):
        support_objective(
            outputs,
            torch.tensor([0, 1]),
            torch.eye(2),
            {key: torch.zeros(2, 2)},
            {key: torch.ones(2, 2)},
            MetaObjectiveConfig(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_guard": -1.0},
        {"lambda_floor": float("nan")},
        {"floor_tau": 0.0},
        {"eps": -1.0},
    ],
)
def test_config_rejects_non_finite_or_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MetaObjectiveConfig(**kwargs)
