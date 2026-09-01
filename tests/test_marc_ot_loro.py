from __future__ import annotations

import torch


def _loro_sampler():
    from cvsrffi.meta_episodes import (
        HierarchicalMetaEpisodeSampler,
        MetaEpisodeSamplerConfig,
        MetaSampleRef,
    )

    refs = []
    dataset_index = 0
    for tx_i in range(3):
        for rx_i in range(3):
            for day_i in range(2):
                for capture_block_i in range(3):
                    for sample_i in range(8):
                        physical_id = (
                            f"tx{tx_i}|rx{rx_i}|day{day_i}|"
                            f"block{capture_block_i}|sig{sample_i}"
                        )
                        refs.append(
                            MetaSampleRef(
                                dataset_index=dataset_index,
                                tx_i=tx_i,
                                rx_i=rx_i,
                                day_i=day_i,
                                eq_i=0,
                                capture_block_i=capture_block_i,
                                physical_sample_id=physical_id,
                                role="L_s",
                                view="leo_clear_weak",
                            )
                        )
                        dataset_index += 1
    return HierarchicalMetaEpisodeSampler(
        refs,
        MetaEpisodeSamplerConfig(
            k_choices=(10,),
            query_per_class=2,
            allowed_roles=("L_s",),
            training=True,
            partial_coverage_probability=1.0,
            partial_class_fraction=(0.5, 0.8),
        ),
    )


def test_rx_holdout_is_a_true_pseudo_target_receiver_episode() -> None:
    """A cross-receiver support pool would leak an easier domain-transfer task."""
    from cvsrffi.meta_episodes import EpisodeKind, validate_episode_semantics

    episode = _loro_sampler().sample_requested(
        kind=EpisodeKind.RX_HOLDOUT,
        k_shot=10,
        seed=713104,
        support_view="leo_clear_weak",
        query_view="leo_clear_weak",
    )
    support_receivers = {row.rx_i for row in episode.support}
    query_receivers = {
        row.rx_i for row in episode.query_adapt + episode.query_guard
    }
    support_ids = {row.physical_sample_id for row in episode.support}
    query_ids = {
        row.physical_sample_id
        for row in episode.query_adapt + episode.query_guard
    }

    assert support_receivers == query_receivers
    assert len(support_receivers) == 1
    assert support_ids.isdisjoint(query_ids)
    audit = validate_episode_semantics(
        episode,
        source_receiver_ids=(0, 1, 2),
    )
    assert audit["pseudo_target_receiver"] == next(iter(query_receivers))
    assert audit["receiver_knowledge_holdout"] is True


def _task_deltas():
    from cvsrffi.meta_weight_bank import DeltaTaskKey

    return {
        DeltaTaskKey("0", "0", "leo_clear_weak", 10, "0"): {
            "id_backbone.t1.weight": torch.tensor([[1.0, 2.0]])
        },
        DeltaTaskKey("1", "0", "leo_clear_weak", 10, "0"): {
            "id_backbone.t1.weight": torch.tensor([[3.0, 4.0]])
        },
        DeltaTaskKey("2", "0", "leo_clear_weak", 10, "0"): {
            "id_backbone.t1.weight": torch.tensor([[5.0, 6.0]])
        },
    }


def test_task_coordinate_bank_keeps_receiver_columns_exact_and_maskable() -> None:
    """An SVD-mixed basis would make a receiver-column mask scientifically false."""
    from cvsrffi.marc_ot_phase1_entry import (
        build_loro_coefficient_mask,
        build_task_coordinate_bank,
    )

    bank = build_task_coordinate_bank(
        "base",
        _task_deltas(),
        max_rank=16,
    )
    assert tuple(key.receiver for key in bank.task_keys) == ("0", "1", "2")
    assert len(bank.entries) == 1
    entry = bank.entries[0]
    assert entry.effective_rank == 3
    assert entry.relative_error == 0.0
    assert torch.equal(
        entry.basis,
        torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
    )
    assert torch.equal(entry.task_coefficients, torch.eye(3))
    assert torch.equal(
        build_loro_coefficient_mask(bank, excluded_receiver=1),
        torch.tensor([1.0, 0.0, 1.0]),
    )


def test_loro_mask_makes_excluded_receiver_basis_value_and_gradient_irrelevant() -> None:
    """Changing d's expert delta must not alter fold-d initialization or its gradient."""
    from cvsrffi.marc_ot_phase1_entry import (
        build_loro_coefficient_mask,
        build_task_coordinate_bank,
    )
    from cvsrffi.meta_bank_trainer import (
        _validate_bank_and_compose_initial,
        apply_bank_coefficient_mask,
    )
    from cvsrffi.meta_support_set_encoder import SupportDomainState

    bank = build_task_coordinate_bank("base", _task_deltas(), max_rank=16)
    entry = bank.entries[0]
    mask = build_loro_coefficient_mask(bank, excluded_receiver=1)
    raw_q = torch.tensor([0.5, 7.0, -0.25], requires_grad=True)
    state = SupportDomainState(
        q=raw_q,
        uncertainty=torch.tensor(0.0, requires_grad=True),
        block_gates=torch.tensor([1.0], requires_grad=True),
        block_lrs=torch.tensor([0.01], requires_grad=True),
    )
    masked = apply_bank_coefficient_mask(state, mask)
    initial, _ = _validate_bank_and_compose_initial(
        {"id_backbone.t1.weight": torch.zeros(1, 2)},
        "base",
        bank,
        masked,
    )
    before = initial["id_backbone.t1.weight"].detach().clone()
    initial["id_backbone.t1.weight"].sum().backward()

    assert raw_q.grad is not None
    assert raw_q.grad[1].item() == 0.0
    assert entry.basis.grad is not None
    assert torch.count_nonzero(entry.basis.grad[:, 1]).item() == 0
    assert torch.count_nonzero(entry.basis.grad[:, (0, 2)]).item() > 0

    with torch.no_grad():
        entry.basis[:, 1].fill_(123456.0)
    changed, _ = _validate_bank_and_compose_initial(
        {"id_backbone.t1.weight": torch.zeros(1, 2)},
        "base",
        bank,
        apply_bank_coefficient_mask(
            SupportDomainState(
                q=raw_q.detach(),
                uncertainty=torch.tensor(0.0),
                block_gates=torch.tensor([1.0]),
                block_lrs=torch.tensor([0.01]),
            ),
            mask,
        ),
    )
    assert torch.equal(changed["id_backbone.t1.weight"], before)
