from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn


def _ref_pool(*, rows_per_domain: int = 22):
    from cvsrffi.meta_episodes import MetaSampleRef

    rows = []
    index = 0
    for tx_i in range(3):
        for rx_i in range(2):
            for day_i in range(2):
                capture_block_i = day_i
                for sample_i in range(rows_per_domain):
                    physical_id = (
                        f"tx{tx_i}|rx{rx_i}|day{day_i}|block{capture_block_i}|sig{sample_i}"
                    )
                    for view in (
                        "clean",
                        "leo_clear_weak",
                        "leo_low_elev_weak",
                        "leo_rain_weak",
                    ):
                        rows.append(
                            MetaSampleRef(
                                dataset_index=index,
                                tx_i=tx_i,
                                rx_i=rx_i,
                                day_i=day_i,
                                eq_i=0,
                                capture_block_i=capture_block_i,
                                physical_sample_id=physical_id,
                                role="L_s",
                                view=view,
                            )
                        )
                        index += 1
    return rows


def _sampler():
    from cvsrffi.meta_episodes import (
        HierarchicalMetaEpisodeSampler,
        MARC_OT_CANONICAL_K,
        MetaEpisodeSamplerConfig,
    )

    return HierarchicalMetaEpisodeSampler(
        _ref_pool(),
        MetaEpisodeSamplerConfig(
            k_choices=MARC_OT_CANONICAL_K,
            query_per_class=2,
            partial_coverage_probability=1.0,
            partial_class_fraction=(0.5, 0.8),
        ),
    )


def test_canonical_schedule_closes_k20_and_every_required_domain_relation() -> None:
    """Dropping K20 or a required scene relation must make coverage fail closed."""
    from cvsrffi.meta_episodes import (
        MARC_OT_CANONICAL_K,
        audit_marc_ot_episode_coverage,
        sample_marc_ot_coverage_schedule,
    )

    episodes = sample_marc_ot_coverage_schedule(_sampler(), seed=713102)
    audit = audit_marc_ot_episode_coverage(
        episodes,
        source_receiver_ids=(0, 1),
        require_complete=True,
    )

    assert MARC_OT_CANONICAL_K == (1, 2, 5, 10, 20)
    assert audit["software_supported_k"] == MARC_OT_CANONICAL_K
    assert audit["receiver_holdout_k"] == MARC_OT_CANONICAL_K
    assert audit["day_capture_holdout_k"] == MARC_OT_CANONICAL_K
    assert audit["clean_to_leo_scenes"] == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )
    assert len(audit["leo_cross_scene_pairs"]) == 6
    assert audit["episode_count"] == 55
    assert audit["semantic_cell_count"] == 55

    without_k20 = tuple(episode for episode in episodes if episode.k_shot != 20)
    with pytest.raises(ValueError, match="K.*20|coverage"):
        audit_marc_ot_episode_coverage(
            without_k20,
            source_receiver_ids=(0, 1),
            require_complete=True,
        )

    duplicate_cell = (*episodes, episodes[0])
    extra_cell = (
        *episodes,
        _sampler().sample_requested(
            kind="Q_SAME_DOMAIN",
            k_shot=1,
            seed=991,
        ),
    )
    for forged in (duplicate_cell, extra_cell):
        with pytest.raises(ValueError, match="55|semantic|coverage"):
            audit_marc_ot_episode_coverage(
                forged,
                source_receiver_ids=(0, 1),
                require_complete=True,
            )


def test_episode_semantics_reject_forged_kind_k_and_partition_overlap() -> None:
    """A forged declaration must fail even when every tensor shape still aligns."""
    from cvsrffi.meta_episodes import (
        EpisodeKind,
        sample_marc_ot_coverage_schedule,
        validate_episode_semantics,
    )

    episode = next(
        row
        for row in sample_marc_ot_coverage_schedule(_sampler(), seed=7)
        if row.kind is EpisodeKind.CLEAN_TO_LEO and row.k_shot == 2
    )
    validate_episode_semantics(episode, source_receiver_ids=(0, 1))

    with pytest.raises(ValueError, match="kind|relation"):
        validate_episode_semantics(
            replace(episode, kind=EpisodeKind.RX_HOLDOUT),
            source_receiver_ids=(0, 1),
        )
    with pytest.raises(ValueError, match="support.*K|k_shot"):
        validate_episode_semantics(
            replace(episode, k_shot=5),
            source_receiver_ids=(0, 1),
        )
    with pytest.raises(ValueError, match="physical.*overlap|disjoint"):
        validate_episode_semantics(
            replace(
                episode,
                query_adapt=(episode.support[0], *episode.query_adapt[1:]),
            ),
            source_receiver_ids=(0, 1),
        )


class _SupportFeatureModel(nn.Module):
    def forward(self, values, return_aux=True):
        assert return_aux is True
        core = values.float().mean(dim=-1)
        weights = torch.linspace(0.25, 1.25, 160, device=values.device).view(1, -1)
        z_id = (core[:, :1] + 1.5) * weights
        t_emb = (core[:, 1:] + 2.0) * weights
        f_emb = (core.mean(dim=1, keepdim=True) + 2.5) * weights.flip(1)
        return {
            "z_id": z_id,
            "aux_id": {"t_emb": t_emb, "f_emb": f_emb},
        }


def _bank_and_state():
    from cvsrffi.meta_weight_bank import (
        BlockSpec,
        DeltaBankEntry,
        DeltaTaskKey,
        WEIGHT_DELTA_BANK_SCHEMA,
        WeightDeltaBank,
    )

    fusion = BlockSpec(
        "fusion",
        ("id_backbone.fusion.bias",),
        ((3,),),
        ("torch.float32",),
    )
    t3 = BlockSpec(
        "t3",
        ("id_backbone.t3.weight",),
        ((2, 3),),
        ("torch.float32",),
    )
    fusion_basis = nn.Parameter(torch.tensor([[0.3], [-0.2], [0.5]]))
    t3_basis = nn.Parameter(
        torch.tensor([[0.6], [-0.4], [0.2], [-0.5], [0.7], [-0.3]])
    )
    bank = WeightDeltaBank(
        schema=WEIGHT_DELTA_BANK_SCHEMA,
        base_checkpoint_id="base-task8",
        task_keys=(
            DeltaTaskKey("rx-0", "day-0", "leo_clear_weak", 2),
        ),
        entries=(
            DeltaBankEntry(
                spec=fusion,
                basis=fusion_basis,
                task_coefficients=torch.ones(1, 1),
                effective_rank=1,
                relative_error=0.0,
            ),
            DeltaBankEntry(
                spec=t3,
                basis=t3_basis,
                task_coefficients=torch.ones(1, 1),
                effective_rank=1,
                relative_error=0.0,
            ),
        ),
    )
    base_state = {
        "id_backbone.fusion.bias": torch.zeros(3),
        "id_backbone.t3.weight": torch.zeros(2, 3),
    }
    return bank, base_state, (fusion, t3), (fusion_basis, t3_basis)


def _episode_batch(episode):
    from cvsrffi.meta_trainer import MetaEpisodeBatch

    def rows(refs):
        class_vectors = {
            0: torch.tensor([1.0, 0.2]),
            1: torch.tensor([-0.2, 1.0]),
            2: torch.tensor([-0.8, -0.7]),
        }
        waveform = torch.linspace(1.0, 0.5, 16)
        values = torch.stack([class_vectors[int(row.tx_i)][:, None] * waveform for row in refs])
        labels = torch.tensor([int(row.tx_i) for row in refs], dtype=torch.long)
        return values, labels

    support_x, support_y = rows(episode.support)
    query_refs = episode.query_adapt + episode.query_guard
    query_x, query_y = rows(query_refs)
    adapt_rows = len(episode.query_adapt)
    return MetaEpisodeBatch(
        episode=episode,
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        adapt_mask=torch.tensor(
            [True] * adapt_rows + [False] * len(episode.query_guard)
        ),
        guard_mask=torch.tensor(
            [False] * adapt_rows + [True] * len(episode.query_guard)
        ),
        frozen_prototypes=torch.zeros(3, 2),
    )


def _run_phase1_test_entry(bundle_path: Path, *, learning_rate: float, selector):
    from cvsrffi.marc_ot_phase1 import run_marc_ot_phase1_bank_training
    from cvsrffi.meta_bank_trainer import MetaBankTrainerConfig
    from cvsrffi.meta_support_set_encoder import SupportSetEncoder

    bank, base_state, specs, bases = _bank_and_state()
    encoder = SupportSetEncoder(
        feature_dim=685,
        coefficient_dim=2,
        block_count=2,
        hidden_dim=5,
        lr_min=0.01,
        lr_max=0.08,
    )
    optimizer = torch.optim.SGD([*encoder.parameters(), *bases], lr=learning_rate)
    pre_step = {
        **{
            f"bank_basis.{entry.spec.name}": entry.basis.detach().clone()
            for entry in bank.entries
        },
        **{
            f"support_encoder.{name}": value.detach().clone()
            for name, value in encoder.state_dict().items()
        },
    }

    def functional_forward(state, values):
        core = values[:, :, 0]
        return core @ state["id_backbone.t3.weight"] + state["id_backbone.fusion.bias"]

    result = run_marc_ot_phase1_bank_training(
        sampler=_sampler(),
        batch_builder=_episode_batch,
        functional_forward=functional_forward,
        base_state=base_state,
        base_checkpoint_id="base-task8",
        bank=bank,
        support_encoder=encoder,
        support_feature_model=_SupportFeatureModel(),
        trainer_config=MetaBankTrainerConfig(
            source_receiver_ids=(0, 1),
            inner_steps=1,
        ),
        optimizer=optimizer,
        expected_block_specs=specs,
        bundle_path=bundle_path,
        training_episode_selector=selector,
        schedule_seed=31,
    )
    return result, bank, encoder, pre_step


def test_real_phase1_entry_runs_bank_step_and_strict_bundle_round_trip(tmp_path: Path) -> None:
    """Replacing the new bank step with the legacy adapter path breaks this integration."""
    result, bank, encoder, pre_step = _run_phase1_test_entry(
        tmp_path / "marc_ot_bundle.pt",
        learning_rate=0.01,
        selector=lambda episodes: tuple(
            row
            for row in episodes
            if row.k_shot == 2 and row.guard_class_ids
        )[:1],
    )

    assert result.entrypoint == "run_marc_ot_phase1_bank_training"
    assert result.bundle_path.is_file()
    assert result.software_coverage["software_supported_k"] == (1, 2, 5, 10, 20)
    assert result.training_coverage["trained_episode_count"] == 1
    assert result.training_coverage["k_shot"] == (2,)
    assert result.training_coverage["training_step_executed"] is True
    assert result.training_coverage["input_provenance"] == "CALLER_SUPPLIED_UNCLAIMED"
    assert result.training_coverage["updated_required_tensor_count"] > 0
    assert "source_training_executed" not in result.training_coverage
    assert result.pilot_executed is False
    assert result.loaded_bundle.base_checkpoint_id == "base-task8"
    assert all(not entry.basis.requires_grad for entry in result.loaded_bundle.bank.entries)
    assert all(not parameter.requires_grad for parameter in result.loaded_bundle.support_encoder.parameters())
    post_step = {
        **{
            f"bank_basis.{entry.spec.name}": entry.basis.detach().clone()
            for entry in bank.entries
        },
        **{
            f"support_encoder.{name}": value.detach().clone()
            for name, value in encoder.state_dict().items()
        },
    }
    loaded = {
        **{
            f"bank_basis.{entry.spec.name}": entry.basis.detach().clone()
            for entry in result.loaded_bundle.bank.entries
        },
        **{
            f"support_encoder.{name}": value.detach().clone()
            for name, value in result.loaded_bundle.support_encoder.state_dict().items()
        },
    }
    assert all(bool(torch.isfinite(value).all()) for value in post_step.values())
    assert any(not torch.equal(post_step[name], pre_step[name]) for name in pre_step)
    assert all(torch.equal(loaded[name], post_step[name]) for name in post_step)
    assert any(not torch.equal(loaded[name], pre_step[name]) for name in pre_step)


def test_phase1_entry_rejects_zero_lr_and_duplicate_selector(tmp_path: Path) -> None:
    """A no-op optimizer or repeated semantic cell cannot claim trained coverage."""

    with pytest.raises(ValueError, match="learning rate|positive"):
        _run_phase1_test_entry(
            tmp_path / "zero_lr.pt",
            learning_rate=0.0,
            selector=lambda episodes: episodes[:1],
        )
    assert not (tmp_path / "zero_lr.pt").exists()

    with pytest.raises(ValueError, match="duplicate|semantic"):
        _run_phase1_test_entry(
            tmp_path / "duplicate.pt",
            learning_rate=0.01,
            selector=lambda episodes: (episodes[0], episodes[0]),
        )
    assert not (tmp_path / "duplicate.pt").exists()

    with pytest.raises(ValueError, match="duplicate semantic"):
        _run_phase1_test_entry(
            tmp_path / "semantic_duplicate.pt",
            learning_rate=0.01,
            selector=lambda episodes: (
                episodes[0],
                replace(episodes[0], seed=episodes[0].seed + 1000),
            ),
        )
    assert not (tmp_path / "semantic_duplicate.pt").exists()

    extra = _sampler().sample_requested(
        kind="Q_SAME_DOMAIN",
        k_shot=1,
        seed=992,
    )
    with pytest.raises(ValueError, match="outside.*schedule"):
        _run_phase1_test_entry(
            tmp_path / "extra.pt",
            learning_rate=0.01,
            selector=lambda _episodes: (extra,),
        )
    assert not (tmp_path / "extra.pt").exists()


def test_support_supcon_k1_zero_and_k2_nonzero_gradient() -> None:
    """Removing positives or detaching normalized support rows must be observable."""
    from cvsrffi.stage2_marc_ot import supervised_contrastive_support_loss

    k1_features = torch.randn(3, 4, requires_grad=True)
    k1 = supervised_contrastive_support_loss(
        k1_features,
        torch.tensor([0, 1, 2]),
        temperature=0.07,
    )
    k1.loss.backward()
    assert k1.mode == "K1_NO_POSITIVE_PAIRS"
    assert k1.valid_anchor_count == 0
    assert float(k1.loss.detach()) == 0.0
    assert k1_features.grad is not None
    assert torch.count_nonzero(k1_features.grad) == 0

    features = torch.tensor(
        [[1.0, 0.0], [0.7, 0.4], [0.0, 1.0], [0.4, 0.7]],
        requires_grad=True,
    )
    k2 = supervised_contrastive_support_loss(
        features,
        torch.tensor([0, 0, 1, 1]),
        temperature=0.07,
    )
    gradient = torch.autograd.grad(k2.loss, features)[0]
    assert k2.mode == "SUPPORT_ONLY_SUPCON"
    assert k2.valid_anchor_count == 4
    assert float(k2.loss.detach()) > 0.0
    assert torch.count_nonzero(gradient)


def test_r2_enables_supcon_while_r1_is_off_and_gradient_differs() -> None:
    """R2 must contain a real positive-weight SupCon gradient absent from R1."""
    from cvsrffi.stage2_marc_ot import supervised_contrastive_support_loss
    from cvsrffi.stage2_marc_ot_runner import (
        MARCOTRunnerConfig,
        supcon_weight_for_arm,
    )

    config = MARCOTRunnerConfig(stage_steps=(0, 0, 0, 0))
    assert supcon_weight_for_arm("R0", config) == 0.0
    assert supcon_weight_for_arm("R1", config) == 0.0
    for arm in ("R2", "R4", "R6", "R8"):
        assert supcon_weight_for_arm(arm, config) == config.supcon_weight > 0.0

    features = torch.tensor(
        [[1.0, 0.0], [0.6, 0.5], [0.0, 1.0], [0.5, 0.6]],
        requires_grad=True,
    )
    supcon = supervised_contrastive_support_loss(
        features,
        torch.tensor([0, 0, 1, 1]),
        temperature=config.supcon_temperature,
    ).loss
    r1_gradient = torch.autograd.grad(0.0 * supcon, features, retain_graph=True)[0]
    r2_gradient = torch.autograd.grad(config.supcon_weight * supcon, features)[0]
    assert torch.count_nonzero(r1_gradient) == 0
    assert torch.count_nonzero(r2_gradient)
    assert not torch.equal(r1_gradient, r2_gradient)


@pytest.mark.parametrize("k_shot", (2, 5, 10, 20))
def test_formal_fold_plan_is_deterministic_for_every_supported_k_above_one(
    k_shot: int,
) -> None:
    from cvsrffi.stage2_marc_ot_runner import build_marc_ot_support_fold_plan

    labels = torch.arange(3).repeat_interleave(k_shot)
    tokens = tuple(f"c{int(label)}-{index}" for index, label in enumerate(labels))
    first = build_marc_ot_support_fold_plan(labels, tokens, fold_count=5, seed=19)
    second = build_marc_ot_support_fold_plan(labels, tokens, fold_count=5, seed=19)

    assert first.mode == "DETERMINISTIC_HELD_OUT_CROSSFIT"
    assert first.held_out_support_evidence is True
    assert len(first.folds) == min(5, k_shot)
    assert [
        (fit.tolist(), validation.tolist()) for fit, validation in first.folds
    ] == [
        (fit.tolist(), validation.tolist()) for fit, validation in second.folds
    ]
    all_rows = set(range(len(labels)))
    for fit, validation in first.folds:
        fit_rows = set(fit.tolist())
        validation_rows = set(validation.tolist())
        assert fit_rows and validation_rows
        assert fit_rows.isdisjoint(validation_rows)
        assert fit_rows | validation_rows == all_rows


def test_formal_runner_k1_falls_back_without_fake_held_out_evidence() -> None:
    from cvsrffi.stage2_marc_ot_runner import MARCOTRunnerConfig, train_marc_ot_arm

    model = nn.Linear(2, 2)
    audit = train_marc_ot_arm(
        model,
        torch.randn(2, 2),
        torch.tensor([0, 1]),
        ("c0", "c1"),
        arm="R2",
        config=MARCOTRunnerConfig(stage_steps=(1, 1, 1, 1)),
        calibration_feature_transform=lambda *_args: pytest.fail(
            "K1 conservative fallback must not train"
        ),
    )

    assert audit.selected_alpha == 0.0
    assert audit.optimizer_steps == 0
    assert audit.held_out_support_evidence is False
    assert audit.stage_audits == (
        {
            "stage": "K1_CONSERVATIVE_FALLBACK",
            "mode": "K1_NO_HELD_OUT_SUPPORT_EVIDENCE",
            "held_out_support_evidence": False,
            "crossfit_fold_count": 0,
            "optimizer_steps": 0,
            "query_rows_used": 0,
        },
    )
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_pilot_config_separates_software_training_and_executed_k() -> None:
    import json

    from cvsrffi.stage2_marc_ot_pilot import validate_pilot_config

    path = Path(__file__).resolve().parents[1] / "configs" / "marc_ot_k10_pilot_20260901.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    payload.pop("support_feature")
    validated = validate_pilot_config(payload)

    assert tuple(validated["software_supported_k"]) == (1, 2, 5, 10, 20)
    assert validated["pilot_k"] == validated["k_shot"] == 10
    assert validated["pilot_executed"] is False
    assert validated["training_coverage_k"] == []
    assert validated["supcon"] == {"temperature": 0.07, "weight": 0.1}
