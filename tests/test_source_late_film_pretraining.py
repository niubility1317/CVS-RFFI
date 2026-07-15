from __future__ import annotations

import numpy as np
import pytest
import torch

from paper_reproduction.scripts.pretrain_cvs_source_late_film import (
    RX_LIGHT5_VIEW_NAMES,
    _load_exact_rx_light5_source_views,
    parse_args,
    select_source_ground_split,
    train_ground_source_film,
)
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS


def test_source_ground_split_excludes_all_target_rows() -> None:
    arrays = {
        "dataset_role": np.asarray(
            ["source", "source", "source", "source", "target_old", "target_new"]
        ),
        "rx_ids": np.asarray(["1-1", "1-1", "2-19", "2-19", "8-8", "8-8"]),
    }
    train, validation, audit = select_source_ground_split(
        arrays,
        source_receivers=["1-1", "2-19"],
        val_receiver="2-19",
    )
    assert train.tolist() == [0, 1]
    assert validation.tolist() == [2, 3]
    assert audit["consumed_roles"] == ["source"]
    assert audit["target_row_count"] == 0
    assert audit["target_query_row_count"] == 0


def test_source_ground_split_rejects_receiver_drift() -> None:
    arrays = {
        "dataset_role": np.asarray(["source", "source"]),
        "rx_ids": np.asarray(["1-1", "2-19"]),
    }
    with pytest.raises(ValueError, match="source receiver mismatch"):
        select_source_ground_split(
            arrays,
            source_receivers=["1-1", "18-2"],
            val_receiver="18-2",
        )


def test_ground_pretraining_cli_caps_epochs_and_steps() -> None:
    args = parse_args(
        [
            "--config",
            "config.json",
            "--ckpt",
            "model.pth",
            "--out_root",
            "out",
            "--epochs",
            "20",
            "--max_optimizer_steps",
            "400",
        ]
    )
    assert args.epochs == 20
    assert args.max_optimizer_steps == 400
    assert args.adapter_type == "late_key_ft"
    with pytest.raises(ValueError, match="ground epochs"):
        parse_args(
            [
                "--config",
                "config.json",
                "--ckpt",
                "model.pth",
                "--out_root",
                "out",
                "--epochs",
                "31",
            ]
        )


def test_exact_rx_light5_is_applied_to_every_formal_scenario() -> None:
    base = np.arange(3 * 2 * 8, dtype=np.float32).reshape(3, 2, 8)
    scenario_arrays = {
        scenario: {"raw_iq": base + float(index)}
        for index, scenario in enumerate(SCENARIOS)
    }
    views, names, audit = _load_exact_rx_light5_source_views(
        scenario_arrays, np.asarray([0, 2], dtype=np.int64)
    )
    assert views.shape == (15, 2, 2, 8)
    assert names == tuple(
        f"{scenario}/{view}"
        for scenario in SCENARIOS
        for view in RX_LIGHT5_VIEW_NAMES
    )
    assert all(audit[scenario]["tta_view_count"] == 5 for scenario in SCENARIOS)
    assert torch.equal(views[1], torch.roll(views[0], shifts=-2, dims=-1))
    assert torch.equal(views[2], torch.roll(views[0], shifts=2, dims=-1))


def test_ground_trainer_selects_source_validation_checkpoint(monkeypatch) -> None:
    import paper_reproduction.scripts.pretrain_cvs_source_late_film as trainer

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.film = torch.nn.Parameter(torch.zeros(1_280))

    model = TinyModel()

    def fake_feature_forward(current_model, rows):
        signal = rows.mean(dim=(1, 2))
        feature = torch.stack(
            [signal + current_model.film[0], -signal + current_model.film[1]],
            dim=1,
        )
        logits = torch.stack([-feature[:, 0], feature[:, 0]], dim=1)
        return feature, logits

    monkeypatch.setattr(trainer, "_feature_forward", fake_feature_forward)
    physical = torch.stack(
        [torch.full((2, 4), value) for value in (-1.0, -0.8, -0.6, 0.6, 0.8, 1.0)]
    )
    views = torch.stack(
        [physical * (0.9 + 0.01 * index) for index in range(15)], dim=0
    )
    view_names = tuple(
        f"{scenario}/{view}"
        for scenario in SCENARIOS
        for view in RX_LIGHT5_VIEW_NAMES
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    teacher_mean = torch.nn.functional.normalize(
        torch.stack([physical.mean(dim=(1, 2)), -physical.mean(dim=(1, 2))], dim=1),
        dim=1,
    )
    trace, state, runtime = train_ground_source_film(
        model,
        views,
        labels,
        torch.tensor([0, 1, 3, 4]),
        torch.tensor([2, 5]),
        teacher_mean,
        epochs=2,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        teacher_weight=0.25,
        batch_size=4,
        grad_clip=1.0,
        max_optimizer_steps=4,
        view_names=view_names,
        multiview_consistency_weight=0.5,
        seed=713101,
        device=torch.device("cpu"),
    )
    assert len(trace) == 2
    assert runtime["optimizer_steps"] == 2
    assert runtime["deployment_optimizer_state_required"] is False
    assert set(state) == {"film"}
    assert state["film"].numel() == 1_280
