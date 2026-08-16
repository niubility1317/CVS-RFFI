"""Behavioral checks for the source-only MIRAGE EMA/SWAD fold trainer."""

from __future__ import annotations

import copy
import io
import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


class _LabeledSource(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, *, seed: int, scene_offset: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.iq = torch.randn(6, 2, 32, generator=generator)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
        self.receiver_ids = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.int64)
        self.day_ids = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)
        self.scene_ids = torch.tensor(
            [scene_offset, scene_offset, scene_offset + 1, scene_offset + 1, scene_offset, scene_offset + 1],
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "iq": self.iq[index],
            "labels": self.labels[index],
            "receiver_ids": self.receiver_ids[index],
            "day_ids": self.day_ids[index],
            "scene_ids": self.scene_ids[index],
        }


class _UnlabeledSource(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, *, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weak_iq = torch.randn(6, 2, 32, generator=generator)
        self.strong_iq = self.weak_iq + 0.01 * torch.randn(6, 2, 32, generator=generator)

    def __len__(self) -> int:
        return int(self.weak_iq.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"weak_iq": self.weak_iq[index], "strong_iq": self.strong_iq[index]}


def _tiny_role_safe_loaders() -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    return {
        "l": DataLoader(_LabeledSource(seed=11), batch_size=6, shuffle=False),
        "u": DataLoader(_UnlabeledSource(seed=12), batch_size=6, shuffle=False),
        "v_cal": DataLoader(_LabeledSource(seed=13, scene_offset=10), batch_size=6, shuffle=False),
        "v_select": DataLoader(_LabeledSource(seed=14, scene_offset=20), batch_size=6, shuffle=False),
    }


def _trainer_api():
    """Import lazily so RED proves the missing trainer module, not a test typo."""

    try:
        from cvsrffi.phase1_mirage import trainer
    except ImportError as error:
        pytest.fail(f"missing source-only MIRAGE trainer: {error}")
    return trainer


def _smoke_config(trainer, *, arm: str = "B0", epochs: int = 2):
    return trainer.TrainConfig(
        arm=arm,
        epochs=epochs,
        formal=False,
        num_classes=3,
        device="cpu",
        git_commit="a" * 40,
        split_sha256="b" * 64,
    )


def _module_bytes(module: torch.nn.Module) -> bytes:
    snapshot = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    stream = io.BytesIO()
    torch.save(snapshot, stream)
    return stream.getvalue()


def test_formal_schedule_and_loader_schema_fail_closed(tmp_path):
    """Catch a non-200 formal trainer or a source run that accepts target/unknown roles."""

    trainer = _trainer_api()
    with pytest.raises(trainer.TrainingProtocolError, match="200"):
        trainer.TrainConfig(arm="B0", epochs=2, formal=True)
    formal = trainer.TrainConfig(arm="C", epochs=200, formal=True)
    assert [trainer._stage_for_epoch(epoch, formal) for epoch in (1, 40, 41, 160, 161, 200)] == [
        "warmup",
        "warmup",
        "joint",
        "joint",
        "stabilization",
        "stabilization",
    ]

    loaders = _tiny_role_safe_loaders()
    config = _smoke_config(trainer)
    for forbidden_key in ("target", "target_known", "target_unknown"):
        candidate = dict(loaders)
        candidate[forbidden_key] = loaders["l"]
        with pytest.raises(trainer.TrainingProtocolError, match="target loader"):
            trainer.train_fold(config, candidate, tmp_path / forbidden_key)

    missing = dict(loaders)
    del missing["u"]
    with pytest.raises(trainer.TrainingProtocolError, match="loader schema"):
        trainer.train_fold(config, missing, tmp_path / "missing")
    extra = dict(loaders)
    extra["targeting_note"] = loaders["l"]
    with pytest.raises(trainer.TrainingProtocolError, match="loader schema"):
        trainer.train_fold(config, extra, tmp_path / "non_target_extra")


def test_cpu_smoke_writes_reloadable_checkpoint_and_nonoverwriting_receipts(tmp_path):
    """Catch a trainer that skips a real step/EMA update, receipt closure, or reload forward."""

    trainer = _trainer_api()
    torch.manual_seed(33)
    output_dir = tmp_path / "run"
    result = trainer.train_fold(
        _smoke_config(trainer),
        _tiny_role_safe_loaders(),
        output_dir,
    )

    assert result.checkpoint_path.is_file()
    assert result.completion_receipt["status"] == "COMPLETED"
    assert result.completion_receipt["epochs_completed"] == 2
    assert result.completion_receipt["selection_source"] == "V_select"
    assert result.completion_receipt["checkpoint_sha256"] == trainer._file_sha256(result.checkpoint_path)
    assert result.completion_receipt["swad_strategy"] == "EMA_FALLBACK_NO_COMPLETE_161_200_WINDOW"
    for name in (
        "metrics_epoch.jsonl",
        "metrics_epoch.csv",
        "split_receipt.json",
        "proxy_receipt.json",
        "resource_receipt.json",
        "completion_receipt.json",
    ):
        assert (output_dir / name).is_file()

    metrics = [json.loads(line) for line in (output_dir / "metrics_epoch.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 2
    assert all(row["optimizer_steps"] == 1 for row in metrics)
    assert all(row["ema_updates"] == row["optimizer_steps"] for row in metrics)
    assert all(row["ema_updated_after_optimizer_step"] for row in metrics)

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["config_sha256"] == trainer._canonical_sha256(checkpoint["config"])
    assert checkpoint["split_sha256"] == "b" * 64
    assert checkpoint["git_commit"] == "a" * 40
    assert {"python", "numpy", "torch", "cuda"} <= set(checkpoint["rng_state"])
    assert {"model", "head", "ema", "swad"} <= set(checkpoint["state_dict_sha256"])

    model, head, _ = trainer.build_models_and_optimizer(_smoke_config(trainer))
    model.load_state_dict(checkpoint["model_state"])
    head.load_state_dict(checkpoint["head_state"])
    model.eval()
    head.eval()
    with torch.no_grad():
        output = head(model(torch.randn(2, 2, 32)).z_id)
    assert torch.isfinite(output.class_scores).all()

    with pytest.raises(trainer.TrainingProtocolError, match="output directory"):
        trainer.train_fold(_smoke_config(trainer), _tiny_role_safe_loaders(), output_dir)


def test_validation_is_read_only_for_model_head_and_ema_state():
    """Catch validation that mutates source model, EMA teacher, head, or prototype bytes."""

    trainer = _trainer_api()
    config = _smoke_config(trainer)
    model, head, _ = trainer.build_models_and_optimizer(config)
    ema = trainer.make_ema_copy(model, head)
    before = {
        "model": _module_bytes(model),
        "head": _module_bytes(head),
        "ema_model": _module_bytes(ema.model),
        "ema_head": _module_bytes(ema.head),
    }

    validation = trainer.run_source_validation(
        model=model,
        head=head,
        ema=ema,
        v_cal=_tiny_role_safe_loaders()["v_cal"],
        v_select=_tiny_role_safe_loaders()["v_select"],
        device=torch.device("cpu"),
    )

    assert {"v_cal_known_macro", "v_select_known_macro", "v_select_worst_scene"} <= set(validation)
    assert before == {
        "model": _module_bytes(model),
        "head": _module_bytes(head),
        "ema_model": _module_bytes(ema.model),
        "ema_head": _module_bytes(ema.head),
    }


@pytest.mark.parametrize("arm", ("B0", "A", "B", "C"))
def test_joint_stage_executes_each_frozen_arm_with_a_real_optimizer_step(arm):
    """Catch an arm that names a frozen mechanism but cannot execute its joint source step."""

    trainer = _trainer_api()
    config = _smoke_config(trainer, arm=arm, epochs=41)
    model, head, optimizer = trainer.build_models_and_optimizer(config)
    ema = trainer.make_ema_copy(model, head)
    loaders = _tiny_role_safe_loaders()

    result = trainer.run_train_epoch(
        model=model,
        head=head,
        ema=ema,
        optimizer=optimizer,
        labeled_loader=loaders["l"],
        unlabeled_loader=loaders["u"],
        epoch=41,
        config=config,
    )

    assert result["stage"] == "joint"
    assert result["optimizer_steps"] == 1
    assert result["ema_updates"] == 1
    assert "total" in result["losses"]
    assert bool(result["proxy_schedule"]) is (arm in {"B", "C"})


def test_v_cal_cannot_select_a_checkpoint_when_v_select_prefers_another_epoch():
    """Catch a best-epoch rule that leaks calibration performance into source model selection."""

    trainer = _trainer_api()
    history = [
        {
            "epoch": 1,
            "v_cal_known_macro": 0.01,
            "v_cal_worst_scene": 0.01,
            "v_select_known_macro": 0.80,
            "v_select_worst_scene": 0.50,
        },
        {
            "epoch": 2,
            "v_cal_known_macro": 1.00,
            "v_cal_worst_scene": 1.00,
            "v_select_known_macro": 0.79,
            "v_select_worst_scene": 1.00,
        },
        {
            "epoch": 3,
            "v_cal_known_macro": 1.00,
            "v_cal_worst_scene": 1.00,
            "v_select_known_macro": 0.80,
            "v_select_worst_scene": 0.50,
        },
    ]

    selected = trainer._select_best_epoch(history)
    assert selected["epoch"] == 1
    assert selected["selection_source"] == "V_select"


def test_state_dict_hash_includes_scalar_head_parameters():
    """Catch a receipt hash that crashes on the head's scalar risk-bias state."""

    trainer = _trainer_api()
    digest = trainer._state_dict_sha256({"risk_bias": torch.tensor(1.0)})
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_completion_receipt_verifies_the_checkpoint_epoch_and_selection_binding(tmp_path):
    """Catch a receipt that trusts caller fields instead of its sealed checkpoint bytes."""

    trainer = _trainer_api()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epochs_completed": 1,
            "selection": {"epoch": 1, "selection_source": "V_select"},
        },
        checkpoint_path,
    )
    selection = {
        "epoch": 2,
        "selection_source": "V_select",
        "v_select_known_macro": 0.7,
        "v_select_worst_scene": 0.6,
    }

    with pytest.raises(trainer.TrainingProtocolError, match="checkpoint epoch"):
        trainer.write_completion_receipt(
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
            status="COMPLETED",
            epochs=2,
            selection=selection,
            swad_strategy="EMA_FALLBACK_NO_COMPLETE_161_200_WINDOW",
        )


def test_completion_receipt_uses_only_the_sealed_checkpoint_selection(tmp_path):
    """Catch a completed receipt that substitutes caller-supplied V_select metrics."""

    trainer = _trainer_api()
    checkpoint_path = tmp_path / "checkpoint.pt"
    sealed_selection = {
        "epoch": 2,
        "selection_source": "V_select",
        "v_select_known_macro": 0.20,
        "v_select_worst_scene": 0.10,
    }
    torch.save({"epochs_completed": 2, "selection": sealed_selection}, checkpoint_path)
    mismatched_selection = {
        **sealed_selection,
        "v_select_known_macro": 0.99,
        "v_select_worst_scene": 0.98,
    }

    with pytest.raises(trainer.TrainingProtocolError, match="checkpoint selection"):
        trainer.write_completion_receipt(
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
            status="COMPLETED",
            epochs=2,
            selection=mismatched_selection,
            swad_strategy="EMA_FALLBACK_NO_COMPLETE_161_200_WINDOW",
        )
    assert not (tmp_path / "completion_receipt.json").exists()

    receipt = trainer.write_completion_receipt(
        output_dir=tmp_path,
        checkpoint_path=checkpoint_path,
        status="COMPLETED",
        epochs=2,
        selection=sealed_selection,
        swad_strategy="EMA_FALLBACK_NO_COMPLETE_161_200_WINDOW",
    )
    persisted = json.loads((tmp_path / "completion_receipt.json").read_text(encoding="utf-8"))
    assert receipt["v_select_known_macro"] == sealed_selection["v_select_known_macro"]
    assert receipt["v_select_worst_scene"] == sealed_selection["v_select_worst_scene"]
    assert persisted["v_select_known_macro"] == sealed_selection["v_select_known_macro"]
    assert persisted["v_select_worst_scene"] == sealed_selection["v_select_worst_scene"]


def test_completion_receipt_rejects_nonfinite_sealed_selection_metrics(tmp_path):
    """Catch NaN or infinity being persisted as a V_select completion metric."""

    trainer = _trainer_api()
    checkpoint_path = tmp_path / "checkpoint.pt"
    selection = {
        "epoch": 2,
        "selection_source": "V_select",
        "v_select_known_macro": float("nan"),
        "v_select_worst_scene": 0.10,
    }
    torch.save({"epochs_completed": 2, "selection": selection}, checkpoint_path)

    with pytest.raises(trainer.TrainingProtocolError, match="finite"):
        trainer.write_completion_receipt(
            output_dir=tmp_path,
            checkpoint_path=checkpoint_path,
            status="COMPLETED",
            epochs=2,
            selection=selection,
            swad_strategy="EMA_FALLBACK_NO_COMPLETE_161_200_WINDOW",
        )
    assert not (tmp_path / "completion_receipt.json").exists()
