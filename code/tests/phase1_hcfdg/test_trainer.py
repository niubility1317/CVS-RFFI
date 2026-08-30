import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cvsrffi.phase1_hcfdg.config import candidate_config
from cvsrffi.phase1_hcfdg.trainer import (
    CheckpointPayload,
    HCFDGTrainer,
    cosface_margin_at,
    learning_rate_at,
)


class _UnlabeledBatch(dict):
    """A source U_s batch that fails if identity labels or forbidden views leak in."""

    def __getitem__(self, key):
        if key in {"tx", "tx_labels", "label", "labels", "target", "query", "truth"}:
            raise AssertionError(f"forbidden Stage0 key opened: {key}")
        return super().__getitem__(key)


class _Stage0StrictBatch(_UnlabeledBatch):
    def __getitem__(self, key):
        if key in {"channel", "channel_id", "scenario", "q_phys", "physical_stats", "phys_stats"}:
            raise AssertionError(f"Stage0 metadata outside IQ/receiver/day opened: {key}")
        return super().__getitem__(key)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = nn.Linear(1, 1)
        self.time_domain_block = nn.Linear(1, 1)
        self.identity_backbone = nn.Linear(1, 1)
        self.common_head = nn.Linear(1, 2)
        self.new_head = nn.Linear(1, 2)
        self.environment_encoder = nn.Linear(3, 2)
        self.forward_calls = 0
        self.stage0_calls = 0
        self.stage0_received_tx = False
        self.freeze_calls = 0
        self.margin_history = []

    def set_cosface_margin(self, value):
        self.margin_history.append(float(value))

    def stage0_step(self, *, iq, receiver, day, env_meta):
        self.stage0_calls += 1
        assert iq is not None
        assert receiver is not None
        assert day is not None
        assert "tx" not in env_meta
        features = torch.stack(
            [iq.reshape(iq.shape[0], -1).mean(dim=1), receiver.float(), day.float()], dim=1
        )
        return self.environment_encoder(features).square().mean()

    def freeze_sinc_and_first_time_domain_block(self):
        self.freeze_calls += 1
        for module in (self.sinc, self.time_domain_block):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, x, *, tx_labels=None, env_meta=None, training_aux=False, **kwargs):
        self.forward_calls += 1
        if x is None:
            x = torch.zeros(1, 1)
        x = x.float().reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        hidden = self.sinc(x)
        hidden = self.time_domain_block(hidden)
        hidden = self.identity_backbone(hidden)
        logits = self.common_head(hidden)
        if tx_labels is None:
            loss = hidden.square().mean()
        else:
            labels = tx_labels.reshape(-1).long().remainder(2)
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "common_logits": logits}


class _SingleViewBuilder:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, augmentor, generator, p_sat=0.30):
        self.calls += 1
        assert p_sat == pytest.approx(0.30)
        assert generator is not None
        return x


class _EnvironmentOnlyEncoder(nn.Module):
    input_dim = 3

    def __init__(self):
        super().__init__()
        self.receiver = nn.Linear(3, 4)
        self.day = nn.Linear(3, 3)
        self.channel = nn.Linear(3, 2)
        self.tx = nn.Linear(3, 2)
        self.calls = 0

    def forward(
        self,
        h_early,
        q_phys=None,
        env_meta=None,
        receiver_labels=None,
        day_labels=None,
        channel_labels=None,
    ):
        self.calls += 1
        return SimpleNamespace(
            receiver_logits=self.receiver(h_early),
            day_logits=self.day(h_early),
            channel_logits=self.channel(h_early),
            tx_from_env_logits=self.tx(h_early),
        )


class _NamedFrontendModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = nn.Linear(1, 1)
        self.t1 = nn.Linear(1, 1)
        self.t2 = nn.Linear(1, 1)
        self.head = nn.Linear(1, 2)

    def forward(self, x, **kwargs):
        hidden = self.t2(self.t1(self.sinc(x.float())))
        return {"common_logits": self.head(hidden)}


class _ParameterlessModel:
    def __call__(self, x, **kwargs):
        return torch.zeros((), requires_grad=True)


class _SingleParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x, **kwargs):
        return self.weight * torch.ones((), device=self.weight.device)


def _labeled_batch():
    return {
        "iq": torch.ones(4, 1),
        "tx": torch.tensor([0, 1, 0, 1]),
        "receiver": torch.tensor([1, 1, 3, 3]),
        "day": torch.tensor([1, 2, 1, 2]),
    }


def _unlabeled_batch():
    return _UnlabeledBatch(
        iq=torch.ones(4, 1),
        receiver=torch.tensor([1, 1, 3, 3]),
        day=torch.tensor([1, 2, 1, 2]),
    )


def _make_trainer(tmp_path, config=None):
    model = _FakeModel()
    builder = _SingleViewBuilder()
    trainer = HCFDGTrainer(
        model=model,
        config=config or candidate_config("A5"),
        labeled_loader=[_labeled_batch()],
        unlabeled_loader=[_unlabeled_batch()],
        validation_loader=[],
        build_single_view_batch=builder,
        device="cpu",
        output_dir=tmp_path,
        source_split="L_s/U_s/V_cal/V_select",
        fold=8,
        seed=392001,
    )
    return trainer, model, builder


def test_v1_runs_exactly_4000_optimizer_updates_and_one_backbone_call_per_update(tmp_path):
    trainer, model, builder = _make_trainer(tmp_path)

    state = trainer.train(candidate_config("A5"))

    assert state.optimizer_updates == 4000
    assert state.backbone_forward_calls == 4000
    assert model.forward_calls == 4000
    assert builder.calls == 4000


def test_v2_stage_counts_total_6300_and_freezes_at_half(tmp_path):
    trainer, model, _ = _make_trainer(tmp_path, candidate_config("A9"))

    state = trainer.train(candidate_config("A9"))

    assert state.stage_updates == {
        "stage0": 700,
        "stage1": 1200,
        "stage2": 2100,
        "stage3": 1700,
        "stage4": 600,
    }
    assert state.optimizer_updates == 6300
    assert state.freeze_update == 3150
    assert model.freeze_calls == 1
    assert not model.sinc.weight.requires_grad
    assert not model.time_domain_block.weight.requires_grad
    assert state.environment_updates == 700 + (5600 // 4)


def test_stage0_never_opens_u_s_tx_or_forbidden_query_keys(tmp_path):
    trainer, model, _ = _make_trainer(tmp_path, candidate_config("A9"))

    state = trainer.train(candidate_config("A9"))

    assert state.stage_updates["stage0"] == 700
    assert model.stage0_calls == 700
    assert model.stage0_received_tx is False


def test_stage0_field_view_is_limited_to_iq_receiver_and_day(tmp_path):
    trainer, _, _ = _make_trainer(tmp_path, candidate_config("A9"))
    raw = _Stage0StrictBatch(
        iq=torch.ones(4, 1),
        receiver=torch.tensor([1, 1, 3, 3]),
        day=torch.tensor([1, 2, 1, 2]),
    )

    source_batch = trainer._prepare_source_batch(raw, allow_tx=False, stage0=True)

    assert set(source_batch.env_meta) == {"receiver", "day"}


def test_stage0_trains_environment_encoder_without_identity_forward(tmp_path):
    trainer, model, _ = _make_trainer(tmp_path, candidate_config("A9"))
    model.stage0_step = None
    model.environment_encoder = _EnvironmentOnlyEncoder()
    raw = _Stage0StrictBatch(
        iq=torch.ones(4, 1),
        receiver=torch.tensor([1, 1, 3, 3]),
        day=torch.tensor([1, 2, 1, 2]),
    )
    source_batch = trainer._prepare_source_batch(raw, allow_tx=False, stage0=True)

    loss = trainer._stage0_loss(source_batch, update=1)

    assert torch.isfinite(loss)
    assert model.forward_calls == 0
    assert model.environment_encoder.calls == 1


def test_custom_loss_function_receives_one_model_output_without_duplicate_output_argument(tmp_path):
    short_config = replace(candidate_config("A5"), optimizer_updates=1)
    calls = []

    def loss_fn(output, tx_labels, env_meta, update, stage):
        calls.append((output, tx_labels, env_meta, update, stage))
        return output["common_logits"].square().mean()

    trainer, _, _ = _make_trainer(tmp_path, short_config)
    trainer.loss_fn = loss_fn

    state = trainer.train(short_config)

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][3:] == (1, "v1")
    assert state.optimizer_updates == 1


@pytest.mark.parametrize(
    "objective",
    (
        None,
        torch.ones(2, requires_grad=True),
        torch.tensor(float("nan"), requires_grad=True),
        torch.tensor(float("inf"), requires_grad=True),
    ),
    ids=("none", "non_scalar", "nan", "inf"),
)
def test_main_invalid_objective_fails_closed_before_optimizer_update(tmp_path, objective):
    short_config = replace(candidate_config("A5"), optimizer_updates=1)
    trainer, _, _ = _make_trainer(tmp_path, short_config)
    trainer.loss_fn = lambda *args, **kwargs: objective
    step_calls = []
    original_step = trainer.optimizer.step

    def counted_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step

    with pytest.raises((ValueError, FloatingPointError), match="objective"):
        trainer.train(short_config)

    assert step_calls == []
    assert not (tmp_path / "metrics.jsonl").exists()


@pytest.mark.parametrize(
    "objective",
    (
        None,
        torch.ones(2, requires_grad=True),
        torch.tensor(float("nan"), requires_grad=True),
        torch.tensor(float("inf"), requires_grad=True),
    ),
    ids=("none", "non_scalar", "nan", "inf"),
)
def test_stage0_invalid_objective_fails_closed_before_optimizer_update(tmp_path, objective):
    trainer, model, _ = _make_trainer(tmp_path, candidate_config("A9"))
    model.stage0_step = lambda **kwargs: objective
    step_calls = []
    original_step = trainer.optimizer.step

    def counted_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step

    with pytest.raises((ValueError, FloatingPointError), match="objective"):
        trainer.train(candidate_config("A9"))

    assert step_calls == []
    assert not (tmp_path / "metrics.jsonl").exists()


def test_freeze_fallback_targets_sinc_and_first_t1_block_only(tmp_path):
    model = _NamedFrontendModel()
    trainer = HCFDGTrainer(
        model=model,
        config=candidate_config("A9"),
        labeled_loader=[],
        device="cpu",
        output_dir=tmp_path,
    )

    frozen = trainer._freeze_frontend()

    assert any(name.startswith("sinc.") for name in frozen)
    assert any(name.startswith("t1.") for name in frozen)
    assert model.sinc.weight.requires_grad is False
    assert model.t1.weight.requires_grad is False
    assert model.t2.weight.requires_grad is True


def test_parameterless_model_adapter_keeps_optimizer_groups_disjoint(tmp_path):
    model = _ParameterlessModel()
    trainer = HCFDGTrainer(
        model=model,
        config=replace(candidate_config("A5"), optimizer_updates=1),
        labeled_loader=[_labeled_batch()],
        device="cpu",
        output_dir=tmp_path,
    )

    first_ids = {id(parameter) for parameter in trainer.optimizer.param_groups[0]["params"]}
    second_ids = {id(parameter) for parameter in trainer.optimizer.param_groups[1]["params"]}

    assert first_ids.isdisjoint(second_ids)


def test_single_parameter_model_adapter_keeps_optimizer_groups_disjoint(tmp_path):
    trainer = HCFDGTrainer(
        model=_SingleParameterModel(),
        config=replace(candidate_config("A5"), optimizer_updates=1),
        labeled_loader=[],
        device="cpu",
        output_dir=tmp_path,
    )

    first_ids = {id(parameter) for parameter in trainer.optimizer.param_groups[0]["params"]}
    second_ids = {id(parameter) for parameter in trainer.optimizer.param_groups[1]["params"]}

    assert first_ids.isdisjoint(second_ids)


def test_exact_optimizer_schedule_and_cosface_ramp():
    assert learning_rate_at(1, total_updates=4000, base_lr=1e-4) == pytest.approx(5e-7)
    assert learning_rate_at(200, total_updates=4000, base_lr=1e-4) == pytest.approx(1e-4)
    assert learning_rate_at(4000, total_updates=4000, base_lr=1e-4) == pytest.approx(1e-6)
    assert learning_rate_at(315, total_updates=6300, base_lr=3e-4) == pytest.approx(3e-4)
    assert cosface_margin_at(1, total_updates=4000) == pytest.approx(0.30 / 800.0)
    assert cosface_margin_at(800, total_updates=4000) == pytest.approx(0.30)
    assert cosface_margin_at(4000, total_updates=4000) == pytest.approx(0.30)


def test_optimizer_has_exact_backbone_and_new_head_groups(tmp_path):
    trainer, _, _ = _make_trainer(tmp_path)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(3e-4)
    assert [group["weight_decay"] for group in trainer.optimizer.param_groups] == [
        pytest.approx(1e-4),
        pytest.approx(1e-4),
    ]
    assert trainer.parameter_group_names == ("backbone", "new_head")


def test_checkpoint_and_telemetry_have_required_schema(tmp_path):
    short_config = replace(candidate_config("A5"), optimizer_updates=4)
    trainer, _, _ = _make_trainer(tmp_path, short_config)

    state = trainer.train(short_config)
    payload = state.checkpoint

    assert isinstance(payload, CheckpointPayload)
    data = payload.to_dict()
    assert data["phase1_method"] == "hcfdg"
    assert data["candidate_id"] == "A5"
    assert data["source_split"] == "L_s/U_s/V_cal/V_select"
    assert data["fold"] == 8
    assert data["seed"] == 392001
    assert data["update"] == 4
    assert data["model_state"]
    assert data["optimizer_state"]
    assert "scaler_state" in data
    assert data["model"] is data["model_state"]
    assert data["optimizer"] is data["optimizer_state"]
    assert data["scaler"] is data["scaler_state"]
    assert data["inference"]["head"] == "common"
    assert data["inference"]["common_head_only"] is True

    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "metrics.csv").exists()
    json_rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(json_rows) == 4
    required = {
        "step_time",
        "samples/s",
        "dataloader_wait",
        "peak_memory",
        "forward_time",
        "backward_time",
        "total_gpu_hours",
    }
    assert required <= set(json_rows[-1])
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 4
    assert required <= set(csv_rows[-1])


def test_trainer_rejects_target_or_query_loader_at_construction(tmp_path):
    model = _FakeModel()
    with pytest.raises(ValueError, match="target/query"):
        HCFDGTrainer(
            model=model,
            config=candidate_config("A5"),
            labeled_loader=[_labeled_batch()],
            target_loader=object(),
            device="cpu",
            output_dir=tmp_path,
        )
