from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from SSDG import train_ssdg
from cvsrffi.phase1_bicad_xr.trainer import BiCADXRBatch, BiCADXRTrainer


class _CountingIQModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Linear(128, 8)
        self.domain = nn.Linear(128, 8)
        self.classifier = nn.Linear(8, 3)
        self.forward_calls = 0
        self.forward_batch_sizes: list[int] = []

    def forward(
        self,
        x: torch.Tensor,
        y_tx: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        del y_tx, return_aux, domain_labels
        self.forward_calls += 1
        self.forward_batch_sizes.append(int(x.size(0)))
        flat = x.flatten(1)
        z_id = self.identity(flat)
        z_dom = self.domain(flat)
        return {
            "tx_logits": self.classifier(z_id),
            "z_id": z_id,
            "z_dom": z_dom,
            "shared_features": flat,
            "identity_features": z_id,
            "domain_features": z_dom,
        }


def _sat_args() -> SimpleNamespace:
    return SimpleNamespace(
        seed=1,
        sat_view_seed=1,
        sat_fs_hz=25e6,
        sat_fc_hz=2.462e9,
    )


def _run_real_concat_step(candidate_id: str, *, update: int):
    args = _sat_args()
    augmenter = train_ssdg._build_bicad_xr_concat_augmenter(args)
    base_model = _CountingIQModel()
    concat_model = train_ssdg._BiCADXRConcatForward(base_model)
    trainer = BiCADXRTrainer(concat_model, candidate_id, num_receivers=4)
    x = torch.ones(2, 2, 64, dtype=torch.float32)
    tx = torch.tensor([0, 1], dtype=torch.long)
    receiver = torch.tensor([0, 1], dtype=torch.long)
    day = torch.tensor([0, 1], dtype=torch.long)

    output, view = train_ssdg._bicad_xr_labeled_step(
        trainer,
        augmenter,
        x,
        tx,
        receiver,
        day,
        args=args,
        epoch=80,
        batch_idx=1,
        update=update,
        total_updates=5000,
    )
    return output, view, base_model, augmenter


def _strict_checkpoint_fixture():
    torch.manual_seed(11)
    source_model = _CountingIQModel()
    source_trainer = BiCADXRTrainer(
        train_ssdg._BiCADXRConcatForward(source_model),
        "F3",
        num_receivers=4,
    )
    with torch.no_grad():
        next(source_model.parameters()).fill_(0.125)
        assert source_trainer.factorized_heads is not None
        next(source_trainer.factorized_heads.parameters()).fill_(0.375)
    source_trainer._backward_control_state.update(
        {
            "firewall_applications": 7,
            "projection_applications": 2,
            "projection_triggers": 1,
            "last_update": 4504,
        }
    )
    source_trainer._swad_state = {
        "candidate_id": "F3",
        "source_loro": True,
        "updates": 3,
        "window_risks": [0.4, 0.3, 0.2],
        "average": {
            name: parameter.detach().clone()
            for name, parameter in source_trainer.named_optimizer_parameters()
        },
    }
    runtime = source_trainer.checkpoint_runtime(update=4504, total_updates=5000)
    checkpoint = {
        "model": deepcopy(source_model.state_dict()),
        "bicad_xr_runtime": runtime,
    }

    torch.manual_seed(22)
    target_model = _CountingIQModel()
    target_trainer = BiCADXRTrainer(
        train_ssdg._BiCADXRConcatForward(target_model),
        "F3",
        num_receivers=4,
    )
    return checkpoint, source_model, source_trainer, target_model, target_trainer


def parse(argv: list[str]):
    return train_ssdg.parse(argv)


def test_bicad_entry_forces_concat_leo_weak_contract() -> None:
    args = parse(["--phase1_method", "bicad_xr", "--candidate_id", "D5"])

    resolved = train_ssdg.resolve_bicad_protocol(args)

    assert resolved.use_concat_sat_channel_aug
    assert resolved.concat_sat_ce_only
    assert resolved.concat_sat_ce_weight == pytest.approx(0.68)
    assert resolved.concat_sat_start_epoch == 80
    assert resolved.sat_train_scenarios == (
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )


def test_bicad_from_scratch_resolves_wisig_sample_rate_before_model_build() -> None:
    args = parse(
        [
            "--phase1_method",
            "bicad_xr",
            "--candidate_id",
            "D5",
            "--from_scratch",
            "true",
        ]
    )
    args.input_len = 256
    args.num_domains = 14
    args.num_classes = 6

    train_ssdg._apply_bicad_xr_model_defaults(args)
    model = train_ssdg.build_baseline_model(args, torch.device("cpu"))

    assert args.sample_rate_hz == pytest.approx(25e6)
    assert isinstance(model, nn.Module)


def test_bicad_route_is_explicit_and_legacy_route_stays_lazy() -> None:
    bicad_args = parse(["--phase1_method", "bicad_xr", "--candidate_id", "D5"])
    legacy_args = parse([])

    assert train_ssdg.route_phase1_method(bicad_args) == "bicad_xr"
    assert train_ssdg.route_phase1_method(legacy_args) == "legacy"

    tree = ast.parse(inspect.getsource(train_ssdg))
    eager_bicad_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            eager_bicad_imports.extend(
                alias.name
                for alias in node.names
                if "phase1_bicad_xr" in alias.name
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "phase1_bicad_xr" in node.module:
                eager_bicad_imports.append(node.module)
    assert eager_bicad_imports == []


def test_bicad_real_leo_concat_uses_one_forward_and_runs_satellite_ce() -> None:
    output, view, model, augmenter = _run_real_concat_step("D5", update=1)
    _, stage = augmenter.stage_for_epoch(80)

    assert stage.scenarios == ("leo_low_elev_weak", "leo_rain_weak")
    assert stage.view_prob == pytest.approx(0.60)
    assert view["applied"] is True
    assert view["scenario"] in {"leo_low_elev_weak", "leo_rain_weak"}
    assert view["clean_batch_size"] == 2
    assert view["total_batch_size"] == 4
    assert not torch.equal(view["satellite_x"], torch.ones_like(view["satellite_x"]))
    assert model.forward_calls == 1
    assert model.forward_batch_sizes == [4]
    component = output.audit["components"]["satellite_tx_ce"]
    assert component["called"] is True
    assert component["skip_reason"] is None
    assert component["weighted"] == pytest.approx(0.68 * component["raw"])


def test_bicad_e3_pair_reuses_the_same_concat_forward() -> None:
    output, _, model, _ = _run_real_concat_step("E3", update=1752)

    assert output.audit["pair_called"] is True
    assert output.audit["pair_source"] == "concat_satellite"
    assert model.forward_calls == 1
    assert model.forward_batch_sizes == [4]


def test_bicad_checkpoint_strictly_restores_model_heads_controls_and_swad() -> None:
    checkpoint, source_model, source_trainer, target_model, target_trainer = (
        _strict_checkpoint_fixture()
    )

    restored_runtime = train_ssdg._load_bicad_xr_checkpoint_strict(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
    )

    assert restored_runtime["strict_reconstruction"] is True
    for name, value in source_model.state_dict().items():
        assert torch.equal(target_model.state_dict()[name], value)
    source_runtime = source_trainer.checkpoint_runtime(update=4504, total_updates=5000)
    target_runtime = target_trainer.checkpoint_runtime(update=4504, total_updates=5000)
    source_heads = source_runtime["training_state"]["factorized_heads"]
    target_heads = target_runtime["training_state"]["factorized_heads"]
    assert source_heads is not None and target_heads is not None
    for name, value in source_heads.items():
        assert torch.equal(target_heads[name], value)
    controls = target_runtime["training_state"]["backward_controls"]
    assert controls["firewall_applications"] == 7
    assert controls["projection_applications"] == 2
    assert controls["projection_triggers"] == 1
    assert target_runtime["swad"]["updates"] == 3
    assert target_runtime["swad"]["state"] is not None


def test_bicad_checkpoint_model_state_is_strict_and_failure_is_unmarked() -> None:
    checkpoint, _, _, target_model, target_trainer = _strict_checkpoint_fixture()
    checkpoint["model"].pop(next(iter(checkpoint["model"])))

    with pytest.raises(RuntimeError):
        train_ssdg._load_bicad_xr_checkpoint_strict(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
        )

    assert "strict_reconstruction" not in checkpoint["bicad_xr_runtime"]


def test_bicad_checkpoint_runtime_is_strict_and_failure_is_unmarked() -> None:
    checkpoint, _, _, target_model, target_trainer = _strict_checkpoint_fixture()
    checkpoint["bicad_xr_runtime"]["candidate_id"] = "D5"

    with pytest.raises(ValueError, match="runtime mismatch"):
        train_ssdg._load_bicad_xr_checkpoint_strict(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
        )

    assert "strict_reconstruction" not in checkpoint["bicad_xr_runtime"]


def test_real_bicad_training_entry_applies_registered_backward_controls() -> None:
    source = inspect.getsource(train_ssdg._train_bicad_xr)

    assert "trainer.apply_backward_controls(step_output)" in source
    assert "step_output.total.backward()" not in source


@pytest.mark.parametrize(
    ("source_receivers", "raw_receivers"),
    [
        ([3, 4, 6, 8], [3, 4, 6, 8]),
        ([1, 3, 4, 6], [1, 3, 4, 6]),
    ],
)
def test_formal_fold_raw_domain_indices_are_remapped_before_domain_ce(
    source_receivers: list[int],
    raw_receivers: list[int],
) -> None:
    receiver = train_ssdg._bicad_xr_local_domain_labels(
        torch.tensor(raw_receivers, dtype=torch.long),
        source_receivers,
        name="receiver",
    )
    day = train_ssdg._bicad_xr_local_domain_labels(
        torch.tensor([1, 2, 3, 1], dtype=torch.long),
        [1, 2, 3],
        name="day",
    )

    model = _CountingIQModel()
    trainer = BiCADXRTrainer(
        train_ssdg._BiCADXRConcatForward(model),
        "D5",
        num_receivers=4,
        num_days=3,
    )
    batch = BiCADXRBatch(
        x=torch.ones(4, 2, 64, dtype=torch.float32),
        tx=torch.tensor([0, 1, 2, 0], dtype=torch.long),
        receiver=receiver,
        day=day,
        channel=torch.zeros(4, dtype=torch.long),
        labeled_mask=torch.ones(4, dtype=torch.bool),
        epoch=1,
    )
    output = trainer.compute_step(batch, update=1, total_updates=5000, epoch=1)

    assert receiver.tolist() == [0, 1, 2, 3]
    assert day.tolist() == [0, 1, 2, 0]
    assert torch.isfinite(output.total)


def test_domain_remap_fails_closed_for_unregistered_source_index() -> None:
    with pytest.raises(ValueError, match="outside the frozen source set"):
        train_ssdg._bicad_xr_local_domain_labels(
            torch.tensor([3, 9], dtype=torch.long),
            [3, 4, 6, 8],
            name="receiver",
        )
