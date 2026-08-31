from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
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


class _ForbiddenTxBatch(list):
    def __getitem__(self, index):
        if index == 1:
            raise AssertionError("PairBiCAD must not read U_s TX labels")
        return super().__getitem__(index)


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


def _run_pair_concat_step(candidate_id: str, *, update: int):
    args = _sat_args()
    augmenter = train_ssdg._build_bicad_xr_concat_augmenter(args)
    base_model = _CountingIQModel()
    concat_model = train_ssdg._BiCADXRConcatForward(base_model)
    trainer = BiCADXRTrainer(concat_model, candidate_id, num_receivers=4, num_days=3)
    x_l = torch.ones(16, 2, 64, dtype=torch.float32)
    x_u = torch.full((32, 2, 64), 2.0, dtype=torch.float32)
    tx = torch.arange(16, dtype=torch.long) % 3
    receiver_l = torch.arange(16, dtype=torch.long) % 4
    day_l = torch.arange(16, dtype=torch.long) % 3
    receiver_u = (torch.arange(32, dtype=torch.long) + 1) % 4
    day_u = (torch.arange(32, dtype=torch.long) + 1) % 3

    output, view = train_ssdg._bicad_xr_labeled_step(
        trainer,
        augmenter,
        x_l,
        tx,
        receiver_l,
        day_l,
        x_u,
        receiver_u,
        day_u,
        args=args,
        epoch=1,
        batch_idx=1,
        update=update,
        total_updates=4000,
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


def test_pairbicad_entry_preserves_registered_candidate_instead_of_legacy_e80() -> None:
    args = parse(["--phase1_method", "bicad_xr", "--candidate_id", "P0"])

    protocol = train_ssdg.resolve_bicad_protocol(args)
    train_ssdg._apply_bicad_xr_entry_protocol(args, protocol)

    assert protocol.strict_pair_concat is True
    assert args.concat_sat_start_epoch == 1
    assert args.sat_training_mode == "ce_only_plus_pair_selfsup"
    assert args.lambda_sat_cls_start == pytest.approx(0.5)
    assert args.lambda_sat_cls_end == pytest.approx(1.0)


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


def test_bicad_audit_jsonable_replaces_nonfinite_tensor_placeholders() -> None:
    payload = train_ssdg._bicad_xr_jsonable(
        {"xdc_donor_query_matrix": torch.tensor([[float("nan"), 0.5]])}
    )

    encoded = json.dumps(payload, allow_nan=False)

    assert json.loads(encoded) == {"xdc_donor_query_matrix": [[None, 0.5]]}


def test_bicad_epoch_loss_rejects_nonfinite_training_values() -> None:
    assert train_ssdg._bicad_xr_mean_epoch_loss([1.0, 2.0]) == pytest.approx(1.5)

    with pytest.raises(FloatingPointError, match="non-finite BiCAD-XR train loss"):
        train_ssdg._bicad_xr_mean_epoch_loss([1.0, float("nan")])


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


def test_pairbicad_real_concat_uses_16l32u_one_forward_and_preserves_u_unlabeled() -> None:
    output, view, model, _ = _run_pair_concat_step("P0", update=1)

    assert model.forward_calls == 1
    assert model.forward_batch_sizes == [96]
    assert view["physical_batch_size"] == 48
    assert view["labeled_count"] == 16
    assert view["unlabeled_count"] == 32
    assert view["network_batch_size"] == 96
    assert output.audit["components"]["satellite_tx_ce"]["called"]


def test_pairbicad_unlabeled_batch_never_reads_tx_slot() -> None:
    batch = _ForbiddenTxBatch(
        [
            torch.ones(2, 2, 64),
            object(),
            torch.tensor([0, 1]),
            {"rx_i": torch.tensor([2, 3]), "day_i": torch.tensor([0, 1])},
        ]
    )

    x_u, extra_u = train_ssdg._move_bicad_xr_unlabeled_batch(batch, torch.device("cpu"))

    assert x_u.shape[0] == 2
    assert extra_u[0].tolist() == [0, 1]


def test_pairbicad_entry_rejects_malformed_labeled_or_unlabeled_sizes_before_forward() -> None:
    args = _sat_args()
    augmenter = train_ssdg._build_bicad_xr_concat_augmenter(args)
    model = _CountingIQModel()
    trainer = BiCADXRTrainer(
        train_ssdg._BiCADXRConcatForward(model), "P0", num_receivers=4, num_days=3
    )

    with pytest.raises(ValueError, match="16 labeled and 32 unlabeled"):
        train_ssdg._bicad_xr_labeled_step(
            trainer,
            augmenter,
            torch.ones(15, 2, 64),
            torch.arange(15) % 3,
            torch.arange(15) % 4,
            torch.arange(15) % 3,
            torch.ones(32, 2, 64),
            torch.arange(32) % 4,
            torch.arange(32) % 3,
            args=args,
            epoch=1,
            batch_idx=1,
            update=1,
            total_updates=4000,
        )
    assert model.forward_calls == 0


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


def _required_bicad_helper(name: str):
    helper = getattr(train_ssdg, name, None)
    assert callable(helper), f"missing required BiCAD-XR helper: {name}"
    return helper


def test_pairbicad_budget_cli_defaults_and_explicit_9000_override() -> None:
    default = parse(["--phase1_method", "bicad_xr", "--candidate_id", "P0"])
    assert getattr(default, "bicad_optimizer_updates", None) == 0
    assert getattr(default, "bicad_loro_receiver", None) == -1
    assert getattr(default, "bicad_loro_eval_interval_updates", None) == 0
    assert getattr(default, "bicad_loro_min_updates", None) == 4000
    assert getattr(default, "bicad_loro_patience", None) == 5

    explicit = parse(
        [
            "--phase1_method",
            "bicad_xr",
            "--candidate_id",
            "P0",
            "--bicad_optimizer_updates",
            "9000",
        ]
    )
    assert explicit.bicad_optimizer_updates == 9000

    from cvsrffi.phase1_bicad_xr.config import candidate_config

    apply_budget = _required_bicad_helper("_bicad_xr_apply_optimizer_budget")
    assert apply_budget(candidate_config("P0"), 0).optimizer_updates == 4000
    assert apply_budget(candidate_config("P0"), 9000).optimizer_updates == 9000


@pytest.mark.parametrize("updates", [-500, 1, 4250, 9500])
def test_pairbicad_optimizer_budget_rejects_invalid_values(updates: int) -> None:
    validate = _required_bicad_helper("_validate_bicad_xr_optimizer_updates")
    with pytest.raises(ValueError, match="500.*4000.*9000"):
        validate(updates)


def test_pairbicad_source_loro_validation_rejects_overlap_and_foreign_receiver() -> None:
    validate = _required_bicad_helper("_validate_bicad_xr_loro_args")
    valid = SimpleNamespace(
        bicad_optimizer_updates=9000,
        bicad_loro_receiver=8,
        bicad_loro_eval_interval_updates=500,
        bicad_loro_min_updates=4000,
        bicad_loro_patience=5,
    )
    settings = validate(
        valid,
        source_receiver_indices=[1, 3, 4, 6],
        source_receiver_values=[1, 3, 4, 6],
        planned_updates=9000,
    )
    assert settings["enabled"] is True
    assert settings["heldout_receiver"] == 8

    overlap = SimpleNamespace(
        bicad_optimizer_updates=9000,
        bicad_loro_receiver=6,
        bicad_loro_eval_interval_updates=500,
        bicad_loro_min_updates=4000,
        bicad_loro_patience=5,
    )
    with pytest.raises(ValueError, match="heldout receiver.*source"):
        validate(
            overlap,
            source_receiver_indices=[1, 3, 4, 6],
            source_receiver_values=[1, 3, 4, 6],
            planned_updates=9000,
        )

    foreign = SimpleNamespace(
        bicad_optimizer_updates=9000,
        bicad_loro_receiver=2,
        bicad_loro_eval_interval_updates=500,
        bicad_loro_min_updates=4000,
        bicad_loro_patience=5,
    )
    with pytest.raises(ValueError, match="source universe"):
        validate(
            foreign,
            source_receiver_indices=[1, 3, 4, 6],
            source_receiver_values=[1, 3, 4, 6],
            planned_updates=9000,
        )

    for field, value, pattern in (
        ("bicad_loro_eval_interval_updates", 100, "interval.*250.*500"),
        ("bicad_loro_min_updates", 4500, "min_updates.*4000"),
        ("bicad_loro_patience", 3, "patience.*4.*6"),
    ):
        invalid = SimpleNamespace(
            bicad_optimizer_updates=9000,
            bicad_loro_receiver=8,
            bicad_loro_eval_interval_updates=500,
            bicad_loro_min_updates=4000,
            bicad_loro_patience=5,
        )
        setattr(invalid, field, value)
        with pytest.raises(ValueError, match=pattern):
            validate(
                invalid,
                source_receiver_indices=[1, 3, 4, 6],
                source_receiver_values=[1, 3, 4, 6],
                planned_updates=9000,
            )


def test_pairbicad_source_loro_accepts_real_receiver_labels_with_index_identity() -> None:
    validate = _required_bicad_helper("_validate_bicad_xr_loro_args")
    settings = validate(
        SimpleNamespace(
            bicad_optimizer_updates=9000,
            bicad_loro_receiver=1,
            bicad_loro_eval_interval_updates=500,
            bicad_loro_min_updates=4000,
            bicad_loro_patience=5,
        ),
        source_receiver_indices=[3, 4, 6, 8],
        source_receiver_values=["18-2", "18-3", "18-5", "18-7"],
        planned_updates=9000,
    )

    assert settings["enabled"] is True
    assert settings["heldout_receiver"] == 1


def test_pairbicad_source_loro_resolves_heldout_receiver_as_payload_index() -> None:
    resolve = _required_bicad_helper("_resolve_bicad_xr_loro_receiver_index")
    payload = {
        "rx_list": [
            "18-1",
            "18-2",
            "18-3",
            "18-4",
            "18-5",
            "18-6",
            "18-7",
            "18-8",
            "18-9",
        ]
    }

    assert resolve(payload, 1) == 1
    assert resolve(payload, 8) == 8


def test_pairbicad_loro_eval_clock_starts_at_4000_and_is_interval_bound() -> None:
    due = _required_bicad_helper("_bicad_xr_loro_eval_due")
    assert not due(3999, planned_updates=9000, min_updates=4000, interval=500)
    assert due(4000, planned_updates=9000, min_updates=4000, interval=500)
    assert not due(4001, planned_updates=9000, min_updates=4000, interval=500)
    assert due(4500, planned_updates=9000, min_updates=4000, interval=500)
    assert due(4250, planned_updates=9000, min_updates=4000, interval=250)
    assert due(6400, planned_updates=6400, min_updates=500, interval=500)
    assert not due(9001, planned_updates=9000, min_updates=4000, interval=500)


def test_pairbicad_source_loro_score_and_patience_use_strict_improvement() -> None:
    score = _required_bicad_helper("_bicad_xr_source_loro_primary_score")
    assert score(80.0, [50.0, 60.0, 70.0]) == pytest.approx(8000.0 / 130.0)
    assert score(0.0, [50.0, 60.0, 70.0]) == 0.0

    update_selection = _required_bicad_helper("_bicad_xr_loro_selection_step")
    state = {
        "best_update": 0,
        "best_score": None,
        "bad_count": 0,
        "stopped_early": False,
    }
    state = update_selection(state, update=4000, score=80.0, patience=5)
    assert state["best_update"] == 4000
    assert state["bad_count"] == 0
    for update in (4500, 5000, 5500, 6000, 6500):
        state = update_selection(state, update=update, score=80.0, patience=5)
    assert state["bad_count"] == 5
    assert state["stopped_early"] is True

    reset = update_selection(state, update=7000, score=80.0 + 2e-12, patience=5)
    assert reset["best_update"] == 7000
    assert reset["bad_count"] == 0


def test_pairbicad_source_loro_loader_reuses_payload_reference() -> None:
    build_loader = _required_bicad_helper("_build_bicad_xr_source_loro_loader")
    payload = {
        "rx_list": [
            "18-1",
            "18-2",
            "18-3",
            "18-4",
            "18-5",
            "18-6",
            "18-7",
            "18-8",
            "18-9",
        ],
        "capture_date_list": [1, 2, 3],
    }
    captured = {}

    class RecordingDataset:
        def __init__(self, ds, **kwargs):
            captured["payload"] = ds
            captured["kwargs"] = kwargs

        def __len__(self):
            return 1

    def fake_loader(dataset, *args, **kwargs):
        captured["dataset"] = dataset
        return dataset

    original_dataset = train_ssdg.WiSigCompactDataset
    original_loader = train_ssdg.make_loader
    train_ssdg.WiSigCompactDataset = RecordingDataset
    train_ssdg.make_loader = fake_loader
    try:
        args = SimpleNamespace(
            wisig_equalized=1,
            wisig_out_len=256,
            wisig_domain="rx_day",
            wisig_max_day123_per_combo=0,
            seed=392001,
            eval_batch_size=32,
            num_workers=0,
            prefetch_factor=2,
        )
        loader = build_loader(
            args,
            {
                "wisig_payload": payload,
                "source_day_indices": [0, 1, 2],
                "source_receiver_indices": [1, 3, 4, 6],
                "source_receiver_values": ["18-2", "18-4", "18-5", "18-7"],
            },
            torch.device("cpu"),
            8,
        )
    finally:
        train_ssdg.WiSigCompactDataset = original_dataset
        train_ssdg.make_loader = original_loader

    assert loader is captured["dataset"]
    assert captured["payload"] is payload
    assert captured["kwargs"]["rx_keep"] == [8]
    assert captured["kwargs"]["day_keep"] == [0, 1, 2]


def test_pairbicad_source_loro_eval_reports_class_floor_and_scenario_seed() -> None:
    evaluate = _required_bicad_helper("_evaluate_bicad_xr_source_loro")

    class FixedPredictionModel(nn.Module):
        def forward(self, x, **kwargs):
            del kwargs
            prediction = (x[:, 0] >= 1.0).long()
            logits = torch.full(
                (x.size(0), 2),
                -1.0,
                dtype=x.dtype,
                device=x.device,
            )
            logits.scatter_(1, prediction[:, None], 1.0)
            return {"tx_logits": logits}

    args = SimpleNamespace(seed=392001, eval_max_batches=0)
    batch = (
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        torch.tensor([0, 0, 1, 1]),
        torch.zeros(4, dtype=torch.long),
        {},
    )
    result = evaluate(
        FixedPredictionModel(),
        [batch],
        torch.device("cpu"),
        args,
        scenario="clean",
        scenario_index=0,
    )

    assert result["seed"] == 392001
    assert result["accuracy"] == pytest.approx(75.0)
    assert result["per_class_accuracy"] == {"0": 50.0, "1": 100.0}
    assert result["floor"] == pytest.approx(50.0)
