from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cvsrffi.phase1_bicad_xr import trainer as trainer_module
from cvsrffi.phase1_bicad_xr.config import BiCADXRStage, candidate_config
from cvsrffi.phase1_bicad_xr.trainer import BiCADXRBatch, BiCADXRTrainer


class _FeatureModel(nn.Module):
    def __init__(self, input_dim: int = 8, feature_dim: int = 8, num_classes: int = 6) -> None:
        super().__init__()
        self.identity = nn.Linear(input_dim, feature_dim)
        self.domain = nn.Linear(input_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        y_tx: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        del y_tx, return_aux, domain_labels
        z_id = self.identity(x)
        z_dom = self.domain(x)
        return {
            "tx_logits": self.classifier(z_id),
            "z_id": z_id,
            "z_dom": z_dom,
            "shared_features": x,
            "identity_features": z_id,
            "domain_features": z_dom,
        }


class _RecordingFeatureModel(_FeatureModel):
    def __init__(self) -> None:
        super().__init__()
        self.received_tx: torch.Tensor | None = torch.empty(0, dtype=torch.long)

    def forward(
        self,
        x: torch.Tensor,
        y_tx: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.received_tx = y_tx
        return super().forward(x, y_tx=y_tx, **kwargs)


class _CountingFeatureModel(_FeatureModel):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.forward_batch_sizes: list[int] = []

    def forward(
        self,
        x: torch.Tensor,
        y_tx: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        self.forward_batch_sizes.append(int(x.size(0)))
        return super().forward(x, y_tx=y_tx, **kwargs)


def _batch(*, with_pair: bool = False) -> BiCADXRBatch:
    count = 48
    tx = torch.arange(count) % 6
    receiver = torch.arange(count) % 4
    day = (torch.arange(count) // 6) % 3
    channel = torch.arange(count) % 4
    values = torch.randn(count, 8)
    pair = {}
    if with_pair:
        pair = {
            "clean_z_id": torch.randn(12, 8),
            "satellite_z_id": torch.randn(12, 8),
            "clean_logits": torch.randn(12, 6),
            "satellite_logits": torch.randn(12, 6),
            "pair_tx": torch.arange(12) % 6,
        }
    return BiCADXRBatch(
        x=values,
        tx=tx,
        receiver=receiver,
        day=day,
        channel=channel,
        **pair,
    )


def _structured_batch(
    *,
    extra_unlabeled: int = 0,
    sparse_cell: tuple[int, int] | None = None,
    shuffle: bool = False,
) -> BiCADXRBatch:
    rows: list[tuple[int, int, int, int]] = []
    physical: list[int] = []
    next_physical = 1000
    for tx in range(6):
        for receiver in range(4):
            samples = 1 if sparse_cell == (tx, receiver) else 2
            for sample in range(samples):
                rows.append((tx, receiver, (tx + receiver + sample) % 3, receiver))
                physical.append(next_physical)
                next_physical += 1
    labeled_count = len(rows)
    for index in range(extra_unlabeled):
        rows.append((index % 6, index % 4, index % 3, index % 4))
        physical.append(next_physical)
        next_physical += 1
    metadata = torch.tensor(rows, dtype=torch.long)
    x = torch.arange(len(rows) * 8, dtype=torch.float32).reshape(len(rows), 8) / 100.0
    if extra_unlabeled:
        x[labeled_count:] += 1000.0
    mask = torch.zeros(len(rows), dtype=torch.bool)
    mask[:labeled_count] = True
    physical_tensor = torch.tensor(physical, dtype=torch.long)
    if shuffle:
        generator = torch.Generator().manual_seed(20260831)
        order = torch.randperm(len(rows), generator=generator)
        metadata = metadata[order]
        x = x[order]
        mask = mask[order]
        physical_tensor = physical_tensor[order]
    return BiCADXRBatch(
        x=x,
        tx=metadata[:, 0],
        receiver=metadata[:, 1],
        day=metadata[:, 2],
        channel=metadata[:, 3],
        physical_indices=physical_tensor,
        labeled_mask=mask,
    )


def _wide_global_label_batch() -> BiCADXRBatch:
    global_tx = (3, 11, 27, 48, 63, 79)
    global_receiver = (7, 19, 42, 88)
    rows: list[tuple[int, int, int, int]] = []
    for tx_index, tx in enumerate(global_tx):
        for receiver_index, receiver in enumerate(global_receiver):
            samples = 1 if (tx, receiver) == (79, 88) else 2
            for sample in range(samples):
                rows.append(
                    (
                        tx,
                        receiver,
                        (tx_index + receiver_index + sample) % 3,
                        receiver_index,
                    )
                )
    metadata = torch.tensor(rows, dtype=torch.long)
    generator = torch.Generator().manual_seed(20260831)
    order = torch.randperm(len(rows), generator=generator)
    metadata = metadata[order]
    return BiCADXRBatch(
        x=torch.randn(len(rows), 8, generator=generator),
        tx=metadata[:, 0],
        receiver=metadata[:, 1],
        day=metadata[:, 2],
        channel=metadata[:, 3],
        physical_indices=10_000 + torch.arange(len(rows)),
        labeled_mask=torch.ones(len(rows), dtype=torch.bool),
    )


class _NoPublicClassifierModel(nn.Module):
    feature_dim = 8
    num_classes = 6

    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Linear(8, 8)
        self.domain = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        z_id = self.identity(x)
        return {
            "tx_logits": z_id[:, :6],
            "z_id": z_id,
            "z_dom": self.domain(x),
        }


class _StemHolder(nn.Module):
    def __init__(self, stem: nn.Module) -> None:
        super().__init__()
        self.sinc = stem


class _SharedStemModel(nn.Module):
    feature_dim = 8
    num_classes = 6

    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(8, 8, bias=False)
        self.id_backbone = _StemHolder(shared)
        self.dom_backbone = _StemHolder(shared)
        self.id_post = nn.Linear(8, 8, bias=False)
        self.dom_post = nn.Linear(8, 8, bias=False)
        self.classifier = nn.Linear(8, 6, bias=False)

    def forward(self, x: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        shared = self.id_backbone.sinc(x)
        z_id = self.id_post(shared)
        z_dom = self.dom_post(shared)
        return {
            "tx_logits": self.classifier(z_id.detach()),
            "z_id": z_id,
            "z_dom": z_dom,
        }


def _trainer(candidate: str) -> BiCADXRTrainer:
    return BiCADXRTrainer(
        _FeatureModel(),
        candidate_config(candidate),
        num_receivers=4,
        num_days=3,
        num_channels=4,
    )


def _pairbicad_batch() -> BiCADXRBatch:
    physical_count = 48
    labeled_count = 16
    clean_receiver = torch.arange(physical_count) % 4
    clean_day = (torch.arange(physical_count) // 4) % 3
    clean_channel = torch.zeros(physical_count, dtype=torch.long)
    return BiCADXRBatch(
        x=torch.randn(2 * physical_count, 8),
        tx=None,
        labeled_tx=torch.arange(labeled_count) % 6,
        receiver=torch.cat((clean_receiver, clean_receiver), dim=0),
        day=torch.cat((clean_day, clean_day), dim=0),
        channel=torch.cat(
            (clean_channel, torch.ones(physical_count, dtype=torch.long)), dim=0
        ),
        labeled_mask=torch.cat(
            (
                torch.ones(labeled_count, dtype=torch.bool),
                torch.zeros(2 * physical_count - labeled_count, dtype=torch.bool),
            ),
            dim=0,
        ),
        pair_tx=torch.arange(labeled_count) % 6,
        epoch=1,
    )


def _pairbicad_trainer(candidate: str):
    from SSDG import train_ssdg

    model = _CountingFeatureModel()
    concat_model = train_ssdg._BiCADXRConcatForward(model)
    concat_model.set_concat_clean_count(48)
    return (
        BiCADXRTrainer(
            concat_model,
            candidate_config(candidate),
            num_receivers=4,
            num_days=3,
            num_channels=2,
        ),
        model,
    )


def test_pairbicad_p0_uses_16l32u_and_one_forward_with_scheduled_satellite_ce() -> None:
    trainer, model = _pairbicad_trainer("P0")

    output = trainer.compute_step(
        _pairbicad_batch(), update=1, total_updates=4000, epoch=1
    )

    runtime = output.audit["pairbicad_runtime"]
    assert model.forward_calls == 1
    assert model.forward_batch_sizes == [96]
    assert runtime["one_forward_count"] == 1
    assert runtime["effective_counts"] == {
        "physical": 48,
        "network": 96,
        "labeled": 16,
        "unlabeled": 32,
        "pair": 48,
    }
    satellite = output.audit["components"]["satellite_tx_ce"]
    assert satellite["called"] is True
    assert satellite["weighted"] == pytest.approx(0.5 * satellite["raw"])
    assert not output.audit["components"]["pair_identity_hinge"]["called"]
    assert not output.audit["components"]["pair_vicreg"]["called"]
    assert not output.audit["components"]["pair_delta_identity_adversary"]["called"]


def test_pairbicad_p3_uses_unlabeled_pair_self_supervision_without_u_tx() -> None:
    trainer, model = _pairbicad_trainer("P3")

    output = trainer.compute_step(
        _pairbicad_batch(), update=1, total_updates=4000, epoch=1
    )

    assert model.forward_calls == 1
    assert output.audit["components"]["tx_ce"]["effective_count"] == 16
    assert output.audit["components"]["satellite_tx_ce"]["effective_count"] == 16
    assert output.audit["components"]["pair_identity_hinge"]["called"]
    assert output.audit["components"]["pair_prediction_js"]["called"]
    assert output.audit["components"]["pair_vicreg"]["called"]
    assert not output.audit["components"]["pair_delta_identity_adversary"]["called"]


def test_pairbicad_p4_records_delta_objectives_and_dynamic_adversarial_dose() -> None:
    trainer, _ = _pairbicad_trainer("P4")

    output = trainer.compute_step(
        _pairbicad_batch(), update=1, total_updates=4000, epoch=1
    )

    runtime = output.audit["pairbicad_runtime"]
    assert runtime["gradient"]["rho_adv"] == pytest.approx(0.2)
    for name in (
        "pair_delta_identity_adversary",
        "pair_delta_channel_prediction",
        "pair_delta_stability",
        "pair_delta_channel_equivariance",
    ):
        assert output.audit["components"][name]["called"]
    assert output.audit["components"]["pair_prediction_js"]["called"]


def test_legacy_runtime_defaults_known_pair_fields_but_keeps_strict_comparison() -> None:
    source = _trainer("D5")
    runtime = deepcopy(source.checkpoint_runtime(504, 5000))
    pair_fields = (
        "strict_pair_concat",
        "pair_identity",
        "pair_vicreg",
        "pair_delta",
        "dynamic_adversarial_dose",
        "satellite_supervision_mode",
        "pair_projector_dim",
        "factor_interaction_dim",
        "lambda_sat_cls_start",
        "lambda_sat_cls_end",
    )
    for field in pair_fields:
        runtime["candidate_config"].pop(field)

    restored = _trainer("D5")
    restored.load_checkpoint_runtime(runtime, strict=True)

    mismatched = deepcopy(runtime)
    mismatched["candidate_config"]["pair_identity"] = True
    with pytest.raises(ValueError, match="runtime mismatch"):
        _trainer("D5").load_checkpoint_runtime(mismatched, strict=True)

    unknown = deepcopy(runtime)
    unknown["candidate_config"]["unexpected_pair_field"] = 1
    with pytest.raises(ValueError, match="runtime mismatch"):
        _trainer("D5").load_checkpoint_runtime(unknown, strict=True)


def test_pairbicad_checkpoint_restores_projectors_and_jsonable_runtime() -> None:
    from SSDG import train_ssdg

    source, _source_model = _pairbicad_trainer("P3")
    output = source.compute_step(
        _pairbicad_batch(), update=1, total_updates=4000, epoch=1
    )
    runtime = deepcopy(output.checkpoint_runtime)
    assert runtime["training_state"]["factorized_projector"] is not None
    assert runtime["training_state"]["pair_projector"] is not None
    assert output.audit["pairbicad_runtime"]["finite"]["loss"] is True

    target_model = _CountingFeatureModel()
    target = train_ssdg._BiCADXRConcatForward(target_model)
    target.set_concat_clean_count(48)
    restored = BiCADXRTrainer(
        target,
        candidate_config("P3"),
        num_receivers=4,
        num_days=3,
        num_channels=2,
    )
    restored.load_checkpoint_runtime(runtime, strict=True)

    assert restored.factorized_projector is not None
    assert restored.pair_projector is not None
    for name, value in source.factorized_projector.state_dict().items():
        torch.testing.assert_close(restored.factorized_projector.state_dict()[name], value)
    for name, value in source.pair_projector.state_dict().items():
        torch.testing.assert_close(restored.pair_projector.state_dict()[name], value)
    encoded = json.dumps(train_ssdg._bicad_xr_jsonable(runtime), allow_nan=False)
    assert "pairbicad_runtime" in encoded


def test_stage0_has_no_grl_xdc_tail_or_tangent() -> None:
    out = _trainer("ADV3B02-BiCAD-XDC-V1").compute_step(
        _batch(), update=1, total_updates=5000
    )

    assert out.audit["stage"] == "stage0"
    assert out.audit["grl_identity"] == 0.0
    assert not out.audit["xdc_called"]
    assert not out.audit["tail_called"]
    assert not out.audit["tangent_called"]


def test_xdc_runs_every_four_steps_only_after_stage2() -> None:
    trainer = _trainer("ADV3B02-BiCAD-XDC-V1")

    before = trainer.compute_step(_batch(), update=1748, total_updates=5000)
    after = trainer.compute_step(_batch(), update=1752, total_updates=5000)

    assert before.audit["stage"] == "stage1"
    assert not before.audit["xdc_called"]
    assert after.audit["stage"] == "stage2"
    assert after.audit["xdc_called"]
    assert after.audit["update"] % 4 == 0


def test_swad_updates_only_for_f3_in_stage4() -> None:
    f3 = _trainer("F3")
    f2 = _trainer("F2")

    evidence = {"source_loro_risk": 0.2, "source_loro_window": True}
    assert not f3.compute_step(_batch(), 4500, 5000, **evidence).audit["swad_updated"]
    assert f3.compute_step(_batch(), 4504, 5000, **evidence).audit["swad_updated"]
    assert not f2.compute_step(_batch(), 4504, 5000, **evidence).audit["swad_updated"]


def test_stage4_scales_domain_dann_and_shared_stem_lr() -> None:
    out = _trainer("D6").compute_step(_batch(), update=4504, total_updates=5000)

    assert out.audit["stage"] == "stage4"
    assert out.audit["domain_dann_scale"] == 0.6
    assert out.audit["shared_stem_lr_scale"] == 0.1


def test_pair_is_only_e3_and_reuses_concat_pair_outputs() -> None:
    e2 = _trainer("E2").compute_step(_batch(with_pair=True), 1752, 5000)
    e3 = _trainer("E3").compute_step(_batch(with_pair=True), 1752, 5000)
    e4 = _trainer("E4").compute_step(_batch(with_pair=True), 1752, 5000)

    assert not e2.audit["pair_called"]
    assert e3.audit["pair_called"]
    assert not e4.audit["pair_called"]
    assert e3.audit["pair_source"] == "concat_satellite"
    assert e3.audit["extra_forward_count"] == 0


def test_v1_disables_pair_kd_d6_tangent_and_swad() -> None:
    out = _trainer("ADV3B02-BiCAD-XDC-V1").compute_step(
        _batch(with_pair=True), 4504, 5000
    )

    assert not out.audit["pair_called"]
    assert not out.audit["xdc_kd_called"]
    assert not out.audit["task_protected_gradient_called"]
    assert not out.audit["tangent_called"]
    assert not out.audit["swad_updated"]


def test_each_loss_audit_has_raw_weighted_call_effective_count_and_skip_reason() -> None:
    out = _trainer("ADV3B02-BiCAD-XDC-V1").compute_step(
        _batch(), update=1, total_updates=5000
    )

    assert out.checkpoint_runtime["candidate_id"] == "ADV3B02-BiCAD-XDC-V1"
    assert out.checkpoint_runtime["stage"] == "stage0"
    assert out.checkpoint_runtime["optimizer_update"] == 1
    for component in out.audit["components"].values():
        assert {"raw", "weighted", "called", "effective_count", "skip_reason"} <= set(component)


def test_runtime_accepts_frozen_config_without_mutating_candidate() -> None:
    cfg = replace(candidate_config("D5"), optimizer_updates=5000)
    trainer = BiCADXRTrainer(_FeatureModel(), cfg, num_receivers=4, num_days=3, num_channels=4)

    output = trainer.compute_step(_batch(), update=5000, total_updates=5000)

    assert output.checkpoint_runtime["phase1_method"] == "bicad_xr"
    assert output.checkpoint_runtime["candidate_id"] == "D5"


def test_unlabeled_source_rows_never_enter_tx_conditioned_losses() -> None:
    base = _trainer("ADV3B02-BiCAD-XDC-V1")
    extended = _trainer("ADV3B02-BiCAD-XDC-V1")
    extended.load_state_dict(base.state_dict())

    base_output = base.compute_step(_structured_batch(), update=3504, total_updates=5000)
    extended_output = extended.compute_step(
        _structured_batch(extra_unlabeled=12), update=3504, total_updates=5000
    )

    for component in (
        "tx_ce",
        "conditional_dann",
        "zdom_tx_adversary",
        "conditional_xcov",
        "xdc_cross_entropy",
        "margin_tail",
    ):
        assert extended_output.audit["components"][component]["raw"] == pytest.approx(
            base_output.audit["components"][component]["raw"], rel=1e-6, abs=1e-6
        )


def test_mixed_unlabeled_source_batch_does_not_pass_tx_into_model_conditioning() -> None:
    model = _RecordingFeatureModel()
    trainer = BiCADXRTrainer(
        model,
        candidate_config("D5"),
        num_receivers=4,
        num_days=3,
        num_channels=4,
    )

    trainer.compute_step(
        _structured_batch(extra_unlabeled=1), update=504, total_updates=5000
    )

    assert model.received_tx is None


def test_sparse_shuffled_xdc_episode_preserves_batch_label_alignment() -> None:
    batch = _structured_batch(sparse_cell=(5, 3), shuffle=True)
    output = _trainer("ADV3B02-BiCAD-XDC-V1").compute_step(
        batch, update=3504, total_updates=5000
    )

    indices = output.audit["xdc_episode_batch_indices"]
    episode_tx = output.audit["xdc_episode_tx"]
    episode_receiver = output.audit["xdc_episode_receiver"]
    assert len(indices) == 46
    assert all(bool(batch.labeled_mask[index]) for index in indices)
    assert episode_tx == [int(batch.tx[index]) for index in indices]
    assert episode_receiver == [int(batch.receiver[index]) for index in indices]
    assert (5, 3) not in set(zip(episode_tx, episode_receiver))
    assert output.audit["xdc_tail_query_tx"] == episode_tx


def test_xdc_remaps_wide_global_labels_to_local_episode_and_back() -> None:
    batch = _wide_global_label_batch()
    trainer = BiCADXRTrainer(
        _FeatureModel(num_classes=80),
        candidate_config("E1"),
        num_receivers=100,
        num_days=3,
        num_channels=4,
    )

    output = trainer.compute_step(batch, update=3504, total_updates=5000)

    indices = output.audit["xdc_episode_batch_indices"]
    episode_tx = output.audit["xdc_episode_tx"]
    episode_receiver = output.audit["xdc_episode_receiver"]
    assert output.logits.shape[1] == 80
    assert output.audit["xdc_called"]
    assert len(indices) == 46
    assert max(indices) < len(batch.x)
    assert episode_tx == [int(batch.tx[index]) for index in indices]
    assert episode_receiver == [int(batch.receiver[index]) for index in indices]
    assert 79 in episode_tx
    assert (79, 88) not in set(zip(episode_tx, episode_receiver))
    assert set(output.audit["xdc_episode_local_tx"]) == set(range(6))
    assert set(output.audit["xdc_episode_local_receiver"]) == set(range(4))
    assert output.audit["xdc_tail_query_tx"] == episode_tx


def test_satellite_ce_starts_at_epoch80_and_pair_consistency_is_e3_only() -> None:
    batch = _batch(with_pair=True)
    d5 = _trainer("D5")

    before = d5.compute_step(batch, 1752, 5000, epoch=79)
    active = d5.compute_step(batch, 1752, 5000, epoch=80)
    e3 = _trainer("E3").compute_step(batch, 1752, 5000, epoch=80)

    assert not before.audit["components"]["satellite_tx_ce"]["called"]
    sat_component = active.audit["components"]["satellite_tx_ce"]
    assert sat_component["called"]
    assert sat_component["weighted"] == pytest.approx(0.68 * sat_component["raw"])
    assert not active.audit["pair_called"]
    assert e3.audit["pair_called"]
    assert active.checkpoint_runtime["protocol"]["lambda_sat_cons"] == 0.0


def test_tangent_fails_closed_without_public_tx_classifier() -> None:
    trainer = BiCADXRTrainer(
        _NoPublicClassifierModel(),
        candidate_config("F1"),
        num_receivers=4,
        num_days=3,
        num_channels=4,
    )

    with pytest.raises(RuntimeError, match="public TX classifier"):
        trainer.compute_step(_batch(), update=3504, total_updates=5000)


def test_backward_controls_scale_only_domain_gradient_into_shared_stem() -> None:
    plain_config = replace(
        candidate_config("D1"),
        candidate_id="D4",
        gradient_firewall=False,
    )
    firewall_config = replace(
        plain_config,
        candidate_id="D5",
        gradient_firewall=True,
    )
    torch.manual_seed(20260831)
    plain = BiCADXRTrainer(_SharedStemModel(), plain_config, num_receivers=4)
    controlled = BiCADXRTrainer(_SharedStemModel(), firewall_config, num_receivers=4)
    controlled.load_state_dict(plain.state_dict())
    batch = _batch()

    plain_output = plain.compute_step(batch, update=504, total_updates=5000)
    plain_output.total.backward()
    controlled_output = controlled.compute_step(batch, update=504, total_updates=5000)
    backward_audit = controlled.apply_backward_controls(controlled_output)

    plain_stem_grad = plain.shared_stem_parameters()[0].grad
    controlled_stem_grad = controlled.shared_stem_parameters()[0].grad
    assert plain_stem_grad is not None and controlled_stem_grad is not None
    torch.testing.assert_close(controlled_stem_grad, plain_stem_grad * 0.05)
    torch.testing.assert_close(
        controlled.factorized_heads.dom_receiver.weight.grad,
        plain.factorized_heads.dom_receiver.weight.grad,
    )
    assert backward_audit["gradient_firewall_applied"]


def test_d6_projection_changes_explicit_shared_stem_gradient_every_four_steps() -> None:
    trainer = BiCADXRTrainer(_SharedStemModel(), candidate_config("D6"), num_receivers=4)
    scheduled = trainer.compute_step(_batch(), update=504, total_updates=5000)
    unscheduled = trainer.compute_step(_batch(), update=503, total_updates=5000)
    assert scheduled.backward_plan.projection_enabled
    assert not unscheduled.backward_plan.projection_enabled

    plan_class = trainer_module.BiCADXRBackwardPlan
    parameter = trainer.shared_stem_parameters()[0]
    task_loss = parameter.sum()
    adversarial_loss = -parameter.sum()
    zero = parameter.sum() * 0.0
    manual_plan = plan_class(
        total=task_loss + adversarial_loss,
        domain_forward=zero,
        adversarial=adversarial_loss,
        task_reference=task_loss,
        stage=BiCADXRStage.stage1,
        update=504,
        firewall_enabled=True,
        projection_enabled=True,
    )
    manual_output = SimpleNamespace(
        total=manual_plan.total,
        backward_plan=manual_plan,
        audit={"components": {}},
    )
    trainer.zero_grad(set_to_none=True)
    audit = trainer.apply_backward_controls(manual_output)

    assert parameter.grad is not None
    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
    assert audit["task_projection_applied"]
    assert audit["projection_triggered"]


def test_f3_swad_requires_explicit_stage4_source_loro_window_and_restores() -> None:
    no_labels = replace(_batch(), tx=None)
    f3 = _trainer("F3")

    no_evidence = f3.compute_step(no_labels, 4504, 5000)
    assert not no_evidence.audit["swad_updated"]
    assert f3.swad_state is None

    accepted = f3.compute_step(
        no_labels,
        4504,
        5000,
        source_loro_risk=0.25,
        source_loro_window=True,
    )
    assert accepted.audit["swad_updated"]
    assert f3.swad_state["updates"] == 1
    runtime = f3.checkpoint_runtime(4504, 5000)

    restored = _trainer("F3")
    restored.load_checkpoint_runtime(runtime, strict=True)
    assert restored.swad_state["updates"] == 1
    assert restored.swad_state["window_risks"] == [pytest.approx(0.25)]
    assert set(restored.swad_state["average"]) == set(f3.swad_state["average"])

    f2 = _trainer("F2")
    f2.compute_step(
        no_labels,
        4504,
        5000,
        source_loro_risk=0.1,
        source_loro_window=True,
    )
    assert f2.swad_state is None


def test_training_heads_are_optimizer_visible_and_strictly_restored() -> None:
    trainer = _trainer("D5")
    assert trainer.factorized_heads is not None
    head_parameter = next(trainer.factorized_heads.parameters())
    assert id(head_parameter) in {id(parameter) for parameter in trainer.optimizer_parameters()}

    runtime = trainer.checkpoint_runtime(504, 5000)
    saved = runtime["training_state"]["factorized_heads"]
    with torch.no_grad():
        head_parameter.add_(10.0)
    trainer.load_checkpoint_runtime(runtime, strict=True)

    torch.testing.assert_close(
        trainer.factorized_heads.state_dict()[next(iter(saved))],
        saved[next(iter(saved))],
    )
    assert runtime["training_state"]["backward_controls"]["firewall_scale"] == 0.05
