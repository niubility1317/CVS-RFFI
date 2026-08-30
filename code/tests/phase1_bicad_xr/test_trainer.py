from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from cvsrffi.phase1_bicad_xr.config import candidate_config
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
        }
    return BiCADXRBatch(
        x=values,
        tx=tx,
        receiver=receiver,
        day=day,
        channel=channel,
        **pair,
    )


def _trainer(candidate: str) -> BiCADXRTrainer:
    return BiCADXRTrainer(
        _FeatureModel(),
        candidate_config(candidate),
        num_receivers=4,
        num_days=3,
        num_channels=4,
    )


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

    assert not f3.compute_step(_batch(), 4500, 5000).audit["swad_updated"]
    assert f3.compute_step(_batch(), 4504, 5000).audit["swad_updated"]
    assert not f2.compute_step(_batch(), 4504, 5000).audit["swad_updated"]


def test_stage4_scales_domain_dann_and_shared_stem_lr() -> None:
    out = _trainer("D6").compute_step(_batch(), update=4504, total_updates=5000)

    assert out.audit["stage"] == "stage4"
    assert out.audit["domain_dann_scale"] == 0.6
    assert out.audit["shared_stem_lr_scale"] == 0.1


def test_pair_is_only_e3_and_reuses_concat_pair_outputs() -> None:
    e2 = _trainer("E2").compute_step(_batch(with_pair=True), 1752, 5000)
    e3 = _trainer("E3").compute_step(_batch(with_pair=True), 1752, 5000)

    assert not e2.audit["pair_called"]
    assert e3.audit["pair_called"]
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
