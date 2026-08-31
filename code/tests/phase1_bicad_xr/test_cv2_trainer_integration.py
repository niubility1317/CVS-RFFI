from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from cvsrffi.phase1_bicad_xr import trainer as trainer_module
from cvsrffi.phase1_bicad_xr.config import CV2_CANDIDATE_IDS, candidate_config
from cvsrffi.phase1_bicad_xr.tailguard import margin_rex_cvar_loss as real_margin_rex_cvar_loss
from cvsrffi.phase1_bicad_xr.trainer import BiCADXRBatch, BiCADXRTrainer


class _CountingFeatureModel(nn.Module):
    def __init__(self, feature_dim: int = 8, num_classes: int = 6) -> None:
        super().__init__()
        self.identity = nn.Linear(feature_dim, feature_dim)
        self.domain = nn.Linear(feature_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.forward_calls = 0

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
        z_id = self.identity(x)
        z_dom = self.domain(x)
        return {
            "tx_logits": self.classifier(z_id),
            "z_id": z_id,
            "z_dom": z_dom,
        }


def _strict_pair_batch() -> BiCADXRBatch:
    physical_count = 48
    labeled_count = 16
    receiver = torch.arange(physical_count, dtype=torch.long) % 4
    day = (torch.arange(physical_count, dtype=torch.long) // 4) % 3
    clean_channel = torch.zeros(physical_count, dtype=torch.long)
    satellite_channel = torch.ones(physical_count, dtype=torch.long)
    clean_pair = torch.zeros(physical_count, 8)
    satellite_pair = torch.zeros(physical_count, 8)
    clean_pair[:, 0] = 1.0
    satellite_pair[:, 1] = 1.0
    return BiCADXRBatch(
        x=torch.randn(2 * physical_count, 8),
        tx=None,
        receiver=torch.cat((receiver, receiver), dim=0),
        day=torch.cat((day, day), dim=0),
        channel=torch.cat((clean_channel, satellite_channel), dim=0),
        labeled_mask=torch.cat(
            (
                torch.ones(labeled_count, dtype=torch.bool),
                torch.zeros(2 * physical_count - labeled_count, dtype=torch.bool),
            )
        ),
        labeled_tx=torch.arange(labeled_count, dtype=torch.long) % 6,
        clean_z_id=clean_pair,
        satellite_z_id=satellite_pair,
        clean_logits=torch.randn(physical_count, 6),
        satellite_logits=torch.randn(physical_count, 6),
        pair_tx=torch.arange(physical_count, dtype=torch.long) % 6,
        epoch=1,
    )


def _ordinary_source_batch() -> BiCADXRBatch:
    count = 8
    return BiCADXRBatch(
        x=torch.randn(count, 8),
        tx=torch.arange(count, dtype=torch.long) % 6,
        receiver=torch.arange(count, dtype=torch.long) % 4,
        day=(torch.arange(count, dtype=torch.long) // 2) % 3,
        channel=torch.arange(count, dtype=torch.long) % 2,
        epoch=1,
    )


def _structured_source_batch() -> BiCADXRBatch:
    rows: list[tuple[int, int, int, int]] = []
    for tx in range(6):
        for receiver in range(4):
            for sample in range(2):
                rows.append((tx, receiver, (tx + receiver + sample) % 3, receiver))
    metadata = torch.tensor(rows, dtype=torch.long)
    return BiCADXRBatch(
        x=torch.randn(len(rows), 8),
        tx=metadata[:, 0],
        receiver=metadata[:, 1],
        day=metadata[:, 2],
        channel=metadata[:, 3],
        epoch=1,
    )


def _trainer_for(candidate_id: str) -> tuple[BiCADXRTrainer, _CountingFeatureModel]:
    model = _CountingFeatureModel()
    config = candidate_config(candidate_id)
    return (
        BiCADXRTrainer(
            model,
            config,
            num_receivers=4,
            num_days=3,
            num_channels=2 if config.strict_pair_concat else 4,
        ),
        model,
    )


@pytest.mark.parametrize("candidate_id", CV2_CANDIDATE_IDS)
def test_every_cv2_compute_step_audits_one_backbone_forward(candidate_id: str) -> None:
    trainer, model = _trainer_for(candidate_id)
    batch = _strict_pair_batch() if trainer.config.strict_pair_concat else _ordinary_source_batch()

    output = trainer.compute_step(batch, update=1, total_updates=5000, epoch=1)

    assert model.forward_calls == 1
    assert output.audit["backbone_forward_count"] == 1


@pytest.mark.parametrize(
    ("candidate_id", "expected_weight"),
    [("CV2-T1", 0.02), ("CV2-T3", 0.02), ("P3", 0.08)],
)
def test_cv2_pair_identity_uses_epsilon005_and_candidate_weight(
    candidate_id: str, expected_weight: float
) -> None:
    trainer, _ = _trainer_for(candidate_id)
    assert trainer.pair_projector is not None
    with torch.no_grad():
        trainer.pair_projector.weight.zero_()
        trainer.pair_projector.bias.zero_()
        trainer.pair_projector.weight[:8].copy_(torch.eye(8))

    output = trainer.compute_step(
        _strict_pair_batch(), update=1, total_updates=5000, epoch=1
    )
    component = output.audit["components"]["pair_identity_hinge"]

    assert component["called"] is True
    assert component["raw"] == pytest.approx(math.sqrt(2.0) - 0.05)
    assert component["weighted"] == pytest.approx(expected_weight * component["raw"])


@pytest.mark.parametrize("candidate_id", ["CV2-T2", "CV2-T3"])
def test_cv2_tailguard_calls_margin_rex_cvar_with_frozen_values(
    candidate_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def recording_loss(
        margins: torch.Tensor,
        groups: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append(
            {
                "margins": margins.detach().clone(),
                "groups": groups.detach().clone(),
                **kwargs,
            }
        )
        return real_margin_rex_cvar_loss(margins, groups, **kwargs)

    monkeypatch.setattr(
        trainer_module, "margin_rex_cvar_loss", recording_loss, raising=False
    )
    trainer, _ = _trainer_for(candidate_id)

    output = trainer.compute_step(
        _strict_pair_batch(), update=3504, total_updates=5000, epoch=1
    )

    assert len(calls) == 1
    assert calls[0]["lambda_rex"] == pytest.approx(0.02)
    assert calls[0]["lambda_cvar"] == pytest.approx(0.05)
    assert calls[0]["tail_fraction"] == pytest.approx(0.2)
    assert output.audit["hard_group_cap"] == pytest.approx(0.30)
    assert output.audit["margin_tail_mode"] == "margin_rex_cvar"
    assert output.audit["components"]["margin_tail"]["called"] is True


def test_historical_margin_tail_keeps_legacy_path() -> None:
    trainer, _ = _trainer_for("E4")

    output = trainer.compute_step(
        _structured_source_batch(), update=3504, total_updates=5000, epoch=1
    )

    assert output.audit["components"]["margin_tail"]["called"] is True
    assert output.audit["margin_tail_mode"] == "legacy_margin_tail"
    assert "hard_group_cap" not in output.audit


def test_cv2_outer_parameter_groups_are_disjoint_and_scoped() -> None:
    trainer, model = _trainer_for("CV2-D3")

    groups = trainer.adversarial_parameter_groups()
    encoder_ids = {id(parameter) for parameter in groups["encoder"]}
    discriminator_ids = {id(parameter) for parameter in groups["discriminator"]}
    all_trainable_ids = {
        id(parameter) for parameter in trainer.parameters() if parameter.requires_grad
    }
    names_by_id = {id(parameter): name for name, parameter in trainer.named_parameters()}

    assert model.forward_calls == 0
    assert encoder_ids.isdisjoint(discriminator_ids)
    assert encoder_ids | discriminator_ids == all_trainable_ids
    assert discriminator_ids
    assert all(
        names_by_id[parameter_id].startswith("factorized_heads.")
        and names_by_id[parameter_id].split(".", 2)[1]
        in {"id_receiver", "id_day", "id_channel", "dom_tx"}
        for parameter_id in discriminator_ids
    )
    assert not any(
        names_by_id[parameter_id].split(".", 2)[1]
        in {"dom_receiver", "dom_day", "dom_channel"}
        for parameter_id in discriminator_ids
    )
    assert groups["local_protection_allowlist"] == (
        "identity_last_block",
        "fusion",
        "projection",
    )


def test_cv2_backward_plan_keeps_two_adversaries_separate() -> None:
    trainer, _ = _trainer_for("CV2-D2")

    output = trainer.compute_step(
        _strict_pair_batch(), update=3504, total_updates=5000, epoch=1
    )

    plan = output.backward_plan
    assert plan.conditional_adversarial.requires_grad
    assert plan.zdom_tx_adversarial.requires_grad
    assert float(plan.adversarial.detach()) == pytest.approx(
        float((plan.conditional_adversarial + plan.zdom_tx_adversarial).detach())
    )


def test_cv2_local_protection_excludes_domain_and_adversarial_heads() -> None:
    trainer, _ = _trainer_for("CV2-D3")
    names_by_id = {
        id(parameter): name for name, parameter in trainer.named_parameters()
    }

    protected = trainer.cv2_local_protection_parameters()
    protected_names = {names_by_id[id(parameter)] for parameter in protected}

    assert protected_names
    assert all("factorized_heads" not in name for name in protected_names)
    assert all("model.domain" not in name for name in protected_names)
    assert all(
        any(token in name for token in ("identity", "id_backbone", "fuse", "proj"))
        for name in protected_names
    )
