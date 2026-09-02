from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import train  # noqa: E402
from cvsrffi import phase1_fcr_losses as fcr_losses  # noqa: E402
from cvsrffi import phase1_fcr_transplant as fcr_transplant  # noqa: E402
from cvsrffi.phase1_fcr_schedule import FCRLambdaConfig, stage_for_epoch  # noqa: E402
from cvsrffi.phase1_fcr_transplant import TransplantLossOutput  # noqa: E402
from cvsrffi.phase1_fcr_types import (  # noqa: E402
    FCRConfig,
    FCRDecodeOutput,
    FCRFactorOutput,
    FCRLossOutput,
    FCRPairBatch,
)
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _resolved_row(row: str):
    values = {
        "phase1_method": "adv3b02_fcr",
        "use_fcr": True,
        "fcr_ablation_row": row,
        "epochs": 200,
        "train_mode": "centralized",
        "use_concat_sat_channel_aug": False,
        "use_meta_ssl_cvs": True,
        "ssl_labeled_ratio": 0.07,
        "ssl_unlabeled_ratio": 0.63,
        "ssl_val_ratio": 0.30,
        "lambda_fcr_self": 1.0,
        "lambda_fcr_swap": 1.0,
        "lambda_fcr_shared": 1.0,
        "lambda_fcr_cycle": 1.0,
        "lambda_fcr_eta": 1.0,
        "lambda_fcr_factor": 1.0,
        "lambda_fcr_need": 1.0,
        "lambda_fcr_phys": 1.0,
    }
    return train.resolve_fcr_training_options(argparse.Namespace(**values))


def _pair(*, labeled: bool = True, valid: bool = True) -> FCRPairBatch:
    batch = 4
    labels = torch.tensor([0, 0, 1, 1]) if labeled else torch.full((batch,), -1)
    label_mask = torch.full((batch,), labeled, dtype=torch.bool)
    if valid:
        nuisance_index = torch.tensor([0, 1, -1, -1])
        content_index = torch.tensor([1, 0, -1, -1])
        fingerprint_index = torch.tensor([3, 2, -1, -1])
    else:
        nuisance_index = content_index = fingerprint_index = torch.full((batch,), -1)
    return FCRPairBatch(
        clean_iq=torch.zeros(batch, 2, 4),
        leo_iq=torch.ones(batch, 2, 4),
        labels=labels,
        label_mask=label_mask,
        receiver_id=torch.zeros(batch, dtype=torch.long),
        day_id=torch.zeros(batch, dtype=torch.long),
        nuisance=torch.zeros(batch, 1),
        nuisance_valid=torch.zeros(batch, 1, dtype=torch.bool),
        physical_sample_id=tuple(f"sample:{index}" for index in range(batch)),
        pair_id=tuple(f"pair:{index}" for index in range(batch)),
        clean_crop_offset=torch.arange(batch),
        leo_crop_offset=torch.arange(batch),
        nuisance_pair_index=nuisance_index,
        content_pair_index=content_index,
        fingerprint_pair_index=fingerprint_index,
        pair_valid_mask={
            "nuisance": nuisance_index >= 0,
            "content": content_index >= 0,
            "fingerprint": fingerprint_index >= 0,
        },
    )


def _factors(offset: float = 0.0) -> FCRFactorOutput:
    base = torch.arange(4, dtype=torch.float32).reshape(4, 1) + float(offset)
    base.requires_grad_()
    return FCRFactorOutput(
        z_s=base.reshape(4, 1, 1),
        z_f_id=torch.cat((base, base + 0.5), dim=1),
        z_tx_state=base + 0.25,
        z_n_parts={"channel": base + 1.0, "receiver": base + 2.0},
        s_hat=base.reshape(4, 1, 1).to(torch.complex64),
        content_confidence=torch.ones(4, 1),
    )


def _cross(reference: torch.Tensor) -> FCRLossOutput:
    components = {
        "self": reference * 1.0,
        "swap": reference * 2.0,
        "shared": reference * 3.0,
        "latent_cycle": reference * 4.0,
        "eta": reference * 5.0,
        "factor": reference * 6.0,
        "anti_collapse": reference * 0.0,
    }
    return FCRLossOutput(total=sum(components.values()), components=components, metrics={})


def _transplant(reference: torch.Tensor) -> TransplantLossOutput:
    return TransplantLossOutput(
        active_pairs=1,
        total=reference * 7.0,
        components={"drop_f": reference * 7.0},
        metrics={"active_pairs": 1.0},
    )


def _configured(args) -> FCRLambdaConfig:
    weights = args.effective_fcr_lambdas
    return FCRLambdaConfig(
        self_reconstruction=weights["self"],
        swap=weights["swap"],
        shared=weights["shared"],
        latent_cycle=weights["latent_cycle"],
        eta=weights["eta"],
        factor=weights["factor"],
        transplant_necessity=weights["need"],
        physical_features=weights["phys"],
    )


def test_row_capabilities_gate_real_objective_terms_not_only_dry_run() -> None:
    reference = torch.tensor(1.0, requires_grad=True)
    cross = _cross(reference)
    targeted = _transplant(reference)
    basic = _transplant(reference * (2.0 / 7.0))
    three_axis = FCRLossOutput(
        total=reference * 8.0,
        components={"nuisance_axis": reference, "content_axis": reference * 3, "fingerprint_axis": reference * 4},
        metrics={},
    )
    physical = {"mrstft": reference * 0.0, "phase": reference * 0.0, "features": reference * 5.0}

    results = {}
    for row in ("R4", "R5", "R6", "R7", "R8"):
        args = _resolved_row(row)
        result = train.combine_fcr_training_losses(
            pair=_pair(),
            cross=cross,
            transplant=targeted,
            basic_necessity=basic,
            three_axis=three_axis,
            identity_per_sample=torch.zeros(4),
            physical_components=physical,
            stage=stage_for_epoch(100, optimizer_step=0, configured=_configured(args)),
            configured=_configured(args),
            capabilities=args.fcr_objective_capabilities,
        )
        results[row] = result

    assert results["R4"].components["transplant"].item() == 0.0
    assert results["R5"].components["transplant"].item() == pytest.approx(2.0)
    assert results["R6"].components["transplant"].item() == pytest.approx(9.0)
    assert results["R6"].components["phys"].item() == 0.0
    assert results["R7"].components["phys"].item() == pytest.approx(5.0)
    assert results["R7"].components["factor"].item() == 0.0
    assert results["R8"].components["factor"].item() == pytest.approx(14.0)

    unlabeled = train.combine_fcr_training_losses(
        pair=_pair(labeled=False),
        cross=cross,
        transplant=targeted,
        basic_necessity=basic,
        three_axis=three_axis,
        identity_per_sample=torch.zeros(4),
        physical_components=physical,
        stage=stage_for_epoch(100, optimizer_step=0, configured=_configured(_resolved_row("R8"))),
        configured=_configured(_resolved_row("R8")),
        capabilities=_resolved_row("R8").fcr_objective_capabilities,
    )
    assert unlabeled.components["transplant"].item() == 0.0


class _DropDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, z_s, z_f_id, z_tx_state, z_n_parts):
        del z_s, z_tx_state, z_n_parts
        value = self.bias + z_f_id.mean(dim=1)
        iq = value[:, None, None].expand(-1, 2, 4)
        return FCRDecodeOutput(
            mu_iq=iq,
            log_variance=torch.zeros_like(iq),
            delta_f=torch.zeros_like(iq),
        )


def test_r5_basic_drop_f_is_an_active_labeled_loss_without_strict_pair() -> None:
    helper = getattr(fcr_transplant, "compute_basic_drop_f_necessity_loss", None)
    assert callable(helper)
    decoder = _DropDecoder()
    factors = _factors()
    result = helper(
        source_factors=factors,
        decoder=decoder,
        fingerprint_residual_error=lambda iq: iq.square().mean(dim=(1, 2)),
        active_mask=torch.tensor([True, False, False, False]),
        necessity_margin=0.05,
        freeze_decoder=False,
    )
    assert result.active_pairs == 1
    assert result.components["drop_f"].item() > 0.0
    result.total.backward()
    assert decoder.bias.grad is not None and decoder.bias.grad.abs().item() > 0.0


def test_r8_three_axis_uses_strict_indices_masks_and_connected_zero_when_missing() -> None:
    helper = getattr(fcr_losses, "compute_three_axis_intervention_loss", None)
    assert callable(helper)
    clean = _factors()
    leo = _factors(offset=0.25)
    clean.z_s.retain_grad()
    result = helper(pair=_pair(), clean_factors=clean, leo_factors=leo, allow_fingerprint=True)
    assert result.components["nuisance_axis"].item() > 0.0
    assert result.components["content_axis"].item() > 0.0
    assert result.components["fingerprint_axis"].item() > 0.0
    assert result.metrics["nuisance_pairs"] == 2.0
    assert result.metrics["content_pairs"] == 2.0
    assert result.metrics["fingerprint_pairs"] == 2.0

    absent = helper(pair=_pair(valid=False), clean_factors=clean, leo_factors=leo, allow_fingerprint=True)
    assert absent.total.item() == 0.0
    assert absent.metrics["nuisance_status"] == "N/A"
    assert absent.metrics["content_status"] == "N/A"
    assert absent.metrics["fingerprint_status"] == "N/A"
    absent.total.backward()
    assert clean.z_s.grad is not None

    unlabeled = helper(pair=_pair(labeled=False), clean_factors=clean, leo_factors=leo, allow_fingerprint=False)
    assert unlabeled.components["fingerprint_axis"].item() == 0.0
    assert unlabeled.metrics["fingerprint_status"] == "N/A"


@pytest.mark.parametrize(
    ("row", "expected_mode"),
    [("R6", "control"), ("R7", "full_physics"), ("R8", "full_physics")],
)
def test_formal_inference_reports_the_row_bound_decoder_mode(row: str, expected_mode: str) -> None:
    args = _resolved_row(row)
    model = build_dual_model(
        num_classes=2,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
        fcr_config=FCRConfig(input_len=64, decoder_mode=args.fcr_decoder_mode),
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2, 64), return_aux=True)
    assert output["fcr_decoder_mode"] == expected_mode
    assert output["fcr_decode"].decoder_mode == expected_mode


def _objective_model_and_pair():
    model = build_dual_model(
        num_classes=2,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=True,
    ).train()
    pair = _pair()
    pair.clean_iq = torch.randn(4, 2, 64)
    pair.leo_iq = pair.clean_iq + 0.01 * torch.randn_like(pair.clean_iq)
    return model, pair


def test_reconstruction_stage_does_not_execute_future_necessity(monkeypatch) -> None:
    model, pair = _objective_model_and_pair()
    args = _resolved_row("R5")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("necessity executed before its scheduled stage")

    monkeypatch.setattr(train, "compute_basic_drop_f_necessity_loss", forbidden)
    result = train.compute_fcr_pair_objective(
        model=model,
        raw_model=model,
        pair=pair,
        role="L_s",
        stage=stage_for_epoch(1, configured=_configured(args)),
        configured=_configured(args),
        frozen_identity_classifier=nn.Identity(),
        capabilities=args.fcr_objective_capabilities,
    )
    assert torch.isfinite(result.total)


def test_reconstruction_stage_does_not_execute_future_physics_or_three_axis(monkeypatch) -> None:
    model, pair = _objective_model_and_pair()
    args = _resolved_row("R8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("future-stage objective executed during reconstruction")

    monkeypatch.setattr(train, "physical_feature_loss", forbidden)
    monkeypatch.setattr(train, "compute_three_axis_intervention_loss", forbidden)
    result = train.compute_fcr_pair_objective(
        model=model,
        raw_model=model,
        pair=pair,
        role="L_s",
        stage=stage_for_epoch(1, configured=_configured(args)),
        configured=_configured(args),
        frozen_identity_classifier=nn.Identity(),
        capabilities=args.fcr_objective_capabilities,
    )
    assert torch.isfinite(result.total)
