from __future__ import annotations

import copy

import pytest
import torch

from model import CVSincNet
from model_dual_cvsincnet import build_dual_model


def _small_backbone(variant: str = "nmfdu_v1") -> CVSincNet:
    return CVSincNet(
        num_classes=3,
        input_len=96,
        sinc_out=8,
        sinc_kernel=15,
        time_bottleneck=8,
        emb_dim=16,
        drop=0.0,
        freq_bands=16,
        pa_memory_depth=1,
        pa_orders=(1, 3, 5),
        time_ch1=8,
        time_ch2=8,
        time_ch3=8,
        dac_ch=8,
        freq_ch1=8,
        freq_ch2=8,
        freq_ch3=8,
        pa_ch1=8,
        pa_ch2=8,
        pa_ch3=8,
        physical_gate_variant=variant,
    )


def test_enabled_identity_path_returns_five_branch_null_aware_diagnostics() -> None:
    torch.manual_seed(71)
    model = _small_backbone().eval()
    x = torch.randn(2, 2, 96)
    with torch.no_grad():
        output = model(
            x,
            return_aux=True,
            return_physical_gate_diag=True,
        )

    assert output["logits"].shape == (2, 3)
    assert output["feat_cls"].shape == (2, 16)
    assert output["feat_joint"].shape == (2, 16)
    assert tuple(output["nmfdu_branch_embeddings"]) == (
        "raw",
        "hom",
        "phase",
        "pa",
        "hos",
    )
    diag = output["physical_gate_diag"]
    sample = diag["per_sample"]
    assert sample["weights"].shape == (2, 5)
    assert sample["null_weight"].shape == (2,)
    assert sample["I"].shape == (2, 5)
    assert sample["D"].shape == (2, 5)
    assert sample["S"].shape == (2, 5)
    assert sample["U"].shape == (2, 5)
    torch.testing.assert_close(
        sample["weights"].sum(dim=-1) + sample["null_weight"],
        torch.ones(2),
    )
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["feat_cls"]).all()


def test_support_discriminability_updates_only_under_explicit_training_flag() -> None:
    torch.manual_seed(72)
    model = _small_backbone().train()
    x = torch.randn(6, 2, 96)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    state = model.nmfdu_gate.evidence_state
    assert state.update_count.item() == 0

    model(x, y=labels, return_aux=True)
    assert state.update_count.item() == 0
    model(x, y=labels, return_aux=True, update_nmfdu_support=True)
    assert state.update_count.item() == 1
    assert state.discriminability_ema.max().item() > 0.0

    model.eval()
    with pytest.raises(RuntimeError, match="training mode"):
        model(x, y=labels, return_aux=True, update_nmfdu_support=True)


def test_labels_cannot_change_physical_evidence_without_authorized_update() -> None:
    torch.manual_seed(73)
    model = _small_backbone().eval()
    model.nmfdu_gate.evidence_state.discriminability_ema.fill_(0.5)
    x = torch.randn(3, 2, 96)
    with torch.no_grad():
        first = model(
            x,
            y=torch.tensor([0, 1, 2]),
            return_aux=True,
            return_physical_gate_diag=True,
        )["physical_gate_diag"]["per_sample"]
        second = model(
            x,
            y=torch.tensor([2, 0, 1]),
            return_aux=True,
            return_physical_gate_diag=True,
        )["physical_gate_diag"]["per_sample"]
    for key in ("I", "D", "S", "U", "weights", "null_weight", "q_sample"):
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)


def test_nmfdu_checkpoint_round_trip_restores_gate_and_ema_state() -> None:
    torch.manual_seed(74)
    model = _small_backbone().eval()
    model.nmfdu_gate.evidence_state.discriminability_ema.copy_(
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    )
    restored = _small_backbone().eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    torch.testing.assert_close(
        restored.nmfdu_gate.evidence_state.discriminability_ema,
        model.nmfdu_gate.evidence_state.discriminability_ema,
    )
    x = torch.randn(2, 2, 96)
    with torch.no_grad():
        expected = model(x, return_aux=False)
        actual = restored(x, return_aux=False)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_dual_model_routes_nmfdu_only_through_identity_backbone() -> None:
    torch.manual_seed(75)
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        model_variant="lite_h",
        input_len=96,
        fast_infer_when_no_aux=False,
        physical_gate_variant="nmfdu_v1",
    ).eval()
    assert model.id_backbone.nmfdu_gate is not None
    assert model.dom_backbone.nmfdu_gate is None
    with torch.no_grad():
        output = model(
            torch.randn(2, 2, 96),
            return_aux=True,
            return_physical_gate_diag=True,
        )
    assert output["z_id"].shape == (2, model.emb_dim)
    assert output["aux_id"]["physical_gate_diag"]["per_sample"]["weights"].shape == (2, 5)
    assert "physical_gate_diag" not in output["aux_dom"]
    torch.testing.assert_close(output["z_id"], output["aux_id"]["feat_joint"])
