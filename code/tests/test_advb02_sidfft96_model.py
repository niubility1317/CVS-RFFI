import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import build_dual_model  # noqa: E402
import post_stage_common  # noqa: E402
from SSDG import train_ssdg as train_module  # noqa: E402
from SSDG.train_ssdg import (  # noqa: E402
    _load_training_checkpoint_state,
    compose_sid_adapter_objective,
    configure_sid_trainable_parameters,
    resolve_phase1_checkpoint_selection,
)


def _write_mask(tmp_path: Path, fft_bins: int = 64) -> Path:
    path = tmp_path / "sid_mask.npz"
    np.savez_compressed(path, mask=np.ones(fft_bins, dtype=np.uint8))
    return path


def _tiny_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        **kwargs,
    )


def test_sid_zero_initialization_matches_raw_logits(tmp_path):
    model = _tiny_model(
        sid_fft96_mode="sid",
        sid_mask_path=str(_write_mask(tmp_path)),
        fast_infer_when_no_aux=False,
    ).eval()
    x = torch.randn(2, 2, 64)
    labels = torch.tensor([0, 1])

    with torch.no_grad():
        output = model(x, y_tx=labels, return_aux=True)

    assert torch.allclose(output["z_id_raw"], output["z_id_sid"], atol=1e-7, rtol=0)
    assert torch.allclose(output["logits_raw"], output["logits_sid"], atol=1e-7, rtol=0)
    assert torch.equal(output["z_id"], output["z_id_sid"])
    assert torch.equal(output["tx_logits"], output["logits_sid"])
    assert output["sid_fft96"].shape == (2, 96)


def test_sid_and_existing_residual_candidates_are_mutually_exclusive(tmp_path):
    mask_path = str(_write_mask(tmp_path))

    with pytest.raises(ValueError, match="independent candidates"):
        _tiny_model(sid_fft96_mode="sid", sid_mask_path=mask_path, use_ntrs=True)
    with pytest.raises(ValueError, match="independent candidates"):
        _tiny_model(sid_fft96_mode="sid", sid_mask_path=mask_path, use_crra=True)


def test_sid_off_preserves_parameter_count_and_fast_path():
    torch.manual_seed(11)
    control = _tiny_model().eval()
    torch.manual_seed(11)
    explicit_off = _tiny_model(sid_fft96_mode="off", sid_mask_path="").eval()

    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in explicit_off.parameters()
    )
    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        assert torch.equal(control(x), explicit_off(x))


def _sid_args(**overrides):
    values = {
        "sid_fft96_mode": "sid",
        "sid_adapter_only": True,
        "sid_guarded_training": True,
        "ntrs_variant": "v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_checkpoint_loader_allows_only_new_sid_keys(tmp_path):
    base = _tiny_model(sid_fft96_mode="off")
    sid = _tiny_model(sid_fft96_mode="sid", sid_mask_path=str(_write_mask(tmp_path)))

    report = _load_training_checkpoint_state(sid, {"model": base.state_dict()}, _sid_args())

    assert report["missing_keys"]
    assert all(key.startswith("sid_fft96.") for key in report["missing_keys"])
    bad = dict(base.state_dict())
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="non-SID checkpoint drift"):
        _load_training_checkpoint_state(sid, {"model": bad}, _sid_args())


def test_sid_only_training_freezes_mature_path(tmp_path):
    model = _tiny_model(sid_fft96_mode="sid", sid_mask_path=str(_write_mask(tmp_path)))

    summary = configure_sid_trainable_parameters(model, _sid_args())
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable
    assert all(name.startswith("sid_fft96.") for name in trainable)
    assert summary["raw_trainable_parameters"] == 0
    assert summary["sid_trainable_parameters"] > 0


def test_sid_adapter_objective_contains_only_tx_satellite_and_identity_anchor():
    z_raw = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_sid = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
    clean_tx = z_sid.sum() * 0.0 + 2.0
    satellite_tx = z_sid.sum() * 0.0 + 3.0

    total, diagnostics = compose_sid_adapter_objective(
        clean_tx_loss=clean_tx,
        satellite_tx_loss=satellite_tx,
        z_sid=z_sid,
        z_raw=z_raw,
        satellite_weight=0.68,
        identity_anchor_weight=0.05,
    )

    expected_anchor = 1.0 - torch.nn.functional.cosine_similarity(z_sid, z_raw, dim=1).mean()
    assert torch.allclose(diagnostics["identity_anchor"], expected_anchor)
    assert torch.allclose(total, clean_tx + 0.68 * satellite_tx + 0.05 * expected_anchor)


def test_sid_adapter_uses_source_validation_checkpoint_selection():
    assert resolve_phase1_checkpoint_selection(
        _sid_args(formal_ablation=False)
    ) == "source_validation_only"
    assert resolve_phase1_checkpoint_selection(
        _sid_args(sid_fft96_mode="off", formal_ablation=False)
    ) == "final_only"


def test_post_stage_model_rebuild_forwards_sid_residual_bound(monkeypatch):
    captured = {}

    class DummyModel:
        def to(self, device):
            captured["device"] = device
            return self

    def fake_build_dual_model(*args, **kwargs):
        captured["positional"] = args
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(post_stage_common, "build_dual_model", fake_build_dual_model)
    args = SimpleNamespace(
        num_classes=3,
        num_domains=2,
        sid_fft96_mode="sid",
        sid_max_residual_ratio=0.10,
    )

    post_stage_common.build_baseline_model(args, device=torch.device("cpu"))

    assert captured["sid_max_residual_ratio"] == pytest.approx(0.10)


def test_hsid_model_keeps_raw_embedding_and_exposes_independent_evidence(tmp_path):
    model = _tiny_model(
        sid_fft96_mode="sid",
        sid_mask_path=str(_write_mask(tmp_path)),
        sid_architecture="hsid",
        sid_fusion_mode="fused",
        sid_spectral_dim=16,
        sid_fusion_alpha_max=0.2,
        fast_infer_when_no_aux=False,
    ).eval()
    x = torch.randn(3, 2, 64)

    with torch.no_grad():
        output = model(x, return_aux=True)

    assert torch.equal(output["z_id"], output["z_id_raw"])
    assert output["sid_spectral_embedding"].shape == (3, 16)
    assert output["sid_spec_logits"].shape == (3, 3)
    assert output["sid_fusion_gate"].shape == (3,)
    assert output["sid_quality"].shape == (3, 7)
    assert torch.equal(output["tx_logits"], output["logits_raw"])


def test_hsid_objective_uses_receiver_risk_and_protects_raw_margin():
    compose = getattr(train_module, "compose_hsid_objective", None)
    assert compose is not None
    labels = torch.tensor([0, 0, 1, 1])
    receivers = torch.tensor([0, 1, 0, 1])
    spectral_embedding = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    spectral_logits = torch.tensor(
        [[3.0, 0.0], [0.2, 2.0], [0.0, 3.0], [2.0, 0.1]],
        requires_grad=True,
    )
    raw_logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    fused_logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [1.0, 3.0], [3.0, 1.0]], requires_grad=True)

    total, diagnostics = compose(
        spectral_logits=spectral_logits,
        spectral_embedding=spectral_embedding,
        fused_logits=fused_logits,
        raw_logits=raw_logits,
        labels=labels,
        receiver_labels=receivers,
        lambda_cross_rx=0.05,
        lambda_receiver_cvar=0.10,
        lambda_interaction=0.02,
        lambda_margin_safety=0.10,
        harm_margin=0.5,
    )

    assert total.requires_grad
    assert diagnostics["receiver_cvar"] > 0
    assert diagnostics["margin_safety"] > 0
    assert diagnostics["cross_rx"] >= 0
    assert diagnostics["interaction"] >= 0


def test_source_hsid_selection_score_includes_receiver_and_receiver_day_floors():
    score_fn = getattr(train_module, "source_hsid_selection_score", None)
    assert score_fn is not None

    score = score_fn(
        clean_acc=90.0,
        sat_mean=80.0,
        sat_floor=70.0,
        receiver_floor=60.0,
        receiver_day_floor=50.0,
    )

    harmonic_sat = 2.0 * 80.0 * 70.0 / (80.0 + 70.0)
    assert score == pytest.approx(0.20 * 90.0 + 0.20 * harmonic_sat + 0.30 * 60.0 + 0.30 * 50.0)
