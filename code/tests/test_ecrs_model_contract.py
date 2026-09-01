from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import ResponseFusionGate, build_dual_model  # noqa: E402


def _tiny_model(**kwargs):
    return build_dual_model(
        num_classes=6,
        num_domains=5,
        model_size="M",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_d",
        branch_ablation="no_dac",
        domain_branch_ablation="no_stats",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        fast_infer_when_no_aux=False,
        **kwargs,
    )


def test_ecrs_off_preserves_legacy_state_and_outputs() -> None:
    torch.manual_seed(20260901)
    legacy = _tiny_model().eval()
    candidate = _tiny_model(use_ecrs=False).eval()
    candidate.load_state_dict(legacy.state_dict(), strict=True)

    assert not any(key.startswith("ecrs") for key in candidate.state_dict())
    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        legacy_out = legacy(x, return_aux=True)
        candidate_out = candidate(x, return_aux=True)
    torch.testing.assert_close(
        candidate_out["tx_logits"], legacy_out["tx_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        candidate_out["z_id"], legacy_out["z_id"], rtol=0.0, atol=0.0
    )


def test_ecrs_on_exposes_report_outputs_and_bounded_residual_fusion() -> None:
    torch.manual_seed(20260902)
    model = _tiny_model(
        use_ecrs=True,
        ecrs_config={"response_basis_dim": 28, "response_dim": 64, "rho_max": 0.25},
    ).eval()
    x = torch.randn(3, 2, 64)
    with torch.no_grad():
        out = model(x, return_aux=True)

    required = {
        "z_id_raw",
        "z_resp",
        "z_id_fused",
        "resp_coef",
        "resp_cov_diag",
        "resp_quality",
        "resp_anchor",
        "nuisance_coef",
        "content_confidence",
        "tx_logits_raw",
        "rho_resp",
    }
    assert required.issubset(out)
    assert out["z_id_raw"].shape == (3, 160)
    assert out["z_resp"].shape == (3, 64)
    assert out["z_id_fused"].shape == (3, 160)
    assert out["resp_coef"].shape == (3, 28)
    assert out["resp_cov_diag"].shape == (3, 28)
    assert out["resp_anchor"].shape == (3, 32)
    assert out["nuisance_coef"].shape == (3, 3)
    assert out["content_confidence"].shape == (3, 64)
    assert torch.all((out["rho_resp"] >= 0.0) & (out["rho_resp"] <= 0.25))
    torch.testing.assert_close(
        out["z_id_fused"].norm(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(out["z_id"], out["z_id_fused"])
    with torch.no_grad():
        single_view_logits = model(x, return_aux=False)
    torch.testing.assert_close(single_view_logits, out["tx_logits"], rtol=0.0, atol=0.0)


def test_ecrs_true_instantiates_only_fixed_v1_dimensions() -> None:
    model = _tiny_model(use_ecrs=True)
    assert model.ecrs.response_basis.block_slices["slew"] == slice(20, 28)
    assert model.ecrs.response_projection.in_features == 64
    assert model.ecrs.response_projection.out_features == 160
    assert model.ecrs.fusion_gate.rho_max == 0.25


def test_response_gate_cannot_backpropagate_into_quality_measurements() -> None:
    gate = ResponseFusionGate()
    quality = {
        key: torch.ones(2, requires_grad=True)
        for key in (
            "log_condition",
            "effective_rank",
            "effective_sample_size",
            "coverage",
            "nmse",
            "snr_db",
        )
    }
    covariance = torch.ones(2, 28, requires_grad=True)
    gate(quality, covariance, sample_count=64).sum().backward()
    assert all(value.grad is None for value in quality.values())
    assert covariance.grad is None
