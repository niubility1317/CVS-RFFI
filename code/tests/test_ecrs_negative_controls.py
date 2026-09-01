from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import ResponseBasis  # noqa: E402
from train import build_ecrs_negative_controls, summarize_ecrs_diagnostics  # noqa: E402


def _synthetic_output():
    batch, samples = 3, 40
    quality = {
        "log_condition": torch.ones(batch),
        "effective_rank": torch.full((batch,), 20.0),
        "effective_sample_size": torch.full((batch,), 30.0),
        "coverage": torch.full((batch,), 0.75),
        "nmse": torch.full((batch,), 0.10),
        "snr_db": torch.full((batch,), 10.0),
        "anchor_variance": torch.ones(batch, 32),
    }
    return {
        "resp_coef": torch.randn(batch, 28, dtype=torch.complex64),
        "resp_cov_diag": torch.ones(batch, 28),
        "resp_quality": quality,
        "response_design": torch.randn(batch, samples, 28, dtype=torch.complex64),
        "canonical_iq": torch.randn(batch, 2, samples),
        "resp_anchor": torch.randn(batch, 32, dtype=torch.complex64),
        "ridge_info": torch.tensor([0, 1, 2]),
        "rho_resp": torch.tensor([0.0, 0.1, 0.2]),
        "z_resp": torch.randn(batch, 64),
    }


def test_section28_negative_controls_are_explicit_and_do_not_enable_free_basis() -> None:
    out = _synthetic_output()
    controls = build_ecrs_negative_controls(out, ["a", "b", "c"])
    assert controls["excitation_shuffle"].shape == out["response_design"].shape
    assert controls["residual_shuffle"].shape == (3, 40)
    assert controls["quality_only_tx_probe"].shape == (3, 7)
    assert controls["raw_coefficient"].shape == (3, 56)
    assert controls["whitened_coefficient"].shape == (3, 56)
    assert controls["anchor_surface"].shape == (3, 64)
    assert controls["pair_id_shuffle"] == ["b", "c", "a"]
    assert controls["basis_controls"][-1] == "free_mlp_forbidden_in_v1"
    assert ResponseBasis("fixed_mp")(torch.randn(2, 40, dtype=torch.complex64)).shape[-1] == 28


def test_diagnostic_record_contains_solver_probe_and_surface_exports() -> None:
    diagnostics = summarize_ecrs_diagnostics(
        _synthetic_output(),
        tx_labels=torch.tensor([0, 1, 2]),
        receiver_labels=torch.tensor([2, 3, 4]),
        same_tx_prediction_error=torch.tensor(0.2),
        different_tx_prediction_error=torch.tensor(1.0),
        raw_correct=torch.tensor([False, True, True]),
        fused_correct=torch.tensor([True, False, True]),
    )
    for key in (
        "response_nmse",
        "gram_log_condition",
        "effective_rank",
        "effective_sample_size",
        "coverage",
        "ridge_fallback_rate",
        "ridge_qr_rate",
        "same_diff_prediction_ratio",
        "gate_rescue_count",
        "gate_harm_count",
        "gate_net_gain",
        "probe_payload",
        "surface_export",
    ):
        assert key in diagnostics
    assert torch.allclose(diagnostics["same_diff_prediction_ratio"], torch.tensor(0.2))
    assert diagnostics["gate_rescue_count"] == 1
    assert diagnostics["gate_harm_count"] == 1
    assert diagnostics["gate_net_gain"] == 0
