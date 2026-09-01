from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import ResponseBasis  # noqa: E402
from train import (  # noqa: E402
    build_ecrs_negative_controls,
    build_ecrs_v1_diagnostic_record,
    ecrs_pair_directional_prediction_errors,
    save_ecrs_v1_diagnostic_artifact,
    summarize_ecrs_diagnostics,
)


def _synthetic_output():
    batch, samples = 3, 40
    quality = {
        "log_condition": torch.ones(batch),
        "effective_rank": torch.full((batch,), 20.0),
        "effective_sample_size": torch.full((batch,), 30.0),
        "coverage": torch.full((batch,), 0.75),
        "nmse": torch.full((batch,), 0.10),
        "snr_db": torch.full((batch,), 10.0),
        "anchor_variance": torch.ones(batch, 8),
    }
    return {
        "resp_coef": torch.randn(batch, 28, dtype=torch.complex64),
        "resp_cov_diag": torch.ones(batch, 28),
        "resp_quality": quality,
        "response_design": torch.randn(batch, samples, 28, dtype=torch.complex64),
        "canonical_iq": torch.randn(batch, 2, samples),
        "resp_anchor": torch.randn(batch, 8, dtype=torch.complex64),
        "ridge_info": torch.tensor([0, 1, 2]),
        "rho_resp": torch.tensor([0.0, 0.1, 0.2]),
        "z_resp": torch.randn(batch, 64),
        "s_hat": torch.randn(batch, samples, dtype=torch.complex64),
        "nuisance_reg_coef": torch.randn(batch, 4, dtype=torch.complex64),
        "nuisance_coef": torch.randn(batch, 3),
        "block_identifiability": torch.rand(batch, 4),
        "response_target": torch.randn(batch, samples, dtype=torch.complex64),
        "response_weights": torch.ones(batch, samples),
    }


def test_section28_negative_controls_are_explicit_and_do_not_enable_free_basis() -> None:
    out = _synthetic_output()
    controls = build_ecrs_negative_controls(out, ["a", "b", "c"])
    assert controls["excitation_shuffle"].shape == out["response_design"].shape
    assert controls["residual_shuffle"].shape == (3, 40)
    assert controls["quality_only_tx_probe"].shape == (3, 7)
    assert controls["raw_coefficient"].shape == (3, 56)
    assert controls["whitened_coefficient"].shape == (3, 56)
    assert controls["anchor_surface"].shape == (3, 16)
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


def test_pair_directional_errors_report_clean_and_leo_directions_separately() -> None:
    clean = _synthetic_output()
    leo = _synthetic_output()
    clean["resp_coef"].zero_()
    leo["resp_coef"].zero_()
    clean["response_target"].fill_(1.0 + 0.0j)
    leo["response_target"].fill_(2.0 + 0.0j)
    clean_to_leo, leo_to_clean = ecrs_pair_directional_prediction_errors(clean, leo)
    assert torch.allclose(clean_to_leo, torch.tensor(1.0))
    assert torch.allclose(leo_to_clean, torch.tensor(1.0))


def test_production_diagnostic_record_is_probe_ready_and_contains_no_tx_truth_alias() -> None:
    clean = _synthetic_output()
    leo = _synthetic_output()
    pair_meta = {
        "physical_sample_id": ["sample:0", "sample:1", "sample:2"] * 2,
        "pair_id": ["pair:0", "pair:1", "pair:2"] * 2,
        "view_type": ["clean"] * 3 + ["leo"] * 3,
        "label_mask": torch.ones(6, dtype=torch.bool),
        "receiver_id": torch.tensor([0, 1, 2, 0, 1, 2]),
        "day_id": torch.tensor([4, 4, 4, 4, 4, 4]),
        "crop_offset": torch.tensor([8, 9, 10, 8, 9, 10]),
        "synchronized_crop": True,
        "clean_mask": torch.tensor([True] * 3 + [False] * 3),
        "leo_mask": torch.tensor([False] * 3 + [True] * 3),
    }
    record = build_ecrs_v1_diagnostic_record(
        clean,
        leo,
        pair_meta,
        tx_labels=torch.tensor([0, 1, 2]),
        include_negative_controls=True,
    )
    assert set(record["probe_payload"]) >= {
        "z_resp", "s_hat_summary", "c_fp", "gamma_nuis", "tx_labels", "receiver_labels", "view_labels"
    }
    assert record["probe_payload"]["z_resp"].shape == (6, 64)
    assert record["surface_export"]["resp_anchor"].shape == (6, 8)
    assert "negative_controls" in record
    assert "query" not in repr(record).lower()
    assert "true_tx_i" not in repr(record)


def test_diagnostic_artifact_save_is_non_overwriting(tmp_path: Path) -> None:
    target = tmp_path / "ecrs_v1_diagnostics.pt"
    payload = {"schema": "adv3b02_ecrs_v1_diagnostics_v1", "records": []}
    save_ecrs_v1_diagnostic_artifact(target, payload)
    assert torch.load(target, weights_only=False)["schema"] == payload["schema"]
    with pytest.raises(FileExistsError):
        save_ecrs_v1_diagnostic_artifact(target, payload)
