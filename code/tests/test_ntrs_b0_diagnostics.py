import torch
import numpy as np
import pytest

from cvsrffi.ntrs_b0_diagnostics import analyze_paired_shift
from scripts.run_ntrs_b0_from_pair_exports import _require_exact_pair, _torch_from_numpy_compatible


def test_b0_diagnostic_reports_rank_oracle_and_continuous_gate():
    clean = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]])
    shift = torch.tensor([[0.0, 1.2], [1.2, 0.0], [0.0, 1.0], [1.0, 0.0]])
    satellite = clean + shift
    labels = torch.tensor([0, 1, 0, 1])
    weight = torch.eye(2)
    result = analyze_paired_shift(
        clean,
        satellite,
        labels,
        weight,
        ranks=(1, 2),
        learned_correction=shift,
        tx_ids=labels,
        scenario_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert result["ranks"]["2"]["explained_variance"] > 0.99
    assert result["full_shift_oracle"]["accuracy"] == 1.0
    assert result["continuous_gate_oracle"]["sample_oracle"]["accuracy"] == 1.0
    assert "tx_main_ratio" in result["variance_decomposition"]
    assert "tx_scenario_interaction_ratio" in result["variance_decomposition"]


def test_b0_tx_scenario_interaction_is_not_tx_main_effect():
    clean = torch.tensor([[2.0, 0.0]] * 4)
    satellite = clean + torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    result = analyze_paired_shift(
        clean,
        satellite,
        torch.zeros(4, dtype=torch.long),
        torch.eye(2),
        ranks=(1,),
        tx_ids=torch.tensor([0, 0, 1, 1]),
        scenario_ids=torch.tensor([0, 1, 0, 1]),
    )
    decomposition = result["variance_decomposition"]
    assert decomposition["tx_main_ratio"] > 0.99
    assert decomposition["tx_scenario_interaction_ratio"] < 1e-8


def test_b0_pair_requires_vcal_manifest_and_exact_physical_ids():
    arrays = {
        "tx_ids": np.asarray(["tx0"]),
        "rx_ids": np.asarray(["rx0"]),
        "day_ids": np.asarray(["day0"]),
        "eq_ids": np.asarray(["1"]),
        "sig_ids": np.asarray(["9"]),
        "raw_labels": np.asarray([0]),
        "domain_labels": np.asarray([0]),
    }
    common = {
        "checkpoint": "/tmp/a.pth",
        "source": {"phase1_role": "V_cal"},
        "source_tx_ids": ["tx0"],
        "source_pair_role": "V_cal",
        "star_ground_channel_impl": "simplified_leo_residual",
    }
    _require_exact_pair(
        arrays,
        dict(arrays),
        {**common, "channel_view": "clean"},
        {**common, "channel_view": "satellite", "sat_scenario": "leo_clear_weak"},
    )
    mismatched = dict(arrays)
    mismatched["sig_ids"] = np.asarray(["10"])
    with pytest.raises(ValueError, match="sig_ids"):
        _require_exact_pair(
            arrays,
            mismatched,
            {**common, "channel_view": "clean"},
            {**common, "channel_view": "satellite", "sat_scenario": "leo_clear_weak"},
        )


def test_b0_numpy_conversion_does_not_depend_on_torch_from_numpy(monkeypatch):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    monkeypatch.setattr(torch, "from_numpy", lambda _value: (_ for _ in ()).throw(TypeError("ABI mismatch")))
    tensor = _torch_from_numpy_compatible(array)
    assert tensor.shape == (3, 4)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, torch.arange(12, dtype=torch.float32).reshape(3, 4))
