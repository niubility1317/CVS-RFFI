import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import (  # noqa: E402
    _crra_satellite_kl_active,
    build_arg_parser,
)
from cvsrffi.crra_training import (  # noqa: E402
    crra_gate_scale,
    validate_crra_phase1_config,
    validate_crra_phase1_scenarios,
)
from cvsrffi.losses import crra_nuisance_huber_loss  # noqa: E402


def test_crra_schedule_has_identity_ramp_and_fixed_tail():
    assert crra_gate_scale(1) == 0.0
    assert crra_gate_scale(16) == 0.0
    assert 0.0 < crra_gate_scale(30) < 1.0
    assert crra_gate_scale(47) == 1.0


def test_phase1_crra_defaults_to_mixed_orbit_and_has_no_target_access():
    args = build_arg_parser().parse_args(["--output_dir", "x"])
    assert args.sat_train_scenario == "mixed_orbit"
    assert args.crra_scenario == "mixed_orbit"
    assert args.crra_target_adapter is False
    assert args.lambda_crra_pair == pytest.approx(0.05)
    assert args.lambda_crra_sat_kl == pytest.approx(0.05)
    assert args.lambda_crra_energy == pytest.approx(0.001)
    assert args.lambda_crra_nuisance == pytest.approx(0.02)
    assert args.lambda_crra_condition_tx_adv == pytest.approx(0.02)


def test_phase1_crra_rejects_wrong_channel_and_target_adapter():
    with pytest.raises(ValueError, match="mixed_orbit"):
        validate_crra_phase1_config(SimpleNamespace(crra_scenario="leo_weak", crra_target_adapter=False))
    with pytest.raises(ValueError, match="target adapter"):
        validate_crra_phase1_config(SimpleNamespace(crra_scenario="mixed_orbit", crra_target_adapter=True))


def test_phase1_crra_rejects_non_historical_satellite_scenarios():
    with pytest.raises(ValueError, match="historical mixed_orbit"):
        validate_crra_phase1_scenarios(["mixed_orbit", "leo_low_elev_weak"])


def test_nuisance_loss_uses_valid_same_view_metadata_only():
    pred = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [2.0, 4.0, 6.0]])
    valid = torch.tensor([True, False, True])
    loss, info = crra_nuisance_huber_loss(pred, target, valid)
    assert info["valid_count"] == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_nuisance_loss_is_zero_with_no_valid_same_view_targets():
    pred = torch.zeros(2, 3, requires_grad=True)
    loss, info = crra_nuisance_huber_loss(pred, None, None)
    assert float(loss.detach()) == 0.0
    assert info["valid_count"] == 0
    loss.backward()
    assert pred.grad is not None


def test_crra_satellite_kl_is_active_without_legacy_sat_cons_weight():
    assert not _crra_satellite_kl_active(
        0.0,
        use_crra=True,
        crra_stage_scale=0.0,
        crra_sat_kl_weight=0.05,
    )
    assert _crra_satellite_kl_active(
        0.0,
        use_crra=True,
        crra_stage_scale=0.5,
        crra_sat_kl_weight=0.05,
    )
    assert _crra_satellite_kl_active(
        0.05,
        use_crra=False,
        crra_stage_scale=0.0,
        crra_sat_kl_weight=0.0,
    )


def test_nuisance_loss_rejects_field_dimension_drift():
    with pytest.raises(ValueError, match="same fixed field dimension"):
        crra_nuisance_huber_loss(
            torch.zeros(2, 8),
            torch.zeros(2, 9),
            torch.tensor([True, True]),
        )
