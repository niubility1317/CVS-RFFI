from types import SimpleNamespace

import pytest

from cvsrffi.ntrs_training import (
    validate_ntrs_phase1_config,
    validate_ntrs_phase1_scenarios,
)


def _valid_args(**overrides):
    values = {
        "use_ntrs": True,
        "use_crra": False,
        "ntrs_target_adapter": False,
        "ntrs_unknown_rescue": False,
        "ntrs_rank": 8,
        "ntrs_alpha_max": 0.20,
        "ntrs_support_tau": 1.0,
        "ntrs_energy_threshold": 0.10,
        "ntrs_slow_ema_decay": 0.95,
        "lambda_ntrs_sat_kl": 0.01,
        "lambda_ntrs_margin": 0.03,
        "lambda_ntrs_relation": 0.02,
        "lambda_ntrs_cond_decorr": 0.01,
        "lambda_ntrs_min_correction": 0.001,
        "lambda_ntrs_subspace": 0.02,
        "lambda_ntrs_correctability": 0.02,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ntrs_rejects_crra_coenablement():
    with pytest.raises(ValueError, match="independent candidates"):
        validate_ntrs_phase1_config(_valid_args(use_crra=True))


def test_ntrs_rejects_phase2_target_adapter_and_unknown_rescue():
    with pytest.raises(ValueError, match="target adapter"):
        validate_ntrs_phase1_config(_valid_args(ntrs_target_adapter=True))
    with pytest.raises(ValueError, match="unknown rescue"):
        validate_ntrs_phase1_config(_valid_args(ntrs_unknown_rescue=True))


def test_ntrs_rejects_mixed_orbit_or_incomplete_leo_weak_family():
    with pytest.raises(ValueError, match="exactly"):
        validate_ntrs_phase1_scenarios(
            ["leo_clear_weak", "leo_low_elev_weak", "mixed_orbit"]
        )
    with pytest.raises(ValueError, match="exactly"):
        validate_ntrs_phase1_scenarios(["leo_clear_weak", "leo_rain_weak"])


def test_ntrs_rejects_unbounded_or_negative_controls():
    with pytest.raises(ValueError, match="alpha_max"):
        validate_ntrs_phase1_config(_valid_args(ntrs_alpha_max=0.21))
    with pytest.raises(ValueError, match="non-negative"):
        validate_ntrs_phase1_config(_valid_args(lambda_ntrs_margin=-0.01))
    with pytest.raises(ValueError, match="slow_ema_decay"):
        validate_ntrs_phase1_config(_valid_args(ntrs_slow_ema_decay=1.0))
