from __future__ import annotations

from types import SimpleNamespace

import pytest

from cvsrffi.crra_training import (
    validate_crra_phase1_config,
    validate_crra_phase1_scenarios,
)


def test_crra_accepts_the_confirmed_leo_weak_family():
    validate_crra_phase1_config(
        SimpleNamespace(crra_scenario="leo_weak", crra_target_adapter=False)
    )


def test_crra_target_adapter_is_rejected_in_phase1():
    with pytest.raises(ValueError, match="target adapter"):
        validate_crra_phase1_config(
            SimpleNamespace(crra_scenario="leo_weak", crra_target_adapter=True)
        )


def test_crra_leo_family_rejects_a_mixed_orbit_training_view():
    with pytest.raises(ValueError, match="leo_weak"):
        validate_crra_phase1_scenarios(
            ["leo_clear_weak", "leo_low_elev_weak", "mixed_orbit"],
            crra_scenario="leo_weak",
        )
