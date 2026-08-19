from __future__ import annotations

from types import SimpleNamespace

import pytest

from cvsrffi.crra_training import (
    validate_crra_phase1_config,
    validate_crra_phase1_scenarios,
)


def test_crra_rejects_wrong_phase1_channel_name():
    with pytest.raises(ValueError, match="mixed_orbit"):
        validate_crra_phase1_config(
            SimpleNamespace(crra_scenario="leo_weak", crra_target_adapter=False)
        )


def test_crra_target_adapter_is_rejected_in_phase1():
    with pytest.raises(ValueError, match="target adapter"):
        validate_crra_phase1_config(
            SimpleNamespace(crra_scenario="mixed_orbit", crra_target_adapter=True)
        )


def test_crra_cannot_schedule_a_leo_weak_training_view():
    with pytest.raises(ValueError, match="historical mixed_orbit"):
        validate_crra_phase1_scenarios(["mixed_orbit", "leo_rain_weak"])
