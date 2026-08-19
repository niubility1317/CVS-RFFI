"""Phase1-safe configuration helpers for ADVB02 CRRA training."""

from __future__ import annotations

from typing import Any

from crra import crra_gate_scale
from training_controls import DEFAULT_CRRA_CHANNEL_FAMILY, LEO_WEAK_SCENARIOS



def _normalize_scenario(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def validate_crra_phase1_config(args: Any) -> None:
    """Reject CRRA configurations that leave the current Phase1 boundary."""

    scenario = _normalize_scenario(getattr(args, "crra_scenario", DEFAULT_CRRA_CHANNEL_FAMILY))
    if scenario not in {"mixed_orbit", "leo_weak"}:
        raise ValueError(
            "ADVB02 CRRA Phase1 channel family must be mixed_orbit or leo_weak; "
            f"got {scenario or '<empty>'}"
        )
    if bool(getattr(args, "crra_target_adapter", False)):
        raise ValueError("CRRA target adapter is not allowed in the Phase1 source-only training path")


def validate_crra_phase1_scenarios(
    scenarios: Any,
    *,
    crra_scenario: str = DEFAULT_CRRA_CHANNEL_FAMILY,
) -> None:
    """Keep CRRA training views inside the configured Phase1 channel family."""

    if isinstance(scenarios, str):
        values = [item for item in scenarios.replace(";", ",").split(",") if item.strip()]
    else:
        values = list(scenarios or [])
    normalized = [_normalize_scenario(value) for value in values if _normalize_scenario(value)]
    family = _normalize_scenario(crra_scenario)
    if family == "mixed_orbit":
        bad = [value for value in normalized if value != "mixed_orbit"]
        if bad:
            raise ValueError(
                "ADVB02 CRRA Phase1 mixed_orbit views must all be mixed_orbit; "
                f"got {bad}"
            )
        return
    if family == "leo_weak":
        scenario_set = set(normalized)
        required_set = set(LEO_WEAK_SCENARIOS)
        bad = sorted(scenario_set - required_set)
        missing = sorted(required_set - scenario_set)
        if bad or missing:
            details = []
            if bad:
                details.append(f"unexpected={bad}")
            if missing:
                details.append(f"missing={missing}")
            raise ValueError(
                "ADVB02 CRRA Phase1 leo_weak training requires exactly "
                f"{list(LEO_WEAK_SCENARIOS)}; " + ", ".join(details)
            )
        return
    raise ValueError(
        "ADVB02 CRRA Phase1 channel family must be mixed_orbit or leo_weak; "
        f"got {family or '<empty>'}"
    )


__all__ = [
    "LEO_WEAK_SCENARIOS",
    "crra_gate_scale",
    "validate_crra_phase1_config",
    "validate_crra_phase1_scenarios",
]
