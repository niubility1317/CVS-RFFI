"""Phase1-safe configuration helpers for ADVB02 CRRA training."""

from __future__ import annotations

from typing import Any

from crra import crra_gate_scale


def validate_crra_phase1_config(args: Any) -> None:
    """Reject CRRA configurations that leave the current Phase1 boundary."""

    scenario = str(getattr(args, "crra_scenario", "mixed_orbit") or "").strip().lower().replace("-", "_")
    if scenario != "mixed_orbit":
        raise ValueError("ADVB02 CRRA Phase1 requires the historical mixed_orbit channel")
    if bool(getattr(args, "crra_target_adapter", False)):
        raise ValueError("CRRA target adapter is not allowed in the Phase1 source-only training path")


def validate_crra_phase1_scenarios(scenarios: Any) -> None:
    """Keep every CRRA training view on the historical mixed_orbit channel."""

    if isinstance(scenarios, str):
        values = [item for item in scenarios.replace(";", ",").split(",") if item.strip()]
    else:
        values = list(scenarios or [])
    normalized = [str(value or "").strip().lower().replace("-", "_") for value in values]
    bad = [value for value in normalized if value and value != "mixed_orbit"]
    if bad:
        raise ValueError(
            "ADVB02 CRRA Phase1 satellite views must remain the historical mixed_orbit channel; "
            f"got {bad}"
        )


__all__ = ["crra_gate_scale", "validate_crra_phase1_config", "validate_crra_phase1_scenarios"]
