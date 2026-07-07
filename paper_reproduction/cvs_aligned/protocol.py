from __future__ import annotations

from typing import Any


RECOMMENDED_LEO_SCENARIOS = {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def validate_stage2_protocol_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stage = str(payload.get("stage") or payload.get("stage2_mode") or "").strip()
    if stage not in {"Stage2-B", "Stage2-C", "B", "C"}:
        raise ValueError("stage must be Stage2-B or Stage2-C")
    normalized_stage = "Stage2-C" if stage.endswith("C") else "Stage2-B"
    source_receivers = {str(v) for v in _list(payload, "source_receiver_labels")}
    target_receivers = {str(v) for v in _list(payload, "target_receiver_labels")}
    if not source_receivers or not target_receivers:
        raise ValueError("source_receiver_labels and target_receiver_labels are required")
    if source_receivers & target_receivers:
        raise ValueError("R_s and R_t must be disjoint")

    old = {str(v) for v in _list(payload, "target_old_tx_labels")}
    new = {str(v) for v in _list(payload, "target_new_tx_labels")}
    unknown = {str(v) for v in _list(payload, "target_unknown_tx_labels")}
    if not old:
        raise ValueError("target_old_tx_labels are required")
    if old & new:
        raise ValueError("Y_old and Y_new must be disjoint")
    if old & unknown:
        raise ValueError("Y_unknown must be disjoint from Y_old")
    if new & unknown:
        raise ValueError("Y_new and Y_unknown must be disjoint")
    if normalized_stage == "Stage2-C" and not new:
        raise ValueError("Stage2-C requires target_new_tx_labels")
    unknown_rejection_enabled = bool(payload.get("unknown_rejection_enabled", True))
    if not unknown and unknown_rejection_enabled:
        raise ValueError("target_unknown_tx_labels are required when unknown_rejection_enabled=true")

    k_shot = int(payload.get("k_shot", 0))
    if k_shot <= 0:
        raise ValueError("k_shot must be a positive integer")

    view = str(payload.get("target_channel_view", "")).strip()
    scenarios = {str(v) for v in _list(payload, "target_channel_scenarios")}
    is_clean = view == "clean"
    is_satellite = view == "satellite/LEO"
    if not (is_clean or is_satellite):
        raise ValueError("target_channel_view must be clean or satellite/LEO")
    if is_clean and scenarios != {"clean"}:
        raise ValueError("clean control line must use target_channel_scenarios=['clean']")
    if is_satellite and not RECOMMENDED_LEO_SCENARIOS.issubset(scenarios):
        raise ValueError("target_channel_scenarios must include recommended simplified LEO scenarios")
    threshold_scope = str(payload.get("threshold_scope") or payload.get("threshold_fit_scope") or "")
    if "unknown_query" in threshold_scope and "no_unknown_query" not in threshold_scope:
        raise ValueError("unknown query must not be used for threshold fitting")

    checked = dict(payload)
    checked.update(
        {
            "stage": normalized_stage,
            "cvs_extension": True,
            "stage2_protocol_valid": True,
            "rs_rt_disjoint": True,
            "rt_cardinality": len(target_receivers),
            "y_old_y_new_disjoint": not bool(old & new),
            "y_unknown_disjoint_from_old_new": not bool(unknown & (old | new)),
            "stage2_seen_new_identity_allowed": normalized_stage == "Stage2-C",
            "unknown_rejection_enabled": unknown_rejection_enabled,
            "is_deployment_primary": bool(is_satellite),
            "is_clean_control": bool(is_clean),
            "unknown_query_used_for_threshold": False,
        }
    )
    return checked
