#!/usr/bin/env python
"""Emit a compact current-decision view of the Stage2 optimizer state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from optimizer_workflow_lib import load_json_compat, write_json


DEFAULT_STAGE2_STATE = Path("automation_reports") / "CV-SincNet" / "stage2_optimizer_state.json"

CURRENT_DECISION_KEYS = (
    "last_updated_local",
    "allowed_hard_launch_blockers",
    "non_blocking_labels",
    "lane_monitor_policy",
    "lane_capacity_policy",
    "idle_lane_execution_policy",
    "training_log_observability_policy",
    "phase1_ground_dg_direction",
    "stage2_sample_protocol",
    "latest_two_lane_monitor_result",
    "latest_monitor_only_result",
    "latest_optimizer_runner_result",
    "latest_controller_review_result",
    "latest_phase1_launch_result",
    "latest_phase2_defer_result",
    "latest_phase2_diagnostic_summary",
    "two_lane_server_landing_policy",
)

AUDIT_ONLY_KEYS = (
    "active_focus",
    "objective_changelog",
    "target_changelog",
    "phase1_ground_dg",
    "phase2_spaceborne_fsl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_json", nargs="?", type=Path, default=DEFAULT_STAGE2_STATE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def byte_len(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def current_view(state: Mapping[str, Any], source_path: Path) -> Dict[str, Any]:
    current = {key: state[key] for key in CURRENT_DECISION_KEYS if key in state}
    audit_only_present = [key for key in AUDIT_ONLY_KEYS if key in state]
    dropped_top_level_keys = sorted(set(state.keys()) - set(current.keys()))
    view: Dict[str, Any] = {
        "schema": "stage2_optimizer_current_state_view_v1",
        "source_path": str(source_path),
        "current_decision_keys": list(current.keys()),
        "audit_only_keys_present": audit_only_present,
        "dropped_top_level_keys": dropped_top_level_keys,
        "state_size_bytes": byte_len(state),
        "current_view_size_bytes": 0,
        "current": current,
    }
    view["current_view_size_bytes"] = byte_len(view)
    return view


def main() -> int:
    args = parse_args()
    state = load_json_compat(args.state_json)
    if not isinstance(state, Mapping):
        raise SystemExit(f"state root must be an object: {args.state_json}")
    payload = current_view(state, args.state_json)
    if args.output:
        write_json(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "source_path": payload["source_path"],
                "current_decision_key_count": len(payload["current_decision_keys"]),
                "audit_only_keys_present": payload["audit_only_keys_present"],
                "state_size_bytes": payload["state_size_bytes"],
                "current_view_size_bytes": payload["current_view_size_bytes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
