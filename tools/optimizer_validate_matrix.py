#!/usr/bin/env python
"""Validate CV-SincNet optimizer candidate matrices before launch."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from optimizer_workflow_lib import item_list, load_json_compat, write_json


REQUIRED_FIELDS = (
    "candidate_id",
    "lane",
    "parent_run",
    "lineage",
    "route_signature",
    "retirement_status",
    "invalidity_status",
    "principle_rejection_ref",
    "experimental_rejection_ref",
    "retirement_evidence_count",
    "retirement_evidence_refs",
    "replacement_reason",
    "hypothesis",
    "control",
    "key_changes",
    "parameters",
    "gpu",
    "estimated_run_path",
    "estimated_log_path",
    "cross_domain_target_metric",
    "satellite_channel_target_metric",
    "allowed_tradeoff",
    "must_not_regress_floor",
    "comparability_status",
    "expected_failure_signals",
    "fallback_or_alternative",
    "exact_command",
    "launchability_status",
)

FED_HARD_CONSTRAINTS = {
    "--wisig_train_ratio": "0.1",
    "--epochs": "200",
    "--fl_rounds": "200",
    "--fl_client_key": "receiver",
}
FED_HARD_CONSTRAINT_FIELDS = {
    "--wisig_train_ratio": "wisig_train_ratio",
    "--epochs": "epochs",
    "--fl_rounds": "fl_rounds",
    "--fl_client_key": "fl_client_key",
}

DEFAULT_STAGE2_SAMPLE_PROTOCOL = {
    "old_tx_ids": [0, 1, 2, 3, 4, 5],
    "cen51_train_receiver_ids": [0, 1, 2, 3, 4, 5, 6],
    "recommended_k_shot_anchors": [1, 2, 5, 10, 15, 20, 50],
    "few_shot_upper_bound": 20,
}

STAGE2_NON_LAUNCHABLE_STATUSES = {
    "local_patch_required",
    "non_launch_diagnostic",
    "retired_route",
    "replaced_by_retirement_policy",
}
DEFERRED_LAUNCHABILITY_TOKENS = (
    "deferred_retry_",
    "monitor_only_continue",
    "user_required_safety_stop",
)
ROUTE_DUPLICATION_TOKENS = (
    "route_duplication",
    "route_duplicate",
    "duplicate_route",
)
PHASE1_SAFE_SSDG_TOKENS = (
    "safe-ssdg-cvs-r01",
    "safe_ssdg_cvs_r01",
    "safe_ssdg",
    "use_safe_ssdg_cvs",
)
PHASE1_CEN51_REFRESH_TOKENS = (
    "cen51_refresh_control",
    "cen51_r04_refresh",
    "matched_cen51",
)
PHASE1_LEGACY_META_TOKENS = (
    "meta-ssl-cvs-r04",
    "meta_ssl",
    "phase1_meta_ssl",
)
PHASE1_LEGACY_ALLOWED_TOKENS = (
    "legacy",
    "diagnostic",
    "protocol_regression",
    "negative_evidence",
    "negative evidence",
    "explicit_user_reopened",
)
PHASE1_PAIC_ROUTE_FAMILY = "CVS-SAT-PAIC"
PHASE1_PAIC_SAT_VIEW_SCHEDULE = (
    "1@0.30:mixed_orbit;"
    "41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;"
    "91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp"
)
PHASE1_PAIC_CONTROL_EXEMPTION_TOKENS = (
    "explicit_cen51_refresh_control",
    "cen51_refresh_control",
    "matched_cen51_r04_control",
    "phase1_star_ground_aug_control_exemption",
)
PHASE1_PAIC_REQUIRED_DIRECT_FLAGS = (
    "--use_concat_sat_channel_aug",
    "--concat_sat_ce_only",
    "--sat_view_schedule",
    "--use_sat_consistency",
)
PHASE1_SAFE_SSDG_EXECUTABLE_ENTRYPOINT_TOKENS = (
    "-m ssdg.train_ssdg",
    "ssdg/train_ssdg.py",
    "code/ssdg/train_ssdg.py",
    "run_phase1_safe_ssdg_candidate",
    "launch_phase1_safe_ssdg",
)
PHASE1_SAFE_SSDG_LOCAL_VERIFY_DEFER_TOKENS = (
    "deferred_retry_local_verify",
    "phase1_training_deferred_local_verify",
    "pending_local_safe_ssdg_launcher_verify",
    "safe_ssdg_launcher_schema_local_verify_required",
    "launcher_schema_required",
    "deferred local verification",
    "# deferred",
)
PHASE1_SAFE_SSDG_INVALID_CLI_TOKENS = (
    "--use_safe_ssdg_cvs",
    "use_safe_ssdg_cvs",
)

CANONICAL_STAGE2_GPU_COUNT = 8
CANONICAL_STAGE2_SLOTS = tuple("ABCDEFGH")
DEFAULT_STAGE2_STATE = Path("automation_reports") / "CV-SincNet" / "stage2_optimizer_state.json"
REQUIRED_ONBOARD_ADAPTATION_BUNDLE_TOKENS = (
    "weibull_evt",
    "target_adapter",
    "pseudo_unknown_energy",
    "seen_new_evidence_gate",
    "seen_new_anchor_gate",
    "siamese_verifier",
    "accepted_only_online_update",
    "stage2_receiver_domain",
)
K_SHOT_FIELDS = (
    "k_shot",
    "k",
    "shots",
    "target_old_support_per_tx",
    "target_old_k",
)
RUN_ID_ENV_RE = re.compile(r"(?<![\w-])RUN_ID=([A-Za-z0-9_.:-]+)")
LAUNCHER_RUN_ID_DEFAULT_RE = re.compile(
    r'^\s*RUN_ID=(?:"\$\{RUN_ID:-([^}]+)\}"|\$\{RUN_ID:-([^}]+)\}|\$\{RUN_ID:-([^}]+)\})',
    re.MULTILINE,
)
PHASE2_LOCAL_PATCH_DEFAULT_RE = re.compile(
    r'^\s*PHASE2_LOCAL_PATCH_REQUIRED=(?:"\$\{PHASE2_LOCAL_PATCH_REQUIRED:-([01])\}"|\$\{PHASE2_LOCAL_PATCH_REQUIRED:-([01])\})',
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_matrix_json", type=Path)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launcher", type=Path, help="Optional local launcher script to bind-check against matrix n607_run_id.")
    parser.add_argument(
        "--repair-launcher-identity",
        action="store_true",
        help="Repair deterministic launcher identity drift before validating --launcher.",
    )
    parser.add_argument("--stage2-state", type=Path, help="Optional stage2 optimizer state with retired route signatures.")
    parser.add_argument("--ignore-retired-routes", action="store_true", help="Skip retired-route validation for legacy audits.")
    return parser.parse_args()


def expected_count_for_matrix(root: Any, cli_value: Optional[int]) -> int:
    if cli_value is not None:
        return int(cli_value)
    if isinstance(root, Mapping) and root.get("expected_count") not in (None, "", []):
        try:
            return int(root.get("expected_count"))
        except (TypeError, ValueError):
            pass
    return 8


def category_for(item: Mapping[str, Any]) -> str:
    value = str(item.get("category") or item.get("type") or "").lower()
    if value in {"conservative", "robust"}:
        return "conservative"
    if value == "aggressive":
        return "aggressive"
    if value in {"old_retention", "unknown_boundary", "seen_new_rescue"}:
        return value
    if value in {"support_quality", "prototype_geometry"}:
        return value
    if value in {"query_free_background_risk", "unknown_separability"}:
        return value
    if value in {"oldqual_oldrisk_fusion", "rollback_calibration"}:
        return value
    if value in {"rollback_safe_retention", "deployment_gate_rescue"}:
        return value
    candidate = str(item.get("candidate_id") or "")
    if re.search(r"_R\d+", candidate):
        return "conservative"
    if re.search(r"_A\d+", candidate):
        return "aggressive"
    return "unknown"


def command_text(item: Mapping[str, Any]) -> str:
    exact_command = str(item.get("exact_command") or "").split("#", 1)[0]
    parts = [exact_command]
    params = item.get("parameters")
    if isinstance(params, Mapping):
        for key, value in params.items():
            parts.append(f"{key} {value}")
    return " ".join(parts)


def _constraint_value_matches(actual: Any, expected: str) -> bool:
    if actual in (None, "", []):
        return False
    try:
        return abs(float(actual) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return normalized_status(actual) == normalized_status(expected)


def fed_constraint_issues(item: Mapping[str, Any]) -> List[str]:
    if item.get("lane") != "federated_vmb":
        return []
    text = command_text(item)
    issues: List[str] = []
    for flag, expected in FED_HARD_CONSTRAINTS.items():
        field = FED_HARD_CONSTRAINT_FIELDS.get(flag)
        if field and _constraint_value_matches(item.get(field), expected):
            continue
        pattern = re.compile(rf"{re.escape(flag)}(?:=|\s+){re.escape(expected)}(?:\s|$)")
        if not pattern.search(text):
            issues.append(f"missing_or_mismatched_{flag.lstrip('-')}")
    return issues


def is_paic_row(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    return normalized_status(item.get("route_family")) == "cvs-sat-paic" or "cvs-sat-paic" in text or "paic" in text


def paic_required_field_issues(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not is_paic_row(item):
        return []
    cid = str(item.get("candidate_id") or "UNKNOWN")
    issues: List[Dict[str, Any]] = []
    if normalized_status(item.get("clean_view_role")) != "control_only":
        issues.append(
            {
                "candidate_id": cid,
                "issue": "paic_clean_view_role_must_be_control_only",
                "clean_view_role": item.get("clean_view_role"),
            }
        )
    if item.get("evidence_level") in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "paic_missing_evidence_level"})
    if item.get("deployment_success_claim_allowed") in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "paic_missing_deployment_success_claim_guard"})
    elif is_true_like(item.get("deployment_success_claim_allowed")):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "paic_deployment_success_claim_must_not_be_preallowed",
                "deployment_success_claim_allowed": item.get("deployment_success_claim_allowed"),
            }
        )
    if is_non_launchable_status(item.get("launchability_status")) and item.get("non_launch_reason") in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "paic_non_launchable_rows_require_non_launch_reason"})

    group = normalized_status(item.get("paic_matrix_group"))
    if group == "federated":
        if not is_true_like(item.get("fl_baseline_view_ce_only")) and normalized_status(item.get("candidate_id")) not in {"f0_fsdg49_anchor"}:
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "paic_fl_requires_ce_only_baseline_view",
                    "fl_baseline_view_ce_only": item.get("fl_baseline_view_ce_only"),
                }
            )
        try:
            ce_weight = float(item.get("fl_baseline_view_ce_weight"))
        except (TypeError, ValueError):
            ce_weight = float("nan")
        if normalized_status(item.get("candidate_id")) != "f0_fsdg49_anchor" and not (ce_weight > 0.0):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "paic_fl_requires_positive_baseline_view_ce_weight",
                    "fl_baseline_view_ce_weight": item.get("fl_baseline_view_ce_weight"),
                }
            )
        if item.get("fl_client_key") not in (None, "", []) and normalized_status(item.get("fl_client_key")) != "receiver":
            issues.append({"candidate_id": cid, "issue": "paic_fl_client_key_must_be_receiver", "fl_client_key": item.get("fl_client_key")})
    if group == "stage2":
        target_view = normalized_status(item.get("target_channel_view"))
        if "satellite" not in target_view and "leo" not in target_view:
            issues.append({"candidate_id": cid, "issue": "paic_stage2_target_channel_view_must_be_satellite_leo"})
        if not is_true_like(item.get("unknown_query_eval_only")):
            issues.append({"candidate_id": cid, "issue": "paic_stage2_unknown_query_must_be_eval_only"})
        if not is_true_like(item.get("target_new_query_not_threshold_fit")):
            issues.append({"candidate_id": cid, "issue": "paic_stage2_target_new_query_must_not_fit_threshold"})
        for field in ("receiver_disjoint_verified", "tx_split_disjoint_verified", "support_query_split_verified"):
            if item.get(field) not in (None, "", []) and not is_true_like(item.get(field)):
                issues.append({"candidate_id": cid, "issue": f"paic_stage2_{field}_must_be_true", field: item.get(field)})
    return issues


def normalize_gpu(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"GPU[0-7]", text):
        return text
    if re.fullmatch(r"[0-7]", text):
        return f"GPU{text}"
    return None


def canonical_stage2_slot_issues(items: List[Mapping[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    if expected_count != CANONICAL_STAGE2_GPU_COUNT * len(CANONICAL_STAGE2_SLOTS):
        return []

    issues: List[Dict[str, Any]] = []
    expected_by_gpu = {
        f"GPU{gpu_idx}": {f"GPU{gpu_idx}/{slot}" for slot in CANONICAL_STAGE2_SLOTS}
        for gpu_idx in range(CANONICAL_STAGE2_GPU_COUNT)
    }
    actual_by_gpu: Dict[str, List[str]] = {gpu: [] for gpu in expected_by_gpu}

    for item in items:
        cid = str(item.get("candidate_id") or "UNKNOWN")
        gpu = normalize_gpu(item.get("gpu"))
        slot = str(item.get("slot") or "").strip().upper()
        if not gpu:
            issues.append({"candidate_id": cid, "issue": "invalid_gpu_for_canonical_slot_matrix", "gpu": item.get("gpu")})
            continue
        actual_by_gpu.setdefault(gpu, []).append(slot)
        if slot not in expected_by_gpu.get(gpu, set()):
            issues.append({"candidate_id": cid, "issue": "invalid_or_mismatched_slot", "gpu": gpu, "slot": slot})
            continue
        match = re.search(r"_GPU([0-7])_([A-H])(?:_|$)", cid.upper())
        if match:
            cid_slot = f"GPU{match.group(1)}/{match.group(2)}"
            if cid_slot != slot:
                issues.append({"candidate_id": cid, "issue": "candidate_id_slot_mismatch", "candidate_id_slot": cid_slot, "slot": slot})

    for gpu, expected_slots in expected_by_gpu.items():
        actual_slots = actual_by_gpu.get(gpu, [])
        actual_set = {slot for slot in actual_slots if slot}
        missing = sorted(expected_slots - actual_set)
        duplicates = sorted({slot for slot in actual_slots if slot and actual_slots.count(slot) > 1})
        if missing or duplicates or len(actual_slots) != len(CANONICAL_STAGE2_SLOTS):
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "canonical_gpu_slot_coverage_failed",
                    "gpu": gpu,
                    "expected_slots": sorted(expected_slots),
                    "actual_slots": actual_slots,
                    "missing_slots": missing,
                    "duplicate_slots": duplicates,
                }
            )
    return issues


def find_default_stage2_state(matrix_path: Path) -> Optional[Path]:
    cwd_candidate = Path.cwd() / DEFAULT_STAGE2_STATE
    if cwd_candidate.exists():
        return cwd_candidate
    try:
        resolved = matrix_path.resolve()
    except OSError:
        resolved = matrix_path
    for parent in [resolved.parent, *resolved.parents]:
        candidate = parent / DEFAULT_STAGE2_STATE
        if candidate.exists():
            return candidate
    return None


def normalized_status(value: Any) -> str:
    return str(value or "").strip().lower()


def is_stage2_non_launchable_status(value: Any) -> bool:
    status = normalized_status(value)
    return any(token in status for token in STAGE2_NON_LAUNCHABLE_STATUSES)


def is_deferred_launchability_status(value: Any) -> bool:
    status = normalized_status(value)
    return any(token in status for token in DEFERRED_LAUNCHABILITY_TOKENS)


def is_non_launchable_status(value: Any) -> bool:
    return is_stage2_non_launchable_status(value)


def is_launchable_status(value: Any) -> bool:
    status = normalized_status(value)
    if not status or is_non_launchable_status(status) or is_deferred_launchability_status(status):
        return False
    return "launchable" in status or "ready" in status or "candidate" in status


def lane_label(item: Mapping[str, Any]) -> str:
    if is_phase1_row(item):
        return "phase1_ground_dg"
    if is_phase2_row(item):
        return "phase2_spaceborne_fsl"
    return str(item.get("lane") or "unknown")


def route_duplication_repair_flag(item: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(item.get(field) or "").lower()
        for field in (
            "launchability_status",
            "defer_reason",
            "replacement_reason",
            "route_signature_family",
            "route_signature",
        )
    )
    return any(token in text for token in ROUTE_DUPLICATION_TOKENS)


def empty_lane_summary() -> Dict[str, Any]:
    return {
        "total": 0,
        "launchable": 0,
        "deferred": 0,
        "non_launchable": 0,
        "local_patch_required": 0,
        "route_duplication_repair": 0,
        "unknown_status": 0,
        "statuses": {},
        "runner_readiness": "LANE_NO_CANDIDATES",
    }


def matrix_launchability_summary(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
    by_lane: Dict[str, Dict[str, Any]] = {}
    totals = empty_lane_summary()
    totals["runner_readiness"] = "NO_CANDIDATES"

    for item in items:
        lane = lane_label(item)
        lane_summary = by_lane.setdefault(lane, empty_lane_summary())
        status = normalized_status(item.get("launchability_status"))
        for summary in (lane_summary, totals):
            summary["total"] += 1
            summary["statuses"][status or "missing"] = summary["statuses"].get(status or "missing", 0) + 1
            if is_launchable_status(status):
                summary["launchable"] += 1
            elif is_deferred_launchability_status(status):
                summary["deferred"] += 1
            elif is_non_launchable_status(status):
                summary["non_launchable"] += 1
            else:
                summary["unknown_status"] += 1
            if "local_patch_required" in status:
                summary["local_patch_required"] += 1
            if route_duplication_repair_flag(item):
                summary["route_duplication_repair"] += 1

    def readiness(summary: Mapping[str, Any], prefix: str) -> str:
        if summary["total"] == 0:
            return f"{prefix}_NO_CANDIDATES"
        if summary["launchable"] > 0:
            return f"{prefix}_HAS_LAUNCHABLE_ROWS"
        if summary["local_patch_required"] > 0:
            return f"{prefix}_LOCAL_PATCH_REQUIRED_NO_LAUNCHABLE_ROWS"
        if summary["deferred"] > 0 and summary["deferred"] == summary["total"]:
            return f"{prefix}_ALL_ROWS_DEFERRED"
        if summary["non_launchable"] > 0:
            return f"{prefix}_NON_LAUNCHABLE_ONLY"
        return f"{prefix}_NO_LAUNCHABLE_ROWS"

    for lane, summary in by_lane.items():
        summary["runner_readiness"] = readiness(summary, "LANE")
    totals["runner_readiness"] = readiness(totals, "MATRIX")
    return {"total": totals, "by_lane": by_lane}


def candidate_search_text(item: Mapping[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True).lower()


def candidate_value_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(candidate_value_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(candidate_value_text(v) for v in value)
    return str(value or "").lower()


def parse_id_set(value: Any) -> set[int]:
    if value in (None, ""):
        return set()
    if isinstance(value, (list, tuple, set)):
        parsed: set[int] = set()
        for part in value:
            parsed.update(parse_id_set(part))
        return parsed
    if isinstance(value, Mapping):
        parsed = set()
        for part in value.values():
            parsed.update(parse_id_set(part))
        return parsed
    return {int(match.group(0)) for match in re.finditer(r"-?\d+", str(value))}


def normalize_receiver_token(value: Any) -> Optional[str]:
    text = str(value or "").strip().strip("[](){}'\"")
    if not text:
        return None
    text = text.lower()
    if re.fullmatch(r"rx\d+", text):
        return text
    if re.fullmatch(r"\d+", text):
        return f"rx{int(text)}"
    return text


def parse_receiver_set(value: Any) -> set[str]:
    """Parse receiver IDs while preserving WiSig labels such as 3-19."""

    if value in (None, ""):
        return set()
    if isinstance(value, (list, tuple, set)):
        parsed: set[str] = set()
        for part in value:
            parsed.update(parse_receiver_set(part))
        return parsed
    if isinstance(value, Mapping):
        parsed = set()
        for part in value.values():
            parsed.update(parse_receiver_set(part))
        return parsed
    parsed = set()
    for raw in re.split(r"[,\s;]+", str(value)):
        token = normalize_receiver_token(raw)
        if token:
            parsed.add(token)
    return parsed


def normalize_tx_label_token(value: Any) -> Optional[str]:
    text = str(value or "").strip().strip("[](){}'\"")
    if re.fullmatch(r"\d+-\d+", text):
        return text
    return None


def parse_tx_label_set(value: Any) -> set[str]:
    """Parse exact WiSig transmitter labels such as 14-10 without splitting on '-'."""

    if value in (None, ""):
        return set()
    if isinstance(value, (list, tuple, set)):
        parsed: set[str] = set()
        for part in value:
            parsed.update(parse_tx_label_set(part))
        return parsed
    if isinstance(value, Mapping):
        parsed = set()
        for part in value.values():
            parsed.update(parse_tx_label_set(part))
        return parsed
    parsed = set()
    for raw in re.split(r"[,\s;]+", str(value)):
        token = normalize_tx_label_token(raw)
        if token:
            parsed.add(token)
    return parsed


def numeric_token_set(value: Any) -> set[int]:
    if value in (None, "") or isinstance(value, Mapping):
        return set()
    if isinstance(value, (list, tuple, set)):
        parsed: set[int] = set()
        for part in value:
            parsed.update(numeric_token_set(part))
        return parsed
    parsed = set()
    for raw in re.split(r"[,\s;]+", str(value)):
        token = raw.strip().strip("[](){}'\"")
        if re.fullmatch(r"\d+", token):
            parsed.add(int(token))
    return parsed


def wisig_candidate_pool(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    pool = protocol.get("confirmed_wisig_candidate_pool")
    if isinstance(pool, Mapping):
        return pool
    pool = protocol.get("confirmed_phase2_candidate_pool")
    if isinstance(pool, Mapping):
        return pool
    return {}


def wisig_old_tx_labels(protocol: Mapping[str, Any]) -> set[str]:
    values: List[Any] = []
    pool = wisig_candidate_pool(protocol)
    values.extend(
        [
            pool.get("old_tx_labels"),
            pool.get("y_old_labels"),
            protocol.get("old_tx_labels"),
        ]
    )
    tx_policy = protocol.get("tx_split_policy")
    if isinstance(tx_policy, Mapping):
        values.extend([tx_policy.get("old_tx_labels"), tx_policy.get("y_old_labels")])
    labels: set[str] = set()
    for value in values:
        labels.update(parse_tx_label_set(value))
    return labels


def looks_like_wisig_manytx_row(item: Mapping[str, Any]) -> bool:
    if item.get("manytx_target_rx_index") not in (None, "", []):
        return True
    for field in ("target_new_tx_labels", "unknown_tx_labels", "new_wisig_pkl", "exact_command"):
        if "manytx" in str(item.get(field) or "").lower():
            return True
    params = item.get("parameters")
    return isinstance(params, Mapping) and "manytx" in candidate_value_text(params)


def tx_labels_for_role(item: Mapping[str, Any], role: str) -> tuple[set[str], Any]:
    if role == "target_new":
        value = first_present(item, ["target_new_tx_labels", "target_new_tx_ids", "new_tx_ids"])
    else:
        value = first_present(item, ["unknown_tx_labels", "unknown_tx_ids", "unseen_new_or_unknown_tx_ids"])
    return parse_tx_label_set(value), value


def first_present(item: Mapping[str, Any], fields: List[str]) -> Any:
    for field in fields:
        value = item.get(field)
        if value not in (None, "", []):
            return value
    return None


def normalize_stage2_sample_protocol(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: Dict[str, Any] = dict(protocol)
    tx_policy = protocol.get("tx_split_policy")
    if "old_tx_ids" not in normalized and isinstance(tx_policy, Mapping):
        old_tx_ids = tx_policy.get("old_tx_ids")
        if old_tx_ids not in (None, "", []):
            normalized["old_tx_ids"] = old_tx_ids
    receiver_policy = protocol.get("receiver_split_policy")
    if "cen51_train_receiver_ids" not in normalized and isinstance(receiver_policy, Mapping):
        train_receiver_ids = receiver_policy.get("cen51_train_receiver_ids")
        if train_receiver_ids not in (None, "", []):
            normalized["cen51_train_receiver_ids"] = train_receiver_ids
    if isinstance(receiver_policy, Mapping):
        for source_key, dest_key in (
            ("source_receiver_labels", "source_receiver_labels"),
            ("cen51_train_receiver_labels", "cen51_train_receiver_labels"),
        ):
            if dest_key not in normalized:
                value = receiver_policy.get(source_key)
                if value not in (None, "", []):
                    normalized[dest_key] = value
    pool = protocol.get("confirmed_wisig_candidate_pool")
    if isinstance(pool, Mapping) and "confirmed_wisig_candidate_pool" not in normalized:
        normalized["confirmed_wisig_candidate_pool"] = pool
    confirmed_pool = protocol.get("confirmed_phase2_candidate_pool")
    if isinstance(confirmed_pool, Mapping) and "confirmed_phase2_candidate_pool" not in normalized:
        normalized["confirmed_phase2_candidate_pool"] = confirmed_pool
    return normalized


def route_retirement_issues(item: Mapping[str, Any], retirement_policy: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not retirement_policy:
        return []
    allowed = {
        normalized_status(value)
        for value in retirement_policy.get("allowed_non_launch_statuses", [])
        if str(value or "").strip()
    }
    if not allowed:
        allowed = {"non_launch_diagnostic", "retired_route", "local_patch_required", "replaced_by_retirement_policy"}
    status = normalized_status(item.get("launchability_status"))
    text = candidate_search_text(item)
    issues: List[Dict[str, Any]] = []
    for rule in retirement_policy.get("retired_route_signatures", []):
        if not isinstance(rule, Mapping):
            continue
        if normalized_status(rule.get("status")) != "retired":
            continue
        match_all = [str(token).lower() for token in rule.get("match_all", []) if str(token or "").strip()]
        match_any = [str(token).lower() for token in rule.get("match_any", []) if str(token or "").strip()]
        if match_all and not all(token in text for token in match_all):
            continue
        if match_any and not any(token in text for token in match_any):
            continue
        if status in allowed:
            continue
        issues.append(
            {
                "issue": "retired_route_signature",
                "route_signature": rule.get("signature_id") or "UNKNOWN",
                "blocker": retirement_policy.get("candidate_launch_blocker_code") or "ROUTE_RETIRED_THREE_STRIKES",
                "evidence_count": rule.get("evidence_count"),
                "allowed_non_launch_statuses": sorted(allowed),
            }
        )
    return issues


def route_invalidity_issues(item: Mapping[str, Any], invalidity_ledger: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not invalidity_ledger:
        return []
    status = normalized_status(item.get("launchability_status"))
    invalidity_status = normalized_status(item.get("invalidity_status"))
    route_signature = str(item.get("route_signature") or "").strip()
    if not route_signature:
        return []
    allowed = {"non_launch_diagnostic", "retired_route", "local_patch_required", "replaced_by_retirement_policy"}
    issues: List[Dict[str, Any]] = []
    for entry in invalidity_ledger.get("entries", []):
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("route_signature") or "").strip() != route_signature:
            continue
        if normalized_status(entry.get("status")) != "principle_and_experiment_rejected":
            continue
        if status in allowed:
            continue
        if invalidity_status == "reopen_requires_new_mechanism_evidence":
            continue
        issues.append(
            {
                "issue": "invalid_route_signature",
                "route_signature": route_signature,
                "blocker": "PRINCIPLE_AND_EXPERIMENT_REJECTED",
                "exploration_count": entry.get("exploration_count"),
                "reopen_policy": entry.get("reopen_policy"),
            }
        )
    return issues


def is_phase2_row(item: Mapping[str, Any]) -> bool:
    lane = normalized_status(item.get("lane"))
    phase_axis = normalized_status(item.get("phase_axis"))
    stage2_axis = normalized_status(item.get("stage2_axis"))
    scenario = normalized_status(item.get("stage2_scenario"))
    return (
        lane == "phase2_spaceborne_fsl"
        or "phase2" in phase_axis
        or "stage2" in stage2_axis
        or "stage2-" in scenario
    )


def load_stage2_sample_protocol(root_state: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(root_state, Mapping):
        return DEFAULT_STAGE2_SAMPLE_PROTOCOL
    phase2 = root_state.get("phase2_spaceborne_fsl", {})
    candidates = [
        root_state.get("stage2_sample_protocol"),
        phase2.get("sample_protocol") if isinstance(phase2, Mapping) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and normalized_status(candidate.get("status")) == "active":
            return normalize_stage2_sample_protocol(candidate)
    return DEFAULT_STAGE2_SAMPLE_PROTOCOL


def stage2_mode_text(item: Mapping[str, Any]) -> str:
    parts = [
        item.get("stage2_mode"),
        item.get("stage2_scenario"),
        item.get("stage2_axis"),
        item.get("protocol"),
        item.get("route_signature"),
        item.get("hypothesis"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


def is_stage2_b_or_c(mode_text: str) -> bool:
    return (
        "stage2-b" in mode_text
        or "old_label_calibration" in mode_text
        or "stage2-c" in mode_text
        or "old_new_enrollment" in mode_text
        or "seen_new" in mode_text
        or "seen-new" in mode_text
    )


def explicit_k_values(item: Mapping[str, Any]) -> List[int]:
    values: List[int] = []
    for field in K_SHOT_FIELDS:
        value = item.get(field)
        if value in (None, "", []):
            continue
        values.extend(sorted(parse_id_set(value)))
    return values


def is_false_like(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return normalized_status(value) in {"false", "0", "no", "n", "off"}


def is_true_like(value: Any) -> bool:
    return normalized_status(value) in {"true", "1", "yes", "y", "on"}


def is_oa_mse_row(item: Mapping[str, Any]) -> bool:
    text = candidate_value_text(item)
    return (
        normalized_status(item.get("route_family")) == "oa_mse_head"
        or normalized_status(item.get("gate_mode")) == "oa_mse"
        or "oa_mse" in text
        or "oa-mse" in text
        or "mse_lite" in text
        or "mse-lite" in text
        or "mse_subspace" in text
        or "mse-subspace" in text
        or "mse-head" in text
        or "mse_head" in text
    )


def oa_mse_required_field_issues(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not is_oa_mse_row(item):
        return []
    cid = str(item.get("candidate_id") or "UNKNOWN")
    required = [
        "route_family",
        "oa_mse_stage",
        "source_target_fusion_policy",
        "fusion_inputs",
        "threshold_selection_label_scope",
        "unknown_query_eval_only",
        "target_new_query_not_threshold_fit",
        "unknown_FAR_target",
        "model_output_semantics",
        "uncertain_policy",
        "seen_new_evidence_gate_calibration_scope",
        "seen_new_evidence_gate_unknown_query_calibration",
        "score_table_required_columns",
    ]
    issues: List[Dict[str, Any]] = []
    missing = [field for field in required if item.get(field) in (None, "", [])]
    if missing:
        issues.append({"candidate_id": cid, "issue": "oa_mse_missing_required_fields", "fields": missing})
    stage_text = normalized_status(
        " ".join(
            candidate_value_text(item.get(field))
            for field in ("stage2_mode", "oa_mse_stage", "target_visibility", "update_module")
        )
    )
    try:
        target_new_k_value = int(float(item.get("target_new_k", item.get("target_new_support_per_tx", -1))))
    except (TypeError, ValueError):
        target_new_k_value = -1
    target_new_ids_text = normalized_status(
        " ".join(
            candidate_value_text(item.get(field))
            for field in ("target_new_tx_ids", "target_new_tx_labels", "new_tx_ids", "target_new_tx_indices")
        )
    )
    cid_upper = str(cid).upper()
    explicit_old_unknown_only = (
        "old_unknown_only" in normalized_status(item.get("k_shot_interpretation"))
        or any(token in cid_upper for token in ("OLDUNK", "BGTRAIN", "RETOLD", "OLDFIRST", "OLDRELAX", "OLDGEOM", "OLDCONF", "OLDBUDGET", "OLDQUAL", "OLDRISK", "OLDFUSE", "ROLLSAFE"))
        or normalized_status(item.get("target_new_leo_query")) == "not_applicable_old_unknown_only"
    )
    old_unknown_only = explicit_old_unknown_only and target_new_k_value == 0 and not target_new_ids_text
    semantics = normalized_status(item.get("model_output_semantics"))
    required_semantics = ["old", "reject", "uncertain", "defer"] if old_unknown_only else ["old", "seen", "reject", "uncertain", "defer"]
    if not semantics or not all(token in semantics for token in required_semantics):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "oa_mse_output_semantics_must_distinguish_defer_uncertain_reject",
                "model_output_semantics": item.get("model_output_semantics"),
            }
        )
    if old_unknown_only and "seen" in semantics:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "oa_mse_old_unknown_only_must_not_claim_seen_new_output",
                "model_output_semantics": item.get("model_output_semantics"),
            }
        )
    try:
        far_target = float(item.get("unknown_FAR_target"))
    except (TypeError, ValueError):
        far_target = float("nan")
    if not (far_target <= 0.05):
        issues.append({"candidate_id": cid, "issue": "oa_mse_unknown_far_target_must_be_at_most_0p05", "unknown_FAR_target": item.get("unknown_FAR_target")})
    if item.get("seen_new_evidence_gate_unknown_query_calibration") not in (None, "", []):
        if is_true_like(item.get("seen_new_evidence_gate_unknown_query_calibration")):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "seen_new_evidence_gate_must_not_use_unknown_query_calibration",
                    "seen_new_evidence_gate_unknown_query_calibration": item.get("seen_new_evidence_gate_unknown_query_calibration"),
                }
            )
    scope = normalized_status(item.get("seen_new_evidence_gate_calibration_scope"))
    if scope and "unknown_query" in scope:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "seen_new_evidence_gate_scope_must_exclude_unknown_query",
                "seen_new_evidence_gate_calibration_scope": item.get("seen_new_evidence_gate_calibration_scope"),
            }
        )
    is_stage2c = "stage2_c" in stage_text or "stage2-c" in stage_text or "oa_mse_head" in stage_text or "seen_new" in stage_text
    success_bundle = normalized_status(item.get("stage2c_success_metric_bundle"))
    if is_stage2c:
        stage2c_required = [
            "target_old_support_per_tx",
            "target_new_support_per_tx",
            "target_old_k",
            "target_new_k",
            "stage2c_success_metric_bundle",
        ]
        missing_stage2c = [field for field in stage2c_required if item.get(field) in (None, "", [])]
        if missing_stage2c:
            issues.append({"candidate_id": cid, "issue": "oa_mse_stage2c_missing_success_scope_fields", "fields": missing_stage2c})
        for token in ("old_acc", "seen_new_acc", "h_old_new", "unknown_far", "unknown_to_seen_new"):
            if token not in success_bundle:
                issues.append({"candidate_id": cid, "issue": "oa_mse_stage2c_success_metric_bundle_incomplete", "missing_token": token})
        try:
            old_k = int(float(item.get("target_old_k")))
            new_k = int(float(item.get("target_new_k")))
        except (TypeError, ValueError):
            old_k = -1
            new_k = -2
        if old_k != new_k or old_k <= 0:
            issues.append({"candidate_id": cid, "issue": "oa_mse_stage2c_old_new_support_k_must_match", "target_old_k": item.get("target_old_k"), "target_new_k": item.get("target_new_k")})
    elif any(token in success_bundle for token in ("seen_new_acc", "h_old_new", "unknown_to_seen_new")):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "non_stage2c_must_not_claim_seen_new_success_metrics",
                "stage2c_success_metric_bundle": item.get("stage2c_success_metric_bundle"),
            }
        )
    if old_unknown_only:
        for field in (
            "target_new_leo_query",
            "new_support_query_split",
            "seen_new_evidence_gate_calibration_scope",
        ):
            value = normalized_status(item.get(field))
            if value and value != "not_applicable_old_unknown_only" and "target_new" in value:
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "oa_mse_old_unknown_only_target_new_residue",
                        "field": field,
                        "value": item.get(field),
                    }
                )
    old_primary_enabled = is_true_like(item.get("oa_mse_old_primary_gate"))
    old_primary_route = any(token in cid_upper for token in ("OLDFIRST", "OLDRELAX", "OLDGEOM", "OLDCONF", "OLDBUDGET"))
    if old_primary_route and is_true_like(item.get("oa_mse_retention_rescue_gate")) and not old_primary_enabled:
        issues.append({"candidate_id": cid, "issue": "old_primary_route_retention_rescue_requires_old_primary_gate"})
    if old_primary_enabled:
        if not is_true_like(item.get("oa_mse_class_envelope_gate")):
            issues.append({"candidate_id": cid, "issue": "old_primary_gate_requires_class_envelope_gate"})
        for field in (
            "old_primary_require_soft_mixture",
            "old_primary_require_support_knn",
        ):
            if not is_true_like(item.get(field)):
                issues.append({"candidate_id": cid, "issue": "old_primary_gate_missing_required_subgate", "field": field})
        if not is_true_like(item.get("old_primary_require_class_envelope")):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "old_primary_gate_missing_required_subgate",
                    "field": "old_primary_require_class_envelope",
                }
            )
        if is_true_like(item.get("oa_mse_retention_rescue_gate")):
            action = normalized_status(item.get("old_primary_unknown_veto_action"))
            if action not in {"reject", "defer", "uncertain"}:
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "old_primary_unknown_veto_action_required_when_rescue_enabled",
                        "old_primary_unknown_veto_action": item.get("old_primary_unknown_veto_action"),
                    }
                )
            if old_primary_route:
                if not is_true_like(item.get("retention_rescue_candidate_only")):
                    issues.append(
                        {
                            "candidate_id": cid,
                            "issue": "old_primary_route_retention_rescue_must_be_candidate_only",
                        }
                    )
                if not is_true_like(item.get("old_primary_promote_rescue_candidates")):
                    issues.append(
                        {
                            "candidate_id": cid,
                            "issue": "old_primary_route_rescue_candidates_must_be_terminally_promoted",
                        }
                    )
    score_columns = normalized_status(item.get("score_table_required_columns"))
    required_score_tokens = ["candidate_label", "candidate_group", "outcome_code"]
    if not old_unknown_only:
        required_score_tokens.extend(["seen_new_minus_old_score", "seen_new_anchor_similarity", "seen_new_anchor_delta"])
    for token in required_score_tokens:
        if token not in score_columns:
            issues.append({"candidate_id": cid, "issue": "oa_mse_score_table_required_columns_incomplete", "missing_token": token})
    if old_primary_enabled:
        required_old_primary_tokens = [
            "old_primary_consistency_pass",
            "old_primary_unknown_veto",
            "old_primary_blocked_accept",
        ]
        if is_true_like(item.get("oa_mse_retention_rescue_gate")):
            required_old_primary_tokens.extend(["old_primary_rescue_promoted", "old_primary_rescue_blocked"])
        for token in required_old_primary_tokens:
            if token not in score_columns:
                issues.append({"candidate_id": cid, "issue": "oa_mse_score_table_required_columns_incomplete", "missing_token": token})
    return issues


def stage2_sample_protocol_issues(item: Mapping[str, Any], sample_protocol: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not is_phase2_row(item):
        return []
    status = normalized_status(item.get("launchability_status"))
    if is_stage2_non_launchable_status(status):
        return []

    protocol = sample_protocol or DEFAULT_STAGE2_SAMPLE_PROTOCOL
    expected_old_tx = parse_id_set(protocol.get("old_tx_ids") or DEFAULT_STAGE2_SAMPLE_PROTOCOL["old_tx_ids"])
    train_rx_value = first_present(
        protocol,
        [
            "cen51_train_receiver_ids",
            "source_receiver_ids",
            "source_receiver_labels",
            "cen51_train_receiver_labels",
        ],
    )
    train_rx = parse_receiver_set(train_rx_value or DEFAULT_STAGE2_SAMPLE_PROTOCOL["cen51_train_receiver_ids"])

    cid = str(item.get("candidate_id") or "UNKNOWN")
    issues: List[Dict[str, Any]] = []

    clean_view_role = normalized_status(item.get("clean_view_role"))
    if clean_view_role != "control_only":
        issues.append(
            {
                "candidate_id": cid,
                "issue": "stage2_clean_view_role_must_be_control_only",
                "clean_view_role": item.get("clean_view_role"),
            }
        )
    if item.get("evidence_level") in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "stage2_missing_evidence_level"})
    if item.get("deployment_success_claim_allowed") in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "stage2_missing_deployment_success_claim_guard"})
    elif is_true_like(item.get("deployment_success_claim_allowed")):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "stage2_deployment_success_claim_must_not_be_preallowed",
                "deployment_success_claim_allowed": item.get("deployment_success_claim_allowed"),
            }
        )
    for field in ("support_query_split_verified", "receiver_disjoint_verified", "tx_split_disjoint_verified"):
        if item.get(field) not in (None, "", []) and not is_true_like(item.get(field)):
            issues.append({"candidate_id": cid, "issue": f"stage2_{field}_must_be_true", field: item.get(field)})

    target_old_tx = parse_id_set(item.get("target_old_tx_ids"))
    if not target_old_tx:
        issues.append({"candidate_id": cid, "issue": "missing_target_old_tx_ids", "expected_old_tx_ids": sorted(expected_old_tx)})
    elif expected_old_tx and target_old_tx != expected_old_tx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "target_old_tx_ids_must_equal_manysig_old_tx_ids",
                "expected_old_tx_ids": sorted(expected_old_tx),
                "actual_target_old_tx_ids": sorted(target_old_tx),
            }
        )

    source_tx = parse_id_set(item.get("source_tx_ids"))
    if source_tx and expected_old_tx and source_tx != expected_old_tx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "source_tx_ids_inconsistent_with_manysig_old_tx_ids",
                "expected_old_tx_ids": sorted(expected_old_tx),
                "actual_source_tx_ids": sorted(source_tx),
            }
        )

    manytx_row = looks_like_wisig_manytx_row(item)
    target_new_label_tx, target_new_label_value = tx_labels_for_role(item, "target_new")
    unknown_label_tx, unknown_label_value = tx_labels_for_role(item, "unknown")
    target_new_uses_labels = bool(target_new_label_tx)
    unknown_uses_labels = bool(unknown_label_tx)
    try:
        target_new_support_k = int(item.get("target_new_support_per_tx") or item.get("target_new_k") or 0)
    except (TypeError, ValueError):
        target_new_support_k = -1
    old_unknown_only_plan = any(token in str(cid).upper() for token in ("OLDUNK", "BGTRAIN", "RETOLD", "OLDFIRST", "OLDRELAX", "OLDGEOM", "OLDCONF", "OLDBUDGET", "OLDQUAL", "OLDRISK", "OLDFUSE", "ROLLSAFE"))
    old_unknown_only_target = (
        target_new_support_k == 0
        and old_unknown_only_plan
        and not str(item.get("target_new_tx_ids") or item.get("new_tx_ids") or "").strip()
        and not str(item.get("target_new_tx_labels") or "").strip()
    )

    target_new_tx = set() if manytx_row and target_new_uses_labels else parse_id_set(item.get("target_new_tx_ids") or item.get("new_tx_ids"))
    unknown_tx = (
        set()
        if manytx_row and unknown_uses_labels
        else parse_id_set(item.get("unknown_tx_ids") or item.get("unseen_new_or_unknown_tx_ids"))
    )
    if not old_unknown_only_target and not target_new_tx and not target_new_label_tx:
        issues.append({"candidate_id": cid, "issue": "missing_target_new_tx_ids"})
    elif target_new_tx & expected_old_tx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "target_new_tx_ids_overlap_manysig_old_tx_ids",
                "overlap_tx_ids": sorted(target_new_tx & expected_old_tx),
            }
        )
    if not unknown_tx and not unknown_label_tx:
        issues.append({"candidate_id": cid, "issue": "missing_unknown_tx_ids"})
    elif unknown_tx & expected_old_tx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "unknown_tx_ids_overlap_manysig_old_tx_ids",
                "overlap_tx_ids": sorted(unknown_tx & expected_old_tx),
            }
        )
    if not old_unknown_only_target and target_new_tx and unknown_tx and target_new_tx & unknown_tx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "target_new_tx_ids_overlap_unknown_tx_ids",
                "overlap_tx_ids": sorted(target_new_tx & unknown_tx),
            }
        )
    if not old_unknown_only_target and target_new_label_tx and unknown_label_tx and target_new_label_tx & unknown_label_tx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "wisig_manytx_target_new_tx_labels_overlap_unknown_tx_labels",
                "overlap_tx_labels": sorted(target_new_label_tx & unknown_label_tx),
            }
        )

    if manytx_row:
        old_labels = wisig_old_tx_labels(protocol)
        explicit_target_new_labels = item.get("target_new_tx_labels")
        explicit_unknown_labels = item.get("unknown_tx_labels")
        if not old_unknown_only_target and not parse_tx_label_set(explicit_target_new_labels):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "wisig_manytx_target_new_tx_labels_not_resolved",
                    "target_new_tx_labels": explicit_target_new_labels,
                    "required": "exact ManyTx tx_list labels such as 1-16, not rank prose or placeholders",
                }
            )
        if not parse_tx_label_set(explicit_unknown_labels):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "wisig_manytx_unknown_tx_labels_not_resolved",
                    "unknown_tx_labels": explicit_unknown_labels,
                    "required": "exact ManyTx tx_list labels such as 10-1, not rank prose or placeholders",
                }
            )
        target_new_numeric_tokens = numeric_token_set(item.get("target_new_tx_ids") or item.get("new_tx_ids"))
        unknown_numeric_tokens = numeric_token_set(item.get("unknown_tx_ids") or item.get("unseen_new_or_unknown_tx_ids"))
        if (
            not old_unknown_only_target
            and target_new_numeric_tokens
            and not parse_tx_label_set(item.get("target_new_tx_ids") or item.get("new_tx_ids"))
        ):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "wisig_manytx_target_new_tx_ids_must_be_resolved_tx_labels",
                    "numeric_tokens": sorted(target_new_numeric_tokens),
                }
            )
        if unknown_numeric_tokens and not parse_tx_label_set(item.get("unknown_tx_ids") or item.get("unseen_new_or_unknown_tx_ids")):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "wisig_manytx_unknown_tx_ids_must_not_be_synthetic_numeric_ranks",
                    "numeric_tokens": sorted(unknown_numeric_tokens),
                }
            )
        if old_labels:
            target_old_label_overlap = set() if old_unknown_only_target else target_new_label_tx & old_labels
            unknown_old_label_overlap = unknown_label_tx & old_labels
            if target_old_label_overlap:
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "wisig_manytx_target_new_tx_labels_overlap_manysig_old_tx_labels",
                        "overlap_tx_labels": sorted(target_old_label_overlap),
                    }
                )
            if unknown_old_label_overlap:
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "wisig_manytx_unknown_tx_labels_overlap_manysig_old_tx_labels",
                        "overlap_tx_labels": sorted(unknown_old_label_overlap),
                    }
                )

    source_rx_value = first_present(
        item,
        [
            "cen51_train_receiver_ids",
            "source_receiver_ids",
            "cen51_train_rxs",
            "source_rxs",
            "train_receiver_ids",
            "source_receiver_labels",
            "cen51_train_receiver_labels",
        ],
    )
    target_rx_value = first_present(
        item,
        [
            "target_receiver_labels",
            "target_receiver_ids",
            "target_rxs",
            "target_rx_ids",
            "target_old_rxs",
            "target_new_rxs",
            "stage2_target_rxs",
        ],
    )
    source_rx = parse_receiver_set(source_rx_value)
    target_rx = parse_receiver_set(target_rx_value)
    if not source_rx:
        issues.append({"candidate_id": cid, "issue": "missing_cen51_train_receiver_ids", "expected_train_receiver_ids": sorted(train_rx)})
    elif train_rx and source_rx != train_rx:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "cen51_train_receiver_ids_must_match_stage2_protocol",
                "expected_train_receiver_ids": sorted(train_rx),
                "actual_train_receiver_ids": sorted(source_rx),
            }
        )
    if not target_rx:
        issues.append({"candidate_id": cid, "issue": "missing_target_receiver_ids"})
    elif target_rx & (source_rx or train_rx):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "target_receiver_ids_overlap_cen51_train_receivers",
                "overlap_receiver_ids": sorted(target_rx & (source_rx or train_rx)),
            }
        )

    missing_split_fields = []
    if first_present(item, ["target_old_leo_query", "target_old_query", "old_support_query_split"]) in (None, "", []):
        missing_split_fields.append("target_old_leo_query_or_old_support_query_split")
    if first_present(item, ["target_new_leo_query", "target_new_query", "new_support_query_split"]) in (None, "", []):
        missing_split_fields.append("target_new_leo_query_or_new_support_query_split")
    if first_present(item, ["unknown_leo_query", "unknown_query", "unseen_new_leo_query"]) in (None, "", []):
        missing_split_fields.append("unknown_leo_query")
    if missing_split_fields:
        issues.append({"candidate_id": cid, "issue": "missing_target_receiver_query_split_fields", "fields": missing_split_fields})

    mode_text = stage2_mode_text(item)
    recommended_k = parse_id_set(
        protocol.get("recommended_k_shot_anchors")
        or protocol.get("allowed_k_shots")
        or DEFAULT_STAGE2_SAMPLE_PROTOCOL["recommended_k_shot_anchors"]
    )
    try:
        few_shot_upper_bound = int(protocol.get("few_shot_upper_bound") or DEFAULT_STAGE2_SAMPLE_PROTOCOL["few_shot_upper_bound"])
    except (TypeError, ValueError):
        few_shot_upper_bound = DEFAULT_STAGE2_SAMPLE_PROTOCOL["few_shot_upper_bound"]
    k_values = explicit_k_values(item)
    if is_stage2_b_or_c(mode_text):
        if not k_values:
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "missing_k_shot_for_stage2_b_or_c",
                    "recommended_k_shot_anchors": sorted(recommended_k),
                }
            )
        for k_value in k_values:
            if k_value <= 0:
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "k_shot_must_be_positive_integer",
                        "k_shot": k_value,
                        "recommended_k_shot_anchors": sorted(recommended_k),
                    }
                )
            if k_value > few_shot_upper_bound:
                interpretation = normalized_status(item.get("k_shot_interpretation") or item.get("shot_interpretation"))
                if "higher" not in interpretation and "medium" not in interpretation and "saturation" not in interpretation:
                    issues.append(
                        {
                            "candidate_id": cid,
                            "issue": "k_gt_fewshot_bound_must_be_labeled_higher_medium_or_saturation",
                            "k_shot": k_value,
                            "few_shot_upper_bound": few_shot_upper_bound,
                        }
                    )
    target_new_support = first_present(item, ["target_new_leo_support", "target_new_support", "seen_new_support"])
    target_new_support_present = target_new_support not in (None, "", []) and not is_false_like(target_new_support)
    if ("stage2-a" in mode_text or "zero_label" in mode_text) and target_new_support_present:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "stage2_a_must_not_use_target_new_support",
                "target_new_support_field": target_new_support,
            }
        )
    if ("stage2-b" in mode_text or "old_label_calibration" in mode_text) and target_new_support_present:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "stage2_b_must_not_use_target_new_support",
                "target_new_support_field": target_new_support,
            }
        )
    new_support_split = normalized_status(item.get("new_support_query_split"))
    declares_seen_new = "seen-new" in mode_text or "seen_new" in mode_text or "old_new_enrollment" in mode_text
    new_support_available = target_new_support_present or (
        "support=empty" not in new_support_split and ("support=target_new" in new_support_split or "seen-new support" in new_support_split)
    )
    if ("stage2-c" in mode_text or declares_seen_new) and declares_seen_new and not new_support_available:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "stage2_c_seen_new_enrollment_requires_target_new_support",
            }
        )

    threshold_scope = normalized_status(item.get("threshold_selection_label_scope"))
    missing_guard_fields = [
        field
        for field in ("unknown_query_eval_only", "target_new_query_not_threshold_fit", "unknown_FAR_target")
        if item.get(field) in (None, "", [])
    ]
    if missing_guard_fields:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "missing_stage2_threshold_guard_fields",
                "fields": missing_guard_fields,
            }
        )
    if not threshold_scope:
        issues.append({"candidate_id": cid, "issue": "missing_threshold_selection_label_scope"})
    if (
        "target_unknown_query" in threshold_scope
        or "unknown_query_calibration" in threshold_scope
        or ("unknown" in threshold_scope and "eval_only" not in threshold_scope and "evaluation_only" not in threshold_scope)
        or (
            item.get("unknown_query_eval_only") not in (None, "", [])
            and not is_true_like(item.get("unknown_query_eval_only"))
        )
    ):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "unknown_query_must_not_calibrate_thresholds",
                "threshold_selection_label_scope": item.get("threshold_selection_label_scope"),
                "unknown_query_eval_only": item.get("unknown_query_eval_only"),
            }
        )
    target_new_scope = normalized_status(item.get("threshold_selection_label_scope"))
    if "target_new_query" in target_new_scope or (
        item.get("target_new_query_not_threshold_fit") not in (None, "", [])
        and not is_true_like(item.get("target_new_query_not_threshold_fit"))
    ):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "target_new_query_must_not_fit_thresholds",
                "threshold_selection_label_scope": item.get("threshold_selection_label_scope"),
                "target_new_query_not_threshold_fit": item.get("target_new_query_not_threshold_fit"),
            }
        )
    if item.get("unknown_FAR_target") not in (None, "", []):
        try:
            unknown_far_target = float(item.get("unknown_FAR_target"))
        except (TypeError, ValueError):
            unknown_far_target = float("nan")
        if not (unknown_far_target <= 0.05):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "unknown_far_target_must_be_at_most_0p05",
                    "unknown_FAR_target": item.get("unknown_FAR_target"),
                }
            )
    low_compute_required = [
        "onboard_low_compute_training",
        "compute_budget_profile",
        "adapter_trainable_params_cap",
        "max_adapt_steps",
        "old_acc_target",
        "seen_new_acc_target",
        "weibull_evt_required",
        "target_adapter_required",
        "pseudo_unknown_energy_required",
        "seen_new_evidence_gate_required",
        "seen_new_anchor_gate_required",
        "siamese_verifier_required",
        "accepted_only_online_update_required",
        "oa_mse_onboard_adaptation_bundle",
    ]
    missing_low_compute = [field for field in low_compute_required if item.get(field) in (None, "", [])]
    if missing_low_compute:
        issues.append({"candidate_id": cid, "issue": "missing_onboard_low_compute_training_fields", "fields": missing_low_compute})
    for field in (
        "onboard_low_compute_training",
        "weibull_evt_required",
        "target_adapter_required",
        "pseudo_unknown_energy_required",
        "seen_new_evidence_gate_required",
        "seen_new_anchor_gate_required",
        "siamese_verifier_required",
        "accepted_only_online_update_required",
    ):
        if item.get(field) not in (None, "", []) and not is_true_like(item.get(field)):
            issues.append({"candidate_id": cid, "issue": f"{field}_must_be_true", field: item.get(field)})
    bundle = normalized_status(item.get("oa_mse_onboard_adaptation_bundle"))
    if bundle:
        missing_bundle_tokens = [token for token in REQUIRED_ONBOARD_ADAPTATION_BUNDLE_TOKENS if token not in bundle]
        if missing_bundle_tokens:
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "oa_mse_onboard_adaptation_bundle_incomplete",
                    "missing_tokens": missing_bundle_tokens,
                    "oa_mse_onboard_adaptation_bundle": item.get("oa_mse_onboard_adaptation_bundle"),
                }
            )
    if item.get("old_acc_target") not in (None, "", []):
        try:
            old_acc_target = float(item.get("old_acc_target"))
        except (TypeError, ValueError):
            old_acc_target = float("nan")
        if not (old_acc_target >= 0.90):
            issues.append({"candidate_id": cid, "issue": "phase2_old_acc_target_must_be_at_least_0p90", "old_acc_target": item.get("old_acc_target")})
    if item.get("seen_new_acc_target") not in (None, "", []):
        try:
            seen_new_acc_target = float(item.get("seen_new_acc_target"))
        except (TypeError, ValueError):
            seen_new_acc_target = float("nan")
        if not (seen_new_acc_target >= 0.75):
            issues.append({"candidate_id": cid, "issue": "phase2_seen_new_acc_target_must_be_at_least_0p75", "seen_new_acc_target": item.get("seen_new_acc_target")})
    issues.extend(oa_mse_required_field_issues(item))

    target_view = normalized_status(item.get("target_channel_view"))
    if "satellite" not in target_view and "leo" not in target_view:
        issues.append({"candidate_id": cid, "issue": "target_channel_view_must_be_satellite_leo", "target_channel_view": item.get("target_channel_view")})

    base_weight = first_present(item, ["cen51_base_checkpoint_or_config", "cen51_base_weight", "cen51_base_weight_policy"])
    if base_weight in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "missing_cen51_strong_generalization_base_weight"})

    return issues


def is_phase1_row(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    lane = normalized_status(item.get("lane"))
    phase_axis = normalized_status(item.get("phase_axis"))
    return (
        lane == "phase1_ground_dg"
        or "phase1" in phase_axis
        or any(token in text for token in PHASE1_SAFE_SSDG_TOKENS)
        or any(token in text for token in PHASE1_CEN51_REFRESH_TOKENS)
        or any(token in text for token in PHASE1_LEGACY_META_TOKENS)
    )


def is_safe_ssdg_or_cen51_phase1_row(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    return any(token in text for token in PHASE1_SAFE_SSDG_TOKENS + PHASE1_CEN51_REFRESH_TOKENS)


def is_safe_ssdg_phase1_row(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    return any(token in text for token in PHASE1_SAFE_SSDG_TOKENS)


def is_legacy_meta_phase1_row(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    return any(token in text for token in PHASE1_LEGACY_META_TOKENS) and not is_safe_ssdg_or_cen51_phase1_row(item)


def is_allowed_legacy_meta_phase1_row(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    return is_legacy_meta_phase1_row(item) and any(token in text for token in PHASE1_LEGACY_ALLOWED_TOKENS)


def is_protocol_only_phase1_row(item: Mapping[str, Any]) -> bool:
    status = normalized_status(item.get("launchability_status"))
    route_signature = normalized_status(item.get("route_signature"))
    ground_scope = normalized_status(item.get("ground_dg_claim_scope"))
    text = candidate_search_text(item)
    protocol_markers = (
        "protocol_regression",
        "protocol_smoke",
        "protocol_stress",
        "protocol_check_only",
        "--meta_ssl_protocol_check_only",
    )
    explicit_protocol_status = any(marker in status for marker in protocol_markers)
    explicit_protocol_route = any(marker in route_signature for marker in protocol_markers)
    explicit_protocol_scope = any(marker in ground_scope for marker in protocol_markers) or "regression_only" in ground_scope
    explicit_protocol_command = "--meta_ssl_protocol_check_only" in text
    return explicit_protocol_status or explicit_protocol_route or explicit_protocol_scope or explicit_protocol_command


def has_phase1_training_command(item: Mapping[str, Any]) -> bool:
    text = candidate_search_text(item)
    safe_ssdg_command_like = any(token in text for token in PHASE1_SAFE_SSDG_EXECUTABLE_ENTRYPOINT_TOKENS)
    legacy_meta_command_like = (
        "launch_phase1_meta_ssl" in text
        or "phase1_meta_ssl" in text
        or "use_meta_ssl_cvs" in text
    )
    command_like = safe_ssdg_command_like or legacy_meta_command_like
    epoch_like = "--epochs 200" in text or "\"epochs\": 200" in text or "'epochs': 200" in text or "full_200e" in text
    return command_like and epoch_like and not is_protocol_only_phase1_row(item)


def truthy_candidate_value(value: Any) -> bool:
    if value is True:
        return True
    return normalized_status(value) in {"true", "1", "yes", "y"}


def candidate_or_parameter_value(item: Mapping[str, Any], field: str) -> Any:
    value = item.get(field)
    if value not in (None, "", []):
        return value
    params = item.get("parameters")
    if isinstance(params, Mapping):
        return params.get(field)
    return None


def truthy_candidate_or_parameter(item: Mapping[str, Any], field: str) -> bool:
    return truthy_candidate_value(candidate_or_parameter_value(item, field))


def phase1_paic_control_exemption(item: Mapping[str, Any]) -> bool:
    policy = candidate_value_text(
        first_present(
            item,
            [
                "phase1_star_ground_aug_policy",
                "phase1_star_ground_aug_exemption",
                "control",
                "route_family",
                "candidate_id",
            ],
        )
    )
    if not any(token in policy for token in PHASE1_PAIC_CONTROL_EXEMPTION_TOKENS):
        return False
    return any(token in candidate_search_text(item) for token in PHASE1_CEN51_REFRESH_TOKENS)


def phase1_paic_schedule_is_canonical(value: Any) -> bool:
    text = candidate_value_text(value)
    required_tokens = (
        "1@0.30",
        "41@0.60",
        "91@0.80",
        "mixed_orbit",
        "low_elev_leo",
        "rain_leo",
        "storm_mp",
    )
    return all(token in text for token in required_tokens)


def phase1_star_ground_aug_default_issues(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not is_phase1_row(item):
        return []
    if is_legacy_meta_phase1_row(item) or phase1_paic_control_exemption(item):
        return []
    if not is_safe_ssdg_or_cen51_phase1_row(item):
        return []

    cid = str(item.get("candidate_id") or "UNKNOWN")
    issues: List[Dict[str, Any]] = []
    text = candidate_search_text(item)
    route_family = first_present(
        item,
        [
            "phase1_star_ground_aug_route_family",
            "star_ground_aug_route_family",
            "satellite_aug_route_family",
        ],
    )
    mode = first_present(
        item,
        [
            "phase1_star_ground_aug_mode",
            "star_ground_aug_mode",
            "satellite_aug_mode",
        ],
    )
    schedule = first_present(item, ["sat_view_schedule", "phase1_sat_view_schedule", "star_ground_sat_view_schedule"])
    axis = first_present(item, ["star_ground_aug_exploration_axis", "phase1_star_ground_aug_exploration_axis"])

    if not (
        truthy_candidate_or_parameter(item, "phase1_star_ground_aug_default_enabled")
        or truthy_candidate_or_parameter(item, "use_concat_sat_channel_aug")
    ):
        issues.append({"candidate_id": cid, "issue": "phase1_star_ground_aug_default_required"})

    if "paic" not in candidate_value_text(route_family) and "cvs-sat-paic" not in text:
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_star_ground_aug_requires_paic_route_family",
                "required": PHASE1_PAIC_ROUTE_FAMILY,
            }
        )

    if "concat_sat_ce_only" not in candidate_value_text(mode) and not truthy_candidate_or_parameter(item, "concat_sat_ce_only"):
        issues.append({"candidate_id": cid, "issue": "phase1_star_ground_aug_requires_concat_ce_only_mode"})

    if not phase1_paic_schedule_is_canonical(schedule):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_star_ground_aug_requires_schedule",
                "required": PHASE1_PAIC_SAT_VIEW_SCHEDULE,
            }
        )

    if not (
        truthy_candidate_or_parameter(item, "use_sat_consistency")
        or "late" in candidate_value_text(axis)
        or "z_id" in candidate_value_text(axis)
    ):
        issues.append({"candidate_id": cid, "issue": "phase1_star_ground_aug_requires_late_consistency_axis"})

    if axis in (None, "", []):
        issues.append({"candidate_id": cid, "issue": "phase1_star_ground_aug_requires_exploration_axis"})

    direct_command = str(item.get("exact_command") or "")
    if "code/train.py" in direct_command:
        missing_flags = [flag for flag in PHASE1_PAIC_REQUIRED_DIRECT_FLAGS if flag not in direct_command]
        if missing_flags:
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "phase1_star_ground_aug_command_missing_required_flags",
                    "missing_flags": missing_flags,
                }
            )

    return issues


def phase1_safe_ssdg_executable_default_issues(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not is_phase1_row(item) or not is_safe_ssdg_phase1_row(item):
        return []
    if phase1_paic_control_exemption(item):
        return []

    cid = str(item.get("candidate_id") or "UNKNOWN")
    status = normalized_status(item.get("launchability_status"))
    if "user_required_safety_stop" in status:
        return []

    issues: List[Dict[str, Any]] = []
    text = candidate_search_text(item)
    exact_command = str(item.get("exact_command") or "").lower()
    runtime_class = normalized_status(item.get("runtime_class"))
    comparability_status = normalized_status(item.get("comparability_status"))
    defer_reason = normalized_status(item.get("defer_reason"))
    defer_text = " ".join((status, runtime_class, comparability_status, defer_reason, exact_command))

    if any(token in defer_text for token in PHASE1_SAFE_SSDG_LOCAL_VERIFY_DEFER_TOKENS):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_safe_ssdg_default_must_be_executable",
                "required": "Use run_phase1_safe_ssdg_candidate or python -m SSDG.train_ssdg; do not default future Phase1 Safe-SSDG rows to local schema deferred.",
            }
        )

    if any(token in exact_command for token in PHASE1_SAFE_SSDG_INVALID_CLI_TOKENS):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_safe_ssdg_unknown_train_py_flag",
                "forbidden": "--use_safe_ssdg_cvs",
                "required_entrypoint": "python -m SSDG.train_ssdg or run_phase1_safe_ssdg_candidate",
            }
        )

    if not any(token in text for token in PHASE1_SAFE_SSDG_EXECUTABLE_ENTRYPOINT_TOKENS):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_safe_ssdg_executable_entrypoint_required",
                "required_entrypoints": list(PHASE1_SAFE_SSDG_EXECUTABLE_ENTRYPOINT_TOKENS),
            }
        )

    return issues


def phase1_safe_ssdg_required_field_issues(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not is_phase1_row(item):
        return []
    cid = str(item.get("candidate_id") or "UNKNOWN")
    issues: List[Dict[str, Any]] = []
    text = candidate_search_text(item)

    if is_legacy_meta_phase1_row(item) and not is_allowed_legacy_meta_phase1_row(item):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_legacy_meta_ssl_not_current_mainline",
                "required": "Safe-SSDG-CVS-R01 or matched CEN51_R04 source-only DG non-regression row",
            }
        )
        return issues

    if not is_safe_ssdg_or_cen51_phase1_row(item):
        issues.append(
            {
                "candidate_id": cid,
                "issue": "phase1_route_family_must_be_safe_ssdg_or_cen51_refresh",
            }
        )
        return issues

    required_present = {
        "route_family": first_present(item, ["route_family", "protocol", "phase1_route_family"]),
        "ground_dg_claim_scope": item.get("ground_dg_claim_scope"),
        "source_ssl_split": first_present(item, ["source_ssl_split", "source_split", "ssl_source_split"]),
        "cen51_base_checkpoint_or_config": first_present(item, ["cen51_base_checkpoint_or_config", "cen51_base_weight", "cen51_base_weight_policy"]),
        "cen51_parent_run_or_control": first_present(item, ["cen51_parent_run_or_control", "cen51_parent_run", "parent_run"]),
        "phase1_non_regression_target": item.get("phase1_non_regression_target"),
        "optimization_target": item.get("optimization_target"),
        "target_lift_over_cen51": item.get("target_lift_over_cen51"),
        "satellite_channel_lift_target": item.get("satellite_channel_lift_target"),
        "pseudo_precision_audit_target": item.get("pseudo_precision_audit_target"),
    }
    for field, value in required_present.items():
        if value in (None, "", []):
            issues.append({"candidate_id": cid, "issue": f"missing_phase1_safe_ssdg_{field}"})

    if not truthy_candidate_value(item.get("no_target_receiver_in_training")):
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_requires_no_target_receiver_in_training_true"})
    if not truthy_candidate_value(first_present(item, ["CEN51_COMPARABLE", "cen51_comparable"])):
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_requires_cen51_comparable_true"})
    if not truthy_candidate_value(item.get("pseudo_coverage_is_risk_metric")):
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_requires_pseudo_coverage_risk_metric"})
    if not truthy_candidate_value(item.get("satellite_channel_primary_metric")):
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_requires_satellite_channel_primary_metric"})
    if not truthy_candidate_value(item.get("forbid_meta_learning_dg_mainline")):
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_requires_forbid_meta_learning_dg_mainline"})

    optimization_text = candidate_value_text(item.get("optimization_target"))
    if "exceed" not in optimization_text or "cen51" not in optimization_text:
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_optimization_target_must_exceed_cen51"})

    lift_text = " ".join(
        candidate_value_text(first_present(item, [field]))
        for field in ("target_lift_over_cen51", "satellite_channel_lift_target")
    )
    if "sat_mean_5" not in lift_text or "sat_floor_5" not in lift_text:
        issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_requires_satellite_lift_targets"})

    floor_text = candidate_value_text(item.get("must_not_regress_floor"))
    for token in ("88.57", "84.87", "79.53", "46.564", "41.52"):
        if token not in floor_text and token not in text:
            issues.append({"candidate_id": cid, "issue": "phase1_safe_ssdg_missing_cen51_floor", "floor_token": token})

    issues.extend(phase1_star_ground_aug_default_issues(item))
    issues.extend(phase1_safe_ssdg_executable_default_issues(item))

    return issues


def phase1_server_landed_training_issues(items: List[Mapping[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    if expected_count != CANONICAL_STAGE2_GPU_COUNT * len(CANONICAL_STAGE2_SLOTS):
        return []

    phase1_rows = [item for item in items if is_phase1_row(item)]
    if not phase1_rows:
        return []

    valid_outcomes = []
    for item in phase1_rows:
        status = normalized_status(item.get("launchability_status"))
        if "user_required_safety_stop" in status:
            valid_outcomes.append(str(item.get("candidate_id") or "UNKNOWN"))
            continue
        if (
            (
                status.startswith("deferred_retry_capacity")
                or status.startswith("deferred_retry_runtime_budget")
                or status.startswith("monitor_only_continue")
            )
            and has_phase1_training_command(item)
        ):
            valid_outcomes.append(str(item.get("candidate_id") or "UNKNOWN"))
            continue
        training_status = any(
            token in status
            for token in (
                "server_landed_training",
                "phase1_training",
                "safe_ssdg_training",
                "meta_ssl_training",
                "full_200e_training",
                "training_candidate",
            )
        )
        if training_status and has_phase1_training_command(item):
            valid_outcomes.append(str(item.get("candidate_id") or "UNKNOWN"))

    if valid_outcomes:
        return []

    return [
        {
            "scope": "matrix",
            "issue": "phase1_server_landed_training_candidate_required",
            "phase1_row_count": len(phase1_rows),
            "protocol_only_phase1_row_count": sum(1 for item in phase1_rows if is_protocol_only_phase1_row(item)),
            "required_outcomes": [
                "server_landed_phase1_training_candidate",
                "DEFERRED_RETRY_CAPACITY_with_exact_phase1_training_command",
                "DEFERRED_RETRY_RUNTIME_BUDGET_with_exact_phase1_training_command",
                "USER_REQUIRED_SAFETY_STOP",
            ],
        }
    ]


def lane_quota_issues(
    items: List[Mapping[str, Any]],
    expected_count: int,
    matrix_root: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if expected_count != CANONICAL_STAGE2_GPU_COUNT * len(CANONICAL_STAGE2_SLOTS):
        return []

    issues: List[Dict[str, Any]] = []
    phase1_rows = [item for item in items if is_phase1_row(item)]
    phase2_rows = [item for item in items if is_phase2_row(item)]
    quota_mode = normalized_status(matrix_root.get("lane_quota_mode")) if isinstance(matrix_root, Mapping) else ""
    phase2_only = quota_mode in {"phase2_only", "phase2-only"}
    expected_phase1 = 0 if phase2_only else CANONICAL_STAGE2_GPU_COUNT
    expected_phase2 = expected_count if phase2_only else expected_count - expected_phase1
    if len(phase1_rows) != expected_phase1 or len(phase2_rows) != expected_phase2:
        issues.append(
            {
                "scope": "matrix",
                "issue": "lane_quota_mismatch",
                "expected_phase1_rows": expected_phase1,
                "actual_phase1_rows": len(phase1_rows),
                "expected_phase2_rows": expected_phase2,
                "actual_phase2_rows": len(phase2_rows),
            }
        )

    for gpu_idx in range(CANONICAL_STAGE2_GPU_COUNT):
        gpu = f"GPU{gpu_idx}"
        gpu_rows = [item for item in items if normalize_gpu(item.get("gpu")) == gpu]
        phase1_count = sum(1 for item in gpu_rows if is_phase1_row(item))
        phase2_count = sum(1 for item in gpu_rows if is_phase2_row(item))
        expected_gpu_phase1 = 0 if phase2_only else 1
        expected_gpu_phase2 = len(CANONICAL_STAGE2_SLOTS) if phase2_only else len(CANONICAL_STAGE2_SLOTS) - 1
        if phase1_count != expected_gpu_phase1 or phase2_count != expected_gpu_phase2:
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "per_gpu_lane_quota_mismatch",
                    "gpu": gpu,
                    "expected_phase1_rows": expected_gpu_phase1,
                    "actual_phase1_rows": phase1_count,
                    "expected_phase2_rows": expected_gpu_phase2,
                    "actual_phase2_rows": phase2_count,
                }
            )
    return issues


def command_registry_uniqueness_issues(items: List[Mapping[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    if expected_count != CANONICAL_STAGE2_GPU_COUNT * len(CANONICAL_STAGE2_SLOTS):
        return []

    issues: List[Dict[str, Any]] = []
    for field in ("registry_key", "command_hash"):
        values = [str(item.get(field) or "").strip() for item in items]
        missing = [
            str(item.get("candidate_id") or "UNKNOWN")
            for item, value in zip(items, values)
            if not value
        ]
        duplicates = sorted({value for value in values if value and values.count(value) > 1})
        if missing:
            issues.append(
                {
                    "scope": "matrix",
                    "issue": f"missing_{field}",
                    "candidate_ids": missing,
                }
            )
        if duplicates:
            issues.append(
                {
                    "scope": "matrix",
                    "issue": f"duplicate_{field}",
                    "values": duplicates,
                }
            )
    return issues


def current_n607_run_id(matrix_root: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(matrix_root, Mapping):
        return None
    for key in ("n607_run_id", "remote_run_id", "stage2_n607_run_id"):
        value = matrix_root.get(key)
        if value not in (None, "", []):
            return str(value).strip()
    return None


def path_contains_run_segment(path_value: Any, segment: str, run_id: str) -> bool:
    text = str(path_value or "").replace("\\", "/")
    return f"/{segment}/{run_id}/" in text or text.endswith(f"/{segment}/{run_id}")


def exact_command_run_ids(command: Any) -> List[str]:
    return [match.group(1) for match in RUN_ID_ENV_RE.finditer(str(command or ""))]


def launcher_default_run_id(launcher_text: str) -> Optional[str]:
    match = LAUNCHER_RUN_ID_DEFAULT_RE.search(launcher_text)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group.strip()
    return None


def phase2_local_patch_default(launcher_text: str) -> Optional[str]:
    match = PHASE2_LOCAL_PATCH_DEFAULT_RE.search(launcher_text)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group.strip()
    return None


def active_launcher_lock_calls(launcher_text: str) -> List[str]:
    calls: List[str] = []
    for line in launcher_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\bstage2_acquire_launcher_lock\b", stripped):
            calls.append(stripped)
    return calls


def current_run_identity_issues(
    items: List[Mapping[str, Any]],
    matrix_root: Optional[Mapping[str, Any]],
    launcher_text: Optional[str] = None,
    launcher_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    expected_run_id = current_n607_run_id(matrix_root)
    if not expected_run_id:
        return []

    issues: List[Dict[str, Any]] = []
    for item in items:
        cid = str(item.get("candidate_id") or "UNKNOWN")
        run_path = item.get("estimated_run_path")
        if run_path not in (None, "") and not path_contains_run_segment(run_path, "runs", expected_run_id):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "estimated_run_path_not_under_current_n607_run_id",
                    "expected_n607_run_id": expected_run_id,
                    "estimated_run_path": str(run_path),
                }
            )
        log_path = item.get("estimated_log_path")
        if log_path not in (None, "") and not path_contains_run_segment(log_path, "logs", expected_run_id):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "estimated_log_path_not_under_current_n607_run_id",
                    "expected_n607_run_id": expected_run_id,
                    "estimated_log_path": str(log_path),
                }
            )
        registry_key = str(item.get("registry_key") or "")
        if registry_key and not registry_key.startswith(f"{expected_run_id}:"):
            issues.append(
                {
                    "candidate_id": cid,
                    "issue": "registry_key_not_bound_to_current_n607_run_id",
                    "expected_prefix": f"{expected_run_id}:",
                    "registry_key": registry_key,
                }
            )
    if launcher_text is None:
        return issues

    launcher_issue_base = {
        "scope": "launcher",
        "launcher_path": launcher_path or "<inline>",
        "expected_n607_run_id": expected_run_id,
    }
    default_run_id = launcher_default_run_id(launcher_text)
    if not default_run_id:
        issues.append({**launcher_issue_base, "issue": "launcher_default_run_id_missing"})
    elif default_run_id != expected_run_id:
        issues.append(
            {
                **launcher_issue_base,
                "issue": "launcher_default_run_id_mismatch",
                "found_run_id": default_run_id,
            }
        )

    if not re.search(r"RUNS_ROOT=.*runs/\$\{?RUN_ID\}?", launcher_text):
        issues.append({**launcher_issue_base, "issue": "launcher_runs_root_not_bound_to_run_id"})
    if not re.search(r"LOG_ROOT=.*logs/\$\{?RUN_ID\}?", launcher_text):
        issues.append({**launcher_issue_base, "issue": "launcher_log_root_not_bound_to_run_id"})

    phase2_summary = matrix_launchability_summary(items).get("by_lane", {}).get("phase2_spaceborne_fsl", {})
    if phase2_summary.get("launchable", 0) > 0 and phase2_local_patch_default(launcher_text) == "1":
        issues.append(
            {
                **launcher_issue_base,
                "issue": "launcher_phase2_local_patch_default_blocks_launchable_rows",
                "phase2_launchable_rows": phase2_summary.get("launchable", 0),
            }
        )

    if "stage2_queue_runner_template.sh" in launcher_text:
        lock_calls = active_launcher_lock_calls(launcher_text)
        if lock_calls:
            issues.append(
                {
                    **launcher_issue_base,
                    "issue": "launcher_must_not_call_stage2_lock_when_template_sources_it",
                    "lock_calls": lock_calls,
                }
            )

    return issues


def replace_or_insert_assignment(
    launcher_text: str,
    name: str,
    assignment: str,
    insert_after: Optional[str] = None,
) -> tuple[str, bool]:
    pattern = re.compile(rf"^\s*{re.escape(name)}=.*$", re.MULTILINE)
    if pattern.search(launcher_text):
        new_text = pattern.sub(assignment, launcher_text, count=1)
        return new_text, new_text != launcher_text
    if insert_after:
        insert_pattern = re.compile(rf"^\s*{re.escape(insert_after)}=.*$", re.MULTILINE)
        match = insert_pattern.search(launcher_text)
        if match:
            pos = match.end()
            return launcher_text[:pos] + "\n" + assignment + launcher_text[pos:], True
    return assignment + "\n" + launcher_text, True


def repair_launcher_identity_text(
    launcher_text: str,
    expected_run_id: str,
    phase2_launchable_rows: int = 0,
) -> tuple[str, List[Dict[str, Any]]]:
    repairs: List[Dict[str, Any]] = []
    text = launcher_text

    text, changed = replace_or_insert_assignment(
        text,
        "RUN_ID",
        f'RUN_ID="${{RUN_ID:-{expected_run_id}}}"',
        insert_after="ROOT",
    )
    if changed:
        repairs.append({"action": "set_launcher_default_run_id", "expected_n607_run_id": expected_run_id})

    text, changed = replace_or_insert_assignment(
        text,
        "RUNS_ROOT",
        'RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"',
        insert_after="RUN_ID",
    )
    if changed:
        repairs.append({"action": "bind_runs_root_to_run_id"})

    text, changed = replace_or_insert_assignment(
        text,
        "LOG_ROOT",
        'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
        insert_after="RUNS_ROOT",
    )
    if changed:
        repairs.append({"action": "bind_log_root_to_run_id"})

    if phase2_launchable_rows > 0:
        text, changed = replace_or_insert_assignment(
            text,
            "PHASE2_LOCAL_PATCH_REQUIRED",
            'PHASE2_LOCAL_PATCH_REQUIRED="${PHASE2_LOCAL_PATCH_REQUIRED:-0}"',
            insert_after="LOG_ROOT",
        )
        if changed:
            repairs.append(
                {
                    "action": "clear_phase2_local_patch_default_for_launchable_rows",
                    "phase2_launchable_rows": phase2_launchable_rows,
                }
            )

    if "stage2_queue_runner_template.sh" in text:
        repaired_lines: List[str] = []
        removed_lock_calls: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "stage2_acquire_launcher_lock":
                removed_lock_calls.append(stripped)
                continue
            repaired_lines.append(line)
        if removed_lock_calls:
            text = "\n".join(repaired_lines) + ("\n" if launcher_text.endswith("\n") else "")
            repairs.append(
                {
                    "action": "remove_direct_template_lock_call",
                    "removed_calls": removed_lock_calls,
                }
            )

    return text, repairs


def validate(
    items: List[Mapping[str, Any]],
    expected_count: int,
    retirement_policy: Optional[Mapping[str, Any]] = None,
    invalidity_ledger: Optional[Mapping[str, Any]] = None,
    sample_protocol: Optional[Mapping[str, Any]] = None,
    matrix_root: Optional[Mapping[str, Any]] = None,
    launcher_text: Optional[str] = None,
    launcher_path: Optional[str] = None,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    ids = [str(item.get("candidate_id") or "") for item in items]
    if len(items) != expected_count:
        issues.append({"scope": "matrix", "issue": "wrong_candidate_count", "expected": expected_count, "actual": len(items)})
    duplicates = sorted({cid for cid in ids if cid and ids.count(cid) > 1})
    if duplicates:
        issues.append({"scope": "matrix", "issue": "duplicate_candidate_id", "candidate_ids": duplicates})
    issues.extend(canonical_stage2_slot_issues(items, expected_count))
    issues.extend(lane_quota_issues(items, expected_count, matrix_root))
    issues.extend(current_run_identity_issues(items, matrix_root, launcher_text, launcher_path))
    for item in items:
        issues.extend(phase1_safe_ssdg_required_field_issues(item))
    issues.extend(phase1_server_landed_training_issues(items, expected_count))
    issues.extend(command_registry_uniqueness_issues(items, expected_count))
    categories = Counter(category_for(item) for item in items)
    for key in ("conservative", "aggressive", "old_retention", "unknown_boundary", "seen_new_rescue", "support_quality", "prototype_geometry", "query_free_background_risk", "unknown_separability", "oldqual_oldrisk_fusion", "rollback_calibration", "rollback_safe_retention", "deployment_gate_rescue", "unknown"):
        categories.setdefault(key, 0)
    expected_per_category = expected_count // 2 if expected_count % 2 == 0 else None
    triage_categories = {"old_retention", "unknown_boundary", "seen_new_rescue"}
    support_quality_categories = {"support_quality", "prototype_geometry"}
    background_risk_categories = {"query_free_background_risk", "unknown_separability"}
    oldfuse_categories = {"oldqual_oldrisk_fusion", "rollback_calibration"}
    rollsafe_categories = {"rollback_safe_retention", "deployment_gate_rescue"}
    if any(categories[key] for key in triage_categories):
        expected_per_triage_category = expected_count // 3 if expected_count % 3 == 0 else None
        if (
            expected_per_triage_category is None
            or any(categories[key] != expected_per_triage_category for key in triage_categories)
            or categories["conservative"] > 0
            or categories["aggressive"] > 0
            or any(categories[key] > 0 for key in support_quality_categories)
            or any(categories[key] > 0 for key in background_risk_categories)
            or any(categories[key] > 0 for key in oldfuse_categories)
            or any(categories[key] > 0 for key in rollsafe_categories)
            or categories["unknown"] > 0
        ):
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "triage_category_count_not_balanced",
                    "expected_per_triage_category": expected_per_triage_category,
                    "categories": dict(categories),
                }
            )
    elif any(categories[key] for key in support_quality_categories):
        expected_per_support_quality_category = expected_count // 2 if expected_count % 2 == 0 else None
        if (
            expected_per_support_quality_category is None
            or any(categories[key] != expected_per_support_quality_category for key in support_quality_categories)
            or categories["conservative"] > 0
            or categories["aggressive"] > 0
            or any(categories[key] > 0 for key in triage_categories)
            or any(categories[key] > 0 for key in background_risk_categories)
            or any(categories[key] > 0 for key in oldfuse_categories)
            or any(categories[key] > 0 for key in rollsafe_categories)
            or categories["unknown"] > 0
        ):
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "support_quality_category_count_not_balanced",
                    "expected_per_support_quality_category": expected_per_support_quality_category,
                    "categories": dict(categories),
                }
            )
    elif any(categories[key] for key in background_risk_categories):
        expected_per_background_risk_category = expected_count // 2 if expected_count % 2 == 0 else None
        if (
            expected_per_background_risk_category is None
            or any(categories[key] != expected_per_background_risk_category for key in background_risk_categories)
            or categories["conservative"] > 0
            or categories["aggressive"] > 0
            or any(categories[key] > 0 for key in triage_categories)
            or any(categories[key] > 0 for key in support_quality_categories)
            or any(categories[key] > 0 for key in oldfuse_categories)
            or any(categories[key] > 0 for key in rollsafe_categories)
            or categories["unknown"] > 0
        ):
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "background_risk_category_count_not_balanced",
                    "expected_per_background_risk_category": expected_per_background_risk_category,
                    "categories": dict(categories),
                }
            )
    elif any(categories[key] for key in oldfuse_categories):
        expected_per_oldfuse_category = expected_count // 2 if expected_count % 2 == 0 else None
        if (
            expected_per_oldfuse_category is None
            or any(categories[key] != expected_per_oldfuse_category for key in oldfuse_categories)
            or categories["conservative"] > 0
            or categories["aggressive"] > 0
            or any(categories[key] > 0 for key in triage_categories)
            or any(categories[key] > 0 for key in support_quality_categories)
            or any(categories[key] > 0 for key in background_risk_categories)
            or any(categories[key] > 0 for key in rollsafe_categories)
            or categories["unknown"] > 0
        ):
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "oldfuse_category_count_not_balanced",
                    "expected_per_oldfuse_category": expected_per_oldfuse_category,
                    "categories": dict(categories),
                }
            )
    elif any(categories[key] for key in rollsafe_categories):
        expected_per_rollsafe_category = expected_count // 2 if expected_count % 2 == 0 else None
        if (
            expected_per_rollsafe_category is None
            or any(categories[key] != expected_per_rollsafe_category for key in rollsafe_categories)
            or categories["conservative"] > 0
            or categories["aggressive"] > 0
            or any(categories[key] > 0 for key in triage_categories)
            or any(categories[key] > 0 for key in support_quality_categories)
            or any(categories[key] > 0 for key in background_risk_categories)
            or any(categories[key] > 0 for key in oldfuse_categories)
            or categories["unknown"] > 0
        ):
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "rollsafe_category_count_not_balanced",
                    "expected_per_rollsafe_category": expected_per_rollsafe_category,
                    "categories": dict(categories),
                }
            )
    elif expected_per_category is not None:
        if categories["conservative"] != expected_per_category or categories["aggressive"] != expected_per_category:
            issues.append(
                {
                    "scope": "matrix",
                    "issue": "category_count_not_balanced",
                    "expected_per_category": expected_per_category,
                    "categories": dict(categories),
                }
            )
    elif categories["unknown"] > 0:
        issues.append({"scope": "matrix", "issue": "unknown_candidate_category", "categories": categories})

    for item in items:
        cid = str(item.get("candidate_id") or "UNKNOWN")
        missing = [field for field in REQUIRED_FIELDS if item.get(field) in (None, "")]
        if missing:
            issues.append({"candidate_id": cid, "issue": "missing_required_fields", "fields": missing})
        issues.extend(paic_required_field_issues(item))
        for issue in fed_constraint_issues(item):
            issues.append({"candidate_id": cid, "issue": issue})
        for issue in route_retirement_issues(item, retirement_policy):
            issues.append({"candidate_id": cid, **issue})
        for issue in route_invalidity_issues(item, invalidity_ledger):
            issues.append({"candidate_id": cid, **issue})
        if is_launchable_status(item.get("launchability_status")) and route_duplication_repair_flag(item):
            issues.append({"candidate_id": cid, "issue": "route_duplication_repair_rows_must_not_be_launchable"})
        for issue in stage2_sample_protocol_issues(item, sample_protocol):
            issues.append(issue)
    return {
        "schema": "optimizer_candidate_matrix_validation_v1",
        "candidate_count": len(items),
        "expected_count": expected_count,
        "categories": categories,
        "launchability_summary": matrix_launchability_summary(items),
        "retired_route_policy": "active" if retirement_policy else "not_loaded",
        "route_invalidity_ledger": "active" if invalidity_ledger else "not_loaded",
        "stage2_sample_protocol": "active" if sample_protocol else "default",
        "issues": issues,
        "verdict": "PASS" if not issues else "FAIL",
    }


def main() -> int:
    args = parse_args()
    root = load_json_compat(args.candidate_matrix_json)
    items = item_list(root)
    expected_count = expected_count_for_matrix(root, args.expected_count)
    retirement_policy: Optional[Mapping[str, Any]] = None
    invalidity_ledger: Optional[Mapping[str, Any]] = None
    sample_protocol: Optional[Mapping[str, Any]] = None
    retirement_policy_source: Optional[Path] = None
    if isinstance(root, Mapping):
        root_sample_protocol = root.get("stage2_sample_protocol") or root.get("sample_protocol")
        if isinstance(root_sample_protocol, Mapping):
            sample_protocol = load_stage2_sample_protocol(root)
    if not args.ignore_retired_routes:
        retirement_policy_source = args.stage2_state or find_default_stage2_state(args.candidate_matrix_json)
        if retirement_policy_source and retirement_policy_source.exists():
            state = load_json_compat(retirement_policy_source)
            if sample_protocol is None and isinstance(state, Mapping):
                sample_protocol = load_stage2_sample_protocol(state)
            phase2 = state.get("phase2_spaceborne_fsl", {}) if isinstance(state, Mapping) else {}
            policy = phase2.get("route_retirement_policy", {}) if isinstance(phase2, Mapping) else {}
            if isinstance(policy, Mapping) and policy.get("status") == "ACTIVE":
                retirement_policy = policy
            ledger = phase2.get("route_invalidity_ledger", {}) if isinstance(phase2, Mapping) else {}
            if isinstance(ledger, Mapping) and ledger.get("status") == "ACTIVE":
                invalidity_ledger = ledger
    launcher_text: Optional[str] = None
    launcher_path: Optional[str] = None
    launcher_read_issue: Optional[Dict[str, Any]] = None
    launcher_identity_repairs: List[Dict[str, Any]] = []
    if args.launcher:
        launcher_path = str(args.launcher)
        try:
            launcher_text = args.launcher.read_text(encoding="utf-8-sig")
        except OSError as exc:
            launcher_text = ""
            launcher_read_issue = {
                "scope": "launcher",
                "issue": "launcher_unreadable",
                "launcher_path": launcher_path,
                "error": str(exc),
            }
    if args.repair_launcher_identity and args.launcher and launcher_text is not None and not launcher_read_issue:
        expected_run_id = current_n607_run_id(root if isinstance(root, Mapping) else None)
        if expected_run_id:
            phase2_summary = matrix_launchability_summary(items).get("by_lane", {}).get("phase2_spaceborne_fsl", {})
            repaired_text, launcher_identity_repairs = repair_launcher_identity_text(
                launcher_text,
                expected_run_id=expected_run_id,
                phase2_launchable_rows=int(phase2_summary.get("launchable") or 0),
            )
            if launcher_identity_repairs:
                args.launcher.write_text(repaired_text, encoding="utf-8", newline="\n")
                launcher_text = repaired_text
    payload = validate(
        items,
        expected_count,
        retirement_policy=retirement_policy,
        invalidity_ledger=invalidity_ledger,
        sample_protocol=sample_protocol,
        matrix_root=root if isinstance(root, Mapping) else None,
        launcher_text=launcher_text,
        launcher_path=launcher_path,
    )
    if launcher_read_issue:
        payload["issues"].insert(0, launcher_read_issue)
        payload["verdict"] = "FAIL"
    if retirement_policy_source:
        payload["retired_route_policy_source"] = str(retirement_policy_source)
    if launcher_path:
        payload["launcher_preflight_source"] = launcher_path
    if args.repair_launcher_identity:
        payload["launcher_identity_repair_mode"] = "enabled"
        payload["launcher_identity_repairs"] = launcher_identity_repairs
    if args.output:
        write_json(args.output, payload)
    launchability = payload["launchability_summary"]["total"]
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "verdict": payload["verdict"],
                "issue_count": len(payload["issues"]),
                "runner_readiness": launchability["runner_readiness"],
                "launchable_count": launchability["launchable"],
                "non_launchable_count": launchability["non_launchable"],
                "deferred_count": launchability["deferred"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
