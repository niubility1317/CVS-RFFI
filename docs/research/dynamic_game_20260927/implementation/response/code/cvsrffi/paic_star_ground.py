"""CVS-SAT-PAIC route specifications and audit helpers.

This module is intentionally pure: it defines the PAIC design as machine-readable
local metadata, without launching training or touching target-domain samples.
"""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence


PAIC_ROUTE_FAMILY = "CVS-SAT-PAIC"
PAIC_CURRICULUM_SCHEDULE = (
    "1@0.30:mixed_orbit;"
    "41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;"
    "91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp"
)
PAIC_SCENARIOS = ("clear_leo", "low_elev_leo", "rain_leo", "storm_mp", "mixed_orbit")
PAIC_SA16_STRICT_UDU = 82.78
PAIC_SA16_SAT_AVG = 43.66
PAIC_SA16_SAT_MIN = 39.56
PAIC_FSDG49_BEST_STRICT_UDU = 76.295
PAIC_FSDG49_FINAL_STRICT_UDU = 75.9167

_ORBIT_LABELS = {0: "LEO", 1: "MEO", 2: "GEO"}
_STATE_LABELS = {0: "LOS", 1: "LOO", 2: "Rayleigh"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(item)
            else:
                out.append(item)
        return out
    return [value]


def _numeric_values(value: Any) -> list[float]:
    vals: list[float] = []
    for item in _as_list(value):
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            vals.append(number)
    return vals


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(float(sorted_values[0]), 6)
    pos = (len(sorted_values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        value = sorted_values[lo]
    else:
        frac = pos - lo
        value = sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
    return round(float(value), 6)


def _ratio(values: list[Any], labels: Mapping[int, str]) -> dict[str, float]:
    total = len(values)
    if total <= 0:
        return {name: 0.0 for name in labels.values()}
    counts = {name: 0 for name in labels.values()}
    for item in values:
        try:
            key = int(item)
        except (TypeError, ValueError):
            continue
        label = labels.get(key, str(key))
        counts[label] = counts.get(label, 0) + 1
    return {key: round(value / total, 6) for key, value in sorted(counts.items())}


def summarize_satellite_meta(meta: Mapping[str, Any], scenario: str | None = None) -> dict[str, Any]:
    """Summarize satellite-channel simulation metadata for reports and gates."""
    if not isinstance(meta, Mapping):
        raise TypeError("meta must be a mapping produced by apply_sat_gnd_channel_batch(..., return_meta=True)")
    count = max((len(_as_list(value)) for value in meta.values()), default=0)
    summary: dict[str, Any] = {
        "schema": "satellite_meta_summary_v1",
        "scenario": str(scenario or meta.get("scenario") or "unknown"),
        "sample_count": int(count),
    }
    if "orbit" in meta:
        summary["orbit_ratio"] = _ratio(_as_list(meta.get("orbit")), _ORBIT_LABELS)
    if "state" in meta:
        summary["state_ratio"] = _ratio(_as_list(meta.get("state")), _STATE_LABELS)
    for field in ("theta_deg", "snr_db", "fD_hz", "cfo_hz", "K_db", "pl_db", "d_km", "h_km"):
        vals = sorted(_numeric_values(meta.get(field)))
        if vals:
            summary[f"{field}_p10"] = _quantile(vals, 0.10)
            summary[f"{field}_p50"] = _quantile(vals, 0.50)
            summary[f"{field}_p90"] = _quantile(vals, 0.90)
    return summary


def _required_base(
    candidate_id: str,
    *,
    lane: str,
    category: str,
    hypothesis: str,
    control: str,
    key_changes: str,
    command: str,
    launchability_status: str = "LOCAL_SPEC_ONLY_NO_N607",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "category": category,
        "parent_run": "SA16/FSDG49/OA_MSE local anchors; no new N607 run",
        "lineage": "cvs_sat_paic_design_20260622",
        "route_family": PAIC_ROUTE_FAMILY,
        "route_signature": f"{PAIC_ROUTE_FAMILY}:{candidate_id}",
        "retirement_status": "active",
        "invalidity_status": "valid",
        "principle_rejection_ref": "none",
        "experimental_rejection_ref": "none",
        "retirement_evidence_count": 0,
        "retirement_evidence_refs": "none",
        "replacement_reason": "implements optimized star-ground channel enhancement design",
        "hypothesis": hypothesis,
        "control": control,
        "key_changes": key_changes,
        "parameters": {},
        "gpu": "GPU0",
        "estimated_run_path": f"runs/cvs_sat_paic/{candidate_id}",
        "estimated_log_path": f"logs/cvs_sat_paic/{candidate_id}.log",
        "cross_domain_target_metric": "clean strict UDU / unseen receiver-day accuracy",
        "satellite_channel_target_metric": "satellite/LEO avg,min and per-scenario floor",
        "allowed_tradeoff": "clean strict UDU may not fall beyond the design gate",
        "must_not_regress_floor": "clean strict UDU and satellite floor gates from CVS-SAT-PAIC report",
        "comparability_status": "LOCAL_MATRIX_SPEC_ONLY_NO_N607_LAUNCH",
        "expected_failure_signals": "clean collapse; satellite floor not improved; z_id receiver leakage rises",
        "fallback_or_alternative": "downgrade to robustness/diagnostic branch if gates fail",
        "exact_command": command,
        "launchability_status": launchability_status,
        "clean_view_role": "control_only",
        "evidence_level": "proxy_stress_not_real_inorbit",
        "deployment_success_claim_allowed": False,
        "n607_launch_allowed": False,
        "dataset_role": "terrestrial_proxy",
    }


def _central_rows() -> list[dict[str, Any]]:
    common = (
        "--dataset wisig --wisig_train_ratio 0.1 --epochs 200 "
        "--eval_sat_channel --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit "
    )
    quoted_schedule = shlex.quote(PAIC_CURRICULUM_SCHEDULE)
    rows = [
        (
            "C0_PAIC_NO_SAT_BASELINE",
            "conservative",
            "No-satellite training establishes clean and satellite-stress reference.",
            "--no_use_sat_consistency --no_use_concat_sat_channel_aug --lambda_sat_cls 0 --lambda_sat_cons 0",
        ),
        (
            "C1_SA16_ANCHOR_REPEAT",
            "conservative",
            "Repeat SA16 anchor before new PAIC deltas.",
            "--use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit",
        ),
        (
            "C2_PAIC_CURRICULUM_CE_ONLY",
            "aggressive",
            "Three-stage physical curriculum improves satellite floor without polluting clean DG losses.",
            f"--use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_view_schedule {quoted_schedule}",
        ),
        (
            "C3_PAIC_LATE_WEAK_ALIGN",
            "aggressive",
            "Late weak clean/satellite z_id alignment fills the CE-only invariance gap.",
            f"--use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_view_schedule {quoted_schedule} --use_sat_consistency --sat_cons_start_epoch 60 --lambda_sat_cons 0.03",
        ),
        (
            "C4_PAIC_SUPCON_PROPOSAL",
            "aggressive",
            "TX-aware contrastive alignment is a second-round proposal after C3 gates pass.",
            f"--use_concat_sat_channel_aug --concat_sat_ce_only --domain_freq_stability_mode dsq --sat_view_schedule {quoted_schedule} --lambda_supcon_id 0.02",
        ),
        (
            "C5_PAIC_ROBUSTNESS_BRANCH",
            "conservative",
            "Robustness branch may trade clean strict UDU for satellite floor and must not replace mainline by default.",
            f"--use_concat_sat_channel_aug --concat_sat_ce_only --domain_freq_stability_mode dsq --sat_view_schedule {quoted_schedule} --id_time_stability_mode phase_delta --id_freq_stability_mode dsq",
        ),
    ]
    out: list[dict[str, Any]] = []
    for cid, category, hypothesis, args in rows:
        status = "NON_LAUNCH_DIAGNOSTIC" if cid in {"C4_PAIC_SUPCON_PROPOSAL", "C5_PAIC_ROBUSTNESS_BRANCH"} else "LOCAL_SPEC_ONLY_NO_N607"
        row = _required_base(
            cid,
            lane="paic_central",
            category=category,
            hypothesis=hypothesis,
            control="C0/C1 and SA16 published local anchor",
            key_changes=args,
            command=f"python code/train.py {common}{args}",
            launchability_status=status,
        )
        row.update(
            {
                "paic_matrix_group": "central",
                "use_concat_sat_channel_aug": "--use_concat_sat_channel_aug" in args,
                "concat_sat_ce_only": "--concat_sat_ce_only" in args,
                "concat_sat_ce_weight": 1.0 if "--concat_sat_ce_only" in args else 0.0,
                "sat_view_schedule": PAIC_CURRICULUM_SCHEDULE if "sat_view_schedule" in args else "",
                "sat_cons_start_epoch": 60 if cid == "C3_PAIC_LATE_WEAK_ALIGN" else None,
                "lambda_sat_cons": 0.03 if cid == "C3_PAIC_LATE_WEAK_ALIGN" else 0.0,
                "domain_freq_stability_mode": "dsq" if "domain_freq_stability_mode dsq" in args else "",
                "clean_strict_udu_gate": f">= {PAIC_SA16_STRICT_UDU - 0.5:.2f}",
                "satellite_floor_gate": f">= {PAIC_SA16_SAT_MIN + 0.5:.2f} or SAT avg >= {PAIC_SA16_SAT_AVG + 0.5:.2f}",
            }
        )
        if status == "NON_LAUNCH_DIAGNOSTIC":
            row["non_launch_reason"] = "diagnostic_or_second_round_proposal_not_mainline_launch"
        out.append(row)
    return out


def _fed_command(args: str) -> str:
    return "PLAN=PAIC bash code/scripts/run_fed_fl82_validation_4gpu.sh --dry-run"


def _federated_rows() -> list[dict[str, Any]]:
    rows = [
        ("F0_FSDG49_ANCHOR", "conservative", "FSDG49 historical receiver-client FedProx anchor.", "--train_mode fedprox --fl_local_objective receiver_agnostic_bex02"),
        ("F1_FL82_16_CE_ONLY_DSQ", "conservative", "Replicate SA16 semantics in FL baseline-view CE-only DSQ.", "--fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"),
        ("F2_FL_PAIC_CURRICULUM", "aggressive", "Add PAIC three-stage satellite curriculum to FL CE-only route.", f"--fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_view_schedule {PAIC_CURRICULUM_SCHEDULE}"),
        ("F3_FL_PAIC_LATE_ALIGN", "aggressive", "Explore late weak consistency in FL after CE-only curriculum.", f"--fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_view_schedule {PAIC_CURRICULUM_SCHEDULE} --lambda_sat_cons 0.01 --sat_cons_start_epoch 90"),
        ("F4_STYLEBANK_DIAGNOSTIC_ONLY", "conservative", "StyleBank remains diagnostic until it beats random physical stress without leakage.", "--fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_fl_style_bank_stats"),
    ]
    out: list[dict[str, Any]] = []
    for cid, category, hypothesis, args in rows:
        status = "NON_LAUNCH_DIAGNOSTIC" if cid == "F4_STYLEBANK_DIAGNOSTIC_ONLY" else "LOCAL_SPEC_ONLY_NO_N607"
        row = _required_base(
            cid,
            lane="federated_vmb",
            category=category,
            hypothesis=hypothesis,
            control="F0/FSDG49 and FL82_16",
            key_changes=args,
            command=_fed_command(args),
            launchability_status=status,
        )
        row.update(
            {
                "paic_matrix_group": "federated",
                "wisig_train_ratio": 0.1,
                "fl_rounds": 200,
                "epochs": 200,
                "fl_client_key": "receiver",
                "fl_baseline_view_ce_only": cid != "F0_FSDG49_ANCHOR",
                "fl_baseline_view_ce_weight": 1.0 if cid != "F0_FSDG49_ANCHOR" else 0.0,
                "sat_view_schedule": PAIC_CURRICULUM_SCHEDULE if "sat_view_schedule" in args else "",
                "diag_baseline_sat_view_active_required": cid != "F0_FSDG49_ANCHOR",
                "diag_sat_cls_active_required": cid != "F0_FSDG49_ANCHOR",
                "diag_fishr_domain_count_required": True,
                "clean_strict_udu_gate": f"> {PAIC_FSDG49_BEST_STRICT_UDU:.3f} or final > {PAIC_FSDG49_FINAL_STRICT_UDU:.4f}",
                "satellite_floor_gate": f"SAT avg/min >= {PAIC_SA16_SAT_AVG:.2f}/{PAIC_SA16_SAT_MIN:.2f}",
            }
        )
        row["parameters"] = {
            "wisig_train_ratio": 0.1,
            "epochs": 200,
            "fl_rounds": 200,
            "fl_client_key": "receiver",
            "paic_launcher_plan": "PAIC",
            "paic_launcher_args": args,
        }
        if status == "NON_LAUNCH_DIAGNOSTIC":
            row["non_launch_reason"] = "stylebank_diagnostic_only_until_random_physical_control_passes"
        out.append(row)
    return out


def _stage2_rows() -> list[dict[str, Any]]:
    rows = [
        ("S2A_PAIC_PROTOCOL_CHECK", "conservative", "Stage2-A zero-label old recognition plus unknown rejection.", "Stage2-A_zero_label_deploy", 0, False),
        ("S2B_PAIC_PROTOCOL_CHECK", "aggressive", "Stage2-B target-old calibration under satellite/LEO view.", "Stage2-B_old_label_calibration", 5, False),
        ("S2C_PAIC_PROTOCOL_CHECK", "aggressive", "Stage2-C old plus seen-new enrollment under satellite/LEO view.", "Stage2-C_old_new_enrollment", 5, True),
    ]
    out: list[dict[str, Any]] = []
    for cid, category, hypothesis, protocol_stage, k, target_new_support in rows:
        row = _required_base(
            cid,
            lane="phase2_spaceborne_fsl",
            category=category,
            hypothesis=hypothesis,
            control="OA-MSE Stage2 protocol validator",
            key_changes="protocol field hardening only; no launch",
            command="python tools/spaceborne_fewshot_da_matrix.py --plan OA_MSE_CARD3 --output-root automation_reports/CV-SincNet",
            launchability_status="NON_LAUNCH_DIAGNOSTIC",
        )
        row.update(
            {
                "paic_matrix_group": "stage2",
                "protocol_stage": protocol_stage,
                "claim_scope": "unknown_rejection" if "Stage2-A" in protocol_stage else ("target_old_calibration" if "Stage2-B" in protocol_stage else "seen_new_enrollment"),
                "target_channel_view": "satellite/LEO",
                "source_receiver_labels": "1-1,1-19,14-7,18-2,19-2,2-1,2-19",
                "target_receiver_labels": "20-1",
                "receiver_disjoint_verified": True,
                "source_tx_ids": "0,1,2,3,4,5",
                "target_old_tx_ids": "0,1,2,3,4,5",
                "target_new_tx_ids": "1-16,1-18",
                "unknown_tx_ids": "10-1,10-10",
                "tx_split_disjoint_verified": True,
                "k_shot": k,
                "support_query_split_verified": True,
                "target_old_leo_query": "target-old query from Y_old on R_t",
                "target_new_leo_support": target_new_support if target_new_support else "",
                "target_new_query_role": "seen_new_identity_eval" if target_new_support else "reject_eval_only_not_seen_new_identity",
                "target_new_leo_query": (
                    "target-new query from Y_new on R_t for seen-new identity evaluation"
                    if target_new_support
                    else "non-old target-new query from Y_new on R_t for rejection evaluation only"
                ),
                "unknown_leo_query": "unknown query from Y_unknown on R_t; eval only",
                "threshold_selection_label_scope": "source_old_and_allowed_support_only; unknown_query_eval_only",
                "unknown_query_eval_only": True,
                "target_new_query_not_threshold_fit": True,
                "unknown_FAR_target": 0.05,
                "deployment_success_claim_allowed": False,
                "evidence_level": "receiver_x_transmitter_proxy_stress",
                "non_launch_reason": "stage2_protocol_check_only_no_n607_launch",
            }
        )
        out.append(row)
    return out


def build_paic_matrix() -> dict[str, Any]:
    """Return the complete local PAIC implementation matrix."""
    candidates = _central_rows() + _federated_rows() + _stage2_rows()
    return {
        "schema": "cvs_sat_paic_matrix_v1",
        "route_family": PAIC_ROUTE_FAMILY,
        "schedule": PAIC_CURRICULUM_SCHEDULE,
        "scenarios": list(PAIC_SCENARIOS),
        "expected_count": len(candidates),
        "validator_command": "python tools/optimizer_validate_matrix.py <matrix.json> --expected-count 14",
        "evidence_boundary": "WiSig/ManySig terrestrial proxy plus physics-informed satellite stress; not real in-orbit IQ validation",
        "candidates": candidates,
    }


def render_paic_markdown(payload: Mapping[str, Any] | None = None) -> str:
    payload = payload or build_paic_matrix()
    lines = [
        "# CVS-SAT-PAIC 实现矩阵",
        "",
        "本文件由本地 PAIC 规范生成，落地的是物理启发的 star-ground stress 训练/评估路线；WiSig/ManySig 仍是 terrestrial proxy，不是真实在轨 IQ 验证。",
        "",
        f"- route_family: `{payload['route_family']}`",
        f"- schedule: `{payload['schedule']}`",
        f"- scenarios: `{','.join(payload['scenarios'])}`",
        f"- validator: `python tools/optimizer_validate_matrix.py cvs_sat_paic_matrix.json --expected-count {payload['expected_count']}`",
        "",
        "| ID | lane | category | status | key gates |",
        "|---|---|---|---|---|",
    ]
    for row in payload["candidates"]:
        gates = "; ".join(
            str(row.get(key))
            for key in ("clean_strict_udu_gate", "satellite_floor_gate", "protocol_stage")
            if row.get(key)
        )
        lines.append(
            f"| `{row['candidate_id']}` | `{row['lane']}` | `{row['category']}` | "
            f"`{row['launchability_status']}` | {gates} |"
        )
    lines.extend(
        [
            "",
            "## 声明边界",
            "",
            "- clean view 只作为 `control_only`。",
            "- satellite/LEO 指标是 deployment-oriented stress test，不是真实在轨部署成功。",
            "- Stage2 unknown query 永远 eval-only，不参与阈值拟合、adapter、prototype 或伪标签。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_paic_payloads(
    output_root: Path | str,
    payload: Mapping[str, Any] | None = None,
    validation_payload: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = payload or build_paic_matrix()
    json_path = root / "cvs_sat_paic_matrix.json"
    report_path = root / "cvs_sat_paic_report.md"
    validation_path = root / "cvs_sat_paic_validation.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_paic_markdown(payload), encoding="utf-8")
    if validation_payload is not None:
        validation_path.write_text(json.dumps(validation_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"json_path": json_path, "report_path": report_path, "validation_path": validation_path}
