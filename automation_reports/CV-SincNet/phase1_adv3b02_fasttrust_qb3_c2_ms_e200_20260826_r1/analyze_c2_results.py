from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "detailed_analysis_inputs"
BASELINE_SUMMARY = (
    HERE.parent
    / "phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1"
    / "analysis_summary.json"
)
OUT = HERE / "c2_final_analysis_summary.json"
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
FATAL_MARKERS = (
    "traceback",
    "cuda out of memory",
    "runtimeerror",
    "killed",
    "deterministic error",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_or_none(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def summarize_c2(path: Path) -> dict:
    required = [
        "final_ssdg.pth",
        "metrics_epoch.jsonl",
        "metrics_epoch.csv",
        "train.log",
        "metrics_clean.json",
        "metrics_joint.json",
        "eval_clean.log",
        "eval_joint.log",
        "phase1_resource_summary.json",
        "phase1_terminal_status.json",
        "phase1_training_completion_receipt.json",
        "status.txt",
    ]
    required += [f"metrics_{scenario}.json" for scenario in SCENARIOS]
    required += [f"eval_{scenario}.log" for scenario in SCENARIOS]
    missing = [name for name in required if not (path / name).is_file()]

    epochs = load_jsonl(path / "metrics_epoch.jsonl")
    with (path / "metrics_epoch.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    clean = load_json(path / "metrics_clean.json")
    joint = load_json(path / "metrics_joint.json")
    resource = load_json(path / "phase1_resource_summary.json")
    terminal = load_json(path / "phase1_terminal_status.json")
    receipt = load_json(path / "phase1_training_completion_receipt.json")

    scenario_metrics = {scenario: load_json(path / f"metrics_{scenario}.json") for scenario in SCENARIOS}
    leo = {scenario: float(scenario_metrics[scenario]["aggregate"]["tx_acc"]) for scenario in SCENARIOS}
    receiver_cells = [float(row["sat_acc"]) for row in joint["rows"]]
    clean_cells = [float(row["tx_acc"]) for row in clean["rows"]]

    all_logs = [path / "train.log", path / "eval_joint.log", path / "eval_clean.log"]
    all_logs += [path / f"eval_{scenario}.log" for scenario in SCENARIOS]
    dispatcher = INPUT / "dispatcher_logs" / f"{path.name}.log"
    if dispatcher.is_file():
        all_logs.append(dispatcher)
    log_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in all_logs)
    lowered = log_text.lower()

    telemetry = []
    for row in epochs:
        if float(row.get("train_rc4_gradient_telemetry_active") or 0.0) <= 0.0:
            continue
        telemetry.append(
            {
                "epoch": int(row["epoch"]),
                "g_L": finite_or_none(row.get("train_rc4_g_L")),
                "g_H": finite_or_none(row.get("train_rc4_g_H")),
                "g_Pset": finite_or_none(row.get("train_rc4_g_Pset")),
                "g_Pcond": finite_or_none(row.get("train_rc4_g_Pcond")),
                "identity_to_labeled": finite_or_none(row.get("train_rc4_g_identity_to_labeled")),
                "cos_H_to_labeled": finite_or_none(row.get("train_rc4_cos_H_to_labeled")),
                "cos_Pset_to_labeled": finite_or_none(row.get("train_rc4_cos_Pset_to_labeled")),
                "hard_effective_coverage": finite_or_none(row.get("train_rc4_hard_effective_coverage")),
                "partial_effective_coverage": finite_or_none(row.get("train_rc4_partial_effective_coverage")),
            }
        )

    audits = [clean["reconstruction_audit"]]
    audits += [scenario_metrics[scenario]["reconstruction_audit"] for scenario in SCENARIOS]
    strict = all(
        bool(audit.get("strict_requested"))
        and bool(audit.get("checkpoint_load_strict"))
        and not bool(audit.get("fallback_used"))
        and int(audit.get("missing_keys") or 0) == 0
        and int(audit.get("unexpected_keys") or 0) == 0
        and int(audit.get("shape_mismatches") or 0) == 0
        for audit in audits
    )

    epoch_numbers = [int(row["epoch"]) for row in epochs]
    result = {
        "candidate": path.name,
        "seed": int(clean["sat_seed"]),
        "training_complete": bool(receipt["phase1_training_complete"]),
        "artifact_complete": not missing,
        "missing_artifacts": missing,
        "epoch_jsonl_rows": len(epochs),
        "epoch_csv_rows": len(csv_rows),
        "epoch_sequence_complete": epoch_numbers == list(range(1, 201)),
        "checkpoint_bytes": (path / "final_ssdg.pth").stat().st_size,
        "checkpoint_epoch": int(clean["checkpoint_epoch"]),
        "terminal_status": terminal["status"],
        "row_status": (path / "status.txt").read_text(encoding="utf-8").strip(),
        "strict_reconstruction": strict,
        "clean_acc": float(clean["aggregate"]["tx_acc"]),
        "clean_receiver_floor": min(clean_cells),
        "leo": leo,
        "leo_mean": statistics.fmean(leo.values()),
        "leo_scenario_floor": min(leo.values()),
        "leo_receiver_scenario_floor": min(receiver_cells),
        "joint_receiver_scenario_cells": len(receiver_cells),
        "scenario_totals": {scenario: int(scenario_metrics[scenario]["aggregate"]["tx_total"]) for scenario in SCENARIOS},
        "scenario_receiver_rows": {scenario: len(scenario_metrics[scenario]["rows"]) for scenario in SCENARIOS},
        "wall_time_seconds": float(resource["wall_time_seconds"]),
        "peak_cuda_memory_allocated_gib": float(resource["peak_cuda_memory_allocated_bytes"]) / 2**30,
        "telemetry": telemetry,
        "fatal_fingerprint_counts": {marker: lowered.count(marker) for marker in FATAL_MARKERS},
        "full_log_files_scanned": [str(p.relative_to(INPUT)) for p in all_logs],
        "full_log_chars_scanned": len(log_text),
    }
    return result


def arm_from_candidate(candidate: str) -> str | None:
    for arm in ("C0", "C2", "C3"):
        if f"_{arm}_" in candidate:
            return arm
    return None


def delta(new: dict, old: dict) -> dict:
    return {
        "clean_delta_pp": new["clean_acc"] - old["clean_acc"],
        "leo_mean_delta_pp": new["leo_mean"] - old["leo_mean"],
        "leo_scenario_floor_delta_pp": new["leo_scenario_floor"] - old["leo_scenario_floor"],
        "leo_receiver_scenario_floor_delta_pp": new["leo_receiver_scenario_floor"] - old["leo_receiver_scenario_floor"],
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        metric: {
            "mean": statistics.fmean(float(row[metric]) for row in rows),
            "sample_std": statistics.stdev(float(row[metric]) for row in rows),
        }
        for metric in ("clean_acc", "leo_mean", "leo_scenario_floor", "leo_receiver_scenario_floor")
    }


def main() -> None:
    c2_rows = [
        summarize_c2(INPUT / "MS_S713101_C2_BC_H_PSET"),
        summarize_c2(INPUT / "MS_S713102_C2_BC_H_PSET"),
    ]
    old = load_json(BASELINE_SUMMARY)
    old_rows = [row for row in old["rows"] if row["family"] in {"formal", "historical_seed392002"}]
    c2_historical = next(row for row in old_rows if row["candidate"] == "E200_C2_BC_H_PSET")
    all_rows = old_rows + c2_rows
    by_seed: dict[int, dict[str, dict]] = {}
    for row in all_rows:
        arm = arm_from_candidate(row["candidate"])
        if arm in {"C0", "C2", "C3"}:
            by_seed.setdefault(int(row["seed"]), {})[arm] = row

    comparisons = []
    for seed in (392002, 713101, 713102):
        arms = by_seed[seed]
        comparisons.append(
            {
                "seed": seed,
                "C0": {key: arms["C0"][key] for key in ("clean_acc", "leo_mean", "leo_scenario_floor", "leo_receiver_scenario_floor")},
                "C2": {key: arms["C2"][key] for key in ("clean_acc", "leo_mean", "leo_scenario_floor", "leo_receiver_scenario_floor")},
                "C3": {key: arms["C3"][key] for key in ("clean_acc", "leo_mean", "leo_scenario_floor", "leo_receiver_scenario_floor")},
                "P_set_C2_minus_C0": delta(arms["C2"], arms["C0"]),
                "P_cond_C3_minus_C2": delta(arms["C3"], arms["C2"]),
                "combined_C3_minus_C0": delta(arms["C3"], arms["C0"]),
            }
        )

    arms_three_seed = {
        arm: [by_seed[seed][arm] for seed in (392002, 713101, 713102)]
        for arm in ("C0", "C2", "C3")
    }
    contribution_means = {}
    for label in ("P_set_C2_minus_C0", "P_cond_C3_minus_C2", "combined_C3_minus_C0"):
        contribution_means[label] = {
            metric: statistics.fmean(row[label][metric] for row in comparisons)
            for metric in (
                "clean_delta_pp",
                "leo_mean_delta_pp",
                "leo_scenario_floor_delta_pp",
                "leo_receiver_scenario_floor_delta_pp",
            )
        }

    output = {
        "schema": "phase1_fasttrust_qb3_c2_final_analysis_v1",
        "evidence_boundary": "C2 full local row copies; every JSONL/CSV record and every train/eval/dispatcher log read in full; C0/C3 from the previously full-parsed same-row analysis summary",
        "c2_rows": c2_rows,
        "historical_c2_seed392002": c2_historical,
        "same_seed_comparisons": comparisons,
        "three_seed_aggregate": {arm: aggregate(rows) for arm, rows in arms_three_seed.items()},
        "three_seed_contribution_means": contribution_means,
        "training_complete": all(row["training_complete"] for row in c2_rows),
        "artifacts_complete": all(row["artifact_complete"] for row in c2_rows),
        "analysis_complete": True,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
