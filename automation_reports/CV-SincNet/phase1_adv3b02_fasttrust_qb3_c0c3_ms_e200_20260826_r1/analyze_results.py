from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
NEW_ROOT = HERE / "detailed_analysis_input"
OLD_ROOT = (
    HERE.parent
    / "phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1"
    / "analysis_input"
)
OUT = HERE / "analysis_summary.json"
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
FATAL_MARKERS = ("traceback", "cuda out of memory", "runtimeerror", "killed")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_epochs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def sum_numeric(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows)


def summarize_row(path: Path, family: str) -> dict:
    epochs = load_epochs(path / "metrics_epoch.jsonl")
    clean = load_json(path / "metrics_clean.json")
    joint = load_json(path / "metrics_joint.json")
    resource = load_json(path / "phase1_resource_summary.json")
    aggregates = {row["scenario"]: float(row["tx_acc"]) for row in joint["aggregates"]}
    clean_acc = float(clean["aggregate"]["tx_acc"])
    clean_floor = min(float(row["tx_acc"]) for row in clean["rows"])
    leo_floor = min(float(row["sat_acc"]) for row in joint["rows"])
    scenario_floor = min(aggregates.values())
    reconstruction_ok = bool(clean["reconstruction_audit"]["checkpoint_load_strict"])
    reconstruction_ok &= not bool(clean["reconstruction_audit"]["fallback_used"])
    reconstruction_ok &= all(
        load_json(path / f"metrics_{scenario}.json")["reconstruction_audit"]["checkpoint_load_strict"]
        and not load_json(path / f"metrics_{scenario}.json")["reconstruction_audit"]["fallback_used"]
        for scenario in SCENARIOS
    )

    train_log = (path / "train.log").read_text(encoding="utf-8", errors="replace")
    eval_logs = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in sorted(path.glob("eval_*.log"))
    )
    full_log = (train_log + "\n" + eval_logs).lower()
    fatal_hits = {marker: full_log.count(marker) for marker in FATAL_MARKERS}

    epoch_numbers = [int(row["epoch"]) for row in epochs]
    expected_epoch_numbers = list(range(1, len(epochs) + 1))
    positive_skip_rates = [
        float(row.get(key) or 0.0)
        for row in epochs
        for key in ("train_skipped_nonfinite_grad", "train_skipped_nonfinite_loss")
        if float(row.get(key) or 0.0) > 0.0
    ]
    inferred_batches = (
        int(round(1.0 / min(positive_skip_rates))) if positive_skip_rates else None
    )
    estimated_grad_skips = (
        round(sum_numeric(epochs, "train_skipped_nonfinite_grad") * inferred_batches)
        if inferred_batches
        else 0
    )
    estimated_loss_skips = (
        round(sum_numeric(epochs, "train_skipped_nonfinite_loss") * inferred_batches)
        if inferred_batches
        else 0
    )
    telemetry_epochs = [
        int(row["epoch"])
        for row in epochs
        if float(row.get("train_rc4_gradient_telemetry_active") or 0.0) > 0.0
    ]
    anomaly_flag_epochs = [
        int(row["epoch"])
        for row in epochs
        if float(row.get("train_rc4_first_anomaly_packet_written") or 0.0) > 0.0
    ]
    stage_keys = (
        "train_muse_time_train_batches_s",
        "train_muse_time_base_validation_s",
        "train_muse_time_heavy_source_validation_s",
        "train_muse_time_checkpoint_io_s",
        "train_muse_time_other_s",
    )
    timing = {key.removeprefix("train_muse_time_"): sum_numeric(epochs, key) for key in stage_keys}
    timing["epoch_total_s"] = sum_numeric(epochs, "epoch_time_s")

    result = {
        "family": family,
        "candidate": path.name,
        "seed": int(clean["sat_seed"]),
        "epochs": len(epochs),
        "epoch_sequence_complete": epoch_numbers == expected_epoch_numbers,
        "status": (path / "status.txt").read_text(encoding="utf-8", errors="replace").strip(),
        "checkpoint_epoch": int(clean["checkpoint_epoch"]),
        "clean_acc": clean_acc,
        "clean_receiver_floor": clean_floor,
        "leo": aggregates,
        "leo_mean": mean(list(aggregates.values())),
        "leo_scenario_floor": scenario_floor,
        "leo_receiver_scenario_floor": leo_floor,
        "strict_reconstruction": reconstruction_ok,
        "wall_time_seconds": float(resource["wall_time_seconds"]),
        "peak_cuda_memory_allocated_gib": float(resource["peak_cuda_memory_allocated_bytes"]) / 2**30,
        "timing_seconds": timing,
        "u_samples_per_s_mean": mean(
            [float(row["train_muse_u_samples_per_s"]) for row in epochs if row.get("train_muse_u_samples_per_s")]
        ),
        "u_forward_samples_per_s_mean": mean(
            [float(row["train_muse_u_forward_samples_per_s"]) for row in epochs if row.get("train_muse_u_forward_samples_per_s")]
        ),
        "inferred_train_batches_per_epoch": inferred_batches,
        "estimated_nonfinite_grad_skips": estimated_grad_skips,
        "estimated_nonfinite_loss_skips": estimated_loss_skips,
        "nonfinite_grad_skip_rate_pct": (
            100.0 * estimated_grad_skips / (len(epochs) * inferred_batches)
            if inferred_batches
            else 0.0
        ),
        "telemetry_epochs": telemetry_epochs,
        "anomaly_packet_written": bool(anomaly_flag_epochs),
        "anomaly_packet_first_reported_epoch": anomaly_flag_epochs[0] if anomaly_flag_epochs else None,
        "fatal_fingerprint_counts": fatal_hits,
        "full_log_chars_scanned": len(full_log),
        "last_epoch": {
            key: epochs[-1].get(key)
            for key in (
                "train_loss",
                "train_tx_acc",
                "val_tx_acc",
                "train_rc4_hard_effective_coverage",
                "train_rc4_partial_effective_coverage",
                "train_rc4_effective_weighted_coverage",
                "train_rc4_g_H",
                "train_rc4_g_Pset",
                "train_rc4_g_Pcond",
                "train_rc4_g_identity_to_labeled",
            )
        },
    }
    for value in result.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite summary value in {path}")
    return result


def main() -> None:
    rows: list[dict] = []
    for path in sorted((NEW_ROOT / "profile").iterdir()):
        if (path / "metrics_joint.json").is_file():
            rows.append(summarize_row(path, "profile"))
    for path in sorted((NEW_ROOT / "formal").iterdir()):
        if (path / "metrics_joint.json").is_file():
            rows.append(summarize_row(path, "formal"))
    for name in (
        "E200_C0_BC_NO_U_ID",
        "E200_C1_BC_H",
        "E200_C2_BC_H_PSET",
        "E200_C3_BC_H_PSET_PCOND",
        "E200_C4_BC_U_FEATURE_ANCHOR",
    ):
        path = OLD_ROOT / name
        rows.append(summarize_row(path, "historical_seed392002"))

    scientific = [
        row
        for row in rows
        if row["family"] in {"formal", "historical_seed392002"}
        and ("C0" in row["candidate"] or "C3" in row["candidate"])
    ]
    by_seed: dict[int, dict[str, dict]] = {}
    for row in scientific:
        arm = "C3" if "C3" in row["candidate"] else "C0"
        by_seed.setdefault(row["seed"], {})[arm] = row
    paired = []
    for seed, arms in sorted(by_seed.items()):
        if set(arms) == {"C0", "C3"}:
            paired.append(
                {
                    "seed": seed,
                    "clean_delta_pp": arms["C3"]["clean_acc"] - arms["C0"]["clean_acc"],
                    "leo_mean_delta_pp": arms["C3"]["leo_mean"] - arms["C0"]["leo_mean"],
                    "leo_scenario_floor_delta_pp": arms["C3"]["leo_scenario_floor"] - arms["C0"]["leo_scenario_floor"],
                    "leo_receiver_scenario_floor_delta_pp": arms["C3"]["leo_receiver_scenario_floor"] - arms["C0"]["leo_receiver_scenario_floor"],
                }
            )

    aggregate = {}
    for arm in ("C0", "C3"):
        arm_rows = [row for row in scientific if arm in row["candidate"]]
        aggregate[arm] = {
            metric: {"mean": mean(values), "sample_std": stdev(values)}
            for metric, values in {
                "clean_acc": [row["clean_acc"] for row in arm_rows],
                "leo_mean": [row["leo_mean"] for row in arm_rows],
                "leo_scenario_floor": [row["leo_scenario_floor"] for row in arm_rows],
                "leo_receiver_scenario_floor": [row["leo_receiver_scenario_floor"] for row in arm_rows],
            }.items()
        }
    aggregate["paired_delta"] = {
        metric: {"mean": mean(values), "sample_std": stdev(values)}
        for metric, values in {
            key: [row[key] for row in paired]
            for key in (
                "clean_delta_pp",
                "leo_mean_delta_pp",
                "leo_scenario_floor_delta_pp",
                "leo_receiver_scenario_floor_delta_pp",
            )
        }.items()
    }

    output = {
        "schema": "phase1_fasttrust_qb3_detailed_analysis_v1",
        "evidence_boundary": "full local copies of every JSONL, CSV, train log and evaluation log listed in each completed row directory",
        "rows": rows,
        "paired_differences": paired,
        "three_seed_aggregate": aggregate,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUT}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
