"""Register and publish the fixed STAR B matrix."""
import argparse
import csv
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_comparison_matrix.plan import CONTEXT, GPUS, GPU_WORKERS, settings
from .b_matrix import (BASELINE, COMMAND, EXISTING, MIN_FREE_MIB, MODULE, OPTIONS, PILOT, PILOT_ROOT,
                       PILOT_ROW, PROJECT, PYTHON, RELEASE, ROOT, RUN, STAR, label, original, rows)
from .register import specification as base_specification


def specification(template):
    spec = base_specification(template)
    example = spec["rows"][0]
    definitions = settings()
    path = Path("docs/reports/pamf_paper_master_design_20261027/data_settings.csv")
    with path.open(encoding="utf-8-sig", newline="") as f:
        paper = {r["legacy_protocol_id"]: r for r in csv.DictReader(f)}
    spec.update(run_id=RUN, group_id="phase1-star-b-practical-matrix", stage="Phase1", kind="comparison",
                display_name="STAR-RFFI B 与原版 STAR-RFFI 全矩阵配对",
                description="复用 B 单种子及全部 57 个原版对照，补齐 56 个同条件 B 训练与评分行。",
                authorization="2026-11-27 用户明确指定以 B 组跑原对比实验矩阵并和原版对比。",
                parent_run_ids=[PILOT, BASELINE, STAR])
    spec["code"].update(cwd=str(RELEASE))
    spec["data"].update(split_id="35 fixed original STAR settings; per-row physical IDs in parent recipes",
                        source_receivers=None, target_receivers=None, source_days=None, target_days=None,
                        roles={"L_s":"Original visible label assignment", "U_s":"Original hidden TX assignment",
                               "V":"Original source-only calibration"}, train_ratio="Exact matching parent assignment",
                        contract_ref=f"{CONTEXT} and original STAR resolved configurations",
                        physical_ids_ref="Original row source contract and prediction handle",
                        label_map_ref="Original row source contract", tx_sets_ref="Original row source contract",
                        leo_config_ref="Reuse original legally generated practical L/U IQ pairs")
    spec["permissions"].update(claim_scope="Fixed-method paired extension on the established WiSig benchmark",
                               external_method_exception="STAR native pseudolabels and clean L-only cross-RX alignment")
    spec["checkpoint"].update(
        initialization="56 rows scratch. Main B seed 2026092701 is a display-only exact prior result.",
        provenance_verdict="Original STAR scores are comparators. Archived B E100 is isolated in source-only smoke.",
        selection_rule="All B rows fixed E100 EMA with source L BN recalibration. The user selected this fixed recipe after the pilot.")
    spec["execution"].update(
        launch_owner="codex-star-b-matrix-20261127", remote_run_root=str(ROOT), remote_log_root=str(ROOT/"logs"),
        local_artifact_root=f"automation_reports/CV-SincNet/{RUN}", launch_command=" ".join(COMMAND),
        gpu_policy=f"Up to {GPU_WORKERS} cards from {GPUS}, one worker/card. Train >= {MIN_FREE_MIB} MiB, predict >= 7000 MiB, one CPU scorer.",
        stop_rule="Fail closed on illegality. Block pending jobs at the same stage after two technical failures, preserve other workers.")
    spec["metrics_plan"].update(
        metric_names=["accuracy", "macro_f1", "paired same-seed delta"],
        dimensions=["setting", "seed", "test population", "scene", "readout", "receiver", "date", "transmitter"],
        prediction_ref=str(ROOT/"readouts/<row>/predictions"),
        scorer_ref="Independent CPU truth-last scorer. Cross-RX raw practical mean is primary; DURA-LU reported separately.")
    spec["expected_artifacts"] = ["smoke.json", "pipeline_state.json", "main_snapshot/paired_complete.json",
        "tables/paired_complete.json", "tables/paired_star_per_seed.csv", "tables/cross_rx_paired_summary.csv",
        "tables/eight_method_columns.csv", "tables/equal_size_receiver_groups.csv"]
    spec["rows"] = []
    for setting, seed in rows():
        reused = (setting, seed) == EXISTING
        name = label(setting, seed)
        source = PILOT_ROOT/"F"/PILOT_ROW/"source" if reused else ROOT/"F"/name/"source"
        config = source/"resolved_config.json" if reused else ROOT/"configs"/f"train__{name}.json"
        parent, old = original(setting, seed)
        d = definitions[setting]
        overrides = (paper[setting] if setting in paper else
                     {k:d[k] for k in ("source_rxs", "source_days", "target_rxs", "counts", "endpoint")})
        spec["rows"].append(dict(example, row_id=name, method="STAR-RFFI B",
            purpose="Exact frozen Main pilot score reuse" if reused else "Fixed B versus same-setting and same-seed native STAR",
            config_ref=str(config), resolved_config_ref=str(source/"resolved_config.json"),
            data_overrides=dict(overrides, original_train_config=str(parent/"configs"/f"train__{old}.json")),
            seeds=dict(example["seeds"], model=seed), hyperparameters=OPTIONS,
            scenario="Clean/six practical. Paper raw+DURA-LU; supplements raw.",
            budget_ref="Original row L32/U256 step schedule, fixed E100 for both models",
            output_root=str(source),
            log_path=str(PILOT_ROOT/"logs"/f"train__{PILOT_ROW}.log" if reused else ROOT/"logs"/f"train__{name}.log"),
            command="Exact historical row reuse; no new execution" if reused else
                    f"{PYTHON} -u -m {MODULE} train --config {config}"))
    spec["notes"] = [
        "B recipe fixed unchanged from the completed pilot: supervised cross-RX clean identity weight 0.20, temperature 0.15, late labeled CE floor 0.30.",
        "33 paper rows and 24 supplements. Reuse F0 seed 2026092701 B; train other 56 from scratch against 57 complete native STAR baselines.",
        "Paper includes Main, receiver reassignments, 1%/5% labels, 10 classes and natural RX counts. Five-source point is derived from the matched common-target Main records.",
        "Supplements preserve three joint label budgets, 16 equal-size receiver groups and five date controls. Dates archived separately from the primary cross-RX conclusion.",
        "Historical pool has seven methods. Publish an eight-column table, same-seed B minus native STAR deltas, and per-setting cross-RX summaries.",
        "Existing pilot result is a reused observed point. This fixed expansion reports matching benchmark comparisons and does not declare a fresh independent target holdout."
    ]
    return spec


def publish(config, output):
    from .publish import publish_bundle

    paths = ["experiments/star_pamf_practical", "experiments/pamf_comparison_matrix",
             "experiments/pamf_advisor_supplements/plan.py", "experiments/practical_full_2x2",
             "experiments/cric_dura_comparison", "code/leo_practical",
             f"automation_reports/CV-SincNet/{RUN}/experiment.json"]
    return publish_bundle(config, output, run=RUN, project=PROJECT, release=RELEASE, run_root=ROOT,
                          python=PYTHON, command=COMMAND, paths=paths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    register = sub.add_parser("register")
    register.add_argument("--template", required=True, type=Path)
    register.add_argument("--output", required=True, type=Path)
    release = sub.add_parser("publish")
    release.add_argument("--ssh-config", required=True, type=Path)
    release.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.action == "register":
        write(args.output, specification(read(args.template)))
    else:
        publish(args.ssh_config, args.output)
