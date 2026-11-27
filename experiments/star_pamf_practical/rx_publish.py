"""Register and release the fixed STAR cross-receiver screen."""
import argparse
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .register import specification as main_specification
from .rx_screen import (BASELINE, COMMAND, COMPARISON, GPUS, MIN_FREE_MIB, MODULE, ORIGINAL,
                        PROJECT, PYTHON, RELEASE, ROOT, RUN, SEED, SOURCE_PAIRS, VARIANTS, row_id)


def specification(template):
    spec = main_specification(template)
    example = spec["rows"][0]
    spec.update(run_id=RUN, group_id="phase1-star-cross-rx-practical",
                display_name="STAR-RFFI 星地增强下的跨接收机小范围优化",
                description="在同一个 Main split 种子上比较两种 STAR-RFFI 跨接收机身份训练配方。",
                authorization="2026-11-27 用户要求发布 STAR-RFFI 小范围性能优化实验。",
                parent_run_ids=[ORIGINAL, COMPARISON], kind="comparison", stage="Phase1",
                tags=["STAR-RFFI", "cross-receiver", "Main split", "practical", "screen"])
    spec["code"].update(cwd=str(RELEASE))
    spec["permissions"].update(
        claim_scope="Single-seed Phase1 screen on the existing Main benchmark; independent confirmation required",
        external_method_exception="Retain the original STAR pseudo objective. New alignment uses only visible source L TX/RX; no target or hidden U TX.")
    spec["checkpoint"].update(
        provenance_verdict="Both candidates start from scratch. Archived same-seed STAR E100 is isolated in source-only smoke.",
        selection_rule="Fixed E100 EMA and source L BN; all two candidates evaluated; no target-selected winner or expansion.")
    spec["execution"].update(
        launch_owner="codex-star-cross-rx-screen-20261127",
        gpu_policy=f"Up to two physical GPUs from {GPUS}, one worker/card above {MIN_FREE_MIB} MiB free; one CPU scorer.",
        remote_run_root=str(ROOT), remote_log_root=str(ROOT / "logs"),
        local_artifact_root=f"automation_reports/CV-SincNet/{RUN}", launch_command=" ".join(COMMAND),
        stop_rule="Fail closed on source/checkpoint/prediction mismatch or technical failure; preserve completed data and healthy independent rows.")
    spec["data"].update(leo_config_ref=f"Original fixed practical L/U pairs at {SOURCE_PAIRS}; existing clean and six practical test views")
    spec["expected_artifacts"] = ["smoke.json", "pipeline_state.json", "paired_comparison.csv", "complete.json"]
    spec["metrics_plan"].update(
        prediction_ref=str(ROOT / "readouts/<row>/predictions"),
        scorer_ref="Independent CPU truth-last scorer. Then seven-method paired-seed comparison from " + str(BASELINE))
    spec["rows"] = []
    for variant, options in VARIANTS.items():
        row = row_id(variant)
        cfg = ROOT / "configs" / f"train__{row}.json"
        source = ROOT / "F" / row / "source"
        spec["rows"].append(dict(example, row_id=row, method=f"STAR-RFFI + {variant}",
            purpose="Cross-RX labeled embedding alignment" if variant == "cross-rx" else
                    "Matched alignment plus an increased late labeled CE floor",
            config_ref=str(cfg), resolved_config_ref=str(source / "resolved_config.json"),
            data_overrides={}, hyperparameters=options, output_root=str(source),
            log_path=str(ROOT / "logs" / f"train__{row}.log"),
            command=f"{PYTHON} -u -m {MODULE} train --config {cfg}", seeds=dict(example["seeds"], model=SEED)))
    spec["notes"] = [
        "Full prior source logs (300 epochs) showed E76–100 U acceptance of 96.08%, 94.48%, 92.32%, motivating a retained labeled identity constraint without threshold tuning.",
        "Both candidates align the 128-d classifier input from clean L only using same-TX different-RX positives, weight 0.20 and temperature 0.15. The original practical CE and pseudo objectives retain their semantics.",
        "One candidate retains the native late true-label CE scale 0.10; the other uses 0.30. All other model/optimizer/pairing/seed/budget settings match the archived STAR baseline.",
        "Evaluate both rows with raw and shared frozen DURA-LU on clean and six practical in all three Main populations. DURA is a reporting condition, absent from training and screen gates.",
        "Preregistered raw-only success: six-scene mean at least 0.50 percentage points above the best of all seven original same-seed methods in both cross-RX and cross-both, and highest three-population mean. Relative to native STAR, no more than 1 pp loss on any population's clean or on cross-date practical mean.",
        "The known benchmark is a falsification screen; do not tune, expand the seed grid, or claim broad superiority based on its target rankings."
    ]
    return spec


def publish(config, output):
    from .publish import publish_bundle

    paths = ["experiments/star_pamf_practical", "experiments/pamf_comparison_matrix",
             "experiments/pamf_advisor_supplements/plan.py", "experiments/practical_full_2x2",
             "experiments/cric_dura_comparison", "code/leo_practical",
             f"automation_reports/CV-SincNet/{RUN}/experiment.json"]
    return publish_bundle(config, output, run=RUN, project=PROJECT, release=RELEASE,
                          run_root=ROOT, python=PYTHON, command=COMMAND, paths=paths)


def publish_rx_report(config, output):
    from .publish import publish_bundle

    reporter = RUN + "-cross-rx-reporting"
    release = PROJECT / "releases" / (reporter + "-release")
    paths = ["experiments/star_pamf_practical", "experiments/pamf_comparison_matrix",
             "experiments/pamf_advisor_supplements/plan.py", "experiments/practical_full_2x2",
             "experiments/cric_dura_comparison", "code/leo_practical"]
    command = [PYTHON, "-u", "-m", "experiments.star_pamf_practical.rx_report", "--root", str(ROOT)]
    return publish_bundle(config, output, run=reporter, project=PROJECT, release=release,
                          run_root=ROOT.parent / reporter, python=PYTHON, command=command, paths=paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    register = actions.add_parser("register")
    register.add_argument("--template", required=True, type=Path)
    register.add_argument("--output", required=True, type=Path)
    for name in ("publish", "publish-cross-rx"):
        release = actions.add_parser(name)
        release.add_argument("--ssh-config", required=True, type=Path)
        release.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "register":
        write(args.output, specification(read(args.template)))
    elif args.action == "publish":
        publish(args.ssh_config, args.output)
    else:
        publish_rx_report(args.ssh_config, args.output)
