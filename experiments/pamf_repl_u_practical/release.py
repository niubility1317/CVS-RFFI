"""Register and publish the one-seed U practical input contrast."""
import argparse
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .protocol import (BASE, BASE_READOUT, COMMAND, CONTEXT, GPUS, MIN_MIB, MODULE, PREFIX,
                       PROJECT, PYTHON, RELEASE, ROOT, ROW, RUN, SEED, SOURCE_OUT, STAR_PAIRS)


def specification(spec):
    spec.update(run_id=RUN, group_id="phase1-pamf-repl-u-practical",
                display_name="PAMF Main split 无标签 practical 单因素对照",
                description="固定三视图 PAMF 标准主法，仅替换完整 REPL 的 U As 半批 IQ。",
                kind="cvs", stage="Phase1", parent_run_ids=[BASE],
                tags=["PAMF", "Tri-SIG-TIME-SSL", "practical", "U As"],
                authorization="2026-11-25 用户要求发布既定单因素优化实验")
    spec["code"].update(commit="Pinned Git commit in runtime.json", checkout="CVS-RFFI Git release",
                        environment=PYTHON, cwd=str(RELEASE))
    data = PROJECT / "runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0"
    contract = data / "source_contract.json"
    spec["data"].update(dataset="WiSig ManySig", version="Existing verified Phase1 Main split F0",
        representation="eq1, FP32 2x256 normalized IQ", split_id="F0",
        contract_ref=str(contract), physical_ids_ref=str(data / "{L_s,U_s,V}_metadata.npz"),
        label_map_ref=str(contract), tx_sets_ref=str(contract), source_receivers=[1, 3, 4, 6, 8],
        target_receivers=[0, 2, 5, 7, 9, 10, 11], source_days=[1, 2, 3], target_days=[0, 1, 2, 3],
        roles={"L_s": "6300 visible", "U_s": "56700 with hidden TX labels", "V": "27000 validation only"},
        train_ratio="7/63/30 percent of source role pool; 10 percent labels in L+U",
        validation_ref=str(CONTEXT), support_query_ref="Phase1 target evaluation only; support is inapplicable.",
        leo_config_ref=f"Inherit validated seven practical test views and reuse source-U residual cache {STAR_PAIRS}")
    spec["permissions"].update(claim_scope="Phase1 Main split matched single-seed pilot with frozen raw and DURA-LU readouts")
    spec["checkpoint"].update(initialization="Resume matching same-seed E20 weights, native segment restart semantics",
        contract_check_ref=str(contract), provenance_verdict="Exact current recipe E20 prefix, scratch, source-only, 4440 updates",
        sources=[{"parent_run_id":BASE,"path":str(PREFIX / "final_ssdg.pth"),"selection":"fixed E20"}],
        selection_rule="Fixed E50, cumulative 11100 updates; no target-based selection")
    spec["execution"].update(host="N607", launch_owner="codex-pamf-repl-u-practical-20261125",
        gpu_policy=f"One own GPU worker at a time, {MIN_MIB} MiB free minimum, cards {GPUS}, independent CPU scorer",
        remote_run_root=str(ROOT), remote_log_root=str(ROOT / "logs"),
        local_artifact_root=f"automation_reports/CV-SincNet/{RUN}", launch_command=" ".join(COMMAND),
        stop_rule="Stop after the first technical failure in smoke, source training, prediction or score; preserve partial output.")
    spec["expected_artifacts"] = ["pipeline_state.json", "smoke.json", "F/<row>/source/final_ssdg.pth",
        "segment30/segment50 repl_state.json", "frozen predictions for all routes/views/populations",
        "independent score", "paired_scores.csv", "complete.json"]
    spec["metrics_plan"].update(metric_names=["accuracy", "macro_f1", "paired accuracy delta", "net corrections"],
        dimensions=["test population", "view", "route", "RX", "TX", "day"],
        prediction_ref=str(ROOT / "readouts" / ROW / "predictions"),
        scorer_ref="Independent CPU practical_full_2x2.evaluation.score after full prediction closure")
    spec["rows"] = [dict(row_id=ROW, method="Tri-SIG-TIME SSL with practical IQ in U As",
        purpose="Single-factor paired confirmation against existing raw U As row",
        config_ref=str(ROOT / "configs/train.json"), resolved_config_ref=str(SOURCE_OUT / "resolved_config.json"),
        data_overrides={}, seeds=dict(model=SEED, split=392005, data=392005,
            augmentation=2026102301, support=None, evaluation=392005),
        seed_notes="Fixed source physical split and target IQ; STAR cache practical draw 2026102301 and session 2026102302; all native seeds retained; no support.",
        k=None, scenario="clean plus six practical, raw and matched DURA-LU",
        optimizer="Native PAMF projected-Adam segment schedule", lr="Original PAMF E21–30 and E31–50",
        epochs=50, budget_ref="E20/4440 reused; 30/6660 new; E50/11100 total; six U student passes and 222 steps per epoch",
        output_root=str(SOURCE_OUT), log_path=str(ROOT / "logs/train.log"),
        command=f"{PYTHON} -u -m {MODULE} train --config {ROOT}/configs/train.json",
        expected_artifacts=["final_ssdg.pth", "complete.json", "source REPL view counts", "blind predictions", "truth-last scores"])]
    spec["notes"] = [
        f"Matched final baseline scores: {BASE_READOUT}/scores/scores.json.",
        "The existing even/odd half As mask remains unchanged. Each designated U sample gets its fixed same-ID practical IQ for the same two original CE passes.",
        "The source cache covers low suburban, mid urban and low urban with fixed probabilities .40/.35/.25. No extra training forward or DURA training use.",
        "Pilot thresholds on cross-RX are low urban raw gain at least 2 pp, raw six-mean nondecreasing, clean raw loss at most 0.5 pp, and DURA-LU six mean loss at most 0.5 pp.",
        "Cross-date and cross-both are reported separately. The threshold verdict is exploratory; no automatic seed expansion or target-guided changes."
    ]
    return spec


def publish(config, output):
    from experiments.star_pamf_practical.publish import publish_bundle

    packages = ["pamf_repl_u_practical", "pamf_repl_view_shuffle", "pamf_repl_completion",
                "pamf_ssl_research", "cric_dura_comparison", "practical_full_2x2", "phase1_external_dg_aug",
                "cvs_phase1_stack"]
    paths = [f"experiments/{pkg}" for pkg in packages]
    paths += ["code/leo_practical", f"automation_reports/CV-SincNet/{RUN}/experiment.json"]
    publish_bundle(config, output, run=RUN, project=PROJECT, release=RELEASE, run_root=ROOT,
                   python=PYTHON, command=COMMAND, paths=paths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="action", required=True)
    register = subs.add_parser("register")
    register.add_argument("--template", required=True, type=Path)
    register.add_argument("--output", required=True, type=Path)
    release = subs.add_parser("publish")
    release.add_argument("--ssh-config", required=True, type=Path)
    release.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.action == "register":
        write(a.output, specification(read(a.template)))
    else:
        publish(a.ssh_config, a.output)
