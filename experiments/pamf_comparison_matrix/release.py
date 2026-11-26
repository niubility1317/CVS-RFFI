"""Register and publish the fixed paper and supplementary comparison matrix."""
import argparse
import csv
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .plan import (ADVISOR, COMMAND, CONTEXT, GPUS, GPU_WORKERS, METHODS, MODULE, NEW_METHODS,
                   PAPER, PROJECT, PYTHON, RELEASE, ROOT, RUN, SHEN, STAR, row_name, settings)


def specification(spec):
    paper_csv = Path("docs/reports/pamf_paper_master_design_20261027/data_settings.csv")
    with paper_csv.open(encoding="utf-8-sig", newline="") as f:
        paper = {row["legacy_protocol_id"]: row for row in csv.DictReader(f)}
    spec.update(run_id=RUN, group_id="phase1-pamf-seven-method-matrix",
                display_name="PAMF 七方法论文及补充对比矩阵",
                kind="comparison", stage="Phase1",
                description="发布 Receiver-Agnostic RFFI r04 和 STAR-RFFI 的匹配训练与读出，汇总全部七种识别方法。",
                authorization="2026-11-26 用户要求直接启动，覆盖原论文矩阵、Word 补充实验和其他方法所缺行。",
                parent_run_ids=[PAPER, ADVISOR, STAR, SHEN],
                tags=["PAMF", "Receiver-Agnostic RFFI", "STAR-RFFI", "practical", "comparison"])
    spec["code"].update(commit="Pinned Git commit in runtime.json", checkout="CVS-RFFI Git release",
                        environment=PYTHON, cwd=str(RELEASE))
    spec["data"].update(dataset="WiSig ManySig and ManyRx", version="Retained paper and advisor Phase1 handles",
        representation="FP32 2x256 eq1 IQ, unchanged source ID and labels", split_id="All 35 preregistered settings",
        contract_ref=f"{CONTEXT} and {PROJECT}/runs/{ADVISOR}/data/<setting>/source_contract.json",
        physical_ids_ref="Exact per-setting L_s/U_s/V metadata and validated target prediction indices",
        label_map_ref="Each locked source_contract.json", tx_sets_ref="Each locked source_contract.json",
        roles={"L_s":"Visible TX labels per setting","U_s":"Hidden TX, STAR only","V":"Source-only calibration or diagnostics"},
        train_ratio="Paper panel natural or 1/5/10% labels; advisor panel exact original budgets",
        validation_ref=str(CONTEXT), support_query_ref="Only fixed Phase1 test handles; no support",
        leo_config_ref="Source practical residual and validated clean plus six practical")
    spec["permissions"].update(claim_scope="Single-packet source-only WiSig domain generalization on simulated practical channels",
        external_method_exception="r04 supervised learner; STAR source-only capped pseudo-labels. Fixed matched DURA-LU at paper evaluation.")
    spec["checkpoint"].update(initialization="Scratch on each new split. Main exact frozen output reuse.",
        sources=[{"parent_run_id":STAR,"selection":"fixed E100 EMA, all three exact finished rows"},
                 {"parent_run_id":SHEN,"selection":"fixed E50 r04, all three saved source weights"}],
        provenance_verdict="Historical weights are reused only on their exact source contract.",
        contract_check_ref=str(CONTEXT), selection_rule="Fixed E50 Shen and fixed E100 STAR; no target feedback")
    spec["execution"].update(host="N607", launch_owner="codex-pamf-seven-method-20261126",
        gpu_policy=f"Up to {GPU_WORKERS} own GPU workers, one per card, candidate cards {GPUS}. Actual free-memory admission. One CPU scorer and one source cache job.",
        remote_run_root=str(ROOT), remote_log_root=str(ROOT/"logs"),
        local_artifact_root=f"automation_reports/CV-SincNet/{RUN}", launch_command=" ".join(COMMAND),
        stop_rule="Fail closed on illegal input or wrong frozen identities; keep healthy tasks and stop affected method/stage after two failures.")
    spec["expected_artifacts"] = ["jobs.json", "pipeline_state.json", "smoke.json", "per-row frozen source/readout files",
        "main_snapshot/seven_method_columns.csv", "tables/seven_method_columns.csv", "tables/equal_size_receiver_groups.csv",
        "tables/summary.csv", "tables/per_seed.csv", "tables/complete.json"]
    spec["metrics_plan"].update(metric_names=["accuracy", "macro_f1"],
        dimensions=["setting", "seed", "cross-RX", "cross-both", "cross-date", "scene", "route", "target strata"],
        prediction_ref=str(ROOT/"readouts/<row>/predictions"), scorer_ref="Independent CPU scorer after all row decisions freeze.")
    spec["rows"] = []
    for setting, d in settings().items():
        for seed in d["seeds"]:
            for method in NEW_METHODS:
                if setting == "F0" and method == "star":
                    continue
                key = row_name(setting, method, seed)
                stage = "predict" if setting == "F0" else "train"
                config = ROOT/"configs"/f"{stage}__{key}.json"
                source = ROOT/"F"/key/"source"
                if setting == "F0":
                    source = PROJECT/"runs"/SHEN/f"s{seed}"/"source"
                spec["rows"].append(dict(row_id=key, method=METHODS[method],
                    purpose="Missing Main readouts only" if stage=="predict" else f"Matched {setting}",
                    config_ref=str(config), resolved_config_ref=str(source/"resolved_config.json") if stage=="train" else str(config),
                    data_overrides=(paper[setting] if d["kind"]=="paper" else
                        {k:d[k] for k in ("source_rxs","source_days","target_rxs","counts","endpoint")}),
                    seeds=dict(model=seed,split=392005,data=392005,augmentation=2026102301 if method=="star" else seed,
                               support=None,evaluation=392005),
                    seed_notes="No support; fixed validated views; native r04 PDP RNG uses seed+1237, STAR application seed 91001.",
                    k=None, scenario="Clean and six practical; raw+DURA-LU on paper settings, raw on advisor settings",
                    optimizer="Native r04 SGD momentum 0.9" if method=="shen" else "Native STAR Adam and clipped gradients",
                    lr="0.001 fixed" if method=="shen" else "0.001/0.01 with existing native schedule",
                    epochs=50 if method=="shen" else 100, budget_ref="Native epochs, row-specific real L batches",
                    output_root=str(ROOT/"readouts"/key),log_path=str(ROOT/"logs"/f"{stage}__{key}.log"),
                    command=f"{PYTHON} -u -m {MODULE} {stage} --method {method} --config {config}",
                    expected_artifacts=["match-bound frozen decisions","scores","seven method summary"]))
    spec["notes"] = [
        "Receiver-Agnostic RFFI is the paper's method name. Fixed WiSig recipe is r04 paper_pdp_half.",
        "Eleven paper source settings with three paired seeds; twenty-four advisor settings with one seed; natural five-source point is a common-six-receiver aggregation.",
        "All 228 existing four-method rows and 57 existing PAMF rows passed scores coverage and truth-last check; no duplicate training or scoring of those rows.",
        "Main reuses three STAR results and three r04 weights plus raw cross-RX/cross-both decisions. Three r04 Main readout extensions and 108 scratch rows are scheduled.",
        "Supplements include all three joint L+V budgets, 16 equal-size receiver combinations and five date holdout or forward settings. Internal PAMF/DURA structure ablations retain their own method matrix.",
        "Receiver-Agnostic RFFI retains the 0.68 practical TX CE and source-only r04 PDP/AWGN. STAR retains the fixed paired-view recipe. A one-RX domain head has one class."
    ]
    assert len(spec["rows"]) == 111
    return spec


def publish(config, output):
    from experiments.star_pamf_practical.publish import publish_bundle
    packages = ["pamf_comparison_matrix","shen_wisig","star_pamf_practical",
                "pamf_advisor_supplements/plan.py","practical_full_2x2","cric_dura_comparison"]
    paths = [f"experiments/{x}" for x in packages] + ["code/leo_practical",
              f"automation_reports/CV-SincNet/{RUN}/experiment.json"]
    return publish_bundle(config, output, run=RUN, project=PROJECT, release=RELEASE, run_root=ROOT,
                          python=PYTHON, command=COMMAND, paths=paths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("register")
    a.add_argument("--template", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("publish")
    a.add_argument("--ssh-config", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.action=="register":
        write(a.output, specification(read(a.template)))
    else:
        publish(a.ssh_config, a.output)
