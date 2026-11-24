"""Generate the preregistration from the executable matrix."""
import argparse
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .plan import RUN, SPLIT_SEED, matrix

PROJECT = "/home/szu2070436088/2510044040/CV-SincNet"
PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
PAPER_RUN = "20261027-phase1-pamf-paper-completion-m119-r03"
CONTEXT = f"{PROJECT}/releases/{PAPER_RUN}-release/context.json"
REMOTE = f"{PROJECT}/runs/{RUN}"
RELEASE = f"{PROJECT}/releases/{RUN}-release"
MODULE = "experiments.pamf_advisor_supplements.runner"
COMMAND = f"{PYTHON} -u -m {MODULE} dispatch --config {RELEASE}/runtime.json"


def specification(template):
    template.update(run_id=RUN, group_id="phase1-pamf-advisor-data-controls",
        display_name="导师补实验：总标签预算、等量接收机与日期留出",
        description="论文原版三视图 PAMF 补实验，固定 E50 末轮和同一目标接收观测",
        kind="comparison", authorization="2026-11-24 用户要求跑 Word 导师提出的补实验",
        tags=["PAMF", "Tri-SIG-TIME-SSL", "label_budget", "receiver_count", "date_holdout"])
    template["code"].update(commit="Git release_commit.txt and runtime.json record the enclosing commit",
        checkout="CVS-RFFI commit c9ebb6db301815159710b4c5cc461c9fd808dc55 plus this adapter",
        environment=PYTHON, cwd=RELEASE)
    template["data"].update(dataset="WiSig ManySig", version="20261023 retained eq1 role and fixed IQ caches",
        representation="2x256 FP32 normalized IQ, eq=1",
        contract_ref=f"{REMOTE}/data/<setting>/source_contract.json",
        physical_ids_ref=f"{REMOTE}/data/<setting>/{{L_s,U_s,V}}_metadata.npz",
        label_map_ref=CONTEXT, tx_sets_ref=CONTEXT, validation_ref=CONTEXT,
        leo_config_ref="Locked seven Phase1 practical-residual views in the October paper context",
        roles={"L_s": "See each row", "U_s": "See each row; TX hidden", "V": "See each row; source diagnostic"},
        target_receivers=[0, 2, 5, 7, 9, 10, 11], target_days=[0, 1, 2, 3],
        train_ratio="Fixed per-family counts in rows")
    template["permissions"]["claim_scope"] = "Phase1 closed-set frozen DG on terrestrial IQ with simulated practical impairments"
    template["checkpoint"].update(initialization="Scratch for every new data contract",
        provenance_verdict="No initial model, teacher or prior data-derived state is loaded",
        contract_check_ref=f"{REMOTE}/data/<setting>/source_contract.json",
        selection_rule="Fixed final E50; V only for source monitoring; no target feedback")
    template["execution"].update(host="N607", launch_owner="codex-pamf-advisor-20261124",
        gpu_policy="GPU 0/1/2/4, one worker per card, 11000 MiB free per start, one CPU scorer",
        remote_run_root=REMOTE, remote_log_root=f"{REMOTE}/logs",
        local_artifact_root=f"automation_reports/CV-SincNet/{RUN}", launch_command=COMMAND,
        stop_rule="Stop new dispatch on first technical failure, retain all existing artifacts and healthy own workers")
    template["expected_artifacts"] = ["smoke.json", "pipeline_state.json", "results.csv",
        "F/<row>/source/complete.json", "readouts/<row>/predictions/complete.json",
        "readouts/<row>/scores/scores.json"]
    template["metrics_plan"].update(metric_names=["accuracy", "macro_f1"],
        dimensions=["row", "receiver", "date", "transmitter", "view", "model_seed"],
        prediction_ref=f"{REMOTE}/readouts/<row>/predictions",
        scorer_ref="Independent experiments.cric_dura_comparison.evaluation.score")
    template["rows"] = []
    for row in matrix():
        key = row["row_id"]
        method = ("Tri-SIG-TIME SSL, fixed window permutation, original REPL"
                  if row["method"] == "ssl" else "PAMF-PL, 0.95 threshold, weight 1, ordered windows")
        template["rows"].append(dict(row_id=key, method=method, purpose=row["family"],
            config_ref=f"{REMOTE}/configs/train__{key}.json",
            resolved_config_ref=f"{REMOTE}/F/{key}/source/resolved_config.json", data_overrides=row,
            seeds=dict(model=row["model_seed"], split=SPLIT_SEED, data=SPLIT_SEED,
                       augmentation=row["model_seed"], support=None, evaluation=SPLIT_SEED),
            seed_notes="Phase1 does not use support. Augmentation follows native training RNG; evaluation uses retained fixed views.",
            k=None, scenario="clean + six practical", optimizer="Same native projected optimizer as paper E50",
            lr="Same per-segment native cosine schedule as paper E50", epochs=50, fl_rounds=None,
            budget_ref="20+10+20 epochs x 222 updates; 11100 total, L128/U256",
            output_root=f"{REMOTE}/F/{key}/source", log_path=f"{REMOTE}/logs/train__{key}.log",
            command=f"{PYTHON} -u -m {MODULE} worker --kind train --config {REMOTE}/configs/train__{key}.json",
            expected_artifacts=["final_ssdg.pth", "complete.json", "seven frozen predictions", "independent scores"]))
    template["notes"] = [
        "This first matrix uses seed 2026092701 and retains every outcome.",
        "Label-budget arm reassigns V only from the original L pool. The original V labels are never opened by the builder or trainer.",
        "Fixed-source-count arm uses three deterministic combinations per size except size six, with its single possible combination.",
        "Fixed-source-count arm keeps 1080 L, 10800 U and 1080 V for every combination. Neither target labels nor prior target metrics select groups.",
        "Held-out dates use the same seven unseen RX. Train D0-D2 / test D3 is also the terminal forward split. Earlier forward split uses D0-D1 / test D2.",
        "Unseen-RX target predictions are fixed across all seven views before independent scoring. Report date-specific rows for date experiments.",
    ]
    return template


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write(Path(a.output), specification(read(a.template)))
