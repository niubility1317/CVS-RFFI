"""Build the registry specification for the executable Main split matrix."""
import argparse
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.plan import SEEDS
from .data import SOURCE, DAYS
from .runner import COMMAND, CONTEXT, GPUS, MIN_FREE_MIB, MODULE, PROJECT, PYTHON, RELEASE, RUN, RUN_ROOT, row_id

SOURCE_CACHE = PROJECT / "runs/20261023-phase1-practical-residual-full2x2-m600-r01/data/source/F0"
CONTRACT = SOURCE_CACHE / "source_contract.json"
METHOD = "Smooth STAR E bn_budget + practical residual"


def specification(spec):
    spec.update(run_id=RUN, group_id="phase1-pamf-main-star-practical",
                display_name="STAR-RFFI 接入 PAMF Main split 主对比（practical 增强）",
                description="匹配 Main split 的三个种子，训练 STAR-RFFI 并与冻结 PAMF 主行比较。",
                authorization="2026-11-25 用户指定用性能最好的 STAR-RFFI 跑 PAMF 主对比，使用 practical 星地信道增强",
                kind="comparison", tags=["STAR-RFFI", "PAMF", "Main split", "practical residual"])
    spec["code"].update(commit="Git commit in runtime.json", checkout="CVS-RFFI Git release archive",
                        environment=PYTHON, cwd=str(RELEASE))
    spec["data"].update(
        dataset="WiSig ManySig", version="20261023 Main split eq1 locked source and test views",
        representation="FP32 unit-RMS 2x256 IQ; native STAR 2x2048 time/FFT packing with empty second segment",
        contract_ref=str(CONTRACT), physical_ids_ref=str(SOURCE_CACHE / "{L_s,U_s,V}_metadata.npz"),
        label_map_ref=str(CONTRACT), tx_sets_ref=str(CONTRACT), source_receivers=list(SOURCE),
        target_receivers=[0, 2, 5, 7, 9, 10, 11], source_days=list(DAYS), target_days=[0, 1, 2, 3],
        roles={"L_s": "6300 labeled", "U_s": "56700 without TX labels", "V": "27000 source calibration only"},
        train_ratio="7/63/30 percent of source; 10 percent labeled among training records",
        split_id="F0", validation_ref=str(CONTEXT),
        leo_config_ref="Matched practical_residual post_sync eq1; three-scene training and six fixed practical tests",
        support_query_ref="Phase1 frozen inference via validated test handles; no support")
    spec["permissions"].update(claim_scope="Closed-set source-only domain generalization on the PAMF Main split",
        external_method_exception="STAR native source-only dual-confidence capped pseudolabel objective retained for this requested method")
    spec["checkpoint"].update(
        initialization="scratch", contract_check_ref=str(CONTRACT),
        provenance_verdict="Historical ten-class STAR weights are read only in separate source-only smoke on Main split L IQ; all fitted weights are freshly trained.",
        selection_rule="Fixed E100 inference EMA, separate BN banks reestimated on source L only.")
    spec["execution"].update(
        host="N607", launch_owner="codex-star-pamf-practical-20261125",
        gpu_policy=f"Physical cards {GPUS}; one worker per card, minimum {MIN_FREE_MIB} MiB free; one CPU scorer",
        remote_run_root=str(RUN_ROOT), remote_log_root=str(RUN_ROOT / "logs"),
        local_artifact_root=f"automation_reports/CV-SincNet/{RUN}", launch_command=" ".join(COMMAND),
        stop_rule="Systemic source/prediction/scoring failure stops new work; preserve immutable outputs and owned healthy workers.")
    spec["expected_artifacts"] = ["smoke.json", "source_pairs/complete.json", "pipeline_state.json",
                                 "paired_per_seed.csv", "paired_summary.csv", "complete.json"]
    spec["metrics_plan"].update(metric_names=["accuracy", "macro_f1", "paired accuracy delta"],
        dimensions=["seed", "receiver", "date", "transmitter", "route", "view", "task"],
        prediction_ref=str(RUN_ROOT / "readouts/<row>/predictions"),
        scorer_ref="Separate CPU process using practical_full_2x2.evaluation.score.")
    spec["rows"] = []
    for seed in SEEDS:
        row = row_id(seed)
        config = RUN_ROOT / "configs" / f"train__{row}.json"
        source = RUN_ROOT / "F" / row / "source"
        spec["rows"].append(dict(row_id=row, method=METHOD, purpose="Main split paper comparison",
             config_ref=str(config), resolved_config_ref=str(source / "resolved_config.json"),
             data_overrides={}, seeds=dict(model=seed, split=392005, data=392005,
                augmentation=2026102301, support=None, evaluation=392005),
             seed_notes="Existing physical split and fixed test views; practical draw 2026102301, session 2026102302, application 91001, batch 92001; no support.",
             k=None, scenario="clean + six fixed practical residual scenes, raw and DURA-LU",
             optimizer="Adam, separate identity/RD optimizers; gradient norm clipping 10",
             lr="0.001 / 0.01, native warmup and cosine floor 0.1", epochs=100,
             budget_ref="197 steps/epoch, 19700 steps; L32/U256 except final L remainder; each epoch L6300/U50432",
             output_root=str(source), log_path=str(RUN_ROOT / "logs" / f"train__{row}.log"),
             command=f"{PYTHON} -u -m {MODULE} train --config {config}",
             expected_artifacts=["final_e100_ema_source_bn.pth", "source calibration", "epoch telemetry",
                                 "all frozen predictions", "independent scores"]))
    spec["notes"] = [
        "Historical STAR A bn_budget had 67.840625 percent mean and 1.279225 pp SD over 8 seeds on the older RX1 Loo benchmark.",
        "The current Main split uses a new six-class head trained with the backbone, one physical IQ packet per item and no inherited recurrent state.",
        "Training pairs replace historical Loo with fixed practical residual draws and epoch-dependent application rates 0.30/0.60/0.80. Three scenes use the paper's 40/35/25 mix.",
        "Actual satellite additive CE weight 0.68. RD/orthogonality operate on clean U, while practical carries TX CE only. The original capped STAR U pseudo objective is retained.",
        "Compare to frozen three-view permuted Tri-SIG-TIME SSL with fixed seeds; report cross-date, cross-receiver and joint populations separately.",
        "Only the Main split is scheduled in this run; label-rate, source-count, receiver-repartition and ten-class panels are outside this row set."
    ]
    return spec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    write(args.output, specification(read(args.template)))
