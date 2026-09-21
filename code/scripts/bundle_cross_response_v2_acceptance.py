"""Copy small inspectable acceptance evidence; leave datasets/tensors in run logs."""
import json
from pathlib import Path
import shutil


def bundle():
    root = Path(__file__).resolve().parents[2]
    source = Path("E:/type10-7/logs/core90_v2_acceptance_20260911")
    target = root / "analysis/core90_v2_acceptance_evidence_20260911"
    target.mkdir(parents=True,exist_ok=False)
    selected = {
        "suite_result.json":"suite_01/pytest_result.json", "pytest.log":"suite_01/pytest.log",
        "fp32_trajectory.json":"fp32_01/trajectory_report.json",
        "amp_trajectory.json":"amp_01/trajectory_main_reanalysis.json",
        "amp_auxiliary_nan.json":"amp_02/trajectory_report.json",
        "amp_auxiliary_inf.json":"amp_inf_01/trajectory_report.json",
        "legacy_fp32_main_reanalysis.json":"legacy_fp32_01/trajectory_main_reanalysis.json",
        "decision_cuda.json":"decision_cuda_benchmark_20260911_v2.json",
        "decision_cuda_initial_failure.json":"decision_cuda_initial_failure_20260911.json",
        "role_reachability.json":"role_reachability_6300.json",
        "actual_v2_branches.json":"branches_fp32_01/branch_acceptance.json",
    }
    for name,relative in selected.items():
        shutil.copyfile(source/relative,target/name)
    shutil.copyfile(root / "analysis/v2_source_update_synthetic_20260911_r2/report.json",target/"source_update.json")
    index = {"status":"IMPLEMENTATION_ACCEPTANCE_WITH_EXPLICIT_HISTORICAL_LIMIT",
        "unit_tests":{"passed":174,"skipped":12,"log":"pytest.log"},
        "full_entrypoint":"real CORE90 synthetic IQ, two epochs, final heldout intentionally omitted",
        "historical_first_divergence":"UNRESOLVED_HISTORICAL_FIRST_DIVERGENCE",
        "real_source_three_condition_gate":"NOT_ESTABLISHED",
        "formal_E200_launched":False,"target_rescored":False,
        "evidence_files":list(selected)+["source_update.json"],
        "large_local_artifacts":{"trajectory_tensors":str(source),
            "source_counterfactual_parameter_deltas":str(root/"analysis/v2_source_update_synthetic_20260911_r2")},
        "large_artifacts_committed":False,
        "interpretation":"Passing synthetic implementation tests does not establish real source benefit or locate historical E200 first divergence."}
    (target/"index.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(target)


if __name__ == "__main__":
    bundle()
