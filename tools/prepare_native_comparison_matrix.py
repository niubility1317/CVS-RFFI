"""Write immutable row configs, native method queue and existing registry records."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

METHODS = ("protonet", "feature_separation", "dadda", "mrior", "twostage", "csil", "mopc_hr", "orthogonal", "radionet_ada")
SEEDS = (392005, 2026092701, 2026092702, 2026092703, 2026092704)
SOURCE_RUN = "20260930-phase1-native-baselines-practical-manysig-m5-r01"
PIPELINE_RUN = "20260930-phase12-native-baselines-practical-m5-r01"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def prepare(root, remote_root, registry_module=None):
    root = Path(root)
    remote_root = remote_root.rstrip("/")
    release = remote_root + "/releases/native_comparisons_20260930_r01"
    python = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
    contract = remote_root + "/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json"
    dataset = remote_root + "/Dataset_WigSig/ManySig.pkl"
    p1 = remote_root + "/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule"
    p2run = remote_root + "/runs/20260927-phase2-practical-data-manytx-s2026092705-r01"
    p2 = p2run + "/capsule"
    for run in (SOURCE_RUN, PIPELINE_RUN):
        if (root / "automation_reports/CV-SincNet" / run).exists():
            raise FileExistsError(f"Already registered {run}; no overwrite or duplicate registration")
    path = Path(registry_module or root / "tools/experiment_registry.py")
    module_spec = importlib.util.spec_from_file_location("comparison_registry", path)
    registry = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(registry)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    combined = {"schema": "native_comparison_launch_v1", "run_id": PIPELINE_RUN,
        "source_run_id": SOURCE_RUN, "code_root": release, "root": release, "python": python,
        "launch_owner": "codex/root/native-comparisons-20260930", "rows": [],
        "p1_truth": remote_root + "/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json",
        "p2_truth": p2run + "/score_only/truth.json",
        "runtime_root": remote_root + "/paper_reproduction/runs/" + PIPELINE_RUN + "/dispatcher",
        "source_gpu_lanes": 1, "phase12_gpu_lanes": 1, "max_training_per_gpu": 2}
    records = {}
    configs = {}
    for run, stage in ((SOURCE_RUN, "Phase1"), (PIPELINE_RUN, "Phase1+Phase2")):
        spec = registry.template()
        spec.update(run_id=run, group_id="native-residual-noeq-exact-manysig-source-manytx-target-20260930",
            display_name="Native comparison source training" if stage == "Phase1" else "Native comparison paired adaptation and registration",
            description="Nine distinct source model families, five fixed seeds, scratch final200; matched residual_noeq capsules and truth-last scoring. Native mechanisms retained; dataset and fixed-budget extensions disclosed.",
            kind="comparison", stage=stage, rows=[], authorization="User 2026-09-30: organize/design the named comparison methods and complete their experiments.",
            tags=["residual_noeq", "source_scratch", "final200", "native_methods", "truth_last"],
            comparison_group_id="manysig-six-source-rx13468-day123-luv076330-residual-noeq-v1")
        spec["code"].update(commit=commit, checkout=str(root), environment=python, cwd=release)
        spec["data"].update(dataset=dataset, version="existing ManySig/ManyTx immutable artifacts", representation="equalized1 center256 unit_rms; residual/post_sync/noeq fs25MHz",
            contract_ref=contract, physical_ids_ref=contract + "#role_ids", label_map_ref="each source checkpoint/classes",
            source_receivers=[1,3,4,6,8], target_receivers=[0,2,5,7,9,10,11], source_days=[1,2,3], target_days=[0,1,2,3],
            roles={"L_s": .07, "U_s": .63, "V": .30}, train_ratio=None,
            capsule_id="residual-noeq-ba667eee4fb061055e4c08b5" if stage != "Phase1" else None,
            support_query_ref=p2 + "/splits" if stage != "Phase1" else None,
            validation_ref=p2 + "/manifest.json" if stage != "Phase1" else None)
        spec["permissions"].update(regime="source_only" if stage == "Phase1" else "registered external method source-access exception and labeled target support",
            query_use="readonly prediction only; independent scorer after prediction completion",
            external_method_exception=None if stage == "Phase1" else "DADDA/MRIOR/twostage/RadioNet ADA source L_s replay per project external-method exception; no target query fitting",
            claim_scope="matched WiSig/LEO extension; not original-paper dataset reproduction")
        spec["checkpoint"].update(initialization="scratch" if stage == "Phase1" else "matching source row last.pt only",
            sources=[] if stage == "Phase1" else ["each row/source_output/last.pt"], contract_check_ref=contract,
            provenance_verdict="SCRATCH_ONLY" if stage == "Phase1" else "CHECK_AT_LOAD: complete scratch contract and final200 metadata",
            selection_rule="fixed epoch200 last; no target or source-V selection")
        spec["execution"].update(host="N607", launch_owner=combined["launch_owner"], remote_run_root=remote_root + "/paper_reproduction/runs/" + run,
            remote_log_root=remote_root + "/paper_reproduction/runs/" + run,
            local_artifact_root="automation_reports/CV-SincNet/" + run,
            launch_command=python + " -m comparison_suite.launch --spec automation_reports/CV-SincNet/" + PIPELINE_RUN + "/launch_spec.json",
            stop_rule="No automatic retry, performance stopping, healthy task intervention, artifact overwrite or duplicate launch. Technical failure preserves artifacts and blocks dependent row.")
        spec["expected_artifacts"] = ["each row/resolved_config.json", "each row/training.log", "each row/last.pt", "each row/base_state.pt", "each row/completion.json"] if stage == "Phase1" else ["each row/phase1_predictions.npz", "each row/phase2_predictions.jsonl", "each row/phase1_complete.json", "each row/phase2_complete.json", "each row/resources.jsonl", "each row/fit_metrics.jsonl", "each row/provenance.json"]
        spec["metrics_plan"].update(metric_names=["clean_accuracy", "target_scene_accuracy", "before_old", "adapted_old", "registered_old", "registered_new", "H", "trainable_parameters", "elapsed_seconds", "peak_memory"],
            dimensions=["method", "model_seed", "RX", "scene", "K", "new_class_count", "support_seed", "stage"], scorer_ref="comparison_suite.score")
        spec["notes"] = ["392005 is disclosed historical optimized seed; primary analysis uses four fresh fixed seeds, all-five sensitivity also reported. No seed cherry-picking.",
            "CSIL/MoPC encoder is explicitly declared six-block IQ CVS extension; native channel separation/KD/Fisher/masks and prototype correction/augmentation/HR retained.",
            "CSIL official_repo_corefix explicitly replaces overflowing author exp(gradient²) with FP64 mean per-example labeled-source CE gradient² empirical Fisher; EWC is active with trainable old-old fingerprints. No silent exponent clamp, identity fallback or literal author-code parity claim.",
            "ProtoNet source training is episodic, not arbitrary frozen embedding NCM. RadioNet ADA uses its own author ADA encoder, not DF Fine-tuning checkpoint.",
            "Current config code.commit is preparation baseline; publisher must record the final committed release before launch.",
            "Source L_s alone supplies supervised source initializers. U_s labels remain hidden and are unused; V only evaluated. Source-step/sample exposure differences are reported."]
        records[run] = spec
        configs[run] = []
    for index, (method, seed) in enumerate((m,s) for m in METHODS for s in SEEDS):
        row_id = method + "-s" + str(seed)
        gpu = index % 8
        source_output = remote_root + "/paper_reproduction/runs/" + SOURCE_RUN + "/" + row_id
        pipeline_output = remote_root + "/paper_reproduction/runs/" + PIPELINE_RUN + "/" + row_id
        optimizer = "SGD" if method in ("dadda", "csil", "mopc_hr", "orthogonal") else "Adam"
        lr = .0001 if method == "dadda" else .0006 if method == "mrior" else .01 if optimizer == "SGD" else .001
        source = {"method": method, "model_seed": seed, "data_seed": seed, "dataset": dataset, "source_contract": contract,
            "output_root": source_output, "device": "cuda:0", "epochs": 200, "batch_size": 128, "lr": lr,
            "optimizer": optimizer, "model_kwargs": {}, "augmentation_start_epoch": 80, "augmentation_weight": .68,
            "augmentation_seed": 2027, "receiver_seed": 2027, "split_seed": 392005, "use_source_ssl_split": True,
            "fs_hz": 25e6, "print_steps": True, "steps_per_epoch": 50, "method_version": "matched_native_v1"}
        if method == "csil":
            source["method_version"] = "csil_official_repo_corefix_empirical_fisher"
        phase12 = {"method": method, "model_seed": seed, "source_output": source_output, "output_root": pipeline_output,
            "device": "cuda:0", "source_contract": contract, "dataset": dataset, "p1_capsule": p1, "p2_capsule": p2,
            "adaptation": {"epochs": 100, "steps_per_epoch": 1, "iterations": 1000, "dann_epochs": 100, "lmmd_epochs": 100, "finetune_epochs": 100},
            "registration": {"csil_old_fingerprint_trainable": True}, "batch_size": 256, "method_version": "matched_native_v1",
            "protocol": "p2_min_v1", "channel_route": "residual_noeq"}
        if method == "csil":
            phase12["method_version"] = "csil_official_repo_corefix_empirical_fisher"
        for run, cfg, module, stage, output in ((SOURCE_RUN,source,"comparison_suite.source","source",source_output), (PIPELINE_RUN,phase12,"comparison_suite.pipeline","phase12",pipeline_output)):
            relative = "automation_reports/CV-SincNet/" + run + "/configs/" + row_id + ".json"
            configs[run].append((relative, cfg))
            remote_config = release + "/" + relative
            log = output + "/launch.stdout.log"
            command = [python, "-u", "-m", module, "--config", remote_config]
            launch_row = {"row_id": row_id, "run_id": run, "stage": stage, "method": method, "model_seed": seed,
                "config": remote_config, "module": module, "gpu": gpu, "output_root": output, "log": log,
                "dependency": None if stage == "source" else source_output + "/completion.json"}
            combined["rows"].append(launch_row)
            records[run]["rows"].append({"row_id": row_id, "method": method, "purpose": "primary", "config_ref": relative,
                "resolved_config_ref": output + "/resolved_config.json", "data_overrides": {},
                "seeds": {"model": seed, "split": 392005, "data": seed if stage == "source" else 2026092705,
                          "augmentation": 2027 if stage == "source" else 2026092707,
                          "support": None, "evaluation": None if stage == "source" else 2026092706},
                "seed_notes": "Source data seed controls sampling/episodes; source V is deterministic (evaluation null), support not applicable. Phase2 data/augmentation/evaluation seeds2026092705/07/06 are existing capsule seeds; support row seeds are2026092711..15 (support null denotes multi-seed matrix). Phase1 target capsule generator seed must be read from existing manifest; unknown here, not inferred. No new target channel draw.",
                "k": None if stage == "source" else [1,5,10,20], "scenario": ["practical_high", "practical_mid", "practical_low_urban"],
                "optimizer": optimizer if stage == "source" else "method-specific matched frozen configuration", "lr": lr if stage == "source" else None,
                "epochs": 200 if stage == "source" else None, "fl_rounds": None, "budget_ref": relative,
                "output_root": output, "log_path": log, "command": command, "expected_artifacts": records[run]["expected_artifacts"]})
    # Complete schema preflight before the first record mutation.
    for spec in records.values():
        errors = registry.validate_spec(spec, ready=True)
        if errors:
            raise ValueError("\n".join(errors))
    for run, spec in records.items():
        prereg = root / "configs/comparison" / (run + ".registration.json")
        write_json(prereg, spec)
        registry.register(root, prereg)
        for relative, config in configs[run]:
            write_json(root / relative, config)
    write_json(root / "automation_reports/CV-SincNet" / PIPELINE_RUN / "launch_spec.json", combined)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-root", default="/home/szu2070436088/2510044040/CV-SincNet")
    parser.add_argument("--registry-module", type=Path)
    args = parser.parse_args()
    spec = prepare(args.root, args.remote_root, args.registry_module)
    print(json.dumps({"run_id": spec["run_id"], "source_rows": 45, "phase12_rows": 45}))
