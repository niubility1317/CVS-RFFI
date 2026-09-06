"""Native local launcher. Default prints an auditable JSON plan without side effects."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvsrffi.ecrs_config import (DEFAULTS, LOSS_DEFAULTS, SOURCE_PROTOCOL, UNAVAILABLE_ROWS,
                                experiment_matrix, should_validate_source,
                                validate_ecrs_revision_args)


def build_plan(root, run_root, wisig_pkl, rows, seed=392005, epochs=200, final_target_eval=False):
    if final_target_eval:
        raise ValueError("final target evaluation requires an independent frozen-checkpoint evaluator; training remains source-only")
    root, run_root, wisig_pkl = Path(root).resolve(), Path(run_root).resolve(), Path(wisig_pkl).resolve()
    if wisig_pkl.name.lower() != "manysig.pkl":
        raise ValueError("source screening requires ManySig.pkl")
    if run_root.exists():
        raise FileExistsError("unique run-root already exists: " + str(run_root))
    if len(set(rows)) != len(rows) or not rows:
        raise ValueError("rows must be nonempty and unique")
    matrix = experiment_matrix()
    common = {
        "dataset": "wisig", "wisig_pkl": str(wisig_pkl), "wisig_protocol": "cvs_day_rx",
        "wisig_equalized": 1, "wisig_domain": "rx_day", "wisig_out_len": 256,
        "wisig_train_days": "1,2,3", "wisig_test_days": "0,1,2,3",
        "wisig_train_rxs": "1,3,4,6,8", "wisig_test_rxs": "0,2,5,7,9,10,11",
        "wisig_split_strategy": "random", "wisig_cap_strategy": "random",
        "wisig_max_test_per_combo": 0, "meta_ssl_max_samples_per_combo_source": 0,
        "seed": seed, "ssl_labeled_ratio": .07, "ssl_unlabeled_ratio": .63,
        "ssl_val_ratio": .30, "model_size": "M", "model_variant": "lite_d",
        "exp_group": "s4_stagewise_full_dual", "slim_group": "none",
        "arch_family": "cvsincnet", "batch_size": 128, "eval_batch_size": 256,
        "lr": .0002, "lr_min": .000001, "wd": .0001, "label_smoothing": .01,
        "branch_ablation": "no_dac", "domain_branch_ablation": "no_stats", "epochs": epochs,
        "test_eval_policy": "interval_final", "test_eval_start_epoch": epochs,
        "test_eval_interval": epochs, "test_eval_final_window": 0, "test_eval_final_interval": 0,
        "concat_sat_start_epoch": 1, "concat_sat_ce_start_epoch": 80,
        "concat_sat_ce_weight": .68, "lambda_sat_cls": .68, "lambda_sat_cons": 0,
        "sat_view_schedule": "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "eval_sat_scenarios": "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "ecrs_alpha_resp": .15, "ecrs_ridge_alpha": .01, "ecrs_basis_mode": "fixed_spline",
        "lambda_ecrs_canonical": .10, "lambda_ecrs_content": .10,
        "lambda_ecrs_cycle": .10, "lambda_ecrs_split_fit": .10,
        "lambda_ecrs_pair_cross": .10, "lambda_ecrs_pair_surface": .03,
        "lambda_ecrs_same_tx": .05, "lambda_ecrs_diff_tx": .03,
        "lambda_ecrs_gate": 0.,
    }
    commands = []
    for row in rows:
        if row in UNAVAILABLE_ROWS:
            raise ValueError(row + " unavailable: " + UNAVAILABLE_ROWS[row])
        if row not in matrix:
            raise ValueError("unknown row: " + row)
        config = {"ecrs_" + k: v for k, v in DEFAULTS.items()}
        config.update(LOSS_DEFAULTS)
        config.update(common)
        config.update(matrix[row])
        if config["ecrs_version"] == "v2" or row in ("B0", "B1"):
            # V2's frozen physical diagnostics are not training objectives.
            for name in common:
                if name.startswith("lambda_ecrs_"):
                    config[name] = 0.
        if row in ("B0", "B1"):
            for name in LOSS_DEFAULTS:
                config[name] = 0.
            config["ecrs_alpha_resp"] = 0.
        config["ecrs_source_screen_only"] = not final_target_eval
        validate_ecrs_revision_args(argparse.Namespace(**config))
        out = run_root / row
        config.update(run_name=row, output_dir=str(out), log_dir=str(out / "logs"),
                      best_save_path=str(out / "best.pth"), latest_save_path=str(out / "latest.pth"))
        cmd = [sys.executable, "-u", str(root / "code" / "train.py"),
               "--use_meta_ssl_cvs", "--wisig_target_receiver_only_eval",
               "--use_concat_sat_channel_aug", "--concat_sat_ce_only",
               "--no_ecrs_enable_learnable_basis", "--no_ecrs_enable_fasttrust"]
        for key, value in config.items():
            if value is None:
                continue
            if key == "use_ecrs":
                if value:
                    cmd.append("--use_ecrs")
            elif isinstance(value, bool):
                cmd.append(("--" if value else "--no_") + key)
            else:
                cmd.extend(["--" + key, str(value)])
        bridge_interpretation = {
            "B3a": "legacy28 old-reference unified-estimator/readout/frozen-physics bridge package; not a solver-only causal contrast against B2",
            "B3b": "relative to B3a replaces the reference/canonicalization path; shares real3 nuisance, W=I, ridge and readout",
            "B3c": "relative to B3b changes dictionary/scaling/redundancy package from legacy28 to compact8; not a dimension-only contrast",
            "B3": "compact8 estimator package, same numerical configuration as B3c",
        }
        commands.append({"row": row, "config": config, "argv": cmd,
                         "bridge_interpretation": bridge_interpretation.get(row)})
    return {"schema": "ecrs_v1r_v2_launch_v1", "run_root": str(run_root),
            "protocol": SOURCE_PROTOCOL, "selection_domain": "source_only",
            "baseline_route": "matched_ECRS_V1_train_py",
            "baseline_scope": "Existing ECRS V1 launcher and train.py preset; not a full SSDG Core90 objective reproduction",
            "final_target_evaluation": final_target_eval,
            "reference_alignment": "not_verified_by_configuration",
            "source_validation_epochs": [e for e in range(1, epochs + 1) if should_validate_source(e, epochs)],
            "unavailable_rows": UNAVAILABLE_ROWS, "commands": commands}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wisig-pkl", type=Path)
    parser.add_argument("--rows", default="B0,B1,B2")
    parser.add_argument("--seed", type=int, default=392005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--final-target-eval", action="store_true",
                        help="unavailable: requires an independent frozen-checkpoint evaluator")
    parser.add_argument("--execute", action="store_true", help="execute sequentially on this host")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.root, args.run_root, args.wisig_pkl or args.root / "Dataset_WigSig/ManySig.pkl",
                          args.rows.split(","), args.seed, args.epochs, args.final_target_eval)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if args.execute:
            if not Path(plan["commands"][0]["config"]["wisig_pkl"]).is_file():
                raise FileNotFoundError("ManySig.pkl does not exist")
            if not (args.root / "code/train.py").is_file():
                raise FileNotFoundError("train.py does not exist")
            args.run_root.mkdir(parents=True, exist_ok=False)
            (args.run_root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            for command in plan["commands"]:
                subprocess.run(command["argv"], cwd=args.root, check=True)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
