"""Emit source-development configs and commands; never launch experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.game_tracking.config import parser as training_parser, validate

DEFAULT_ROWS = ("B0", "B1", "B2", "B4", "B5", "B6", "S1", "S3", "S4", "C1", "C2")
DEFAULT_SEEDS = (392002, 392003)
REAUDIT_PRIORITY = ('J1','H1','B7','B8','C2','S4','S3','S1','C1')


def row_menu(replay_path=""):
    """B3 candidates require source-only selection; none is claimed tuned."""
    rows = {
        "B0": ("matched CORE90 scratch baseline without audits", {"game_no_audit": True}),
        "B1": ("source audits with ordinary updates", {}),
        "B2": ("identity-domain adversarial field disabled; domain retention remains", {"lambda_adv": 0.}),
        "B3": ("source learning-rate candidate: half baseline", {"lr": 1e-4}),
        "B3_lr_high": ("source learning-rate candidate: twice baseline", {"lr": 4e-4}),
        "B3_head_slow": ("source head-timescale candidate", {"game_head_lr_ratio": .5}),
        "B3_head_fast": ("source head-timescale candidate", {"game_head_lr_ratio": 2.}),
        "B3_head_scale": ("separate head CE scale; encoder keeps adversarial coefficient", {"game_head_scale":"separate_head_scale"}),
        "B3_adv_low": ("source adversarial-scale candidate", {"lambda_adv": .175}),
        "B3_adv_high": ("source adversarial-scale candidate", {"lambda_adv": .7}),
        "B4": ("alternating with two total head steps", {"game_solver": "alternating", "game_fixed_head_steps": 2}),
        "B4_fixedk": ("fixed head catch-up plus ordinary coupled update", {"game_fixed_head_steps": 2}),
        "B5": ("complete extragradient", {"game_solver": "extragradient"}),
        "B6": ("Heun average-gradient update", {"game_solver": "heun"}),
        "B7": ("optimistic field update with event resets", {"game_solver": "optimistic"}),
        "B8": ("head-only predictor approximation", {"game_solver": "head_lookahead"}),
        "S1": ("source-triggered bounded head catch-up", {"game_control": "catchup"}),
        "S2": ("fixed correction schedule cost control", {"game_control": "fixed"}),
        "S2_random": ("random correction schedule cost control", {"game_control": "random"}),
        "S3": ("source-triggered correction", {"game_control": "correction"}),
        "S4": ("source-triggered catch-up and correction", {"game_control": "both"}),
        "C1": ("source capability curriculum with ordinary updates", {"game_curriculum": "capability"}),
        "C2": ("source capability curriculum and both controls", {"game_control": "both", "game_curriculum": "capability"}),
        "C3": ("capability curriculum with random correction timing", {"game_control": "random", "game_curriculum": "capability"}),
        "H1": ("optional implicit head-response experiment", {"game_response_tracking": True}),
        "J1": ("full active-objective local Jacobian diagnostic", {"game_jacobian_interval":500}),
    }
    if replay_path:
        rows["C4"] = ("cross-seed replay of a frozen source action schedule",
                      {"game_control": "replay", "game_actions_replay": str(replay_path)})
    return rows


def build_matrix(*, rows=DEFAULT_ROWS, seeds=DEFAULT_SEEDS, runs_root="runs/core90_game",
                 replay_path="", dataset_path=None, device="cuda:0"):
    if not seeds or len(set(seeds)) != len(seeds) or any(int(seed) < 0 for seed in seeds):
        raise ValueError("Specify distinct training seeds")
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Specify distinct experiment rows")
    menu = row_menu(replay_path)
    unknown = set(rows) - set(menu)
    if unknown:
        raise ValueError(f"Unknown or unavailable rows {sorted(unknown)}; C4 requires explicit source replay path")
    result = []
    p = training_parser()
    known = {action.dest for action in p._actions}
    for row in rows:
        description, override = menu[row]
        for seed in seeds:
            run_id = f"{row}_seed{int(seed)}"
            config = {
                "epochs": 200, "label_epochs": 130, "pseudo_epochs": 70,
                "seed": int(seed), "game_split_seed": 392002,
                "from_scratch": True, "baseline_ckpt": "", "game_resume": "",
                "labeled_ratio": .07, "unlabeled_ratio": .63, "source_val_ratio": .30,
                "game_solver": "simultaneous", "game_optimizer": "adamw",
                "game_control": "off", "game_curriculum": "fixed", "game_no_audit": False,
                "game_fixed_head_steps": 0, "game_response_tracking": False,
                "game_actions_replay": "", "game_max_steps_per_epoch": 0,
                "game_synthetic": False, "game_skip_final_eval": False,
                "game_export_source_predictions": True, "game_correction_fraction": .20,
                "output_dir": str(Path(runs_root) / run_id), "run_name": run_id,
                "metrics_csv": "", "metrics_jsonl": "", "device": device,
            }
            if dataset_path is not None:
                config["wisig_pkl"] = str(dataset_path)
            config.update(override)
            if set(config) - known:
                raise ValueError("Matrix used unknown training configuration keys")
            args = p.parse_args(["--output_dir", config["output_dir"]])
            for key, value in config.items():
                setattr(args, key, value)
            validate(args)
            result.append({"row": row, "run_id": run_id, "seed": seed,
                           "description": description, "config": config})
    return result


def write_matrix(output_dir, **kwargs):
    output = Path(output_dir)
    rows = build_matrix(**kwargs)
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / (row["run_id"] + ".json") for row in rows]
    if any(path.exists() for path in destinations) or (output / "matrix.json").exists():
        raise FileExistsError("Matrix artifacts already exist; choose a new output directory")
    for row, destination in zip(rows, destinations):
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(row["config"], stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        # Keep an argv array: schedulers must not interpolate it through a shell.
        row["config_path"] = str(destination.resolve())
        row["argv"] = ["code/SSDG/train_core90_game.py", "--output_dir", row["config"]["output_dir"],
                       "--game_config_json", str(destination.resolve())]
    manifest = {"schema": "core90_game_matrix_v1", "status": "CONFIGURED_NOT_LAUNCHED",
                "source_development_only": True, "target_evaluation": False,
                "checkpoint_selection": "fixed_final_E200", "split_seed": 392002,
                "budget_matching": "Compare realized field/head/audit counts and wall time; fractions alone are not matched cost",
                "initial_default_rows": list(DEFAULT_ROWS), "runs": rows}
    with (output / "matrix.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runs-root", default="runs/core90_game")
    p.add_argument("--rows", default=",".join(DEFAULT_ROWS), help="Comma-separated IDs or all")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--replay-path", default="")
    p.add_argument("--dataset-path")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)
    rows = (tuple(row_menu(a.replay_path)) if a.rows == "all" else
            REAUDIT_PRIORITY+tuple(r for r in row_menu(a.replay_path) if r not in REAUDIT_PRIORITY)
            if a.rows=='reaudit' else tuple(a.rows.split(",")))
    manifest = write_matrix(a.output_dir, rows=rows, seeds=tuple(int(v) for v in a.seeds.split(",")),
                            runs_root=a.runs_root, replay_path=a.replay_path,
                            dataset_path=a.dataset_path, device=a.device)
    print(json.dumps({"status": manifest["status"], "runs": len(manifest["runs"]),
                      "manifest": str((Path(a.output_dir) / "matrix.json").resolve())}))


if __name__ == "__main__":
    main()
