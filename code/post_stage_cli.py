from __future__ import annotations

import argparse
from typing import Any, Dict


MAIN_SAT_EVAL_ON = "test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx"


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def default_data_args() -> Dict[str, Any]:
    return {
        "dataset": "wisig",
        "dataset_dir": "./Dataset_ORALCE",
        "run_name": "post_stage",
        "wisig_pkl": "./Dataset_WigSig/ManySig.pkl",
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
        "wisig_out_len": 256,
        "wisig_train_ratio": 0.2,
        "wisig_val_ratio": -1.0,
        "wisig_guard_gap": 8,
        "wisig_train_days": "0,1",
        "wisig_test_days": "2,3",
        "wisig_train_rxs": "0,1,2,3,4,5,6",
        "wisig_test_rxs": "7,8,9,10,11",
        "wisig_split_seed": -1,
        "allow_source_target_day_overlap_by_disjoint_rx": False,
        "wisig_split_strategy": "random",
        "wisig_cap_strategy": "random",
        "wisig_max_day123_per_combo": 0,
        "wisig_max_train_per_combo": 0,
        "wisig_max_val_per_combo": 0,
        "wisig_max_test_per_combo": 0,
        "num_classes": 16,
        "batch_size": 128,
        "eval_batch_size": 256,
        "num_workers": 4,
        "prefetch_factor": 2,
        "device": "cuda:0",
        "seed": 1337,
        "eval_max_batches": 0,
        "sample_rate_hz": 0.0,
    }


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    defaults = default_data_args()
    parser.add_argument("--dataset", type=str, default=defaults["dataset"], choices=["wisig", "oralce"])
    parser.add_argument("--dataset_dir", type=str, default=defaults["dataset_dir"])
    parser.add_argument("--run_name", type=str, default=defaults["run_name"])
    parser.add_argument("--wisig_pkl", type=str, default=defaults["wisig_pkl"])
    parser.add_argument("--wisig_equalized", type=str, default=defaults["wisig_equalized"])
    parser.add_argument("--wisig_domain", type=str, default=defaults["wisig_domain"], choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=defaults["wisig_out_len"])
    parser.add_argument("--wisig_train_ratio", type=float, default=defaults["wisig_train_ratio"])
    parser.add_argument("--wisig_val_ratio", type=float, default=defaults["wisig_val_ratio"])
    parser.add_argument("--wisig_guard_gap", type=int, default=defaults["wisig_guard_gap"])
    parser.add_argument("--wisig_train_days", type=str, default=defaults["wisig_train_days"])
    parser.add_argument("--wisig_test_days", type=str, default=defaults["wisig_test_days"])
    parser.add_argument("--wisig_train_rxs", type=str, default=defaults["wisig_train_rxs"])
    parser.add_argument("--wisig_test_rxs", type=str, default=defaults["wisig_test_rxs"])
    parser.add_argument("--wisig_split_seed", type=int, default=defaults["wisig_split_seed"])
    parser.add_argument(
        "--allow_source_target_day_overlap_by_disjoint_rx",
        type=str2bool,
        default=defaults["allow_source_target_day_overlap_by_disjoint_rx"],
    )
    parser.add_argument("--wisig_split_strategy", type=str, default=defaults["wisig_split_strategy"], choices=["random", "contiguous"])
    parser.add_argument("--wisig_cap_strategy", type=str, default=defaults["wisig_cap_strategy"], choices=["random", "front"])
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=defaults["wisig_max_day123_per_combo"])
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=defaults["wisig_max_train_per_combo"])
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=defaults["wisig_max_val_per_combo"])
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=defaults["wisig_max_test_per_combo"])
    parser.add_argument("--num_classes", type=int, default=defaults["num_classes"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--prefetch_factor", type=int, default=defaults["prefetch_factor"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--eval_max_batches", type=int, default=defaults["eval_max_batches"])
    parser.add_argument("--sample_rate_hz", type=float, default=defaults["sample_rate_hz"])


def add_optional_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, nargs="?", const=True, type=str2bool, help=help_text)
    group.add_argument(f"--no_{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})


def add_sat_eval_args(parser: argparse.ArgumentParser) -> None:
    add_optional_bool_flag(parser, "eval_sat_channel", False, "Enable satellite-channel OOD evaluation after each epoch")
    parser.add_argument(
        "--eval_sat_scenarios",
        type=str,
        default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        help=(
            "Satellite scenarios to evaluate. The CVS deployment-primary default is "
            "leo_clear_weak,leo_low_elev_weak,leo_rain_weak; legacy full-physics scenarios "
            "must be requested explicitly as diagnostic stress."
        ),
    )
    parser.add_argument(
        "--eval_sat_on",
        type=str,
        default=MAIN_SAT_EVAL_ON,
        help="Named test loaders for satellite evaluation: main, all, or comma-separated names. Defaults to the three main OOD splits.",
    )
    parser.add_argument("--sat_eval_max_batches", type=int, default=-1)
    parser.add_argument("--sat_seed", type=int, default=2027)
