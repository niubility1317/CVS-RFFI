"""Resume-safe Stage2-C 5/10/20-new extreme-light matrix runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from paper_reproduction.common.config import load_json_config
from paper_reproduction.cvs_aligned.cvs_method_runner import run, validate_config


DEFAULT_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
DEV_SEEDS = (713101, 713102, 713103, 713104, 713105)
CONFIRM_SEEDS = (713106, 713107, 713108, 713109, 713110)
NEW_CLASS_COUNTS = (5, 10, 20)
ARMS: dict[str, dict[str, Any]] = {
    "baseline_single_qknn": {
        "qknnv42_head_mode": "qknn",
        "qknnv42_aux_score_weight": 0.34,
        "qknnv42_labelprop_mode": "support_prototype",
        "qknnv42_support_representation": "all_support",
        "qknnv42_feature_adapter_mode": "support_diag_whiten_fisher",
        "qknnv42_decision_mode": "per_sample_argmax",
    },
    "el_diag_aux1p5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 1.5,
    },
    "el_diag_aux2p0": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
    },
    "el_zid_anchor5_e5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 0.0,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_epochs": 5,
    },
    "el_aux0p5_anchor5_e5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 0.5,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_epochs": 5,
    },
    "el_aux1p0_anchor5_e5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 1.0,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_epochs": 5,
    },
    "el_aux1p5_anchor5_e5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 1.5,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_epochs": 5,
    },
    "el_aux2p0_anchor5_e5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_epochs": 5,
    },
    "el_aux1p5_anchor20_e5": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 1.5,
        "extreme_light_prototype_anchor_weight": 20.0,
        "extreme_light_epochs": 5,
    },
    "el_aux2p0_anchor5_e20": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_epochs": 20,
    },
    "el_aux2p0_anchor20_e20": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_prototype_anchor_weight": 20.0,
        "extreme_light_epochs": 20,
    },
    "el_aux2p0_anchor5_noise5_e20": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_prototype_anchor_weight": 5.0,
        "extreme_light_feature_noise_std": 0.05,
        "extreme_light_epochs": 20,
    },
    "el_aux2p0_anchor20_noise5_e20": {
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_prototype_anchor_weight": 20.0,
        "extreme_light_feature_noise_std": 0.05,
        "extreme_light_epochs": 20,
    },
    "el_proto_zid": {
        "qknnv42_head_mode": "extreme_light_prototype_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 0.0,
        "extreme_light_epochs": 0,
    },
    "el_proto_aux0p5": {
        "qknnv42_head_mode": "extreme_light_prototype_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 0.5,
        "extreme_light_epochs": 0,
    },
    "el_proto_aux1p0": {
        "qknnv42_head_mode": "extreme_light_prototype_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 1.0,
        "extreme_light_epochs": 0,
    },
    "el_proto_aux1p5": {
        "qknnv42_head_mode": "extreme_light_prototype_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 1.5,
        "extreme_light_epochs": 0,
    },
    "el_proto_aux2p0": {
        "qknnv42_head_mode": "extreme_light_prototype_cosine",
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_decision_mode": "per_sample_argmax",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_epochs": 0,
    },
}

CORE_ARMS = ("baseline_single_qknn", "el_diag_aux1p5", "el_diag_aux2p0")


@dataclass(frozen=True)
class MatrixRow:
    index: int
    arm: str
    new_class_count: int
    receiver: str
    seed: int
    k_shot: int
    experiment_id: str
    run_dir: str
    log_path: str


def _strings(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("string grid must not be empty")
    return values


def _ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer grid must contain positive values")
    return values


def build_rows(
    *,
    arms: tuple[str, ...],
    new_class_counts: tuple[int, ...],
    receivers: tuple[str, ...],
    seeds: tuple[int, ...],
    k_grid: tuple[int, ...],
    output_root: Path,
    log_root: Path,
) -> list[MatrixRow]:
    unknown_arms = sorted(set(arms) - set(ARMS))
    if unknown_arms:
        raise ValueError(f"unknown arms: {unknown_arms}")
    if not set(new_class_counts) <= set(NEW_CLASS_COUNTS):
        raise ValueError(f"new-class counts must be a subset of {NEW_CLASS_COUNTS}")
    rows: list[MatrixRow] = []
    for arm in arms:
        for count in new_class_counts:
            for receiver in receivers:
                for seed in seeds:
                    for k_shot in k_grid:
                        experiment_id = (
                            f"{arm}_n{count}_rx{receiver}_k{k_shot}_seed{seed}"
                        )
                        run_dir = (
                            output_root
                            / arm
                            / f"new_{count}"
                            / f"rx_{receiver}"
                            / f"seed_{seed}"
                            / f"k_{k_shot}"
                        )
                        log_path = (
                            log_root
                            / arm
                            / f"new_{count}"
                            / f"rx_{receiver}"
                            / f"seed_{seed}"
                            / f"k_{k_shot}.log"
                        )
                        rows.append(
                            MatrixRow(
                                index=len(rows),
                                arm=arm,
                                new_class_count=count,
                                receiver=receiver,
                                seed=seed,
                                k_shot=k_shot,
                                experiment_id=experiment_id,
                                run_dir=str(run_dir),
                                log_path=str(log_path),
                            )
                        )
    return rows


def row_config(base: dict[str, Any], row: MatrixRow, *, device: str) -> dict[str, Any]:
    labels = [str(value) for value in base["target_new_tx_labels"]]
    if len(labels) < int(row.new_class_count):
        raise ValueError("base config does not contain enough nested target-new labels")
    config = dict(base)
    config.update(ARMS[row.arm])
    config.update(
        {
            "experiment_id": row.experiment_id,
            "target_receiver_labels": [row.receiver],
            "target_new_tx_labels": labels[: int(row.new_class_count)],
            "seed": int(row.seed),
            "split_seed": int(row.seed),
            "k_shot": int(row.k_shot),
            "_runtime_device": str(device),
            "matrix_arm": row.arm,
            "matrix_new_class_count": int(row.new_class_count),
        }
    )
    validate_config(config)
    return config


def _signature(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _complete(run_dir: Path, signature: str) -> bool:
    required = (
        "metrics.json",
        "split_manifest.json",
        "resolved_config.json",
        "score_table.csv",
        "detailed_metrics.json",
        "detailed_metrics.csv",
        "loss_trace.json",
        "loss_trace.csv",
        "row_manifest.json",
    )
    if any(not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0 for name in required):
        return False
    manifest = json.loads((run_dir / "row_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("config_sha256")) != signature:
        raise ValueError(f"run-path collision with different config: {run_dir}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "dev", "confirm"), required=True)
    parser.add_argument("--arms", default=None)
    parser.add_argument("--receivers", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--k-grid", default=None)
    parser.add_argument("--new-class-counts", default="5,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0,shard_count)")
    defaults = {
        "smoke": {
            "arms": CORE_ARMS,
            "receivers": ("20-1", "8-8"),
            "seeds": DEV_SEEDS[:2],
            "k_grid": (20,),
        },
        "dev": {
            "arms": CORE_ARMS,
            "receivers": DEFAULT_RECEIVERS,
            "seeds": DEV_SEEDS,
            "k_grid": (10, 15, 20),
        },
        "confirm": {
            "arms": ("el_diag_aux2p0",),
            "receivers": DEFAULT_RECEIVERS,
            "seeds": CONFIRM_SEEDS,
            "k_grid": (20,),
        },
    }[args.mode]
    arms = _strings(args.arms) if args.arms else defaults["arms"]
    receivers = _strings(args.receivers) if args.receivers else defaults["receivers"]
    seeds = _ints(args.seeds) if args.seeds else defaults["seeds"]
    k_grid = _ints(args.k_grid) if args.k_grid else defaults["k_grid"]
    counts = _ints(args.new_class_counts)
    base = load_json_config(args.config)
    rows = build_rows(
        arms=arms,
        new_class_counts=counts,
        receivers=receivers,
        seeds=seeds,
        k_grid=k_grid,
        output_root=args.output_root,
        log_root=args.log_root,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "cvs_stage2c_extreme_light_matrix_v1",
        "mode": args.mode,
        "arms": list(arms),
        "new_class_counts": list(counts),
        "receivers": list(receivers),
        "seeds": list(seeds),
        "k_grid": list(k_grid),
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
        "success_thresholds": base.get("success_thresholds", {}),
        "query_labels_used_for_training": False,
        "query_role_or_quota_oracle": False,
    }
    manifest_path = args.output_root / "matrix_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"matrix manifest collision: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    shard_rows = [row for row in rows if row.index % args.shard_count == args.shard_index]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "total_rows": len(rows),
                    "shard_rows": len(shard_rows),
                    "manifest": str(manifest_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    counts_out = {"completed": 0, "skipped": 0, "failed": 0}
    for row in shard_rows:
        config = row_config(base, row, device=args.device)
        signature = _signature(config)
        run_dir = Path(row.run_dir)
        log_path = Path(row.log_path)
        if _complete(run_dir, signature):
            counts_out["skipped"] += 1
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write("[ROW-START] " + json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
                handle.write("[CONFIG] " + json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n")
            result = run(config, run_dir)
            trace = json.loads((run_dir / "loss_trace.json").read_text(encoding="utf-8"))
            with log_path.open("a", encoding="utf-8") as handle:
                for trace_row in trace:
                    handle.write("[LOSS-TRACE] " + json.dumps(trace_row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.write("[ROW-COMPLETE] " + json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True) + "\n")
            (run_dir / "row_manifest.json").write_text(
                json.dumps(
                    {"config_sha256": signature, "row": asdict(row)},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if not _complete(run_dir, signature):
                raise ValueError("row artifact completion check failed")
            counts_out["completed"] += 1
        except Exception:
            counts_out["failed"] += 1
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("[ROW-FAILED]\n" + traceback.format_exc() + "\n")
    summary = {
        "mode": args.mode,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_row_count": len(shard_rows),
        **counts_out,
    }
    (args.output_root / f"worker_{args.shard_index}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if counts_out["failed"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
