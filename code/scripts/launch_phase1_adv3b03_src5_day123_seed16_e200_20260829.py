#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Sequence


RUN_ID_DEFAULT = "phase1_adv3b03_src5_day123_seed16_e200_20260829_r1"
FORMAL_SEEDS = tuple(range(713101, 713117))
SCENARIOS = "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SAT_SCHEDULE = (
    "1@0.30:leo_clear_weak;"
    "41@0.60:leo_low_elev_weak,leo_rain_weak;"
    "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
)


class PlanRow(NamedTuple):
    seed: int
    gpu: int
    candidate_id: str


def build_plan(seeds: Sequence[int] | None = None) -> list[PlanRow]:
    selected = list(seeds) if seeds is not None else list(FORMAL_SEEDS)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("seeds must be a non-empty unique sequence")
    return [
        PlanRow(
            seed=int(seed),
            gpu=index % 8,
            candidate_id=f"S{int(seed)}_ADV3B03_MU10_ALPHA20_E200",
        )
        for index, seed in enumerate(selected)
    ]


def build_train_command(
    row: PlanRow,
    *,
    root: Path,
    code_root: Path,
    python: Path,
    runs_root: Path,
    wisig_pkl: Path,
    epochs: int,
    run_id: str = RUN_ID_DEFAULT,
) -> list[str]:
    label_epochs = min(130, int(epochs))
    pseudo_epochs = max(0, int(epochs) - label_epochs)
    candidate_root = runs_root / row.candidate_id
    return [
        str(python),
        "-u",
        str(code_root / "code" / "SSDG" / "train_ssdg.py"),
        "--wisig_pkl", str(wisig_pkl),
        "--wisig_equalized", "1",
        "--wisig_train_days", "1,2,3",
        "--wisig_test_days", "",
        "--wisig_train_rxs", "1,3,4,6,8",
        "--wisig_test_rxs", "",
        "--wisig_allow_shared_days_if_receivers_disjoint", "false",
        "--phase1_source_only_eval", "true",
        "--phase1_external_final_eval", "true",
        "--split_mode", "tx_rx_day_1_7_2",
        "--labeled_ratio", "0.07",
        "--unlabeled_ratio", "0.63",
        "--source_val_ratio", "0.30",
        "--source_cal_ratio", "0.15",
        "--source_select_ratio", "0.15",
        "--phase1_source_role_protocol", "l_s_u_s_v_cal_v_select",
        "--output_dir", str(candidate_root),
        "--run_id", str(run_id),
        "--candidate_id", row.candidate_id,
        "--base_candidate", "ADV3B03_MU10_ALPHA20_E200",
        "--epochs", str(int(epochs)),
        "--batch_size", "128",
        "--eval_batch_size", "512",
        "--label_epochs", str(label_epochs),
        "--pseudo_epochs", str(pseudo_epochs),
        "--from_scratch", "true",
        "--phase1_source_val_selection_only", "true",
        "--checkpoint_selection", "final_only",
        "--best_metric", "source_val_sat_hmean",
        "--source_val_heavy_eval_start_epoch", str(int(epochs)),
        "--source_val_heavy_eval_interval", "1",
        "--source_val_heavy_eval_final_window", "1",
        "--source_val_heavy_eval_final_interval", "1",
        "--paic_guard_enabled", "true",
        "--paic_guard_sat_ce_delta", "0.12",
        "--paic_guard_grad_delta", "3.0",
        "--paic_guard_reliable_drop", "0.01",
        "--paic_guard_cooldown_epochs", "1",
        "--paic_guard_sat_scale", "0.75",
        "--use_feature_masks", "true",
        "--use_txrx_geometry_losses", "true",
        "--use_tx_rx_balanced_sampler", "false",
        "--phase1_distribution_audit_only", "true",
        "--lambda_tx_proto", "0",
        "--lambda_rx_proto", "0",
        "--lambda_mask_aux", "0",
        "--lambda_tx_supcon_masked", "0",
        "--lambda_rx_supcon_masked", "0",
        "--lambda_txrx_rect", "0",
        "--use_proto_memory", "true",
        "--lambda_proto", "0.0032",
        "--proto_domain_align_weight", "0.10",
        "--proto_margin", "0.15",
        "--proto_push_weight", "0.10",
        "--proto_min_count", "2",
        "--lambda_open_world_feat", "0.0024",
        "--ow_feat_start_epoch", "12",
        "--ow_feat_warmup_epochs", "25",
        "--ow_feat_radius_deg", "12",
        "--ow_feat_inter_margin_deg", "55",
        "--ow_feat_sample_margin_deg", "5",
        "--ow_feat_domain_align_weight", "0",
        "--ow_feat_min_classes", "2",
        "--ow_feat_min_samples_per_class", "1",
        "--ow_feat_tail_mode", "robust_3sigma",
        "--ow_feat_tail_weight", "0.14",
        "--ow_feat_cvar_alpha", "0.95",
        "--ow_feat_vacuum_weight", "0.40",
        "--ow_feat_vacuum_width_deg", "6",
        "--ow_feat_vacuum_hard_k", "3",
        "--lambda_zid_compact", "0.032",
        "--zid_compact_start_epoch", "8",
        "--zid_compact_warmup_epochs", "25",
        "--zid_compact_supcon_weight", "0.30",
        "--zid_compact_radius_weight", "0.35",
        "--zid_compact_cvar_weight", "0.35",
        "--zid_compact_cvar_alpha", "0.95",
        "--zid_compact_radius_deg", "40",
        "--zid_compact_domain_aware", "true",
        "--lambda_proxy_unknown", "0.0050",
        "--proxy_unknown_start_epoch", "45",
        "--proxy_unknown_warmup_epochs", "25",
        "--proxy_unknown_holdout_tx_per_batch", "1",
        "--proxy_unknown_virtual_count", "48",
        "--proxy_unknown_virtual_mode", "hard",
        "--proxy_unknown_energy_margin", "0.0",
        "--proxy_unknown_energy_temperature", "1.0",
        "--proxy_unknown_placeholder_weight", "0.0",
        "--proxy_unknown_virtual_detach", "false",
        "--proxy_unknown_vacuum_weight", "0.55",
        "--proxy_unknown_vacuum_width_deg", "5",
        "--proxy_unknown_vacuum_hard_k", "3",
        "--proxy_unknown_vacuum_radius_deg", "40",
        "--proxy_unknown_core_quantile", "0.85",
        "--proxy_unknown_accept_quantile", "0.80",
        "--proxy_unknown_tail_quantile", "0.92",
        "--proxy_unknown_overflow_quantile", "0.97",
        "--proxy_unknown_vaccept_weight", "1.00",
        "--proxy_unknown_core_accept_weight", "0.35",
        "--proxy_unknown_component_gate_weight", "0.65",
        "--proxy_unknown_tail_quarantine_weight", "0.20",
        "--proxy_unknown_source_safe_weight", "0.20",
        "--proxy_unknown_vaccept_cvar_alpha", "0.20",
        "--proxy_unknown_unknown_margin", "0.10",
        "--proxy_unknown_known_margin", "0.05",
        "--proxy_unknown_energy_softplus_temperature", "0.04",
        "--proxy_unknown_component_temperature_deg", "3.0",
        "--proxy_unknown_component_margin_deg", "4.0",
        "--proxy_unknown_component_margin_temperature_deg", "3.0",
        "--proxy_unknown_shell_width_deg", "4.0",
        "--lambda_soft_unknown_mixup", "0.0045",
        "--soft_unknown_mixup_start_epoch", "25",
        "--soft_unknown_mixup_warmup_epochs", "25",
        "--soft_unknown_mixup_count", "24",
        "--soft_unknown_mixup_order", "3",
        "--soft_unknown_mixup_alpha", "0.5",
        "--soft_unknown_mixup_energy_margin", "1.0",
        "--soft_unknown_mixup_ce_weight", "0.60",
        "--soft_unknown_mixup_energy_weight", "1.0",
        "--soft_unknown_mixup_vacuum_weight", "0.35",
        "--soft_unknown_mixup_vacuum_width_deg", "6",
        "--soft_unknown_mixup_vacuum_hard_k", "3",
        "--soft_unknown_mixup_detach", "false",
        "--lambda_source_episode", "0.0035",
        "--source_episode_start_epoch", "20",
        "--source_episode_warmup_epochs", "25",
        "--source_episode_min_domains", "2",
        "--source_episode_radius_cap_deg", "33",
        "--source_episode_mixup_weight", "0.75",
        "--source_episode_mixup_hard_k", "3",
        "--use_sat_consistency",
        "--use_concat_sat_channel_aug",
        "--concat_sat_ce_only",
        "--concat_sat_ce_weight", "1.0",
        "--sat_training_mode", "concat_masked",
        "--sat_train_scenario", "leo_clear_weak",
        "--sat_train_scenarios", SCENARIOS,
        "--sat_view_schedule", SAT_SCHEDULE,
        "--sat_cons_start_epoch", "80",
        "--lambda_sat_cls", "0.68",
        "--lambda_sat_cons", "0",
        "--lambda_zid_channel_invariance", "0",
        "--zid_channel_pair_weight", "1.0",
        "--lambda_u", "0.16",
        "--lambda_ent", "0.01",
        "--lambda_domain", "1",
        "--lambda_adv", "0.35",
        "--lambda_group_ce", "0.16",
        "--lambda_fishr", "0.04",
        "--max_grad_norm", "5",
        "--tau_min", "0.92",
        "--tau_max", "0.97",
        "--pseudo_quantile", "0.86",
        "--use_ema_teacher", "true",
        "--eval_sat_channel", "true",
        "--eval_sat_scenarios", SCENARIOS,
        "--sat_eval_max_batches", "-1",
        "--device", "cuda:0",
        "--seed", str(row.seed),
    ]


def build_eval_command(
    row: PlanRow,
    *,
    code_root: Path,
    python: Path,
    candidate_root: Path,
    eval_batch_size: int,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(code_root / "code" / "scripts" / "eval_ssdg_sat_per_rx.py"),
        "--ckpt", str(candidate_root / "final_ssdg.pth"),
        "--output_json", str(candidate_root / "metrics_joint.json"),
        "--eval_on", "source_v_select",
        "--group_loader", "source_v_select",
        "--scenarios", SCENARIOS,
        "--device", "cuda:0",
        "--max_batches", "-1",
        "--eval_batch_size", str(int(eval_batch_size)),
        "--sat_seed", str(row.seed),
        "--strict_reconstruction",
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_metrics(candidate_root: Path) -> None:
    joint_path = candidate_root / "metrics_joint.json"
    data = json.loads(joint_path.read_text(encoding="utf-8"))
    audit = data.get("reconstruction_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("missing strict reconstruction audit")
    if audit.get("strict_requested") is not True or audit.get("checkpoint_load_strict") is not True:
        raise RuntimeError("strict reconstruction was not proven")
    if audit.get("fallback_used") is not False:
        raise RuntimeError("fallback reconstruction is forbidden")
    for key in ("missing_keys", "unexpected_keys", "shape_mismatches"):
        if int(audit.get(key, -1)) != 0:
            raise RuntimeError(f"strict reconstruction reported {key}")
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("joint evaluator returned no rows")

    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        prefix = "clean" if scenario == "clean" else "sat"
        selected = rows if scenario == "clean" else [r for r in rows if r.get("scenario") == scenario]
        if not selected:
            raise RuntimeError(f"missing rows for {scenario}")
        normalized: list[dict[str, object]] = []
        seen_clean: set[tuple[object, ...]] = set()
        for source in selected:
            identity = (
                source.get("name"),
                source.get("rx_idx"),
                source.get("rx_label"),
                json.dumps(source.get("days_label"), ensure_ascii=False, sort_keys=True),
            )
            if scenario == "clean" and identity in seen_clean:
                continue
            seen_clean.add(identity)
            correct = int(source[f"{prefix}_correct"])
            total = int(source[f"{prefix}_total"])
            accuracy = float(source[f"{prefix}_acc"])
            if total <= 0 or abs(accuracy - 100.0 * correct / total) > 1e-6:
                raise RuntimeError(f"invalid metric counts for {scenario}")
            normalized.append(
                {
                    "name": source.get("name"),
                    "rx_idx": source.get("rx_idx"),
                    "rx_label": source.get("rx_label"),
                    "days_label": source.get("days_label"),
                    "scenario": scenario,
                    "tx_acc": accuracy,
                    "tx_correct": correct,
                    "tx_total": total,
                }
            )
        correct = sum(int(r["tx_correct"]) for r in normalized)
        total = sum(int(r["tx_total"]) for r in normalized)
        payload = {
            "schema": "ssdg_phase1_scenario_eval_v1",
            "checkpoint": data.get("checkpoint"),
            "checkpoint_epoch": data.get("checkpoint_epoch"),
            "reconstruction_audit": audit,
            "eval_on": data.get("eval_on"),
            "group_loader": data.get("group_loader"),
            "scenario": scenario,
            "aggregate": {
                "scenario": scenario,
                "tx_acc": 100.0 * correct / total,
                "tx_correct": correct,
                "tx_total": total,
            },
            "rows": normalized,
        }
        _write_json(candidate_root / f"metrics_{scenario}.json", payload)
        (candidate_root / f"eval_{scenario}.log").write_text(
            f"scenario={scenario} tx_acc={payload['aggregate']['tx_acc']:.6f} "
            f"correct={correct} total={total}\n",
            encoding="utf-8",
        )


def _run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("wb") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
    return int(completed.returncode)


def run_row(row: PlanRow, args: argparse.Namespace) -> dict[str, object]:
    candidate_root = args.runs_root / row.candidate_id
    candidate_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{args.code_root / 'code'}:{args.code_root}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(row.gpu)
    train = build_train_command(
        row,
        root=args.root,
        code_root=args.code_root,
        python=args.python,
        runs_root=args.runs_root,
        wisig_pkl=args.wisig_pkl,
        epochs=args.epochs,
        run_id=args.run_id,
    )
    evaluate = build_eval_command(
        row,
        code_root=args.code_root,
        python=args.python,
        candidate_root=candidate_root,
        eval_batch_size=args.eval_batch_size,
    )
    _write_json(
        candidate_root / "config.json",
        {
            "run_id": args.run_id,
            "candidate_id": row.candidate_id,
            "method": "ADV3B03_MU10_ALPHA20_E200",
            "seed": row.seed,
            "gpu": row.gpu,
            "epochs": args.epochs,
            "train_days": [1, 2, 3],
            "train_receivers": [1, 3, 4, 6, 8],
            "target_access": False,
            "source_roles": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},
            "final_evaluation": ["clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
            "train_command": train,
            "eval_command": evaluate,
        },
    )
    (candidate_root / "status.txt").write_text("RUNNING\n", encoding="utf-8")
    train_status = _run_logged(train, candidate_root / "train.log", env)
    if train_status != 0:
        (candidate_root / "status.txt").write_text(f"TRAIN_FAILED exit={train_status}\n", encoding="utf-8")
        return {"candidate_id": row.candidate_id, "status": "TRAIN_FAILED", "exit": train_status}
    checkpoint = candidate_root / "final_ssdg.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        (candidate_root / "status.txt").write_text("FINAL_CHECKPOINT_MISSING\n", encoding="utf-8")
        return {"candidate_id": row.candidate_id, "status": "FINAL_CHECKPOINT_MISSING"}
    eval_status = _run_logged(evaluate, candidate_root / "eval_joint.log", env)
    if eval_status != 0:
        (candidate_root / "status.txt").write_text(f"EVAL_FAILED_JOINT exit={eval_status}\n", encoding="utf-8")
        return {"candidate_id": row.candidate_id, "status": "EVAL_FAILED_JOINT", "exit": eval_status}
    try:
        _split_metrics(candidate_root)
    except Exception as exc:
        (candidate_root / "metrics_split_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        (candidate_root / "status.txt").write_text("EVAL_FAILED_METRICS_SPLIT\n", encoding="utf-8")
        return {"candidate_id": row.candidate_id, "status": "EVAL_FAILED_METRICS_SPLIT"}
    (candidate_root / "status.txt").write_text("ARTIFACTS_COMPLETE\n", encoding="utf-8")
    return {"candidate_id": row.candidate_id, "status": "ARTIFACTS_COMPLETE"}


def _parse_seeds(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/szu2070436088/2510044040/CV-SincNet"))
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path("/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"))
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--wisig-pkl", type=Path)
    parser.add_argument("--seeds", default=",".join(str(v) for v in FORMAL_SEEDS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.code_root = args.code_root or args.root
    args.runs_root = args.runs_root or args.root / "runs" / args.run_id
    args.log_root = args.log_root or args.root / "logs" / args.run_id
    args.wisig_pkl = args.wisig_pkl or args.root / "Dataset_WigSig" / "ManySig.pkl"
    selected_seeds = _parse_seeds(args.seeds)
    if args.smoke:
        if args.run_id == RUN_ID_DEFAULT or "smoke" not in args.run_id.lower():
            print("[ADV3B03-ERROR] smoke mode requires a distinct smoke run ID", file=sys.stderr)
            return 2
        if selected_seeds != [FORMAL_SEEDS[0]] or args.epochs != 1:
            print(
                f"[ADV3B03-ERROR] smoke mode is fixed to seed{FORMAL_SEEDS[0]} and E1",
                file=sys.stderr,
            )
            return 2
    else:
        if args.run_id != RUN_ID_DEFAULT:
            print("[ADV3B03-ERROR] formal mode requires the frozen formal run ID", file=sys.stderr)
            return 2
        if tuple(selected_seeds) != FORMAL_SEEDS or args.epochs != 200:
            print(
                f"[ADV3B03-ERROR] formal mode requires seeds"
                f"{','.join(str(seed) for seed in FORMAL_SEEDS)} and E200",
                file=sys.stderr,
            )
            return 2
    rows = build_plan(selected_seeds)
    if args.smoke:
        rows = [
            row._replace(candidate_id=f"S{row.seed}_ADV3B03_MU10_ALPHA20_E200_SMOKE_E1")
            for row in rows
        ]
    plan = []
    for row in rows:
        candidate_root = args.runs_root / row.candidate_id
        plan.append(
            {
                "seed": row.seed,
                "gpu": row.gpu,
                "candidate_id": row.candidate_id,
                "candidate_root": str(candidate_root),
                "train_command": build_train_command(
                    row,
                    root=args.root,
                    code_root=args.code_root,
                    python=args.python,
                    runs_root=args.runs_root,
                    wisig_pkl=args.wisig_pkl,
                    epochs=args.epochs,
                    run_id=args.run_id,
                ),
                "eval_command": build_eval_command(
                    row,
                    code_root=args.code_root,
                    python=args.python,
                    candidate_root=candidate_root,
                    eval_batch_size=args.eval_batch_size,
                ),
            }
        )
    if args.dry_run:
        print(json.dumps({"run_id": args.run_id, "rows": plan}, ensure_ascii=False, indent=2))
        return 0
    if args.runs_root.exists() or args.log_root.exists():
        print("[ADV3B03-ERROR] refusing to overwrite existing run/log root", file=sys.stderr)
        return 3
    args.runs_root.mkdir(parents=True, exist_ok=False)
    args.log_root.mkdir(parents=True, exist_ok=False)
    _write_json(args.log_root / "plan.json", {"run_id": args.run_id, "rows": plan})
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = {pool.submit(run_row, row, args): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"candidate_id": row.candidate_id, "status": "DISPATCH_FAILED", "error": str(exc)}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    _write_json(args.log_root / "final_status.json", sorted(results, key=lambda item: str(item["candidate_id"])))
    return 0 if all(item.get("status") == "ARTIFACTS_COMPLETE" for item in results) else 4


if __name__ == "__main__":
    raise SystemExit(main())
