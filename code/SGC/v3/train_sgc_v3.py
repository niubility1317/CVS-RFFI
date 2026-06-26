from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Any, Dict, Mapping

import torch

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from post_stage_common import (
    build_standard_data,
    ensure_dir,
    load_baseline_from_checkpoint,
    load_yaml_or_json,
    mean_logs,
    move_batch,
    resolve_device,
    save_payload,
    set_seed,
)
from cvsrffi.eval import (
    aggregate_named_stats,
    apply_sat_channel_for_scenario,
    evaluate_loader,
    evaluate_named_loaders,
    evaluate_sat_scenarios,
    format_named_test_lines,
    format_sat_test_lines,
)
from cvsrffi.tensors import (
    make_torch_generator,
)
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario

from SGC.v3.losses_v3 import compute_sgc_v3_losses
from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SGC v3 safe adapter on a frozen base teacher.")
    parser.add_argument("--config", type=str, default="SGC/configs/sgc_v3_safe.yaml")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ipfa_blrc", choices=["blrc_only", "ipfa_only", "ipfa_blrc", "target_adapt"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_sgc", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sat_train_scenario", type=str, default="mixed_orbit")
    parser.add_argument("--freeze_teacher", type=str2bool, default=True)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    add_sat_eval_args(parser)
    parser.set_defaults(eval_sat_channel=True)
    return parser


def _cfg_from_file(path: str, num_classes: int, feature_dim: int) -> SGCv3Config:
    raw = load_yaml_or_json(path)
    model_cfg: Dict[str, Any] = dict(raw.get("sgc_v3", raw) or {})
    model_cfg.setdefault("num_classes", int(num_classes))
    model_cfg.setdefault("feature_dim", int(feature_dim))
    return SGCv3Config.from_mapping(model_cfg)


def _infer_feature_dim(teacher, x: torch.Tensor) -> int:
    with torch.no_grad():
        if hasattr(teacher, "extract_feature"):
            return int(teacher.extract_feature(x[:1]).shape[-1])
        out = teacher(x[:1], return_aux=True)
        for key in ("z_id_raw", "z_id", "feat_joint", "feat_cls", "base"):
            value = out.get(key) if isinstance(out, dict) else None
            if torch.is_tensor(value):
                return int(value.shape[-1])
    raise AttributeError("Cannot infer teacher feature_dim.")


def _apply_mode_freeze(model: SGCv3Model, mode: str) -> None:
    if mode == "blrc_only":
        for param in model.feature_adapter.parameters():
            param.requires_grad = False
    elif mode == "ipfa_only":
        for param in model.logit_calibrator.parameters():
            param.requires_grad = False
    for param in model.base_teacher.parameters():
        param.requires_grad = False


class SGCv3EvalAdapter(torch.nn.Module):
    """Expose SGC v3 outputs through the backbone evaluator contract."""

    def __init__(self, sgc_model: SGCv3Model) -> None:
        super().__init__()
        self.sgc_model = sgc_model

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        out = self.sgc_model(x)
        tx_logits = out["logits_final"]
        dom_logits = None
        teacher = self.sgc_model.base_teacher
        try:
            with torch.no_grad():
                teacher_out = teacher(x, y_tx=y_tx, grl_lambda=grl_lambda, return_aux=True, domain_labels=domain_labels)
            if isinstance(teacher_out, Mapping):
                dom_logits = teacher_out.get("dom_logits")
        except Exception:
            dom_logits = None
        if not torch.is_tensor(dom_logits):
            dom_logits = tx_logits.new_zeros((tx_logits.size(0), 1))
        return {
            "tx_logits": tx_logits,
            "dom_logits": dom_logits,
            "sgc_v3": out,
        }


def _fmt(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def _log_value(logs: Mapping[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(logs.get(key, default))
    except Exception:
        return float(default)


def _safe_pct(value: float) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "nan"


def print_sgc_v3_config_block(args, *, device, cfg: SGCv3Config, trainable_params: int, data_ctx: Mapping[str, Any]) -> None:
    sep = "=" * 132
    split_info = data_ctx.get("split_info")
    print(sep, flush=True)
    print("[CONFIG-BEGIN] sgc_v3_train.py resolved experiment configuration", flush=True)
    print(
        f"[CONFIG-RUN] time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} run_name={args.run_name} "
        f"seed={args.seed} device={device} amp={int(bool(args.amp))} mode={args.mode}",
        flush=True,
    )
    print(
        f"[CONFIG-DATA] dataset={args.dataset} wisig_domain={args.wisig_domain} pkl={args.wisig_pkl} "
        f"train_ratio={float(args.wisig_train_ratio):.3f} val_ratio={float(args.wisig_val_ratio):.3f} "
        f"batch={args.batch_size} eval_batch={args.eval_batch_size} workers={args.num_workers} "
        f"input_len={data_ctx.get('input_len')} num_classes={args.num_classes} num_domains={data_ctx.get('num_domains')}",
        flush=True,
    )
    if split_info is not None:
        print(
            f"[CONFIG-SPLIT] train_days={split_info.get('train_days_label', [])} "
            f"train_rxs={split_info.get('train_rxs_idx', [])} "
            f"test_days={split_info.get('test_days_label', [])} "
            f"test_rxs={split_info.get('test_rxs_idx', [])} guard_gap={split_info.get('guard_gap', '-')}",
            flush=True,
        )
    print(
        f"[CONFIG-SGC] feature_dim={cfg.feature_dim} scenario_dim={cfg.scenario_dim} experts={cfg.num_experts} "
        f"adapter_rank={cfg.adapter_rank} epsilon_z={cfg.epsilon_z:.4f} topk_only={cfg.topk_only} "
        f"epsilon_logit={cfg.epsilon_logit:.4f} trainable_params={trainable_params}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] optimizer=AdamW lr_sgc={float(args.lr_sgc):.3e} wd={float(args.weight_decay):.3e} "
        f"epochs={args.epochs} grad_clip={float(args.grad_clip):.3f} freeze_teacher={int(bool(args.freeze_teacher))}",
        flush=True,
    )
    print(
        f"[CONFIG-SAT] train_scenario={args.sat_train_scenario} eval_enabled={int(bool(args.eval_sat_channel))} "
        f"eval_scenarios={','.join(getattr(args, 'eval_sat_scenario_list', []))} eval_on={args.eval_sat_on} "
        f"eval_max_batches={args.sat_eval_max_batches} sat_seed={args.sat_seed}",
        flush=True,
    )
    print(
        f"[CONFIG-CKPT] teacher={args.teacher_ckpt} latest={str(ensure_dir(args.output_dir) / 'latest_sgc_v3.pth')} "
        f"best={str(ensure_dir(args.output_dir) / 'best_sgc_v3.pth')}",
        flush=True,
    )
    print("[CONFIG-END]", flush=True)
    print(sep, flush=True)


def format_sgc_v3_epoch_report(
    *,
    epoch: int,
    epochs: int,
    lr: float,
    epoch_time_s: float,
    logs: Mapping[str, float],
    val_stats: Mapping[str, float],
    test_stats: Mapping[str, float],
    named_test_stats: Dict[str, Dict[str, float]],
    named_test_meta: Dict[str, Dict[str, Any]],
    sat_test_stats: Dict[str, Dict[str, Any]] | None,
    best_score: float,
    best_epoch: int,
    latest_path: str,
    best_path: str,
    best_updated: bool,
) -> str:
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(f"[EPOCH-BEGIN] E{epoch:03d}/{epochs:03d} | time={epoch_time_s:.1f}s | lr={lr:.2e}")
    lines.append(minor)
    lines.append(
        "[SGC-LOSS] "
        f"total={_fmt(_log_value(logs, 'train/loss_total'))} "
        f"clean_kl={_fmt(_log_value(logs, 'train/loss_clean_kl'))} "
        f"clean_feat={_fmt(_log_value(logs, 'train/loss_clean_feat'))} "
        f"clean_margin={_fmt(_log_value(logs, 'train/loss_clean_margin'))} "
        f"pair_feat={_fmt(_log_value(logs, 'train/loss_pair_feat'))} "
        f"pair_logit={_fmt(_log_value(logs, 'train/loss_pair_logit'))} "
        f"proto={_fmt(_log_value(logs, 'train/loss_proto'))} "
        f"sat_ce={_fmt(_log_value(logs, 'train/loss_sat_ce'))} "
        f"gate_safe={_fmt(_log_value(logs, 'train/loss_gate_safety'))}"
    )
    lines.append(
        "[SGC-METRIC] "
        f"net_gain={_fmt(_log_value(logs, 'sgc/net_gain'))} "
        f"wrong_to_right={_fmt(_log_value(logs, 'sgc/wrong_to_right'))} "
        f"right_to_wrong={_fmt(_log_value(logs, 'sgc/right_to_wrong'))} "
        f"top1_flip={_fmt(_log_value(logs, 'sgc/top1_flip_rate'))} "
        f"pseudo_cov={_fmt(_log_value(logs, 'target/pseudo_coverage'))}"
    )
    lines.append(
        "[SGC-CLEAN] "
        f"gate_clean={_fmt(_log_value(logs, 'sgc/gate_clean_mean'))} "
        f"gate_sat={_fmt(_log_value(logs, 'sgc/gate_sat_mean'))} "
        f"dz_mean={_fmt(_log_value(logs, 'sgc/delta_z_ratio_mean'))} "
        f"dz_p95={_fmt(_log_value(logs, 'sgc/delta_z_ratio_p95'))} "
        f"dlogit={_fmt(_log_value(logs, 'sgc/delta_logit_norm_mean'))}"
    )
    lines.append(minor)
    lines.append(f"[VAL]   tx={_safe_pct(val_stats.get('tx_acc', float('nan')))}% dom={_safe_pct(val_stats.get('dom_acc', float('nan')))}%")
    lines.append(
        f"[TEST]  overall_tx={_safe_pct(test_stats.get('tx_acc', float('nan')))}% "
        f"({int(test_stats.get('tx_correct', 0))}/{int(test_stats.get('tx_total', 0))})"
    )
    lines.append("[TEST-SPLIT]")
    lines.extend(format_named_test_lines(named_test_stats, named_test_meta))
    lines.append(f"[BEST-SGC] net_gain={best_score:.4f} @ E{best_epoch:03d}")
    lines.append(f"[CKPT]  latest -> {latest_path} (saved) | best -> {best_path}{' (updated)' if best_updated else ''}")
    if sat_test_stats:
        lines.extend(format_sat_test_lines(sat_test_stats))
    lines.append(f"[EPOCH-END] E{epoch:03d}/{epochs:03d}")
    lines.append(sep)
    return "\n".join(lines)


def train(args) -> int:
    args.sat_train_scenario = str(args.sat_train_scenario or "mixed_orbit").strip().lower().replace("-", "_")
    args.eval_sat_scenario_list = parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []
    sat_channel_config_for_scenario(args.sat_train_scenario)
    for scenario in args.eval_sat_scenario_list:
        sat_channel_config_for_scenario(scenario)

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    data_ctx = build_standard_data(args, device)
    teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=bool(args.freeze_teacher))
    first_batch = next(iter(data_ctx["train_loader"]))
    x0, _, _ = move_batch(first_batch, device)
    cfg = _cfg_from_file(args.config, int(args.num_classes), _infer_feature_dim(teacher, x0))
    model = SGCv3Model(teacher, cfg).to(device)
    _apply_mode_freeze(model, str(args.mode))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr_sgc), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    sat_gen = make_torch_generator(device, int(args.seed) + 991)
    eval_model = SGCv3EvalAdapter(model)

    print(
        f"[SGC-V3] mode={args.mode} epochs={args.epochs} lr={args.lr_sgc:.3e} "
        f"scenario={args.sat_train_scenario} trainable_params={sum(p.numel() for p in trainable)}",
        flush=True,
    )
    print_sgc_v3_config_block(args, device=device, cfg=cfg, trainable_params=sum(p.numel() for p in trainable), data_ctx=data_ctx)
    if args.dry_run:
        print("[DRY-RUN] Built SGC v3 model and skipped optimization.", flush=True)
        return 0

    best_score = float("-inf")
    best_epoch = 0
    main_test_keys = (
        ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
        if str(args.dataset).lower() == "wisig"
        else list(data_ctx["named_test_loaders"].keys())
    )
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        model.base_teacher.eval()
        logs_epoch = []
        for batch in data_ctx["train_loader"]:
            x_clean, y, _ = move_batch(batch, device)
            with torch.no_grad():
                x_sat, _ = apply_sat_channel_for_scenario(x_clean, str(args.sat_train_scenario), args, gen=sat_gen, return_meta=False)
            train_batch = {"x_clean": x_clean, "x_sat": x_sat, "y": y, "scenario": torch.ones_like(y)}
            if str(args.mode) == "target_adapt":
                train_batch["x_target"] = x_sat.detach()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                loss, logs = compute_sgc_v3_losses(model, train_batch)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            logs_epoch.append(logs)
        logs_mean = mean_logs(logs_epoch)
        score = float(logs_mean.get("sgc/net_gain", 0.0))
        payload = {"sgc_v3": model.state_dict(), "epoch": epoch, "args": vars(args), "config": cfg, "logs": logs_mean}
        latest_path = out_dir / "latest_sgc_v3.pth"
        best_path = out_dir / "best_sgc_v3.pth"
        save_payload(latest_path, payload)
        best_updated = False
        if score > best_score and float(logs_mean.get("sgc/gate_clean_mean", 1.0)) <= 0.05:
            best_score = score
            best_epoch = epoch
            best_updated = True
            save_payload(best_path, payload)
        val_stats = evaluate_loader(
            eval_model,
            data_ctx["val_loader"],
            device,
            domain_label_map=data_ctx["domain_label_map"],
            max_batches=int(args.eval_max_batches),
        )
        named_test_stats = evaluate_named_loaders(
            eval_model,
            data_ctx["named_test_loaders"],
            device,
            domain_label_map=data_ctx["domain_label_map"],
            max_batches=int(args.eval_max_batches),
        )
        test_stats = aggregate_named_stats(named_test_stats, [k for k in main_test_keys if k in named_test_stats])
        sat_test_stats: Dict[str, Dict[str, Any]] = {}
        if bool(args.eval_sat_channel) and len(args.eval_sat_scenario_list) > 0:
            sat_eval_max_batches = int(args.sat_eval_max_batches)
            if sat_eval_max_batches < 0:
                sat_eval_max_batches = int(args.eval_max_batches)
            sat_test_stats = evaluate_sat_scenarios(
                eval_model,
                data_ctx["named_test_loaders"],
                device,
                domain_label_map=data_ctx["domain_label_map"],
                scenario_names=args.eval_sat_scenario_list,
                args=args,
                max_batches=sat_eval_max_batches,
            )
        print(
            format_sgc_v3_epoch_report(
                epoch=epoch,
                epochs=int(args.epochs),
                lr=optimizer.param_groups[0]["lr"],
                epoch_time_s=time.time() - t0,
                logs=logs_mean,
                val_stats=val_stats,
                test_stats=test_stats,
                named_test_stats=named_test_stats,
                named_test_meta=data_ctx["named_test_meta"],
                sat_test_stats=sat_test_stats,
                best_score=best_score,
                best_epoch=best_epoch,
                latest_path=str(latest_path),
                best_path=str(best_path),
                best_updated=best_updated,
            ),
            flush=True,
        )
    print(f"Training finished. best_sgc_net_gain={best_score:.4f} at epoch {best_epoch} -> {out_dir / 'best_sgc_v3.pth'}")
    return 0


def main() -> int:
    return train(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
