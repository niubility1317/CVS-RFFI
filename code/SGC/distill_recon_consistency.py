from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from post_stage_cli import add_common_data_args, str2bool
from post_stage_common import build_standard_data, ensure_dir, load_baseline_from_checkpoint, load_yaml_or_json, mean_logs, move_batch, resolve_device, save_payload, set_seed
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator

from SGC.recon.channel_losses import residual_constraint_loss
from SGC.recon.cx_consistency import CxConsistency
from SGC.recon.cx_resdiff import CxResDiff
from SGC.recon.cx_unet_1d import count_parameters
from SGC.recon.identity_losses import identity_preservation_loss
from SGC.recon.stft_losses import stft_mag_phase_loss


def _fmt(value, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def _log_value(logs, key: str, default: float = 0.0) -> float:
    try:
        return float(logs.get(key, default))
    except Exception:
        return float(default)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill PhyCon CxResDiff into CxConsistency.")
    parser.add_argument("--config", type=str, default="SGC/configs/recon_cxconsistency_020m.yaml")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_recon", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sat_train_scenario", type=str, default="mixed_orbit")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    return parser


def print_recon_cm_config_block(args, *, device, cfg, trainable_params: int, data_ctx=None, has_diff_teacher: bool = False) -> None:
    sep = "=" * 132
    split_info = (data_ctx or {}).get("split_info")
    model_cfg = cfg.get("model") or {}
    loss_cfg = cfg.get("loss") or {}
    cons_cfg = cfg.get("consistency") or {}
    print(sep, flush=True)
    print("[CONFIG-BEGIN] distill_recon_consistency.py resolved experiment configuration", flush=True)
    print(
        f"[CONFIG-RUN] run_name={args.run_name} seed={args.seed} device={device} amp={int(bool(args.amp))} "
        f"dry_run={int(bool(args.dry_run))} scenario={args.sat_train_scenario}",
        flush=True,
    )
    print(
        f"[CONFIG-DATA] dataset={args.dataset} wisig_domain={args.wisig_domain} pkl={args.wisig_pkl} "
        f"train_ratio={float(args.wisig_train_ratio):.3f} batch={args.batch_size} eval_batch={args.eval_batch_size} "
        f"workers={args.num_workers} input_len={getattr(args, 'wisig_out_len', '-')}",
        flush=True,
    )
    if split_info is not None:
        print(
            f"[CONFIG-SPLIT] train_days={split_info.get('train_days_label', [])} train_rxs={split_info.get('train_rxs_idx', [])} "
            f"test_days={split_info.get('test_days_label', [])} test_rxs={split_info.get('test_rxs_idx', [])}",
            flush=True,
        )
    print(
        f"[CONFIG-MODEL] type={model_cfg.get('type', 'cx_residual_consistency_unet_1d')} "
        f"channels={model_cfg.get('channels', [32, 48, 64, 72])} condition_dim={model_cfg.get('condition_dim', 24)} "
        f"trainable_params={trainable_params}",
        flush=True,
    )
    print(
        f"[CONFIG-CM] enabled={cons_cfg.get('enabled', True)} steps_train={cons_cfg.get('steps_train', [1, 2, 4])} "
        f"steps_eval={cons_cfg.get('steps_eval', [1, 2, 4])} loss_type={cons_cfg.get('loss_type', 'pseudo_huber')} "
        f"has_diff_teacher={int(bool(has_diff_teacher))}",
        flush=True,
    )
    print(
        f"[CONFIG-LOSS] cm={loss_cfg.get('lambda_cm', 1.0)} teacher={loss_cfg.get('lambda_teacher', 0.5)} "
        f"id={loss_cfg.get('lambda_id', 0.5)} res={loss_cfg.get('lambda_res', 2.0)} tf={loss_cfg.get('lambda_tf', 0.2)}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] optimizer=AdamW lr_recon={float(args.lr_recon):.3e} wd={float(args.weight_decay):.3e} "
        f"epochs={args.epochs} grad_clip={float(args.grad_clip):.3f}",
        flush=True,
    )
    print(f"[CONFIG-CKPT] base_teacher={args.teacher_ckpt} diffusion_teacher={args.diffusion_ckpt or '<clean-pair-target>'} output_dir={args.output_dir}", flush=True)
    print("[CONFIG-END]", flush=True)
    print(sep, flush=True)


def format_recon_cm_epoch_report(*, epoch: int, epochs: int, lr: float, epoch_time_s: float, logs, latest_path: str, best_path: str) -> str:
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(f"[EPOCH-BEGIN] RECON-CM E{epoch:03d}/{epochs:03d} | time={epoch_time_s:.1f}s | lr={lr:.2e}")
    lines.append(minor)
    lines.append(
        "[RECON-CM-LOSS] "
        f"total={_fmt(_log_value(logs, 'train/loss_total'))} "
        f"cm={_fmt(_log_value(logs, 'train/loss_cm'))} "
        f"teacher={_fmt(_log_value(logs, 'train/loss_teacher'))} "
        f"id_feat={_fmt(_log_value(logs, 'loss_id_feat'))} "
        f"id_ce={_fmt(_log_value(logs, 'loss_id_ce'))} "
        f"res={_fmt(_log_value(logs, 'train/loss_res'))} "
        f"tf={_fmt(_log_value(logs, 'train/loss_tf'))}"
    )
    lines.append(
        "[RECON-CM-METRIC] "
        f"id_cos={_fmt(_log_value(logs, 'id_feature_cos'), 4)} "
        f"residual_p95={_fmt(_log_value(logs, 'train/residual_ratio_p95'), 4)}"
    )
    lines.append(f"[CKPT] latest -> {latest_path} (saved) | best -> {best_path} (saved)")
    lines.append(f"[EPOCH-END] RECON-CM E{epoch:03d}/{epochs:03d}")
    lines.append(sep)
    return "\n".join(lines)


def _load_diffusion_teacher(path: str, device) -> CxResDiff | None:
    if not path:
        return None
    ckpt = torch.load(path, map_location=device)
    teacher = CxResDiff().to(device)
    teacher.load_state_dict(ckpt.get("recon", ckpt), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def train(args) -> int:
    cfg = load_yaml_or_json(args.config)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    model = CxConsistency().to(device)
    print(f"[RECON-CM] params={count_parameters(model):,} dry_run={int(bool(args.dry_run))}", flush=True)
    if args.dry_run:
        print_recon_cm_config_block(args, device=device, cfg=cfg, trainable_params=count_parameters(model, trainable_only=True), data_ctx=None, has_diff_teacher=bool(args.diffusion_ckpt))
        return 0
    out_dir = ensure_dir(args.output_dir)
    data_ctx = build_standard_data(args, device)
    base_teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=True)
    diff_teacher = _load_diffusion_teacher(args.diffusion_ckpt, device)
    loss_cfg = cfg.get("loss") or {}
    residual_cfg = cfg.get("residual") or {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr_recon), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    sat_gen = make_torch_generator(device, int(args.seed) + 2701)
    print_recon_cm_config_block(args, device=device, cfg=cfg, trainable_params=count_parameters(model, trainable_only=True), data_ctx=data_ctx, has_diff_teacher=diff_teacher is not None)
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        logs_epoch = []
        for batch in data_ctx["train_loader"]:
            x_clean, labels, _ = move_batch(batch, device)
            with torch.no_grad():
                y_sat, meta = apply_sat_channel_for_scenario(x_clean, args.sat_train_scenario, args, gen=sat_gen, return_meta=True)
            c = model.encode_condition(y_sat, meta)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                corrected = model.correct(y_sat, c=c, steps=2, rho=float(residual_cfg.get("rho_max", 0.15)))
                x_hat = corrected["x_hat"]
                with torch.no_grad():
                    target = diff_teacher.correct(y_sat, c=c, rho=float(residual_cfg.get("rho_max", 0.15)))["x_hat"] if diff_teacher is not None else x_clean
                loss_cm = model.consistency_loss(x_hat, target, loss_type=str((cfg.get("consistency") or {}).get("loss_type", "pseudo_huber")))
                loss_teacher = F.mse_loss(x_hat, target)
                loss_id, log_id = identity_preservation_loss(base_teacher, x_hat, x_clean, labels)
                loss_res = residual_constraint_loss(x_hat, y_sat, r_max=float(residual_cfg.get("r_max_late", 0.15)))
                loss_tf = stft_mag_phase_loss(x_hat, x_clean)
                loss = (
                    float(loss_cfg.get("lambda_cm", 1.0)) * loss_cm
                    + float(loss_cfg.get("lambda_teacher", 0.5)) * loss_teacher
                    + float(loss_cfg.get("lambda_id", 0.5)) * loss_id
                    + float(loss_cfg.get("lambda_res", 2.0)) * loss_res
                    + float(loss_cfg.get("lambda_tf", 0.2)) * loss_tf
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            logs_epoch.append(
                {
                    "train/loss_total": loss.detach(),
                    "train/loss_cm": loss_cm.detach(),
                    "train/loss_teacher": loss_teacher.detach(),
                    "train/loss_res": loss_res.detach(),
                    "train/loss_tf": loss_tf.detach(),
                    "train/residual_ratio_p95": torch.quantile(corrected["residual_ratio"].detach().float(), 0.95),
                    **log_id,
                }
            )
        logs_mean = mean_logs(logs_epoch)
        payload = {"recon": model.state_dict(), "epoch": epoch, "args": vars(args), "config": cfg, "logs": logs_mean}
        latest_path = out_dir / "latest_recon_consistency.pth"
        best_path = out_dir / "best_recon_consistency.pth"
        save_payload(latest_path, payload)
        save_payload(best_path, payload)
        print(format_recon_cm_epoch_report(
            epoch=epoch,
            epochs=int(args.epochs),
            lr=optimizer.param_groups[0]["lr"],
            epoch_time_s=time.time() - t0,
            logs=logs_mean,
            latest_path=str(latest_path),
            best_path=str(best_path),
        ), flush=True)
    return 0


def main() -> int:
    return train(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
