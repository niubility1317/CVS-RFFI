from __future__ import annotations

import argparse
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from post_stage_common import build_standard_data, ensure_dir, load_baseline_from_checkpoint, load_yaml_or_json, mean_logs, move_batch, resolve_device, save_payload, set_seed
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator

from SGC.recon.channel_losses import channel_consistency_loss, residual_constraint_loss
from SGC.recon.cx_resdiff import CxResDiff
from SGC.recon.cx_unet_1d import count_parameters
from SGC.recon.diff_sat_channel import DifferentiableSatChannel
from SGC.recon.identity_losses import identity_preservation_loss
from SGC.recon.stft_losses import stft_mag_phase_loss


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PhyCon CxResDiff IQ reconstruction frontend.")
    parser.add_argument("--config", type=str, default="SGC/configs/recon_cxresdiff_020m.yaml")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr_recon", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sat_train_scenario", type=str, default="mixed_orbit")
    parser.add_argument("--enable_channel_loss", type=str2bool, default=False)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    add_sat_eval_args(parser)
    return parser


def _loss_weights(cfg: Mapping[str, Any]) -> dict[str, float]:
    raw = dict((cfg.get("loss") or {}) if isinstance(cfg, Mapping) else {})
    return {
        "diff": float(raw.get("lambda_diff", 1.0)),
        "pair": float(raw.get("lambda_pair", 0.2)),
        "chan": float(raw.get("lambda_chan", 0.0)),
        "id": float(raw.get("lambda_id", 0.5)),
        "res": float(raw.get("lambda_res", 1.5)),
        "tf": float(raw.get("lambda_tf", 0.2)),
    }


def _rho_for_epoch(epoch: int, cfg: Mapping[str, Any]) -> tuple[float, float]:
    residual = dict(cfg.get("residual") or {})
    if int(epoch) <= 20:
        return float(residual.get("rho_start", 0.05)), float(residual.get("r_max_start", 0.05))
    if int(epoch) <= 60:
        return float(residual.get("rho_mid", 0.10)), float(residual.get("r_max_mid", 0.10))
    return float(residual.get("rho_max", 0.15)), float(residual.get("r_max_late", 0.15))


def _fmt(value: Any, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def _log_value(logs: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(logs.get(key, default))
    except Exception:
        return float(default)


def print_recon_diff_config_block(args, *, device, cfg: Mapping[str, Any], trainable_params: int, data_ctx: Mapping[str, Any] | None) -> None:
    sep = "=" * 132
    split_info = (data_ctx or {}).get("split_info")
    model_cfg = cfg.get("model") or {}
    loss_cfg = cfg.get("loss") or {}
    residual_cfg = cfg.get("residual") or {}
    print(sep, flush=True)
    print("[CONFIG-BEGIN] train_recon_diffusion.py resolved experiment configuration", flush=True)
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
            f"test_days={split_info.get('test_days_label', [])} test_rxs={split_info.get('test_rxs_idx', [])} "
            f"guard_gap={split_info.get('guard_gap', '-')}",
            flush=True,
        )
    print(
        f"[CONFIG-MODEL] type={model_cfg.get('type', 'cx_residual_consistency_unet_1d')} "
        f"channels={model_cfg.get('channels', [32, 48, 64, 72])} condition_dim={model_cfg.get('condition_dim', 24)} "
        f"target_params={model_cfg.get('target_params', '0.20M')} trainable_params={trainable_params}",
        flush=True,
    )
    print(
        f"[CONFIG-LOSS] diff={loss_cfg.get('lambda_diff', 1.0)} pair={loss_cfg.get('lambda_pair', 0.2)} "
        f"chan={loss_cfg.get('lambda_chan', 0.0)} id={loss_cfg.get('lambda_id', 0.5)} "
        f"res={loss_cfg.get('lambda_res', 1.5)} tf={loss_cfg.get('lambda_tf', 0.2)} "
        f"enable_channel_loss={int(bool(args.enable_channel_loss))}",
        flush=True,
    )
    print(
        f"[CONFIG-RESIDUAL] rho_start={residual_cfg.get('rho_start', 0.05)} rho_mid={residual_cfg.get('rho_mid', 0.10)} "
        f"rho_max={residual_cfg.get('rho_max', 0.15)} r_start={residual_cfg.get('r_max_start', 0.05)} "
        f"r_mid={residual_cfg.get('r_max_mid', 0.10)} r_late={residual_cfg.get('r_max_late', 0.15)}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] optimizer=AdamW lr_recon={float(args.lr_recon):.3e} wd={float(args.weight_decay):.3e} "
        f"epochs={args.epochs} grad_clip={float(args.grad_clip):.3f}",
        flush=True,
    )
    print(f"[CONFIG-CKPT] teacher={args.teacher_ckpt} output_dir={args.output_dir}", flush=True)
    print("[CONFIG-END]", flush=True)
    print(sep, flush=True)


def format_recon_diff_epoch_report(
    *,
    epoch: int,
    epochs: int,
    lr: float,
    epoch_time_s: float,
    rho: float,
    r_max: float,
    logs: Mapping[str, Any],
    latest_path: str,
    best_path: str,
    best_updated: bool,
) -> str:
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(f"[EPOCH-BEGIN] RECON-DIFF E{epoch:03d}/{epochs:03d} | time={epoch_time_s:.1f}s | lr={lr:.2e} | rho={rho:.3f} r_max={r_max:.3f}")
    lines.append(minor)
    lines.append(
        "[RECON-LOSS] "
        f"total={_fmt(_log_value(logs, 'train/loss_total'))} "
        f"diff={_fmt(_log_value(logs, 'loss_diff'))} "
        f"pair={_fmt(_log_value(logs, 'train/loss_pair'))} "
        f"id_feat={_fmt(_log_value(logs, 'loss_id_feat'))} "
        f"id_ce={_fmt(_log_value(logs, 'loss_id_ce'))} "
        f"res={_fmt(_log_value(logs, 'train/loss_res'))} "
        f"tf={_fmt(_log_value(logs, 'train/loss_tf'))} "
        f"chan={_fmt(_log_value(logs, 'loss_chan'))}"
    )
    lines.append(
        "[RECON-METRIC] "
        f"id_cos={_fmt(_log_value(logs, 'id_feature_cos'), 4)} "
        f"residual_mean={_fmt(_log_value(logs, 'train/residual_ratio_mean'), 4)} "
        f"residual_p95={_fmt(_log_value(logs, 'train/residual_ratio_p95'), 4)} "
        f"chan_time={_fmt(_log_value(logs, 'loss_chan_time'))} "
        f"chan_tf={_fmt(_log_value(logs, 'loss_chan_tf'))}"
    )
    lines.append(f"[CKPT] latest -> {latest_path} (saved) | best -> {best_path}{' (updated)' if best_updated else ''}")
    lines.append(f"[EPOCH-END] RECON-DIFF E{epoch:03d}/{epochs:03d}")
    lines.append(sep)
    return "\n".join(lines)


def _build_model_from_cfg(cfg: Mapping[str, Any]) -> CxResDiff:
    train_timesteps = int((cfg.get("diffusion") or {}).get("train_timesteps", 1000))
    condition_dim = int((cfg.get("model") or {}).get("condition_dim", 24))
    return CxResDiff(train_timesteps=train_timesteps, condition_dim=condition_dim)


def train(args) -> int:
    cfg = load_yaml_or_json(args.config)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    model = _build_model_from_cfg(cfg).to(device)
    print(f"[RECON-DIFF] params={count_parameters(model):,} dry_run={int(bool(args.dry_run))}", flush=True)
    if args.dry_run:
        print_recon_diff_config_block(args, device=device, cfg=cfg, trainable_params=count_parameters(model, trainable_only=True), data_ctx=None)
        return 0

    out_dir = ensure_dir(args.output_dir)
    data_ctx = build_standard_data(args, device)
    base_teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=True)
    channel = DifferentiableSatChannel(
        fs_hz=float(getattr(args, "sat_fs_hz", 25e6)),
        fc_hz=float(getattr(args, "sat_fc_hz", 2.462e9)),
    ).to(device)
    weights = _loss_weights(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr_recon), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    sat_gen = make_torch_generator(device, int(args.seed) + 1701)
    print_recon_diff_config_block(args, device=device, cfg=cfg, trainable_params=count_parameters(model, trainable_only=True), data_ctx=data_ctx)

    best_score = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        rho, r_max = _rho_for_epoch(epoch, cfg)
        logs_epoch = []
        for batch in data_ctx["train_loader"]:
            x_clean, labels, _ = move_batch(batch, device)
            with torch.no_grad():
                y_sat, meta = apply_sat_channel_for_scenario(x_clean, args.sat_train_scenario, args, gen=sat_gen, return_meta=True)
            c = model.encode_condition(y_sat, meta)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                loss_diff, log_diff = model.diffusion_loss(x_clean, y_sat, c)
                corrected = model.correct(y_sat, c=c, rho=rho)
                x_hat = corrected["x_hat"]
                loss_pair = F.l1_loss(x_hat, x_clean) + 0.5 * F.mse_loss(x_hat, x_clean)
                loss_id, log_id = identity_preservation_loss(base_teacher, x_hat, x_clean, labels)
                loss_res = residual_constraint_loss(x_hat, y_sat, r_max=r_max)
                loss_tf = stft_mag_phase_loss(x_hat, x_clean)
                loss_chan = x_clean.new_tensor(0.0)
                log_chan = {}
                if bool(args.enable_channel_loss) or weights["chan"] > 0:
                    loss_chan, log_chan = channel_consistency_loss(channel, x_hat, y_sat, meta)
                loss = (
                    weights["diff"] * loss_diff
                    + weights["pair"] * loss_pair
                    + weights["id"] * loss_id
                    + weights["res"] * loss_res
                    + weights["tf"] * loss_tf
                    + weights["chan"] * loss_chan
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            logs = {
                "train/loss_total": loss.detach(),
                "train/loss_pair": loss_pair.detach(),
                "train/loss_res": loss_res.detach(),
                "train/loss_tf": loss_tf.detach(),
                "train/residual_ratio_mean": corrected["residual_ratio"].detach().mean(),
                "train/residual_ratio_p95": torch.quantile(corrected["residual_ratio"].detach().float(), 0.95),
                **log_diff,
                **log_id,
                **log_chan,
            }
            logs_epoch.append(logs)
        logs_mean = mean_logs(logs_epoch)
        score = float(logs_mean.get("train/loss_total", float("inf")))
        payload = {"recon": model.state_dict(), "epoch": epoch, "args": vars(args), "config": cfg, "logs": logs_mean}
        latest_path = out_dir / "latest_recon_diffusion.pth"
        best_path = out_dir / "best_recon_diffusion.pth"
        save_payload(latest_path, payload)
        best_updated = False
        if score < best_score:
            best_score = score
            best_updated = True
            save_payload(best_path, payload)
        print(format_recon_diff_epoch_report(
            epoch=epoch,
            epochs=int(args.epochs),
            lr=optimizer.param_groups[0]["lr"],
            epoch_time_s=time.time() - t0,
            rho=rho,
            r_max=r_max,
            logs=logs_mean,
            latest_path=str(latest_path),
            best_path=str(best_path),
            best_updated=best_updated,
        ), flush=True)
    return 0


def main() -> int:
    return train(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
