from __future__ import annotations

import argparse
import time
from typing import Any, Mapping
from typing import Iterable

import torch
import torch.nn as nn

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from post_stage_common import build_standard_data, ensure_dir, load_baseline_from_checkpoint, load_yaml_or_json, mean_logs, move_batch, resolve_device, save_payload, set_seed
from cvsrffi.eval import apply_sat_channel_for_scenario, evaluate_loader, evaluate_named_loaders
from cvsrffi.tensors import make_torch_generator
from training_test_eval import TrainingTestEvalResult, evaluate_training_tests

from SGC.recon.channel_losses import residual_constraint_loss
from SGC.recon.cx_consistency import CxConsistency
from SGC.recon.cx_unet_1d import count_parameters
from SGC.recon.identity_losses import identity_preservation_loss
from SGC.v3.losses_v3 import compute_sgc_v3_losses
from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model


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
    parser = argparse.ArgumentParser(description="Joint fine-tune PhyCon recon frontend with SGC v3 safe adapters.")
    parser.add_argument("--config", type=str, default="SGC/configs/recon_sgc_joint.yaml")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--recon_ckpt", type=str, default="")
    parser.add_argument("--sgc_v3_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_recon", type=float, default=2e-5)
    parser.add_argument("--lr_sgc", type=float, default=5e-5)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--eval_recon_steps", type=int, default=2)
    parser.add_argument("--eval_recon_rho", type=float, default=-1.0)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    add_sat_eval_args(parser)
    return parser


def set_recon_joint_trainable(recon: CxConsistency) -> None:
    for param in recon.parameters():
        param.requires_grad = False
    for module in (recon.model.head, recon.residual_gate):
        for param in module.parameters():
            param.requires_grad = True


def set_sgc_joint_trainable(sgc_model: nn.Module) -> None:
    for param in sgc_model.parameters():
        param.requires_grad = False
    for attr in ("feature_adapter", "logit_calibrator"):
        module = getattr(sgc_model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


class ReconSGCJointEvalAdapter(nn.Module):
    """Expose recon+SGC joint inference through the backbone evaluator contract."""

    def __init__(self, recon: CxConsistency, sgc_model: SGCv3Model, *, steps: int = 2, rho: float = 0.10) -> None:
        super().__init__()
        self.recon = recon
        self.sgc_model = sgc_model
        self.steps = int(steps)
        self.rho = float(rho)

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        rec = self.recon.correct(x, steps=self.steps, rho=self.rho)
        x_hat = rec["x_hat"]
        sgc_out = self.sgc_model(x_hat)
        tx_logits = sgc_out["logits_final"]
        dom_logits = None
        teacher = self.sgc_model.base_teacher
        try:
            with torch.no_grad():
                teacher_out = teacher(x_hat, y_tx=y_tx, grl_lambda=grl_lambda, return_aux=True, domain_labels=domain_labels)
            if isinstance(teacher_out, Mapping):
                dom_logits = teacher_out.get("dom_logits")
        except Exception:
            dom_logits = None
        if not torch.is_tensor(dom_logits):
            dom_logits = tx_logits.new_zeros((tx_logits.size(0), 1))
        return {
            "tx_logits": tx_logits,
            "dom_logits": dom_logits,
            "recon_x": x_hat,
            "recon": rec,
            "sgc_v3": sgc_out,
        }


def trainable_params(modules: Iterable[nn.Module]) -> list[torch.nn.Parameter]:
    return [p for module in modules for p in module.parameters() if p.requires_grad]


def print_joint_config_block(args, *, device, cfg, data_ctx=None, recon_trainable: int = 0, sgc_trainable: int = 0) -> None:
    sep = "=" * 132
    split_info = (data_ctx or {}).get("split_info")
    joint_cfg = cfg.get("joint_finetune") or {}
    loss_cfg = cfg.get("loss") or {}
    print(sep, flush=True)
    print("[CONFIG-BEGIN] train_recon_sgc_joint.py resolved experiment configuration", flush=True)
    print(
        f"[CONFIG-RUN] run_name={args.run_name} seed={args.seed} device={device} amp={int(bool(args.amp))} dry_run={int(bool(args.dry_run))}",
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
        f"[CONFIG-JOINT] epochs={joint_cfg.get('epochs', args.epochs)} rho_max={joint_cfg.get('rho_max', 0.10)} "
        f"r_max={joint_cfg.get('r_max', 0.10)} recon_trainable={recon_trainable} sgc_trainable={sgc_trainable}",
        flush=True,
    )
    print(
        f"[CONFIG-LOSS] id={loss_cfg.get('lambda_id', 0.8)} res={loss_cfg.get('lambda_res', 3.0)} "
        f"clean_preserve={loss_cfg.get('lambda_clean_preserve', 1.0)} sgc={loss_cfg.get('lambda_sgc', 1.0)}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] optimizer=AdamW lr_recon={float(args.lr_recon):.3e} lr_sgc={float(args.lr_sgc):.3e} epochs={args.epochs}",
        flush=True,
    )
    print(f"[CONFIG-CKPT] teacher={args.teacher_ckpt} recon={args.recon_ckpt or '<init>'} sgc_v3={args.sgc_v3_ckpt or '<init>'} output_dir={args.output_dir}", flush=True)
    print("[CONFIG-END]", flush=True)
    print(sep, flush=True)


def format_joint_epoch_report(
    *,
    epoch: int,
    epochs: int,
    lr_recon: float,
    lr_sgc: float,
    epoch_time_s: float,
    rho: float,
    r_max: float,
    logs,
    latest_path: str,
    best_path: str,
    eval_result: TrainingTestEvalResult | None = None,
) -> str:
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(
        f"[EPOCH-BEGIN] RECON-SGC-JOINT E{epoch:03d}/{epochs:03d} | time={epoch_time_s:.1f}s "
        f"| lr_recon={lr_recon:.2e} lr_sgc={lr_sgc:.2e} | rho={rho:.3f} r_max={r_max:.3f}"
    )
    lines.append(minor)
    lines.append(
        "[JOINT-LOSS] "
        f"total={_fmt(_log_value(logs, 'train/loss_total'))} "
        f"id={_fmt(_log_value(logs, 'train/loss_id'))} "
        f"res={_fmt(_log_value(logs, 'train/loss_res'))} "
        f"clean_preserve={_fmt(_log_value(logs, 'train/loss_clean_preserve'))} "
        f"sgc={_fmt(_log_value(logs, 'train/loss_sgc'))}"
    )
    lines.append(
        "[JOINT-METRIC] "
        f"id_cos={_fmt(_log_value(logs, 'id_feature_cos'), 4)} "
        f"residual_p95={_fmt(_log_value(logs, 'train/residual_ratio_p95'), 4)} "
        f"net_gain={_fmt(_log_value(logs, 'sgc/net_gain'), 5)} "
        f"wrong_to_right={_fmt(_log_value(logs, 'sgc/wrong_to_right'), 5)} "
        f"right_to_wrong={_fmt(_log_value(logs, 'sgc/right_to_wrong'), 5)} "
        f"gate_sat={_fmt(_log_value(logs, 'sgc/gate_sat_mean'), 4)}"
    )
    if eval_result is not None:
        val_stats = eval_result.val_stats
        lines.append(f"[VAL]   tx={_fmt(val_stats.get('tx_acc', float('nan')), 2)}% dom={_fmt(val_stats.get('dom_acc', float('nan')), 2)}%")
        lines.extend(eval_result.lines)
    lines.append(f"[CKPT] latest -> {latest_path} (saved) | best -> {best_path} (saved)")
    lines.append(f"[EPOCH-END] RECON-SGC-JOINT E{epoch:03d}/{epochs:03d}")
    lines.append(sep)
    return "\n".join(lines)


def _infer_feature_dim(teacher: nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        if hasattr(teacher, "extract_feature"):
            return int(teacher.extract_feature(x[:1]).shape[-1])
        out = teacher(x[:1], return_aux=True)
        if isinstance(out, Mapping):
            for key in ("z_id_raw", "z_id", "feat_joint", "feat_cls", "base"):
                value = out.get(key)
                if torch.is_tensor(value):
                    return int(value.shape[-1])
    raise AttributeError("Cannot infer teacher feature dimension.")


def _build_or_load_sgc(args, teacher: nn.Module, feature_dim: int, device) -> SGCv3Model:
    if args.sgc_v3_ckpt:
        ckpt = torch.load(args.sgc_v3_ckpt, map_location=device)
        raw_cfg = ckpt.get("config")
        if isinstance(raw_cfg, SGCv3Config):
            cfg = raw_cfg
        elif isinstance(raw_cfg, Mapping):
            cfg = SGCv3Config.from_mapping(raw_cfg)
        else:
            cfg = SGCv3Config(num_classes=int(args.num_classes), feature_dim=int(feature_dim))
        model = SGCv3Model(teacher, cfg).to(device)
        model.load_state_dict(ckpt.get("sgc_v3", ckpt), strict=False)
        return model
    cfg = SGCv3Config(num_classes=int(args.num_classes), feature_dim=int(feature_dim))
    return SGCv3Model(teacher, cfg).to(device)


def main_train(args) -> int:
    cfg = load_yaml_or_json(args.config)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    recon = CxConsistency().to(device)
    if args.recon_ckpt:
        ckpt = torch.load(args.recon_ckpt, map_location=device)
        recon.load_state_dict(ckpt.get("recon", ckpt), strict=False)
    set_recon_joint_trainable(recon)
    print(
        f"[RECON-SGC-JOINT] recon_params={count_parameters(recon):,} "
        f"trainable_recon={sum(p.numel() for p in recon.parameters() if p.requires_grad):,} "
        f"dry_run={int(bool(args.dry_run))}",
        flush=True,
    )
    if args.dry_run:
        print_joint_config_block(args, device=device, cfg=cfg, data_ctx=None, recon_trainable=sum(p.numel() for p in recon.parameters() if p.requires_grad), sgc_trainable=0)
        return 0
    out_dir = ensure_dir(args.output_dir)
    data_ctx = build_standard_data(args, device)
    base_teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=True)
    first_batch = next(iter(data_ctx["train_loader"]))
    x0, _, _ = move_batch(first_batch, device)
    sgc = _build_or_load_sgc(args, base_teacher, _infer_feature_dim(base_teacher, x0), device)
    set_sgc_joint_trainable(sgc)
    params_recon = [p for p in recon.parameters() if p.requires_grad]
    params_sgc = [p for p in sgc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": params_recon, "lr": float(args.lr_recon)},
            {"params": params_sgc, "lr": float(args.lr_sgc)},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    sat_gen = make_torch_generator(device, int(args.seed) + 4701)
    joint_cfg: Mapping[str, Any] = cfg.get("joint_finetune") or {}
    loss_cfg: Mapping[str, Any] = cfg.get("loss") or {}
    rho = float(joint_cfg.get("rho_max", 0.10))
    r_max = float(joint_cfg.get("r_max", 0.10))
    eval_rho = float(args.eval_recon_rho) if float(args.eval_recon_rho) > 0.0 else rho
    eval_model = ReconSGCJointEvalAdapter(
        recon,
        sgc,
        steps=int(args.eval_recon_steps),
        rho=eval_rho,
    )
    print(
        f"[RECON-SGC-JOINT] trainable_sgc={sum(p.numel() for p in params_sgc):,} "
        f"rho={rho:.3f} r_max={r_max:.3f} eval_steps={int(args.eval_recon_steps)} eval_rho={eval_rho:.3f}",
        flush=True,
    )
    print_joint_config_block(
        args,
        device=device,
        cfg=cfg,
        data_ctx=data_ctx,
        recon_trainable=sum(p.numel() for p in params_recon),
        sgc_trainable=sum(p.numel() for p in params_sgc),
    )
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        recon.train()
        sgc.train()
        sgc.base_teacher.eval()
        logs_epoch = []
        for batch in data_ctx["train_loader"]:
            x_clean, labels, _ = move_batch(batch, device)
            with torch.no_grad():
                y_sat, meta = apply_sat_channel_for_scenario(x_clean, "mixed_orbit", args, gen=sat_gen, return_meta=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                rec = recon.correct(y_sat, meta=meta, steps=2, rho=rho)
                x_hat = rec["x_hat"]
                loss_id, log_id = identity_preservation_loss(base_teacher, x_hat, x_clean, labels)
                loss_res = residual_constraint_loss(x_hat, y_sat, r_max=r_max)
                sgc_loss, sgc_logs = compute_sgc_v3_losses(
                    sgc,
                    {"x_clean": x_clean, "x_sat": x_hat, "y": labels, "scenario": torch.ones_like(labels)},
                )
                clean_rec = recon.correct(x_clean, steps=1, rho=min(rho, 0.05))["x_hat"]
                loss_clean_preserve = torch.nn.functional.mse_loss(clean_rec, x_clean)
                loss = (
                    float(loss_cfg.get("lambda_id", 0.8)) * loss_id
                    + float(loss_cfg.get("lambda_res", 3.0)) * loss_res
                    + float(loss_cfg.get("lambda_clean_preserve", 1.0)) * loss_clean_preserve
                    + float(loss_cfg.get("lambda_sgc", 1.0)) * sgc_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_recon + params_sgc, 1.0)
            scaler.step(optimizer)
            scaler.update()
            logs_epoch.append(
                {
                    "train/loss_total": loss.detach(),
                    "train/loss_id": loss_id.detach(),
                    "train/loss_res": loss_res.detach(),
                    "train/loss_clean_preserve": loss_clean_preserve.detach(),
                    "train/loss_sgc": sgc_loss.detach(),
                    "train/residual_ratio_p95": torch.quantile(rec["residual_ratio"].detach().float(), 0.95),
                    **log_id,
                    **sgc_logs,
                }
            )
        logs_mean = mean_logs(logs_epoch)
        eval_result = evaluate_training_tests(
            model=eval_model,
            val_loader=data_ctx["val_loader"],
            named_test_loaders=data_ctx["named_test_loaders"],
            device=device,
            domain_label_map=data_ctx["domain_label_map"],
            named_test_meta=data_ctx["named_test_meta"],
            dataset=args.dataset,
            max_batches=int(args.eval_max_batches),
            evaluate_loader_fn=evaluate_loader,
            evaluate_named_loaders_fn=evaluate_named_loaders,
        )
        payload = {"recon": recon.state_dict(), "sgc_v3": sgc.state_dict(), "epoch": epoch, "args": vars(args), "config": cfg, "logs": logs_mean}
        payload["eval"] = {
            "val": eval_result.val_stats,
            "test": eval_result.test_stats,
            "named_test": eval_result.named_test_stats,
            "eval_recon_steps": int(args.eval_recon_steps),
            "eval_recon_rho": eval_rho,
        }
        latest_path = out_dir / "latest_recon_sgc_joint.pth"
        best_path = out_dir / "best_recon_sgc_joint.pth"
        save_payload(latest_path, payload)
        save_payload(best_path, payload)
        print(format_joint_epoch_report(
            epoch=epoch,
            epochs=int(args.epochs),
            lr_recon=optimizer.param_groups[0]["lr"],
            lr_sgc=optimizer.param_groups[1]["lr"],
            epoch_time_s=time.time() - t0,
            rho=rho,
            r_max=r_max,
            logs=logs_mean,
            latest_path=str(latest_path),
            best_path=str(best_path),
            eval_result=eval_result,
        ), flush=True)
    return 0


def main() -> int:
    return main_train(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
