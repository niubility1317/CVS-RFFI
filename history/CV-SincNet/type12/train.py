import math
import time
import argparse
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
try:
    from dataset_wisig_modified import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test
except Exception:
    from dataset_wisig import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test

from model_dual_cvsincnet import build_dual_model

try:
    from DataAugmentation_v2 import build_augmentor
except Exception:
    build_augmentor = None


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        self.sum += float(v) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def fp32_zero(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=ref.device, dtype=torch.float32)


def sanitize_batch_tensor(x: torch.Tensor, abs_clip: float = 0.0) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if abs_clip > 0:
        x = x.clamp(min=-abs_clip, max=abs_clip)
    return x


def tensor_is_finite(x: Optional[torch.Tensor]) -> bool:
    return torch.is_tensor(x) and bool(torch.isfinite(x).all().item())


def find_nonfinite_tensors(named_tensors: Dict[str, Optional[torch.Tensor]]) -> Tuple[str, ...]:
    bad = []
    for name, value in named_tensors.items():
        if torch.is_tensor(value) and (not tensor_is_finite(value)):
            bad.append(name)
    return tuple(bad)


def sanitize_gradients(model: nn.Module, value_clip: float = 5.0, limit: int = 8) -> Tuple[str, ...]:
    repaired = []
    do_clip = float(value_clip) > 0
    clip_val = float(value_clip)
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        had_bad = not torch.isfinite(grad).all()
        if had_bad:
            if len(repaired) < limit:
                repaired.append(name)
            param.grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad = param.grad
        if do_clip:
            grad.clamp_(min=-clip_val, max=clip_val)
    return tuple(repaired)


def extract_domain_from_extra(extra, device) -> Optional[torch.Tensor]:
    if extra is None or len(extra) == 0:
        return None
    d0 = extra[0]
    if torch.is_tensor(d0):
        return d0.to(device, non_blocking=True).view(-1)
    try:
        return torch.as_tensor(d0, device=device).view(-1)
    except Exception:
        return None


def extract_meta_from_extra(extra) -> Optional[Dict[str, Any]]:
    if extra is None or len(extra) < 2:
        return None
    meta = extra[1]
    return meta if isinstance(meta, dict) else None


def meta_get_tensor(meta: Optional[Dict[str, Any]], key: str, device) -> Optional[torch.Tensor]:
    if meta is None or key not in meta:
        return None
    value = meta[key]
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True).view(-1)
    try:
        return torch.as_tensor(value, device=device).view(-1)
    except Exception:
        return None


def get_nuisance_target(extra, device, target: str = "auto") -> Optional[torch.Tensor]:
    target = str(target).lower().strip()
    d = extract_domain_from_extra(extra, device)
    meta = extract_meta_from_extra(extra)
    rx = meta_get_tensor(meta, "rx_i", device)
    day = meta_get_tensor(meta, "day_i", device)
    if target == "rx":
        return rx if rx is not None else d
    if target == "day":
        return day if day is not None else d
    if target == "domain":
        return d
    if target == "auto":
        return rx if rx is not None else d
    return d


def infer_nuisance_classes(train_ds, target: str = "auto") -> int:
    target = str(target).lower().strip()
    index = getattr(train_ds, "index", None)
    if index is not None and len(index) > 0:
        if target == "rx":
            return len(sorted({int(it.rx_i) for it in index}))
        if target == "day":
            return len(sorted({int(it.day_i) for it in index}))
        if target == "auto":
            vals = sorted({int(it.rx_i) for it in index})
            if len(vals) > 0:
                return len(vals)
        if hasattr(train_ds, "_domain_lut"):
            return len(sorted({int(train_ds._domain_lut[(it.rx_i, it.day_i)]) for it in index}))
    if hasattr(train_ds, "rx_list") and target in ("rx", "auto"):
        return max(1, len(getattr(train_ds, "rx_list")))
    if hasattr(train_ds, "day_list") and target == "day":
        return max(1, len(getattr(train_ds, "day_list")))
    return 1


def covariance_decorrelation_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean(dim=0, keepdim=True)
    b = b.float() - b.float().mean(dim=0, keepdim=True)
    n = a.size(0)
    if n <= 1:
        return fp32_zero(a)
    cov = (a.t() @ b) / float(n - 1)
    return torch.mean(cov * cov)


def same_tx_diff_nuisance_consistency(z: torch.Tensor, y: torch.Tensor, nlab: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    if nlab is None:
        return fp32_zero(z), float("nan")
    z = F.normalize(z.float(), dim=1, eps=1e-4)
    y = y.view(-1)
    nlab = nlab.view(-1)
    losses = []
    sims = []
    for cls in torch.unique(y):
        m_cls = (y == cls)
        nuis = torch.unique(nlab[m_cls])
        if nuis.numel() < 2:
            continue
        cents = []
        for nv in nuis:
            m = m_cls & (nlab == nv)
            if m.sum() == 0:
                continue
            cents.append(F.normalize(z[m].mean(dim=0, keepdim=True), dim=1, eps=1e-4).squeeze(0))
        if len(cents) < 2:
            continue
        c = torch.stack(cents, dim=0)
        sim = c @ c.t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        pair_sim = sim[iu[0], iu[1]]
        losses.append((1.0 - pair_sim).mean())
        sims.append(pair_sim.mean().item())
    if len(losses) == 0:
        return fp32_zero(z), float("nan")
    return torch.stack(losses).mean(), float(np.mean(sims))


def center_compact_loss(z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
    if labels is None:
        return fp32_zero(z)
    z = F.normalize(z.float(), dim=1, eps=1e-4)
    labels = labels.view(-1)
    uniq = torch.unique(labels)
    losses = []
    for cls in uniq:
        m = labels == cls
        if int(m.sum()) < 2:
            continue
        c = F.normalize(z[m].mean(dim=0, keepdim=True), dim=1, eps=1e-4)
        losses.append(1.0 - (z[m] * c).sum(dim=1).mean())
    if len(losses) == 0:
        return fp32_zero(z)
    return torch.stack(losses).mean()


def batch_corrcoef(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return float("nan")
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    n = min(x.numel(), y.numel())
    if n < 2:
        return float("nan")
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    den = x.norm().item() * y.norm().item()
    if den <= 1e-12:
        return float("nan")
    return float((x * y).sum().item() / den)


def stage_scales(epoch: int, epochs: int, warmup_frac: float = 0.15, stage2_frac: float = 0.50):
    r = float(epoch) / max(1.0, float(epochs))
    if r <= warmup_frac:
        return {
            "rx": 0.30,
            "adv_rx": 0.0,
            "adv_tx": 0.0,
            "sep": 0.0,
            "center": 0.10,
            "cons": 0.0,
            "probe": 0.0,
            "proxy": 0.60,
        }
    if r <= stage2_frac:
        return {
            "rx": 0.60,
            "adv_rx": 0.35,
            "adv_tx": 0.20,
            "sep": 0.40,
            "center": 0.40,
            "cons": 0.50,
            "probe": 0.30,
            "proxy": 0.35,
        }
    return {
        "rx": 1.00,
        "adv_rx": 1.00,
        "adv_tx": 1.00,
        "sep": 1.00,
        "center": 1.00,
        "cons": 1.00,
        "probe": 0.40,
        "proxy": 0.20,
    }


def set_sinc_trainable(model: nn.Module, trainable: bool = True):
    changed = 0
    for name, param in model.named_parameters():
        if "id_backbone.sinc." in name:
            param.requires_grad = bool(trainable)
            changed += 1
    return changed


@torch.no_grad()
def evaluate(model, loader, device, nuisance_target: str = "auto", num_nuisance_train: int = 1, max_batches: int = 0):
    model.eval()
    meters = {k: AverageMeter() for k in [
        "tx_acc", "rx_acc", "probe_rx_acc", "probe_tx_acc"
    ]}
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)
        nlab = get_nuisance_target(extra, device, target=nuisance_target)
        out = model(x, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True)
        meters["tx_acc"].update(accuracy_from_logits(out["tx_logits"], y), x.size(0))
        if nlab is not None and int(num_nuisance_train) > 1:
            meters["rx_acc"].update(accuracy_from_logits(out["rx_logits"], nlab), x.size(0))
            meters["probe_rx_acc"].update(accuracy_from_logits(out["probe_rx_logits"], nlab), x.size(0))
        meters["probe_tx_acc"].update(accuracy_from_logits(out["probe_tx_logits"], y), x.size(0))
    return {k: v.avg for k, v in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wisig", choices=["oralce", "wisig"])
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_eq", type=str, default="1")
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_domain", type=str, default="day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_train_days", type=str, default="0,1,2")
    parser.add_argument("--wisig_full_test_days", type=str, default="3")
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--sample_rate_hz", type=float, default=25e6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sinc_lr_scale", type=float, default=0.1)
    parser.add_argument("--freeze_sinc_epochs", type=int, default=5)
    parser.add_argument("--rx_dim", type=int, default=128)
    parser.add_argument("--rx_branch_dim", type=int, default=64)
    parser.add_argument("--lambda_rx", type=float, default=1.0)
    parser.add_argument("--lambda_adv_rx", type=float, default=0.15)
    parser.add_argument("--lambda_adv_tx", type=float, default=0.08)
    parser.add_argument("--lambda_sep", type=float, default=0.02)
    parser.add_argument("--lambda_center_rx", type=float, default=0.05)
    parser.add_argument("--lambda_cons_tx", type=float, default=0.08)
    parser.add_argument("--lambda_probe", type=float, default=0.02)
    parser.add_argument("--lambda_pa_proxy", type=float, default=0.20)
    parser.add_argument("--lambda_dac_proxy", type=float, default=0.15)
    parser.add_argument("--nuisance_target", type=str, default="auto", choices=["auto", "domain", "rx", "day"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="best_model_dual_v12.pth")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--amp_warmup_epochs", type=int, default=8)
    parser.add_argument("--grad_value_clip", type=float, default=2.0)
    parser.add_argument("--input_abs_clip", type=float, default=4.0)
    parser.add_argument("--grl_lambda", type=float, default=1.0)
    parser.add_argument("--skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_true")
    parser.add_argument("--no_skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_false")
    parser.set_defaults(skip_nonfinite_batches=True)
    parser.add_argument("--nonfinite_log_limit", type=int, default=6)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} | AMP: {args.amp}")

    split_info = None
    if args.dataset.lower() == "wisig":
        tx_days = [int(s) for s in args.wisig_train_days.split(",") if str(s).strip() != ""]
        test_days = [int(s) for s in args.wisig_full_test_days.split(",") if str(s).strip() != ""]
        ds = load_wisig_compact_pkl(args.wisig_pkl)
        train_ds, test_ds, split_info = make_day123_randomsplit_plus_day4_test(
            ds,
            out_len=args.wisig_out_len,
            crop_mode="center",
            normalize=True,
            equalized=args.wisig_eq,
            domain=args.wisig_domain,
            train_days=tx_days,
            full_test_days=test_days,
            train_ratio=0.8,
            seed=args.seed,
        )
        num_classes = len(getattr(train_ds, "tx_list", [])) if hasattr(train_ds, "tx_list") else 6
        print(f"[WISIG] overriding num_classes 16 -> {num_classes}")
        print(f"[WISIG] pkl={args.wisig_pkl} eq={args.wisig_eq} out_len={args.wisig_out_len} domain={args.wisig_domain}")
        if split_info is not None:
            print(f"[WISIG] TRAIN SOURCES: {split_info.get('train_days_label')} -> random 80%")
            print(f"[WISIG] TEST SOURCES : same days remaining 20% + full {split_info.get('full_test_days_label')}")
            print(f"[WISIG] split_info={split_info}")
    else:
        train_ds = WiFiRFFIDataset(args.dataset_dir, split="train")
        test_ds = WiFiRFFIDataset(args.dataset_dir, split="test")
        num_classes = int(getattr(train_ds, "num_classes", 16))

    num_nuisance = infer_nuisance_classes(train_ds, target=args.nuisance_target)
    print(f"[NUISANCE] target={args.nuisance_target} classes={num_nuisance}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = build_dual_model(
        num_classes=num_classes,
        num_domains=num_nuisance,
        model_size=args.model_size,
        dataset=args.dataset,
        input_len=args.wisig_out_len if args.dataset.lower() == "wisig" else 1024,
        sample_rate_hz=args.sample_rate_hz,
        rx_dim=args.rx_dim,
        rx_branch_dim=args.rx_branch_dim,
    ).to(device)
    print(f"[MODEL] DualRxADCDisentangle tx_dim={model.tx_dim} rx_dim={model.rx_dim}")

    sinc_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "id_backbone.sinc." in name:
            sinc_params.append(p)
        else:
            other_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": sinc_params, "lr": args.lr * args.sinc_lr_scale, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    ce_tx = nn.CrossEntropyLoss()
    ce_rx = nn.CrossEntropyLoss()
    augmentor_main = None
    augmentor_proxy = None
    if args.augment and (build_augmentor is not None):
        augmentor_main = build_augmentor(sample_rate_hz=args.sample_rate_hz)
        augmentor_proxy = build_augmentor(sample_rate_hz=args.sample_rate_hz)
        print("[AUG] enabled. main defects offloaded to proxy views; class_signature=False")
    else:
        print("[AUG] disabled.")

    best_tx = -1.0
    best_epoch = -1
    nonfinite_logs = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= int(args.freeze_sinc_epochs):
            set_sinc_trainable(model, trainable=False)
        elif epoch == int(args.freeze_sinc_epochs) + 1:
            set_sinc_trainable(model, trainable=True)

        model.train()
        t0 = time.time()
        scale = stage_scales(epoch, args.epochs)
        use_amp_epoch = bool(args.amp) and (epoch > int(args.amp_warmup_epochs))
        meters = {k: AverageMeter() for k in [
            "loss", "tx", "rx", "adv_rx", "adv_tx", "sep", "center", "cons", "probe_rx", "probe_tx",
            "pa_proxy", "dac_proxy", "tx_acc", "rx_acc", "probe_rx_acc", "probe_tx_acc", "corr_pa", "corr_dac"
        ]}
        cons_cos_vals = []
        nonfinite_batches = 0

        for batch in train_loader:
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long().view(-1)
            nlab = get_nuisance_target(extra, device, target=args.nuisance_target)

            clip_val = float(args.input_abs_clip)
            if augmentor_main is not None and augmentor_proxy is not None:
                x_main = augmentor_main(x, labels=y, no_dac=True, no_pa=True)
                x_pa, s_p = augmentor_proxy(x, labels=y, pa_only=True, return_pa_strength=True)
                x_dac, s_d = augmentor_proxy(x, labels=y, dac_only=True, return_dac_strength=True)
            else:
                x_main = x
                x_pa = x
                x_dac = x
                s_p = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
                s_d = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)

            x_main = sanitize_batch_tensor(x_main, abs_clip=clip_val)
            x_pa = sanitize_batch_tensor(x_pa, abs_clip=clip_val)
            x_dac = sanitize_batch_tensor(x_dac, abs_clip=clip_val)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp_epoch):
                grl_rx = float(args.grl_lambda) * float(scale["adv_rx"])
                grl_tx = float(args.grl_lambda) * float(scale["adv_tx"])
                out_main = model(x_main, y_tx=y, grl_rx_on_tx=grl_rx, grl_tx_on_rx=grl_tx, return_aux=True)
                out_pa = model(x_pa, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True)
                out_dac = model(x_dac, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True)

                tx_logits = out_main["tx_logits"]
                rx_logits = out_main["rx_logits"]
                adv_rx_logits = out_main["adv_rx_logits"]
                adv_tx_logits = out_main["adv_tx_logits"]
                probe_rx_logits = out_main["probe_rx_logits"]
                probe_tx_logits = out_main["probe_tx_logits"]
                z_tx = out_main["z_tx"]
                z_rx = out_main["z_rx"]
                pa_pred_pa = out_pa["aux_id"].get("pa_pred", None)
                dac_pred_dac = out_dac["aux_id"].get("dac_pred", None)

                loss_ref = tx_logits.float().mean()
                loss_tx = ce_tx(tx_logits.float(), y)
                if nlab is not None and int(num_nuisance) > 1:
                    loss_rx = ce_rx(rx_logits.float(), nlab)
                    loss_adv_rx = ce_rx(adv_rx_logits.float(), nlab)
                    loss_probe_rx = ce_rx(probe_rx_logits.float(), nlab)
                    loss_center = center_compact_loss(z_rx, nlab)
                    loss_cons, cons_cos = same_tx_diff_nuisance_consistency(z_tx, y, nlab)
                    if not math.isnan(cons_cos):
                        cons_cos_vals.append(cons_cos)
                    meters["rx_acc"].update(accuracy_from_logits(rx_logits, nlab), x.size(0))
                    meters["probe_rx_acc"].update(accuracy_from_logits(probe_rx_logits, nlab), x.size(0))
                else:
                    loss_rx = fp32_zero(loss_ref)
                    loss_adv_rx = fp32_zero(loss_ref)
                    loss_probe_rx = fp32_zero(loss_ref)
                    loss_center = fp32_zero(loss_ref)
                    loss_cons = fp32_zero(loss_ref)
                    cons_cos = float("nan")
                loss_adv_tx = ce_tx(adv_tx_logits.float(), y)
                loss_probe_tx = ce_tx(probe_tx_logits.float(), y)
                loss_sep = covariance_decorrelation_loss(z_tx, z_rx)
                if torch.is_tensor(pa_pred_pa):
                    loss_pa_proxy = F.mse_loss(pa_pred_pa.float().view(-1), s_p.float().view(-1))
                else:
                    loss_pa_proxy = fp32_zero(loss_ref)
                if torch.is_tensor(dac_pred_dac):
                    loss_dac_proxy = F.mse_loss(dac_pred_dac.float().view(-1), s_d.float().view(-1))
                else:
                    loss_dac_proxy = fp32_zero(loss_ref)

                total = (
                    loss_tx
                    + float(args.lambda_rx) * scale["rx"] * loss_rx
                    + float(args.lambda_adv_rx) * loss_adv_rx
                    + float(args.lambda_adv_tx) * loss_adv_tx
                    + float(args.lambda_sep) * scale["sep"] * loss_sep
                    + float(args.lambda_center_rx) * scale["center"] * loss_center
                    + float(args.lambda_cons_tx) * scale["cons"] * loss_cons
                    + float(args.lambda_probe) * scale["probe"] * (loss_probe_rx + loss_probe_tx)
                    + float(args.lambda_pa_proxy) * scale["proxy"] * loss_pa_proxy
                    + float(args.lambda_dac_proxy) * scale["proxy"] * loss_dac_proxy
                )
                bad = find_nonfinite_tensors({
                    "loss": total, "tx_logits": tx_logits, "rx_logits": rx_logits, "adv_rx_logits": adv_rx_logits,
                    "adv_tx_logits": adv_tx_logits, "z_tx": z_tx, "z_rx": z_rx,
                })
                loss = total if len(bad) == 0 else loss_ref.new_tensor(float("nan"))

            if not tensor_is_finite(loss):
                nonfinite_batches += 1
                if nonfinite_logs < int(args.nonfinite_log_limit):
                    print(f"[WARN][E{epoch:03d}] skip batch due to non-finite values", flush=True)
                    nonfinite_logs += 1
                if bool(args.skip_nonfinite_batches):
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            repaired = sanitize_gradients(model, value_clip=float(args.grad_value_clip))
            if len(repaired) > 0 and nonfinite_logs < int(args.nonfinite_log_limit):
                print(f"[WARN][E{epoch:03d}] repaired non-finite grads: {', '.join(repaired)}", flush=True)
                nonfinite_logs += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.is_tensor(grad_norm):
                grad_norm = float(grad_norm.item())
            if not math.isfinite(float(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                nonfinite_batches += 1
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            bsz = x.size(0)
            meters["loss"].update(loss.item(), bsz)
            meters["tx"].update(loss_tx.item(), bsz)
            meters["rx"].update(loss_rx.item(), bsz)
            meters["adv_rx"].update(loss_adv_rx.item(), bsz)
            meters["adv_tx"].update(loss_adv_tx.item(), bsz)
            meters["sep"].update(loss_sep.item(), bsz)
            meters["center"].update(loss_center.item(), bsz)
            meters["cons"].update(loss_cons.item(), bsz)
            meters["probe_rx"].update(loss_probe_rx.item(), bsz)
            meters["probe_tx"].update(loss_probe_tx.item(), bsz)
            meters["pa_proxy"].update(loss_pa_proxy.item(), bsz)
            meters["dac_proxy"].update(loss_dac_proxy.item(), bsz)
            meters["tx_acc"].update(accuracy_from_logits(tx_logits, y), bsz)
            meters["probe_tx_acc"].update(accuracy_from_logits(probe_tx_logits, y), bsz)
            corr_pa = batch_corrcoef(pa_pred_pa, s_p)
            corr_dac = batch_corrcoef(dac_pred_dac, s_d)
            if not math.isnan(corr_pa):
                meters["corr_pa"].update(corr_pa, bsz)
            if not math.isnan(corr_dac):
                meters["corr_dac"].update(corr_dac, bsz)

        scheduler.step()
        cons_cos_epoch = float(np.mean(cons_cos_vals)) if len(cons_cos_vals) > 0 else float("nan")

        eval_stats = None
        if epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            eval_stats = evaluate(
                model,
                test_loader,
                device,
                nuisance_target=args.nuisance_target,
                num_nuisance_train=num_nuisance,
                max_batches=int(args.eval_max_batches),
            )
            if eval_stats["tx_acc"] > best_tx:
                best_tx = eval_stats["tx_acc"]
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_tx_acc": best_tx,
                        "args": vars(args),
                        "split_info": split_info,
                        "num_nuisance": num_nuisance,
                    },
                    args.save_path,
                )

        msg = (
            f"[E{epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e} loss={meters['loss'].avg:.4f} "
            f"tx={meters['tx'].avg:.4f} rx={meters['rx'].avg:.4f} adv_rx={meters['adv_rx'].avg:.4f} adv_tx={meters['adv_tx'].avg:.4f} "
            f"sep={meters['sep'].avg:.4f} center={meters['center'].avg:.4f} cons={meters['cons'].avg:.4f} "
            f"probe_rx={meters['probe_rx'].avg:.4f} probe_tx={meters['probe_tx'].avg:.4f} "
            f"pa_proxy={meters['pa_proxy'].avg:.4f} dac_proxy={meters['dac_proxy'].avg:.4f} | "
            f"tx_acc={meters['tx_acc'].avg:.2f}% rx_acc={meters['rx_acc'].avg if meters['rx_acc'].count else float('nan'):.2f}% "
            f"probe_rx_acc={meters['probe_rx_acc'].avg if meters['probe_rx_acc'].count else float('nan'):.2f}% "
            f"probe_tx_acc={meters['probe_tx_acc'].avg:.2f}% cons_cos={cons_cos_epoch:.4f} "
            f"corr_pa={meters['corr_pa'].avg if meters['corr_pa'].count else float('nan'):.4f} "
            f"corr_dac={meters['corr_dac'].avg if meters['corr_dac'].count else float('nan'):.4f} "
            f"stage=rx{scale['rx']:.2f}/arx{scale['adv_rx']:.2f}/atx{scale['adv_tx']:.2f}/sep{scale['sep']:.2f}/ctr{scale['center']:.2f}/con{scale['cons']:.2f}/px{scale['proxy']:.2f} "
            f"nf_skip={nonfinite_batches} | time={time.time()-t0:.1f}s"
        )
        if eval_stats is not None:
            msg += (
                f" | val_tx={eval_stats['tx_acc']:.2f}% val_rx={eval_stats['rx_acc']:.2f}% "
                f"val_probe_rx={eval_stats['probe_rx_acc']:.2f}% val_probe_tx={eval_stats['probe_tx_acc']:.2f}% "
                f"best={best_tx:.2f}%@E{best_epoch:03d}"
            )
        print(msg, flush=True)

    print(f"Training finished. best_tx_acc={best_tx:.2f}% at epoch {best_epoch}")
    if split_info is not None:
        print(f"Final split info: {split_info}")


if __name__ == "__main__":
    main()
