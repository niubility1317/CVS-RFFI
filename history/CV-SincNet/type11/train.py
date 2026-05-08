import math
import time
import argparse
import random
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
try:
    from dataset_wisig import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test
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


def sanitize_gradients(model: nn.Module, value_clip: float = 5.0, limit: int = 6) -> Tuple[str, ...]:
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
        # Receiver-style nuisance preferred when available.
        return rx if rx is not None else d
    return d


def infer_nuisance_classes(train_ds, target: str = "auto") -> int:
    target = str(target).lower().strip()

    def _get_index(ds):
        return getattr(ds, "index", None)

    index = _get_index(train_ds)
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


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=1, eps=1e-4)
    b = F.normalize(b.float(), dim=1, eps=1e-4)
    return (a * b).sum(dim=1).mean()


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


def stage_scales(epoch: int, epochs: int, warmup_frac: float = 0.2, stage2_frac: float = 0.5):
    r = float(epoch) / max(1.0, float(epochs))
    if r <= warmup_frac:
        return {
            "dom": 0.5,
            "adv": 0.0,
            "decor": 0.0,
            "pa_con": 0.0,
            "compact": 0.2,
            "proxy": 1.0,
        }
    if r <= stage2_frac:
        return {
            "dom": 1.0,
            "adv": 0.3,
            "decor": 0.2,
            "pa_con": 1.0,
            "compact": 0.6,
            "proxy": 0.7,
        }
    return {
        "dom": 1.0,
        "adv": 1.0,
        "decor": 1.0,
        "pa_con": 1.0,
        "compact": 1.0,
        "proxy": 0.35,
    }


@torch.no_grad()
def evaluate(model, loader, device, nuisance_target: str = "auto", num_nuisance_train: int = 1, max_batches: int = 0):
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    probe_correct = probe_total = 0
    style_tx_probe_correct = style_tx_probe_total = 0
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        nlab = get_nuisance_target(extra, device, target=nuisance_target)
        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        tx_logits = out["tx_logits"]
        tx_correct += int((tx_logits.argmax(dim=1) == y).sum().item())
        tx_total += int(y.numel())
        style_tx_probe_correct += int((out["style_tx_probe_logits"].argmax(dim=1) == y).sum().item())
        style_tx_probe_total += int(y.numel())
        if nlab is not None:
            valid = nlab < int(num_nuisance_train)
            if valid.any():
                yy = nlab[valid]
                dom_correct += int((out["dom_logits"][valid].argmax(dim=1) == yy).sum().item())
                dom_total += int(yy.numel())
                probe_correct += int((out["probe_dom_logits"][valid].argmax(dim=1) == yy).sum().item())
                probe_total += int(yy.numel())
        if max_batches > 0 and (bi + 1) >= max_batches:
            break
    return {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": 100.0 * probe_correct / max(1, probe_total) if probe_total > 0 else float("nan"),
        "style_tx_probe_acc": 100.0 * style_tx_probe_correct / max(1, style_tx_probe_total) if style_tx_probe_total > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wisig", choices=["wisig", "oralce"])
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_equalized", type=str, default="1")
    parser.add_argument("--wisig_domain", type=str, default="day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_train_ratio", type=float, default=0.8)
    parser.add_argument("--wisig_train_days", type=str, default="0,1,2")
    parser.add_argument("--wisig_full_test_days", type=str, default="3")
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)

    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--model_size", type=str, default="M")

    parser.add_argument("--lambda_dom", type=float, default=0.5)
    parser.add_argument("--lambda_adv", type=float, default=0.2)
    parser.add_argument("--lambda_decor", type=float, default=0.01)
    parser.add_argument("--lambda_pa_con", type=float, default=0.10)
    parser.add_argument("--lambda_rx_compact", type=float, default=0.20)
    parser.add_argument("--lambda_probe", type=float, default=0.10)
    parser.add_argument("--lambda_style_tx_probe", type=float, default=0.05)
    parser.add_argument("--lambda_pa_proxy", type=float, default=0.30)
    parser.add_argument("--lambda_dac_proxy", type=float, default=0.20)
    parser.add_argument("--grl_lambda", type=float, default=1.0)

    parser.add_argument("--nuisance_target", type=str, default="auto", choices=["auto", "domain", "rx", "day"])
    parser.add_argument("--warmup_frac", type=float, default=0.20)
    parser.add_argument("--stage2_frac", type=float, default=0.50)

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="best_model_v11.pth")

    parser.add_argument("--use_aug", dest="use_aug", action="store_true")
    parser.add_argument("--no_aug", dest="use_aug", action="store_false")
    parser.set_defaults(use_aug=True)
    parser.add_argument("--aug_p_dac", type=float, default=0.20)
    parser.add_argument("--aug_p_pa", type=float, default=0.25)
    parser.add_argument("--aug_enable_class_signature", dest="aug_enable_class_signature", action="store_true")
    parser.add_argument("--aug_disable_class_signature", dest="aug_enable_class_signature", action="store_false")
    parser.set_defaults(aug_enable_class_signature=False)
    parser.add_argument("--aug_class_sig_mix", type=float, default=0.0)
    parser.add_argument("--proxy_defect_apply_channel", dest="proxy_defect_apply_channel", action="store_true")
    parser.add_argument("--proxy_no_defect_apply_channel", dest="proxy_defect_apply_channel", action="store_false")
    parser.set_defaults(proxy_defect_apply_channel=False)

    parser.add_argument("--detach_imp_gate", dest="detach_imp_gate", action="store_true")
    parser.add_argument("--no_detach_imp_gate", dest="detach_imp_gate", action="store_false")
    parser.set_defaults(detach_imp_gate=True)
    parser.add_argument("--disable_freq_stats_to_shared", dest="disable_freq_stats_to_shared", action="store_true")
    parser.add_argument("--enable_freq_stats_to_shared", dest="disable_freq_stats_to_shared", action="store_false")
    parser.set_defaults(disable_freq_stats_to_shared=True)
    parser.add_argument("--input_abs_clip", type=float, default=6.0)
    parser.add_argument("--skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_true")
    parser.add_argument("--no_skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_false")
    parser.set_defaults(skip_nonfinite_batches=True)
    parser.add_argument("--nonfinite_log_limit", type=int, default=5)
    parser.add_argument("--grad_value_clip", type=float, default=5.0)

    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(f"[WARN] Ignored unknown args: {unknown}")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Starting Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} | AMP: {use_amp}")

    if float(args.sample_rate_hz) <= 0.0:
        args.sample_rate_hz = 25e6 if args.dataset == "wisig" else 5e6

    split_info = None
    input_len = 1024
    if args.dataset == "wisig":
        ds_w = load_wisig_compact_pkl(args.wisig_pkl)
        infer_nc = len(ds_w.get("tx_list", []))
        if infer_nc > 0 and args.num_classes != infer_nc:
            print(f"[WISIG] overriding num_classes {args.num_classes} -> {infer_nc}")
            args.num_classes = infer_nc
        eq2 = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
        max_tr = None if int(args.wisig_max_train_per_combo) <= 0 else int(args.wisig_max_train_per_combo)
        max_te = None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo)

        def parse_days(s: str):
            s = str(s).strip()
            if s == "":
                return None
            out = []
            for item in s.split(","):
                item = item.strip()
                if item == "":
                    continue
                try:
                    out.append(int(item))
                except Exception:
                    out.append(item)
            return out if len(out) > 0 else None

        train_ds, test_ds, split_info = make_day123_randomsplit_plus_day4_test(
            ds_w,
            equalized=eq2,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform=None,
            train_ratio=float(args.wisig_train_ratio),
            train_days=parse_days(args.wisig_train_days),
            full_test_days=parse_days(args.wisig_full_test_days),
            max_samples_per_combo_train=max_tr,
            max_samples_per_combo_test=max_te,
            seed=int(args.seed),
        )
        input_len = int(args.wisig_out_len)
        print(f"[WISIG] pkl={args.wisig_pkl} eq={eq2} out_len={input_len} domain={args.wisig_domain}")
        print(f"[WISIG] TRAIN SOURCES: {split_info['train_days_label']} -> random {int(100 * split_info['train_ratio'])}%")
        print(f"[WISIG] TEST SOURCES : same days remaining {100 - int(100 * split_info['train_ratio'])}% + full {split_info['full_test_days_label']}")
        print(f"[WISIG] split_info={split_info}")
    else:
        train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train", run_name=args.run_name)
        test_ds = WiFiRFFIDataset(args.dataset_dir, mode="test", run_name=args.run_name)
        try:
            x0, _ = train_ds[0]
            input_len = int(x0.shape[-1])
        except Exception:
            input_len = 1024
        print(f"[ORALCE] dir={args.dataset_dir} run={args.run_name} input_len={input_len}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    num_nuisance = infer_nuisance_classes(train_ds, target=args.nuisance_target)
    print(f"[NUISANCE] target={args.nuisance_target} classes={num_nuisance}")

    model = build_dual_model(
        args.num_classes,
        num_nuisance,
        model_size=args.model_size,
        dataset=args.dataset,
        input_len=input_len,
        sample_rate_hz=float(args.sample_rate_hz),
        detach_imp_gate=bool(args.detach_imp_gate),
        disable_freq_stats_to_shared=bool(args.disable_freq_stats_to_shared),
    ).to(device)
    print(f"[MODEL] RefactorDualCVSincNet emb_dim={model.emb_dim} nuisance_dim={model.nuisance_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_tx = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    ce_dom = nn.CrossEntropyLoss()

    augmentor_main = None
    augmentor_proxy = None
    if bool(args.use_aug) and build_augmentor is not None:
        augmentor_main = build_augmentor(
            p_dac=float(args.aug_p_dac),
            p_pa=float(args.aug_p_pa),
            enable_class_signature=bool(args.aug_enable_class_signature),
            class_sig_mix=float(args.aug_class_sig_mix),
            defect_apply_channel=False,
        )
        augmentor_proxy = build_augmentor(
            p_dac=1.0,
            p_pa=1.0,
            enable_class_signature=False,
            class_sig_mix=0.0,
            defect_apply_channel=bool(args.proxy_defect_apply_channel),
        )
        print(f"[AUG] enabled. main defects offloaded to proxy views; class_signature={args.aug_enable_class_signature}")
    elif bool(args.use_aug):
        print("[AUG] requested but build_augmentor not found, disabled.")

    best_tx = -1.0
    best_epoch = 0
    nonfinite_logs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meters = {k: AverageMeter() for k in [
            "loss", "cls", "dom", "adv", "decor", "pa_con", "compact", "probe", "style_tx_probe",
            "pa_proxy", "dac_proxy", "tx_acc", "dom_acc", "probe_dom_acc", "style_tx_probe_acc",
            "corr_pa", "corr_dac"
        ]}
        pa_con_cos_vals = []
        nonfinite_batches = 0
        scale = stage_scales(epoch, args.epochs, warmup_frac=args.warmup_frac, stage2_frac=args.stage2_frac)

        for batch in train_loader:
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            nlab = get_nuisance_target(extra, device, target=args.nuisance_target)
            if nlab is not None:
                nlab = nlab.view(-1)

            clip_val = float(args.input_abs_clip)
            x = sanitize_batch_tensor(x, abs_clip=clip_val)
            if augmentor_main is not None:
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
            loss = None
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_main = model(x_main, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True)
                out_pa = model(x_pa, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True)
                out_dac = model(x_dac, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True)

                tx_logits = out_main["tx_logits"]
                dom_logits = out_main["dom_logits"]
                adv_dom_logits = out_main["adv_dom_logits"]
                probe_dom_logits = out_main["probe_dom_logits"]
                style_tx_probe_logits = out_main["style_tx_probe_logits"]
                z_pa = out_main["z_pa"]
                z_rxc = out_main["z_rxc"]
                pa_pred_pa = out_pa["aux_id"].get("pa_pred", None)
                dac_pred_dac = out_dac["aux_id"].get("dac_pred", None)

                loss_ref = tx_logits.float().mean()
                loss_cls = ce_tx(tx_logits.float(), y)
                if nlab is not None and int(num_nuisance) > 1:
                    loss_dom = ce_dom(dom_logits.float(), nlab)
                    loss_adv = ce_dom(adv_dom_logits.float(), nlab)
                    loss_probe = ce_dom(probe_dom_logits.float(), nlab)
                    loss_pa_con, pa_con_cos = same_tx_diff_nuisance_consistency(z_pa, y, nlab)
                    if not math.isnan(pa_con_cos):
                        pa_con_cos_vals.append(pa_con_cos)
                    loss_compact = center_compact_loss(z_rxc, nlab)
                    loss_decor = covariance_decorrelation_loss(z_pa, z_rxc)
                    meters["dom_acc"].update(accuracy_from_logits(dom_logits, nlab), x.size(0))
                    meters["probe_dom_acc"].update(accuracy_from_logits(probe_dom_logits, nlab), x.size(0))
                else:
                    loss_dom = fp32_zero(loss_ref)
                    loss_adv = fp32_zero(loss_ref)
                    loss_probe = fp32_zero(loss_ref)
                    loss_pa_con = fp32_zero(loss_ref)
                    loss_compact = fp32_zero(loss_ref)
                    loss_decor = fp32_zero(loss_ref)
                    pa_con_cos = float("nan")

                loss_style_tx_probe = ce_tx(style_tx_probe_logits.float(), y)

                if torch.is_tensor(pa_pred_pa):
                    loss_pa_proxy = F.mse_loss(pa_pred_pa.float().view(-1), s_p.float().view(-1))
                else:
                    loss_pa_proxy = fp32_zero(loss_ref)
                if torch.is_tensor(dac_pred_dac):
                    loss_dac_proxy = F.mse_loss(dac_pred_dac.float().view(-1), s_d.float().view(-1))
                else:
                    loss_dac_proxy = fp32_zero(loss_ref)

                total = (
                    loss_cls
                    + float(args.lambda_dom) * scale["dom"] * loss_dom
                    + float(args.lambda_adv) * scale["adv"] * loss_adv
                    + float(args.lambda_decor) * scale["decor"] * loss_decor
                    + float(args.lambda_pa_con) * scale["pa_con"] * loss_pa_con
                    + float(args.lambda_rx_compact) * scale["compact"] * loss_compact
                    + float(args.lambda_probe) * loss_probe
                    + float(args.lambda_style_tx_probe) * loss_style_tx_probe
                    + float(args.lambda_pa_proxy) * scale["proxy"] * loss_pa_proxy
                    + float(args.lambda_dac_proxy) * scale["proxy"] * loss_dac_proxy
                )
                bad = find_nonfinite_tensors({
                    "loss": total,
                    "tx_logits": tx_logits,
                    "dom_logits": dom_logits,
                    "adv_dom_logits": adv_dom_logits,
                    "z_pa": z_pa,
                    "z_rxc": z_rxc,
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
            meters["cls"].update(loss_cls.item(), bsz)
            meters["dom"].update(loss_dom.item(), bsz)
            meters["adv"].update(loss_adv.item(), bsz)
            meters["decor"].update(loss_decor.item(), bsz)
            meters["pa_con"].update(loss_pa_con.item(), bsz)
            meters["compact"].update(loss_compact.item(), bsz)
            meters["probe"].update(loss_probe.item(), bsz)
            meters["style_tx_probe"].update(loss_style_tx_probe.item(), bsz)
            meters["pa_proxy"].update(loss_pa_proxy.item(), bsz)
            meters["dac_proxy"].update(loss_dac_proxy.item(), bsz)
            meters["tx_acc"].update(accuracy_from_logits(tx_logits, y), bsz)
            meters["style_tx_probe_acc"].update(accuracy_from_logits(style_tx_probe_logits, y), bsz)

            corr_pa = batch_corrcoef(pa_pred_pa, s_p)
            corr_dac = batch_corrcoef(dac_pred_dac, s_d)
            if not math.isnan(corr_pa):
                meters["corr_pa"].update(corr_pa, bsz)
            if not math.isnan(corr_dac):
                meters["corr_dac"].update(corr_dac, bsz)

        scheduler.step()
        pa_con_cos_epoch = float(np.mean(pa_con_cos_vals)) if len(pa_con_cos_vals) > 0 else float("nan")

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
                    },
                    args.save_path,
                )

        msg = (
            f"[E{epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e} loss={meters['loss'].avg:.4f} "
            f"cls={meters['cls'].avg:.4f} dom={meters['dom'].avg:.4f} adv={meters['adv'].avg:.4f} "
            f"decor={meters['decor'].avg:.4f} pa_con={meters['pa_con'].avg:.4f} compact={meters['compact'].avg:.4f} "
            f"probe={meters['probe'].avg:.4f} style_tx_probe={meters['style_tx_probe'].avg:.4f} "
            f"pa_proxy={meters['pa_proxy'].avg:.4f} dac_proxy={meters['dac_proxy'].avg:.4f} | "
            f"tx_acc={meters['tx_acc'].avg:.2f}% dom_acc={meters['dom_acc'].avg if meters['dom_acc'].count else float('nan'):.2f}% "
            f"probe_dom_acc={meters['probe_dom_acc'].avg if meters['probe_dom_acc'].count else float('nan'):.2f}% "
            f"style_tx_probe_acc={meters['style_tx_probe_acc'].avg:.2f}% pa_con_cos={pa_con_cos_epoch:.4f} "
            f"corr_pa={meters['corr_pa'].avg if meters['corr_pa'].count else float('nan'):.4f} "
            f"corr_dac={meters['corr_dac'].avg if meters['corr_dac'].count else float('nan'):.4f} "
            f"stage=dom{scale['dom']:.1f}/adv{scale['adv']:.1f}/decor{scale['decor']:.1f}/pc{scale['pa_con']:.1f}/cmp{scale['compact']:.1f}/px{scale['proxy']:.2f} "
            f"nf_skip={nonfinite_batches} | time={time.time()-t0:.1f}s"
        )
        if eval_stats is not None:
            msg += (
                f" | val_tx={eval_stats['tx_acc']:.2f}% val_dom={eval_stats['dom_acc']:.2f}% "
                f"val_probe={eval_stats['probe_dom_acc']:.2f}% val_style_tx_probe={eval_stats['style_tx_probe_acc']:.2f}% "
                f"best={best_tx:.2f}%@E{best_epoch:03d}"
            )
        print(msg, flush=True)

    print(f"Training finished. best_tx_acc={best_tx:.2f}% at epoch {best_epoch}")
    if split_info is not None:
        print(f"Final split info: {split_info}")


if __name__ == "__main__":
    main()
