import math
import time
import argparse
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
from dataset_wisig import load_wisig_compact_pkl, make_day123_trainval_day4_test
from model_dual_cvsincnet import build_dual_model


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


class NanMeter:
    def __init__(self):
        self.values = []

    def update(self, v):
        if v is None:
            return
        try:
            fv = float(v)
        except Exception:
            return
        if math.isnan(fv):
            return
        self.values.append(fv)

    @property
    def avg(self):
        return float(np.mean(self.values)) if len(self.values) > 0 else float("nan")

    @property
    def count(self):
        return len(self.values)


def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra


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


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def normalized_accuracy(acc_pct: float, n_cls: int) -> float:
    if n_cls <= 1:
        return 0.0
    chance = 100.0 / float(n_cls)
    return max(0.0, (float(acc_pct) - chance) / max(1e-6, 100.0 - chance))


def covariance_orth_loss(z_id: torch.Tensor, z_dom: torch.Tensor) -> torch.Tensor:
    z_id = z_id.float() - z_id.float().mean(dim=0, keepdim=True)
    z_dom = z_dom.float() - z_dom.float().mean(dim=0, keepdim=True)
    n = z_id.size(0)
    if n <= 1:
        return z_id.new_tensor(0.0)
    cov = (z_id.t() @ z_dom) / float(n - 1)
    return torch.mean(cov * cov)


def same_tx_cross_domain_consistency(z_id: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    if d is None:
        return z_id.new_tensor(0.0), float("nan")
    z = F.normalize(z_id.float(), dim=1)
    y = y.view(-1)
    d = d.view(-1)
    losses = []
    sims = []
    for cls in torch.unique(y):
        m_cls = (y == cls)
        doms = torch.unique(d[m_cls])
        if doms.numel() < 2:
            continue
        cents = []
        for dom in doms:
            m = m_cls & (d == dom)
            if m.sum() == 0:
                continue
            cents.append(F.normalize(z[m].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        if len(cents) < 2:
            continue
        C = torch.stack(cents, dim=0)
        sim = C @ C.t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        pair_sim = sim[iu[0], iu[1]]
        losses.append((1.0 - pair_sim).mean())
        sims.append(pair_sim.mean().item())
    if len(losses) == 0:
        return z_id.new_tensor(0.0), float("nan")
    return torch.stack(losses).mean(), float(np.mean(sims))


def should_warn_domain_bias(dbi_hist, patience: int = 3, threshold: float = 0.05) -> bool:
    if len(dbi_hist) < patience:
        return False
    tail = dbi_hist[-patience:]
    return all(v > threshold for v in tail) and all(tail[i] >= tail[i - 1] - 1e-6 for i in range(1, len(tail)))


@torch.no_grad()
def evaluate(model, loader, device, num_domains_train: int = 1, max_batches: int = 0):
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    probe_correct = probe_total = 0
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d = extract_domain_from_extra(extra, device)
        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        tx_logits = out["tx_logits"]
        tx_correct += int((tx_logits.argmax(dim=1) == y).sum().item())
        tx_total += int(y.numel())
        if d is not None:
            valid = d < int(num_domains_train)
            if valid.any():
                dom_y = d[valid]
                dom_correct += int((out["dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                dom_total += int(dom_y.numel())
                probe_correct += int((out["probe_dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                probe_total += int(dom_y.numel())
        if max_batches > 0 and (bi + 1) >= max_batches:
            break
    return {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": 100.0 * probe_correct / max(1, probe_total) if probe_total > 0 else float("nan"),
    }


def save_checkpoint(path: str, *, model, optimizer, scheduler, scaler, epoch: int, args, split_info, stats: dict):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
        "split_info": split_info,
        "stats": stats,
    }
    torch.save(payload, path)


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
    parser.add_argument("--wisig_guard_gap", type=int, default=8)
    parser.add_argument("--wisig_train_days", type=str, default="0,1,2")
    parser.add_argument("--wisig_test_days", type=str, default="3")
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)

    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--model_size", type=str, default="M")

    parser.add_argument("--lambda_dom", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.5)
    parser.add_argument("--lambda_orth", type=float, default=0.05)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--lambda_probe", type=float, default=1.0)
    parser.add_argument("--grl_lambda", type=float, default=1.0)

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--best_save_path", type=str, default="best_model_v10_type3.pth")
    parser.add_argument("--latest_save_path", type=str, default="latest_modelv10_type3.pth")
    args = parser.parse_args()

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
    val_ds = None
    test_ds = None

    if args.dataset == "wisig":
        ds_w = load_wisig_compact_pkl(args.wisig_pkl)
        infer_nc = len(ds_w.get("tx_list", []))
        if infer_nc > 0 and args.num_classes != infer_nc:
            print(f"[WISIG] overriding num_classes {args.num_classes} -> {infer_nc}")
            args.num_classes = infer_nc

        eq2 = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
        max_day123 = None if int(args.wisig_max_day123_per_combo) <= 0 else int(args.wisig_max_day123_per_combo)
        max_tr = None if int(args.wisig_max_train_per_combo) <= 0 else int(args.wisig_max_train_per_combo)
        max_va = None if int(args.wisig_max_val_per_combo) <= 0 else int(args.wisig_max_val_per_combo)
        max_te = None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo)

        train_ds, val_ds, test_ds, split_info = make_day123_trainval_day4_test(
            ds_w,
            equalized=eq2,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform_train=None,
            transform_eval=None,
            train_ratio=float(args.wisig_train_ratio),
            guard_gap=int(args.wisig_guard_gap),
            train_days=parse_days(args.wisig_train_days),
            test_days=parse_days(args.wisig_test_days),
            max_samples_per_combo_day123=max_day123,
            max_samples_per_combo_test=max_te,
            max_samples_per_combo_train=max_tr,
            max_samples_per_combo_val=max_va,
            seed=int(args.seed),
        )
        input_len = int(args.wisig_out_len)
        print(f"[WISIG] pkl={args.wisig_pkl} eq={eq2} out_len={input_len} domain={args.wisig_domain}")
        print(f"[WISIG] TRAIN DAYS: {split_info['train_days_label']} -> contiguous front {int(100*split_info['train_ratio'])}%")
        print(f"[WISIG] VAL   DAYS: same day1-3 tail (guard_gap={split_info['guard_gap']})")
        print(f"[WISIG] TEST  DAYS: {split_info['test_days_label']} (full day, never merged into val)")
        print(f"[WISIG] split_info={split_info}")
    else:
        train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train", run_name=args.run_name)
        test_ds = WiFiRFFIDataset(args.dataset_dir, mode="test", run_name=args.run_name)
        val_ds = test_ds
        try:
            x0, _ = train_ds[0]
            input_len = int(x0.shape[-1])
        except Exception:
            input_len = 1024
        print(f"[ORALCE] dir={args.dataset_dir} run={args.run_name} input_len={input_len}")
        print("[WARN] ORALCE currently has no separate val set in this script; val=test only for compatibility.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    num_domains = 1
    if hasattr(train_ds, "index") and hasattr(train_ds, "_domain_lut"):
        try:
            doms = sorted({int(train_ds._domain_lut[(it.rx_i, it.day_i)]) for it in train_ds.index})
            num_domains = max(1, len(doms))
            print(f"[DOMAIN] train domains={doms} (used for domain heads / DBI monitor)")
        except Exception:
            num_domains = 1

    model = build_dual_model(
        args.num_classes,
        num_domains,
        model_size=args.model_size,
        dataset=args.dataset,
        input_len=input_len,
        sample_rate_hz=float(args.sample_rate_hz),
    ).to(device)
    print(f"[MODEL] DualCVSincNetDisentangle emb_dim={model.emb_dim} num_domains={num_domains}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_tx = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    ce_dom = nn.CrossEntropyLoss()

    best_val_tx = -1.0
    best_test_tx_at_best_val = float("nan")
    best_epoch = -1
    dbi_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        m_loss, m_cls, m_dom, m_adv, m_orth, m_cons, m_probe = [AverageMeter() for _ in range(7)]
        m_txacc, m_domacc, m_probeacc = AverageMeter(), NanMeter(), NanMeter()
        cons_cos_vals = []

        for batch in train_loader:
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d = extract_domain_from_extra(extra, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True)
                tx_logits = out["tx_logits"]
                dom_logits = out["dom_logits"]
                adv_dom_logits = out["adv_dom_logits"]
                probe_dom_logits = out["probe_dom_logits"]
                z_id = out["z_id"]
                z_dom = out["z_dom"]

                loss_cls = ce_tx(tx_logits.float(), y)
                if d is not None and num_domains > 1:
                    loss_dom = ce_dom(dom_logits.float(), d)
                    loss_adv = ce_dom(adv_dom_logits.float(), d)
                    loss_probe = ce_dom(probe_dom_logits.float(), d)
                    loss_cons, cons_cos = same_tx_cross_domain_consistency(z_id, y, d)
                    if not math.isnan(cons_cos):
                        cons_cos_vals.append(cons_cos)
                    m_domacc.update(accuracy_from_logits(dom_logits, d))
                    m_probeacc.update(accuracy_from_logits(probe_dom_logits, d))
                else:
                    loss_dom = z_id.new_tensor(0.0)
                    loss_adv = z_id.new_tensor(0.0)
                    loss_probe = z_id.new_tensor(0.0)
                    loss_cons = z_id.new_tensor(0.0)
                loss_orth = covariance_orth_loss(z_id, z_dom)
                loss = (
                    loss_cls
                    + float(args.lambda_dom) * loss_dom
                    + float(args.lambda_adv) * loss_adv
                    + float(args.lambda_orth) * loss_orth
                    + float(args.lambda_cons) * loss_cons
                    + float(args.lambda_probe) * loss_probe
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            m_loss.update(loss.item(), x.size(0))
            m_cls.update(loss_cls.item(), x.size(0))
            m_dom.update(loss_dom.item(), x.size(0))
            m_adv.update(loss_adv.item(), x.size(0))
            m_orth.update(loss_orth.item(), x.size(0))
            m_cons.update(loss_cons.item(), x.size(0))
            m_probe.update(loss_probe.item(), x.size(0))
            m_txacc.update(accuracy_from_logits(tx_logits, y), x.size(0))

        scheduler.step()

        train_dbi = normalized_accuracy(m_probeacc.avg if m_probeacc.count else 0.0, num_domains) - normalized_accuracy(m_txacc.avg, args.num_classes)
        dbi_hist.append(train_dbi)
        bias_flag = should_warn_domain_bias(dbi_hist, patience=3, threshold=0.05)
        cons_cos_epoch = float(np.mean(cons_cos_vals)) if len(cons_cos_vals) > 0 else float("nan")

        val_stats = evaluate(model, val_loader, device, num_domains_train=num_domains, max_batches=int(args.eval_max_batches))
        test_stats = evaluate(model, test_loader, device, num_domains_train=num_domains, max_batches=int(args.eval_max_batches))

        is_best = val_stats["tx_acc"] > best_val_tx
        if is_best:
            best_val_tx = val_stats["tx_acc"]
            best_test_tx_at_best_val = test_stats["tx_acc"]
            best_epoch = epoch
            save_checkpoint(
                args.best_save_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                args=args,
                split_info=split_info,
                stats={
                    "train_tx_acc": m_txacc.avg,
                    "val_tx_acc": val_stats["tx_acc"],
                    "val_dom_acc": val_stats["dom_acc"],
                    "val_probe_dom_acc": val_stats["probe_dom_acc"],
                    "test_tx_acc": test_stats["tx_acc"],
                    "test_dom_acc": test_stats["dom_acc"],
                    "test_probe_dom_acc": test_stats["probe_dom_acc"],
                    "best_epoch": epoch,
                },
            )

        save_checkpoint(
            args.latest_save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
            split_info=split_info,
            stats={
                "train_tx_acc": m_txacc.avg,
                "val_tx_acc": val_stats["tx_acc"],
                "val_dom_acc": val_stats["dom_acc"],
                "val_probe_dom_acc": val_stats["probe_dom_acc"],
                "test_tx_acc": test_stats["tx_acc"],
                "test_dom_acc": test_stats["dom_acc"],
                "test_probe_dom_acc": test_stats["probe_dom_acc"],
                "best_val_tx_acc_so_far": best_val_tx,
                "best_epoch_so_far": best_epoch,
            },
        )

        msg = (
            f"[E{epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e} "
            f"loss={m_loss.avg:.4f} cls={m_cls.avg:.4f} dom={m_dom.avg:.4f} adv={m_adv.avg:.4f} "
            f"orth={m_orth.avg:.4f} cons={m_cons.avg:.4f} probe={m_probe.avg:.4f} | "
            f"train_tx={m_txacc.avg:.2f}% train_dom={m_domacc.avg:.2f}% train_probe={m_probeacc.avg:.2f}% "
            f"cons_cos={cons_cos_epoch:.4f} DBI={train_dbi:.4f} bias_flag={int(bias_flag)} | "
            f"val_tx={val_stats['tx_acc']:.2f}% val_dom={val_stats['dom_acc']:.2f}% val_probe={val_stats['probe_dom_acc']:.2f}% | "
            f"test_tx={test_stats['tx_acc']:.2f}% test_dom={test_stats['dom_acc']:.2f}% test_probe={test_stats['probe_dom_acc']:.2f}% | "
            f"best_val={best_val_tx:.2f}%@E{best_epoch:03d} best_test_at_best_val={best_test_tx_at_best_val:.2f}% | "
            f"time={time.time() - t0:.1f}s"
        )
        print(msg, flush=True)
        print(
            f"[CKPT] latest -> {args.latest_save_path} | best -> {args.best_save_path} {'(updated)' if is_best else ''}",
            flush=True,
        )
        if bias_flag:
            print(f"[BIAS-WARN] Epoch {epoch}: DBI连续升高且超过阈值，模型可能正在由学指纹转向学域偏置。", flush=True)

    print(f"Training finished. best_val_tx_acc={best_val_tx:.2f}% at epoch {best_epoch}")
    print(f"At best-val epoch, test_tx_acc={best_test_tx_at_best_val:.2f}%")
    if split_info is not None:
        print(f"Final split info: {split_info}")

    # 训练结束后，再用 best checkpoint 做一次最终 test 汇报（不影响训练过程）
    try:
        ckpt = torch.load(args.best_save_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        final_test = evaluate(model, test_loader, device, num_domains_train=num_domains, max_batches=0)
        print(
            f"[FINAL-BEST] test_tx={final_test['tx_acc']:.2f}% test_dom={final_test['dom_acc']:.2f}% "
            f"test_probe={final_test['probe_dom_acc']:.2f}%",
            flush=True,
        )
    except Exception as e:
        print(f"[WARN] final best-checkpoint test failed: {e}", flush=True)


if __name__ == "__main__":
    main()
