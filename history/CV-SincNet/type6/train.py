# train.py
import os
import math
import time
import copy
import inspect
import argparse
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 1337):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits: [B, C]
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item() * 100.0


def safe_to_float(x: torch.Tensor) -> torch.Tensor:
    # avoid fp16 overflow inside CE / log-softmax
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x




def _call_model(model, x, return_aux: bool = False):
    # Try to call model with a return_aux keyword; if the model does not support it, fall back.
    try:
        return model(x, return_aux=return_aux)
    except TypeError:
        return model(x)


def _unpack_model_output(out):
    # Unpack common model output conventions.
    # Returns: logits, dac_pred, feat_cls, feat_con, cos_logits
    logits = dac_pred = feat_cls = feat_con = cos_logits = None

    if isinstance(out, dict):
        for k in ("logits", "pred", "y_pred", "out"):
            if k in out:
                logits = out[k]
                break
        dac_pred = out.get("dac_pred", out.get("dac", None))
        feat_cls = out.get("feat_cls", out.get("feat", None))
        feat_con = out.get("feat_con", out.get("proj", None))
        cos_logits = out.get("cos_logits", out.get("cos", None))

    elif isinstance(out, (tuple, list)):
        if len(out) >= 1:
            logits = out[0]
        if len(out) >= 2:
            dac_pred = out[1]
        if len(out) >= 3:
            feat_cls = out[2]
        if len(out) >= 4:
            feat_con = out[3]
        # Some models may return (logits, feat) only.
        if feat_con is None and feat_cls is None and len(out) == 2:
            feat_con = out[1]

    else:
        # torch.Tensor logits only
        try:
            import torch
            if torch.is_tensor(out):
                logits = out
            else:
                raise TypeError
        except Exception as e:
            raise RuntimeError(f"Unsupported model output type: {type(out)}") from e

    return logits, dac_pred, feat_cls, feat_con, cos_logits
def compute_ramp(epoch_idx: int, warmup_epochs: int, ramp_epochs: int) -> float:
    # epoch_idx: 1-based
    if epoch_idx <= warmup_epochs:
        return 0.0
    t = epoch_idx - warmup_epochs
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, t / ramp_epochs)))


def compute_lr(epoch_idx: int, total_epochs: int, lr_base: float, warmup_epochs: int) -> float:
    # epoch_idx: 1-based
    if epoch_idx <= warmup_epochs:
        return lr_base * (epoch_idx / max(1, warmup_epochs))
    # cosine decay after warmup
    t = (epoch_idx - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return lr_base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


# -----------------------
# Prototype bank (no dependency on contrastive_loss.py)
# -----------------------
class PrototypeBank:
    """
    momentum prototype bank: prototypes[c] = m*proto + (1-m)*mean(feats_of_class_c)
    also maintains counts to mark valid classes
    """
    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9, device="cuda"):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.device = device

        self.prototypes = torch.zeros(num_classes, feat_dim, device=device, dtype=torch.float32)
        self.counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: [B, D] (already normalized or not)
        labels: [B]
        """
        feats = feats.detach().float()
        labels = labels.detach()
        for c in labels.unique():
            c = int(c.item())
            idx = (labels == c)
            if idx.sum() == 0:
                continue
            mean_feat = feats[idx].mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=0)
            if self.counts[c] == 0:
                self.prototypes[c] = mean_feat
            else:
                self.prototypes[c] = F.normalize(
                    self.momentum * self.prototypes[c] + (1.0 - self.momentum) * mean_feat,
                    dim=0
                )
            self.counts[c] += idx.sum()

    def valid_mask(self) -> torch.Tensor:
        return self.counts > 0


def proto_ce_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    bank: PrototypeBank,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    feats: [B, D] normalized
    prototypes: [C, D] normalized
    use CE over prototypes (safe fp16: compute in fp32, mask invalid with -1e4)
    """
    feats = F.normalize(feats.float(), dim=1)
    prot = F.normalize(bank.prototypes.float(), dim=1)

    logits = (feats @ prot.t()) / max(1e-6, tau)  # [B, C] float32
    valid = bank.valid_mask()  # [C]
    if valid.sum().item() == 0:
        return logits.new_tensor(0.0)

    # mask invalid prototypes with fp16-safe negative
    neg = -1e4
    logits[:, ~valid] = neg

    # if a sample's class prototype invalid, ignore it
    tgt = labels.clone()
    invalid_tgt = ~valid[tgt]
    if invalid_tgt.any():
        tgt[invalid_tgt] = -100  # ignore_index

    loss = F.cross_entropy(logits, tgt, ignore_index=-100)
    return loss


# -----------------------
# SupCon (safe fp16)
# -----------------------
def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    feats: [B, V, D]  (float/half ok)
    labels: [B]
    """
    B, V, D = feats.shape
    x = F.normalize(feats.float(), dim=2)  # [B,V,D] float32

    # flatten views as anchors
    x = x.reshape(B * V, D)  # [BV, D]
    y = labels.repeat_interleave(V)  # [BV]

    # similarity
    logits = (x @ x.t()) / max(1e-6, tau)  # [BV, BV]
    # mask self
    diag = torch.eye(B * V, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag, -1e4)

    # positives mask: same class, not self
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~diag)  # [BV, BV]

    # log-softmax
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # [BV,BV]

    # mean over positives
    pos_cnt = pos.sum(dim=1)  # [BV]
    # avoid div0
    loss = -(log_prob * pos.float()).sum(dim=1) / pos_cnt.clamp_min(1.0)
    # only anchors with >=1 positive contribute
    loss = loss[pos_cnt > 0].mean() if (pos_cnt > 0).any() else logits.new_tensor(0.0)
    return loss


# -----------------------
# Augmentor loader (auto-detect)
# -----------------------
def build_augmentor_safely(device: str):
    try:
        import importlib
        mod = importlib.import_module("DataAugmentation_v2")

        # 1) build_augmentor() function
        if hasattr(mod, "build_augmentor") and callable(mod.build_augmentor):
            aug = mod.build_augmentor()
            return aug, "[OK] build_augmentor()"

        # 2) common class names
        preferred = ["DataAugmentationV2", "DataAugmentation", "Augmentor", "RFFIAugmentor"]
        for name in preferred:
            if hasattr(mod, name) and isinstance(getattr(mod, name), type):
                cls = getattr(mod, name)
                try:
                    aug = cls()
                except TypeError:
                    # try passing device if accepted
                    sig = inspect.signature(cls.__init__)
                    kwargs = {}
                    if "device" in sig.parameters:
                        kwargs["device"] = device
                    aug = cls(**kwargs)
                return aug, f"[OK] class {name}"

        # 3) fallback: find any class with __call__
        candidates = []
        for k, v in vars(mod).items():
            if isinstance(v, type) and hasattr(v, "__call__"):
                candidates.append(k)
        if len(candidates) == 0:
            raise RuntimeError("No augmentor class found in DataAugmentation_v2.py")

        # pick the most likely one (name contains 'Aug' first)
        candidates.sort(key=lambda s: (("aug" not in s.lower()), len(s)))
        cls = getattr(mod, candidates[0])
        try:
            aug = cls()
        except TypeError:
            sig = inspect.signature(cls.__init__)
            kwargs = {}
            if "device" in sig.parameters:
                kwargs["device"] = device
            aug = cls(**kwargs)
        return aug, f"[OK] fallback class {candidates[0]} (candidates={candidates})"

    except Exception as e:
        return None, f"[WARN] Failed to import/construct DataAugmentation_v2 augmentor. Aug disabled. Reason: {e}"


def call_augmentor(aug, x: torch.Tensor, return_dac_strength=False, dac_only=False):
    """
    Try calling aug with multiple possible signatures.
    Must return:
      - x_aug: Tensor like x
      - (optional) dac_strength: Tensor [B] or [B,1]
    """
    if aug is None:
        if return_dac_strength:
            return x, torch.zeros(x.size(0), device=x.device)
        return x

    # Try direct call with kwargs
    try:
        out = aug(x, return_dac_strength=return_dac_strength, dac_only=dac_only)
        if return_dac_strength:
            return out[0], out[1]
        return out
    except TypeError:
        pass

    # Try method augment()
    if hasattr(aug, "augment") and callable(aug.augment):
        try:
            out = aug.augment(x, return_dac_strength=return_dac_strength, dac_only=dac_only)
            if return_dac_strength:
                return out[0], out[1]
            return out
        except TypeError:
            pass

    # Try DAC simulate separately
    if dac_only and hasattr(aug, "simulate_dac") and callable(aug.simulate_dac):
        # if aug provides internal sampling, just call simulate_dac
        y = aug.simulate_dac(x, dac_strength=None)
        if return_dac_strength:
            return y, torch.ones(x.size(0), device=x.device)
        return y

    # last resort: identity
    if return_dac_strength:
        return x, torch.zeros(x.size(0), device=x.device)
    return x


# -----------------------
# Model loader (auto-pick)
# -----------------------
def build_model_safely(num_classes: int, device: str, model_size: str = "M"):
    import importlib
    mod = importlib.import_module("model")

    # common factory
    if hasattr(mod, "build_model") and callable(mod.build_model):
        m = mod.build_model(num_classes=num_classes, model_size=model_size)
        return m, "[OK] model.build_model"

    # preferred class names
    preferred = [
        "DACSpecializedCVSincNet",
        "DAC_CV_SincNet",
        "CVSincNet",
        "SincNet",
        "Model",
        "Net",
    ]
    for name in preferred:
        if hasattr(mod, name) and isinstance(getattr(mod, name), type):
            cls = getattr(mod, name)
            if not issubclass(cls, nn.Module):
                continue
            sig = inspect.signature(cls.__init__)
            kwargs = {}
            if "num_classes" in sig.parameters:
                kwargs["num_classes"] = num_classes
            elif "n_classes" in sig.parameters:
                kwargs["n_classes"] = num_classes
            if "model_size" in sig.parameters:
                kwargs["model_size"] = model_size
            m = cls(**kwargs)
            return m, f"[OK] class {name}"

    # fallback: pick first nn.Module subclass with 'Net' or 'Model' in name
    cands = []
    for k, v in vars(mod).items():
        if isinstance(v, type) and issubclass(v, nn.Module):
            cands.append(k)
    if len(cands) == 0:
        raise RuntimeError("No nn.Module model class found in model.py")

    def score(n):
        n2 = n.lower()
        s = 0
        if "net" in n2: s -= 5
        if "model" in n2: s -= 4
        if "sinc" in n2: s -= 3
        if "dac" in n2: s -= 2
        return (s, len(n))

    cands.sort(key=score)
    cls = getattr(mod, cands[0])
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes
    elif "n_classes" in sig.parameters:
        kwargs["n_classes"] = num_classes
    if "model_size" in sig.parameters:
        kwargs["model_size"] = model_size
    m = cls(**kwargs)
    return m, f"[OK] fallback class {cands[0]} (candidates={cands})"


# -----------------------
# Checkpoint
# -----------------------
@dataclass
class BestState:
    best_acc: float = -1.0
    best_epoch: int = -1
    best_blob: dict = None  # model/optim/scaler etc


def pack_state(model, optimizer, scaler, epoch, lr_base, best_acc, best_epoch, global_step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "lr_base": lr_base,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "global_step": global_step,
    }


def unpack_state(blob, model, optimizer, scaler):
    model.load_state_dict(blob["model"], strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    if scaler is not None and blob.get("scaler", None) is not None:
        scaler.load_state_dict(blob["scaler"])
    return blob.get("epoch", 0), blob.get("lr_base", None), blob.get("global_step", 0)


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr_base_init", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--ramp_epochs", type=int, default=80)

    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--tau_proto", type=float, default=0.07)
    parser.add_argument("--lambda_con", type=float, default=0.05)
    parser.add_argument("--lambda_proto", type=float, default=0.2)

    parser.add_argument("--dac_lambda", type=float, default=5.0)
    parser.add_argument("--dac_zero_weight", type=float, default=0.2)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Cosine-margin branch metrics (optional). Default ON for backward-compat with older logs.
    mg = parser.add_mutually_exclusive_group()
    mg.add_argument("--use_cosine_margin", dest="use_cosine_margin", action="store_true",
                    help="Enable cosine-margin branch metrics (default: on).")
    mg.add_argument("--no_cosine_margin", dest="use_cosine_margin", action="store_false",
                    help="Disable cosine-margin branch metrics.")
    parser.set_defaults(use_cosine_margin=True)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    # rollback
    parser.add_argument("--rollback_enable", action="store_true", default=True)
    parser.add_argument("--rollback_drop_abs", type=float, default=12.0)
    parser.add_argument("--rollback_shrink", type=float, default=0.5)

    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--model_size", type=str, default="M")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Starting Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} | AMP: {args.amp}")
    print(f"DATASET_DIR={args.dataset_dir}")
    print(f"NUM_CLASSES={args.num_classes}, BATCH_SIZE={args.batch_size}, EPOCHS={args.epochs}")
    print(f"LR_BASE_INIT={args.lr_base_init}, WD={args.wd}")
    print(f"N_VIEWS={args.n_views}, TAU={args.tau}, TAU_PROTO={args.tau_proto}")
    print(f"LAMBDA_CON={args.lambda_con}, LAMBDA_PROTO={args.lambda_proto}")
    print(f"DAC_LAMBDA={args.dac_lambda}, DAC_ZERO_WEIGHT={args.dac_zero_weight}")
    print(f"WARMUP_EPOCHS={args.warmup_epochs}, RAMP_EPOCHS={args.ramp_epochs}")
    print(f"GRAD_CLIP={args.grad_clip}")
    print(f"ROLLBACK: enable={args.rollback_enable}, drop_abs={args.rollback_drop_abs}, shrink={args.rollback_shrink}")

    # dataset
    train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train")
    test_ds = WiFiRFFIDataset(args.dataset_dir, mode="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # model
    model, model_msg = build_model_safely(args.num_classes, str(device), model_size=args.model_size)
    model.to(device)
    print(model_msg)

    # infer feat_dim for prototype bank: do a tiny forward
    feat_dim = None
    model.eval()
    # Infer feat_dim for prototype bank (prefer projection head feat_con)
    feat_dim = None
    with torch.no_grad():
        x0 = torch.zeros(2, 2, 1024, device=device)
        try:
            out0 = _call_model(model, x0, return_aux=True)
            logits0, dac0, feat_cls0, feat_con0, _ = _unpack_model_output(out0)
            feat0 = feat_con0 if feat_con0 is not None else feat_cls0
            if feat0 is not None and torch.is_tensor(feat0):
                feat_dim = int(feat0.shape[-1])
        except Exception as e:
            feat_dim = None
    if feat_dim is None:
        feat_dim = 256
        print(f"[WARN] Cannot infer feat_dim from model output. Fallback feat_dim={feat_dim}")

    proto_bank = PrototypeBank(args.num_classes, feat_dim, momentum=0.9, device=device)

    # optimizer
    lr_base = args.lr_base_init
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=args.wd)

    # AMP scaler
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=2.0**8)

    # augmentor
    augmentor, aug_msg = build_augmentor_safely(str(device))
    print(aug_msg)

    best = BestState(best_acc=-1.0, best_epoch=-1, best_blob=None)

    # -----------------------
    # train loop
    # -----------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ramp = compute_ramp(epoch, args.warmup_epochs, args.ramp_epochs)
        lr_now = compute_lr(epoch, args.epochs, lr_base, args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()

        loss_sum = 0.0
        ce_sum = 0.0
        con_sum = 0.0
        proto_sum = 0.0
        dac_sum = 0.0

        correct = 0
        total = 0

        # optional margin stats if model returns cos logits
        correct_margin = 0
        margin_viol = 0
        margin_total = 0

        skipped_steps = 0
        gnorm_avg = 0.0
        gnorm_max = 0.0
        gnorm_cnt = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)  # [B,2,1024]
            y = y.to(device, non_blocking=True)

            B = x.size(0)

            # ---- build views ----
            views = []
            dac_strength = torch.zeros(B, device=device)

            if augmentor is not None:
                # view0: weak/normal
                v0 = call_augmentor(augmentor, x, return_dac_strength=False, dac_only=False)
                views.append(v0)

                # mid views: normal (could be stronger if your augmentor supports)
                for _ in range(args.n_views - 2):
                    vv = call_augmentor(augmentor, x, return_dac_strength=False, dac_only=False)
                    views.append(vv)

                # last view: dac-only + return strength
                vd, ds = call_augmentor(augmentor, x, return_dac_strength=True, dac_only=True)
                views.append(vd)
                dac_strength = ds
            else:
                for _ in range(args.n_views):
                    views.append(x)

            # concat forward once (cheaper & consistent)
            x_cat = torch.cat(views, dim=0)  # [B*V,2,1024]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = _call_model(model, x_cat, return_aux=True)

            logits_all, dac_pred_all, feat_cls_all, feat_con_all, cos_logits_all = _unpack_model_output(out)
            if logits_all is None:
                raise RuntimeError("Model output has no logits. Please ensure model returns logits or dict['logits'].")

            # Use cosine logits if the model provides it; otherwise fall back to logits
            cos_all = None
            if args.use_cosine_margin:
                cos_src = cos_logits_all if cos_logits_all is not None else logits_all
                cos_all = torch.nan_to_num(cos_src.float(), nan=0.0, posinf=0.0, neginf=0.0)

            # ---- losses in float32 to avoid fp16 overflow (especially for exp/logsumexp) ----
            logits_all_f = torch.nan_to_num(logits_all.float(), nan=0.0, posinf=0.0, neginf=0.0)
            logits0_f = logits_all_f[:B]
            ce_loss = F.cross_entropy(logits0_f, y)

            # acc
            with torch.no_grad():
                correct += (logits0_f.argmax(dim=1) == y).sum().item()
                total += B

            # ---- contrastive / proto / dac losses (after warmup via ramp) ----
            con_loss = logits0_f.new_tensor(0.0)
            pr_loss = logits0_f.new_tensor(0.0)
            d_loss = logits0_f.new_tensor(0.0)

            if ramp > 0.0 and feat_con_all is not None:
                feat_con_all_f = torch.nan_to_num(feat_con_all.float(), nan=0.0, posinf=0.0, neginf=0.0)
                # split feat_con into [B,V,D]
                feat_con = feat_con_all_f.view(args.n_views, B, -1).permute(1, 0, 2).contiguous()

                # SupCon over feat_con
                con_loss = supcon_loss(feat_con, y, tau=args.tau)

                # Prototype loss: use view0 features
                feat0 = F.normalize(feat_con[:, 0, :], dim=1)
                proto_bank.update(feat0, y)
                pr_loss = proto_ce_loss(feat0, y, proto_bank, tau=args.tau_proto)

                # DAC loss: use last view dac_pred if available
                if dac_pred_all is not None:
                    dac_pred = dac_pred_all[B * (args.n_views - 1): B * args.n_views]
                    dac_pred = torch.nan_to_num(dac_pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    dac_pred = dac_pred.view(B, -1).squeeze(-1)
                    target = dac_strength.view(B, -1).squeeze(-1).float()

                    w = torch.ones_like(target)
                    w = torch.where(target.abs() < 1e-12, w * args.dac_zero_weight, w)
                    # huber is more stable than mse under strong DAC
                    d_loss = (F.smooth_l1_loss(dac_pred, target, reduction="none") * w).mean()

            loss = ce_loss + ramp * (args.lambda_con * con_loss + args.lambda_proto * pr_loss + args.dac_lambda * d_loss)

            # ---- backward ----
            if not torch.isfinite(loss.detach()):
                skipped_steps += 1
                continue

            if use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()

                # grad clip (must unscale first)
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                else:
                    scaler.unscale_(optimizer)
                    gn = torch.norm(
                        torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]),
                        p=2
                    ) if any(p.grad is not None for p in model.parameters()) else torch.tensor(0.0, device=device)

                # step
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                # if scale dropped, step was effectively skipped due to inf/nan
                if scale_after < scale_before:
                    skipped_steps += 1

            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            # grad norm stats
            if torch.is_tensor(gn):
                g = float(gn.detach().item())
                gnorm_avg += g
                gnorm_max = max(gnorm_max, g)
                gnorm_cnt += 1

            # meters
            loss_sum += float(loss.detach().item())
            ce_sum += float(ce_loss.detach().item())
            con_sum += float(con_loss.detach().item())
            proto_sum += float(pr_loss.detach().item())
            dac_sum += float(d_loss.detach().item())

            # optional margin stats if cos logits exist
            if cos_all is not None:
                cos0 = cos_all[:B]
                with torch.no_grad():
                    correct_margin += (cos0.argmax(dim=1) == y).sum().item()
                    margin_total += B
                    # rough "violation": true cos <= max other cos  (you can refine using your margin m)
                    tmp = cos0.clone()
                    tmp[torch.arange(B, device=device), y] = -1e4
                    other_max = tmp.max(dim=1).values
                    true_cos = cos0[torch.arange(B, device=device), y]
                    margin_viol += (true_cos <= other_max).sum().item()

        # epoch stats
        train_acc = (correct / max(1, total)) * 100.0
        train_acc_margin = (correct_margin / max(1, margin_total)) * 100.0 if margin_total > 0 else 0.0
        viol_pct = (margin_viol / max(1, margin_total)) * 100.0 if margin_total > 0 else 0.0

        denom = max(1, len(train_loader))
        loss_avg = loss_sum / denom
        ce_avg = ce_sum / denom
        con_avg = con_sum / denom
        proto_avg = proto_sum / denom
        dac_avg = dac_sum / denom

        gavg = (gnorm_avg / max(1, gnorm_cnt)) if gnorm_cnt > 0 else 0.0

        # ---- eval ----
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = _call_model(model, x, return_aux=False)
                logits, _, _, _, _ = _unpack_model_output(out)
                if logits is None:
                    raise RuntimeError("Eval: model did not return logits")
                test_correct += (logits.argmax(dim=1) == y).sum().item()
                test_total += y.numel()
        test_acc = (test_correct / max(1, test_total)) * 100.0

        epoch_time = time.time() - t0

        # ---- auto AMP fail-safe: if almost all steps skipped, disable AMP next epoch ----
        if use_amp and skipped_steps >= int(0.9 * len(train_loader)):
            print(f"  !!! AMP unstable: skipped_steps={skipped_steps}/{len(train_loader)}. Disabling AMP & resetting scaler.")
            use_amp = False
            scaler = None  # disable

        # print
        msg = (
            f"Epoch [{epoch}/{args.epochs}] Time: {epoch_time:.1f}s | "
            f"Loss: {loss_avg:.4f} (CE: {ce_avg:.4f}, Con: {con_avg:.4f}, Proto: {proto_avg:.4f}, DAC: {dac_avg:.4f}) | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"ramp={ramp:.2f} | lr={lr_now:.6g} | "
            f"Train Acc (margin): {train_acc_margin:.2f}% | Margin Viol: {viol_pct:.2f}% | "
            f"gnorm(avg/max): {gavg:.2f}/{gnorm_max:.2f}"
        )
        if skipped_steps > 0:
            msg += f" | skipped_steps: {skipped_steps}"
        print(msg)

        # ---- save best ----
        if test_acc > best.best_acc:
            best.best_acc = test_acc
            best.best_epoch = epoch
            blob = pack_state(model, optimizer, scaler, epoch, lr_base, best.best_acc, best.best_epoch, global_step=0)
            best.best_blob = copy.deepcopy(blob)
            torch.save(blob, args.save_path)
            print(f"  >>> Best Model Saved! (Acc: {test_acc:.2f}%)")

        # ---- rollback on collapse ----
        if args.rollback_enable and best.best_acc >= 0 and (best.best_blob is not None):
            if test_acc < best.best_acc - args.rollback_drop_abs:
                print(
                    f"  !!! Collapse detected: test_acc={test_acc:.2f}% < best_acc={best.best_acc:.2f}% - {args.rollback_drop_abs}%. "
                    f"Rolling back to best and shrinking lr_base."
                )
                # restore best
                unpack_state(best.best_blob, model, optimizer, scaler if scaler is not None else optimizer)  # scaler ignored if None
                lr_base = lr_base * args.rollback_shrink

                # IMPORTANT: reset scaler to avoid “all steps skipped”
                if device.type == "cuda":
                    use_amp = bool(args.amp)
                    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=2.0**6)
                else:
                    use_amp = False
                    scaler = None

                # apply new lr
                lr_now = compute_lr(epoch, args.epochs, lr_base, args.warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                print(f"  >>> Rollback done. new lr_base={lr_base:.6g}, lr_now={lr_now:.6g}")

    print(f"Training done. Best Acc={best.best_acc:.2f}% at epoch {best.best_epoch}. Saved to {args.save_path}")


if __name__ == "__main__":
    main()
