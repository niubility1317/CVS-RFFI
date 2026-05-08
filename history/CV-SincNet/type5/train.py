# train.py
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
from model import CVSincNet
from DataAugmentation_v2 import RFFIAugmentor
from contrastive_loss import supcon_loss, PrototypeMemory, prototype_contrastive_loss


# ================= 配置区域 =================
DATASET_DIR = "./Dataset_ORALCE"
LOG_DIR = "./log"
WEIGHT_DIR = "./weight"

NUM_CLASSES = 16
BATCH_SIZE = 400
EPOCHS = 300
LR = 1e-3
WD = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 多视图设置
# view0: 原始 (no aug)
# view1: channel impairments (NO DAC)  -> encourage domain invariance
# view2: DAC-only (ONLY DAC)           -> isolate DAC for aux head + invariance on feat_id
N_VIEWS = 3
TAU = 0.10
TAU_PROTO = 0.07

# 损失权重（建议 warmup）
LAMBDA_CON = 0.05
LAMBDA_PROTO = 0.2

DAC_LAMBDA = 5.0
DAC_ZERO_WEIGHT = 0.2

WARMUP_EPOCHS = 5
RAMP_EPOCHS = 30

GRAD_CLIP = 5.0
# ===========================================


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")

    def log_func(message):
        print(message, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    return log_func


def weighted_dac_loss(pred: torch.Tensor, target: torch.Tensor, zero_weight: float = 0.2) -> torch.Tensor:
    """
    pred/target: (N,) in [0,1]
    target==0 means "no DAC applied", give it smaller weight.
    """
    pred = pred.float()
    target = target.float()
    per = F.smooth_l1_loss(pred, target, reduction="none")
    w = torch.where(target <= 0.0, torch.full_like(target, float(zero_weight)), torch.ones_like(target))
    return (per * w).mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / max(1, total)


def main():
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    logger = get_logger(LOG_DIR)

    logger(f"Starting Training at {datetime.datetime.now()}")
    logger(f"Device: {DEVICE}")
    logger(f"N_VIEWS={N_VIEWS}, TAU={TAU}, TAU_PROTO={TAU_PROTO}")
    logger(f"LAMBDA_CON={LAMBDA_CON}, LAMBDA_PROTO={LAMBDA_PROTO}")
    logger(f"DAC_LAMBDA={DAC_LAMBDA}, DAC_ZERO_WEIGHT={DAC_ZERO_WEIGHT}")
    logger(f"WARMUP_EPOCHS={WARMUP_EPOCHS}, RAMP_EPOCHS={RAMP_EPOCHS}")

    # 1) Data
    train_ds = WiFiRFFIDataset(DATASET_DIR, mode="train")
    test_ds = WiFiRFFIDataset(DATASET_DIR, mode="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 2) Augmentors (split DAC vs channel)
    # Channel-only augmentor: disable DAC
    augmentor_channel = RFFIAugmentor(
        sampling_rate=5e6,
        p_phase_rotate=0.5,
        p_amp_scale=0.4,
        p_noise=0.5,
        p_cfo=0.25,
        p_iq_imbalance=0.2,
        p_dac=0.0,                # important: no DAC here
        snr_db_range=(18.0, 35.0),
        cfo_max_hz=800.0,
        dac_strength_range=(0.1, 1.0),  # unused
    )

    # DAC-only augmentor: only DAC, no other impairments (probabilities set to 0)
    augmentor_dac = RFFIAugmentor(
        sampling_rate=5e6,
        p_phase_rotate=0.0,
        p_amp_scale=0.0,
        p_noise=0.0,
        p_cfo=0.0,
        p_iq_imbalance=0.0,
        p_dac=1.0,                # always apply DAC (strength label nonzero)
        snr_db_range=(18.0, 35.0),  # unused
        cfo_max_hz=800.0,           # unused
        dac_strength_range=(0.1, 1.0),
    )

    # 3) Model
    model = CVSincNet(num_classes=NUM_CLASSES).to(DEVICE)

    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)

    # Prototype memory（类中心）
    with torch.no_grad():
        dummy = torch.zeros(2, 2, 1024, device=DEVICE)
        _, _, feat_dummy = model(dummy, return_aux=True)
        feat_dim = int(feat_dummy.size(-1))

    proto_mem = PrototypeMemory(NUM_CLASSES, feat_dim, momentum=0.9).to(DEVICE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        # ramp：warmup 后逐步把 con/proto/dac 加起来
        if epoch < WARMUP_EPOCHS:
            ramp = 0.0
        else:
            ramp = min(1.0, (epoch - WARMUP_EPOCHS + 1) / max(1, RAMP_EPOCHS))

        lam_con = LAMBDA_CON * ramp
        lam_proto = LAMBDA_PROTO * ramp
        lam_dac = DAC_LAMBDA * ramp

        train_correct, train_total = 0, 0
        total_loss = 0.0
        total_ce = 0.0
        total_con = 0.0
        total_proto = 0.0
        total_dac = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            B = x.size(0)

            # --------- Multi-view ----------
            # view0: 原始
            views = [x]
            dac_strengths = [torch.zeros(B, device=DEVICE)]

            with torch.no_grad():
                if N_VIEWS >= 2:
                    x1 = augmentor_channel(x, return_dac_strength=False, no_dac=True)  # channel impairments, no DAC
                    views.append(x1)
                    dac_strengths.append(torch.zeros(B, device=DEVICE))

                if N_VIEWS >= 3:
                    x2, s2 = augmentor_dac(x, return_dac_strength=True, dac_only=True)  # DAC only
                    views.append(x2)
                    dac_strengths.append(s2.to(DEVICE))

                # if N_VIEWS > 3: fill remaining with channel-only
                while len(views) < N_VIEWS:
                    xa = augmentor_channel(x, return_dac_strength=False, no_dac=True)
                    views.append(xa)
                    dac_strengths.append(torch.zeros(B, device=DEVICE))

            V = len(views)
            x_all = torch.cat(views, dim=0)              # (B*V,2,1024)
            y_all = y.repeat(V)                          # (B*V,)
            dac_t = torch.cat(dac_strengths, dim=0)      # (B*V,)

            # --------- Forward ----------
            logits_all, dac_pred_all, feat_all = model(x_all, return_aux=True)  # feat_all: (B*V,D) (feat_id)

            # --------- CE on view0 ----------
            logits_v0 = logits_all[:B]
            ce = ce_loss_fn(logits_v0, y)

            # --------- SupCon (all views) ----------
            if lam_con > 0:
                D = feat_all.size(-1)
                feat_views = feat_all.view(V, B, D).permute(1, 0, 2).contiguous()  # (B,V,D)
                con = supcon_loss(feat_views, y, temperature=TAU)
            else:
                con = torch.zeros([], device=DEVICE)

            # --------- Prototype contrast (all views) ----------
            protos, counts = proto_mem.get()
            if lam_proto > 0:
                proto = prototype_contrastive_loss(feat_all, y_all, protos, counts, temperature=TAU_PROTO)
            else:
                proto = torch.zeros([], device=DEVICE)

            # --------- DAC loss (all views) ----------
            if lam_dac > 0:
                dac = weighted_dac_loss(dac_pred_all, dac_t, zero_weight=DAC_ZERO_WEIGHT)
            else:
                dac = torch.zeros([], device=DEVICE)

            loss = ce + lam_con * con + lam_proto * proto + lam_dac * dac

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP is not None and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # --------- update prototypes using view0 (detached) ----------
            with torch.no_grad():
                feat_v0 = feat_all[:B].detach()
                proto_mem.update(feat_v0, y)

            # --------- stats ----------
            total_loss += loss.item()
            total_ce += ce.item()
            total_con += float(con.item()) if con.numel() == 1 else 0.0
            total_proto += float(proto.item()) if proto.numel() == 1 else 0.0
            total_dac += float(dac.item()) if dac.numel() == 1 else 0.0

            pred = logits_v0.argmax(dim=1)
            train_total += B
            train_correct += (pred == y).sum().item()

        scheduler.step()

        train_acc = 100.0 * train_correct / max(1, train_total)
        test_acc = evaluate(model, test_loader)

        ep_time = time.time() - start_time
        n_steps = max(1, len(train_loader))

        logger(
            f"Epoch [{epoch+1}/{EPOCHS}] Time: {ep_time:.1f}s | "
            f"Loss: {total_loss/n_steps:.4f} (CE: {total_ce/n_steps:.4f}, "
            f"Con: {total_con/n_steps:.4f}, Proto: {total_proto/n_steps:.4f}, DAC: {total_dac/n_steps:.4f}) | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | ramp={ramp:.2f}"
        )

        # save
        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_model_run1_Aug_CL3_silm.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "best_model_run1_Aug_CL3_silm.pth"))
            logger(f"  >>> Best Model Saved! (Acc: {best_acc:.2f}%)")

    logger(f"Training done. Best Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
