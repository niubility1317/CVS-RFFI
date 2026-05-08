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


# ================== Config ==================
DATASET_DIR = "./Dataset_ORALCE"
LOG_DIR = "./log"
WEIGHT_DIR = "./weight"

BATCH_SIZE = 128
EPOCHS = 200
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AUGMENTATION = True

# ===== DAC auxiliary loss =====
DAC_LAMBDA = 0.5          # 建议 0.2~1.0 之间网格试
DAC_ZERO_WEIGHT = 0.2     # 未施加DAC样本的权重（小一些防止全预测0）
# ==============================


def get_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")

    def log_func(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log_func


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # return_aux=False
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def weighted_dac_loss(dac_pred: torch.Tensor, dac_strength: torch.Tensor, zero_weight: float = 0.2):
    """
    dac_pred: (B,) in [0,1]
    dac_strength: (B,) in [0,1]
    加权 SmoothL1：
      - dac_strength>0 的样本权重大
      - dac_strength==0 的样本权重小（防止全预测0）
    """
    # mask: 施加DAC的样本（注意：我在增强里让 smin>=0.1，所以这里 mask 可用）
    mask = (dac_strength > 0).float()
    w = mask + (1.0 - mask) * float(zero_weight)  # applied=1, non-applied=zero_weight

    per = F.smooth_l1_loss(dac_pred, dac_strength, reduction="none")  # (B,)
    loss = (per * w).sum() / (w.sum() + 1e-8)
    return loss


def main():
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    logger = get_logger(LOG_DIR)
    logger(f"Starting Training at {datetime.datetime.now()}")
    logger(f"Device: {DEVICE}")
    logger(f"Augmentation: {USE_AUGMENTATION}")
    logger(f"DAC_LAMBDA={DAC_LAMBDA}, DAC_ZERO_WEIGHT={DAC_ZERO_WEIGHT}")

    # dataset
    train_ds = WiFiRFFIDataset(DATASET_DIR, mode="train")
    test_ds = WiFiRFFIDataset(DATASET_DIR, mode="test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # augmentor (建议 CFO 用 Hz 模式更可控)

    augmentor = RFFIAugmentor(
        sampling_rate=5e6,

    # probabilities (建议先温和，确保能收敛)
        p_phase_rotate=0.5,
        p_amp_scale=0.4,
        p_noise=0.5,
        p_cfo=0.25,
        p_iq_imbalance=0.2,
        p_dac=0.3,

    # noise
        snr_db_range=(18.0, 35.0),

    # amp scale
        scale_range=0.10,

    # CFO (Hz)
        cfo_max_hz=300.0,

    # IQ imbalance (small)
        amp_imbalance_max=0.03,
        phase_imbalance_max_deg=2.0,

    # DAC label + impairment
        dac_strength_range=(0.1, 1.0),
        dac_bits_range=(7, 12),
        dac_poly_a3_max=0.18,
        dac_poly_a5_max=0.06,
        dac_slew_delta_range=(0.05, 0.50),
        dac_dither=True,
    )

    # model
    model = CVSincNet(num_classes=16, sample_rate=5e6).to(DEVICE)

    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_dac = 0.0

        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # ---- augmentation + dac strength label ----
            if USE_AUGMENTATION:
                with torch.no_grad():
                    inputs, dac_strength = augmentor(inputs, return_dac_strength=True)
            else:
                dac_strength = torch.zeros(inputs.size(0), device=DEVICE, dtype=torch.float32)

            optimizer.zero_grad()

            logits, dac_pred, feat = model(inputs, return_aux=True)

            # ---- losses ----
            ce = ce_loss_fn(logits, labels)
            dac = weighted_dac_loss(dac_pred, dac_strength, zero_weight=DAC_ZERO_WEIGHT)
            loss = ce + DAC_LAMBDA * dac

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += ce.item()
            running_dac += dac.item()

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / max(total, 1)
        test_acc = eval_acc(model, test_loader, DEVICE)

        avg_loss = running_loss / max(len(train_loader), 1)
        avg_ce = running_ce / max(len(train_loader), 1)
        avg_dac = running_dac / max(len(train_loader), 1)

        dt = time.time() - t0
        logger(
            f"Epoch [{epoch+1}/{EPOCHS}] Time: {dt:.1f}s | "
            f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, DAC: {avg_dac:.4f}) | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "best_model_run1_Aug.pth"))
            logger(f"  >>> Best Model Saved! (Acc: {best_acc:.2f}%)")

        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_model_run1_Aug.pth"))

    logger(f"Training Done. Best Test Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
