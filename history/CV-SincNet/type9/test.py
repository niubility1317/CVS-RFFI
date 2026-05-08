# test.py
# ------------------------------------------------------------
# Run2 evaluator for CV-SincNet (CVSincNet)
#
# Fixes:
# 1) Robust checkpoint loading:
#    - supports checkpoints saved by train.py pack_state(): {"model": ..., "optimizer": ..., ...}
#    - supports {"state_dict": ...} or pure state_dict
#    - strips "module." prefix (DataParallel)
#    - tries strict=True first; if fails, falls back to shape-safe partial loading with clear warnings
#
# 2) Robust model forward output:
#    - model(x) may return logits tensor, or (logits, ...) tuple/list, or dict with "logits"
# ------------------------------------------------------------

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import CVSincNet
from DataAugmentation_v2 import apply_receiver_dg

SAMPLE_RATE_HZ = 5e6



class NpyRFFIDataset(Dataset):
    """
    直接从指定的 X_test.npy / Y_test.npy 读取，绕开 dataset.py，避免误读 run1。
    """
    def __init__(self, x_path: str, y_path: str):
        self.x_path = os.path.realpath(x_path)
        self.y_path = os.path.realpath(y_path)

        print(f"[TEST] Loading data from {self.x_path} ...")
        self.X = np.load(self.x_path, mmap_mode="r")  # memmap
        self.Y = np.load(self.y_path)

        print(f"[TEST] Loaded. Shape: {self.X.shape}")
        print(f"[DEBUG] memmap file: {self.X.filename}")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # X: (2, 1024), float32
        signal = torch.from_numpy(self.X[idx].copy()).float()
        label = torch.tensor(int(self.Y[idx]), dtype=torch.long)

        # 与你原 dataset.py 里保持一致：能量归一化
        energy = torch.sqrt(torch.sum(signal ** 2))
        if energy > 1e-8:
            signal = signal / energy
        return signal, label


def _strip_module_prefix(sd: dict) -> dict:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _load_state_dict(weights_path: str, map_location="cpu") -> dict:
    """
    Accepts:
      - pure state_dict: {param_name: Tensor, ...}
      - {"state_dict": state_dict}
      - {"model": state_dict, "optimizer": ..., ...}  (from train.py pack_state)
      - other common aliases: "model_state", "net", "weights"
    """
    ckpt = torch.load(weights_path, map_location=map_location)

    if isinstance(ckpt, dict):
        # nested common keys
        for key in ("state_dict", "model", "model_state", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                # a few frameworks nest again
                if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                return _strip_module_prefix(sd)

        # pure state_dict (dict of tensors)
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return _strip_module_prefix(ckpt)

    raise ValueError(f"Unrecognized checkpoint format: {weights_path}")


def _safe_load_into_model(model: torch.nn.Module, sd: dict, strict: bool = True) -> None:
    """
    Try strict load first; if fails and strict=True, do a best-effort partial load:
      - keeps only keys that exist in model.state_dict()
      - drops keys with mismatched shapes
    """
    try:
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if (len(missing) == 0 and len(unexpected) == 0):
            print("[OK] load_state_dict strict=True (exact match).")
        else:
            print(f"[WARN] load_state_dict finished with missing={len(missing)}, unexpected={len(unexpected)} (strict={strict}).")
            if len(missing) > 0:
                print("  missing keys (first 20):", missing[:20])
            if len(unexpected) > 0:
                print("  unexpected keys (first 20):", unexpected[:20])
        return
    except RuntimeError as e:
        if not strict:
            raise

        print("[WARN] strict=True load failed, will fallback to shape-safe partial load.")
        print("       Error:", str(e).splitlines()[0])

        model_sd = model.state_dict()
        kept = {}
        dropped_missing = 0
        dropped_shape = 0

        for k, v in sd.items():
            if k not in model_sd:
                dropped_missing += 1
                continue
            if hasattr(model_sd[k], "shape") and hasattr(v, "shape") and model_sd[k].shape != v.shape:
                dropped_shape += 1
                continue
            kept[k] = v

        print(f"[INFO] filtered state_dict: kept={len(kept)}/{len(sd)} "
              f"(drop_not_in_model={dropped_missing}, drop_shape_mismatch={dropped_shape})")

        missing2, unexpected2 = model.load_state_dict(kept, strict=False)
        print(f"[OK] partial load done (strict=False). missing={len(missing2)}, unexpected={len(unexpected2)}")
        if len(missing2) > 0:
            print("  missing keys (first 30):", missing2[:30])
        if len(unexpected2) > 0:
            print("  unexpected keys (first 30):", unexpected2[:30])


def _get_logits(model_out):
    """
    model(x) could return:
      - logits Tensor
      - (logits, ...) tuple/list
      - dict with 'logits' / 'out' / 'pred' / 'y_pred'
    """
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 1:
        return model_out[0]
    if isinstance(model_out, dict):
        for k in ("logits", "out", "pred", "y_pred"):
            if k in model_out:
                return model_out[k]
    raise RuntimeError(f"Unsupported model output type: {type(model_out)}")


def evaluate(model, loader, device, num_classes: int, args):
    model.eval()
    correct = 0
    total = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # optional DC block
            if getattr(args, 'dc_block', False):
                x = x - x.mean(dim=-1, keepdim=True)
                # re-normalize energy to match training
                rms = torch.sqrt(torch.mean(x ** 2, dim=(1,2), keepdim=True) + 1e-12)
                x = x / rms

            if getattr(args, 'tta', False):
                n_tta = max(1, int(getattr(args, 'tta_n', 1)))
                envs = max(1, int(getattr(args, 'tta_envs', 8)))
                logits_acc = None
                for _ in range(n_tta):
                    env_id = torch.randint(0, envs, (x.size(0),), device=x.device)
                    x_aug = apply_receiver_dg(x, fs=SAMPLE_RATE_HZ, env_id=env_id,
                                           p_lowpass=float(getattr(args, 'tta_p_lowpass', 0.7)),
                                           p_multipath=float(getattr(args, 'tta_p_multipath', 0.7)))
                    # re-normalize after DG
                    rms = torch.sqrt(torch.mean(x_aug ** 2, dim=(1,2), keepdim=True) + 1e-12)
                    x_aug = x_aug / rms
                    out_i = model(x_aug)
                    logits_i = _get_logits(out_i)
                    logits_acc = logits_i if logits_acc is None else (logits_acc + logits_i)
                logits = logits_acc / float(n_tta)
            else:
                out = model(x)
                logits = _get_logits(out)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            for t, p in zip(y_np, p_np):
                conf[int(t), int(p)] += 1

    acc = 100.0 * correct / max(total, 1)

    per_class_total = conf.sum(axis=1)
    per_class_correct = np.diag(conf)
    per_class_acc = (per_class_correct / np.maximum(per_class_total, 1)) * 100.0

    return acc, conf, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run2_dir", type=str, default="./Dataset_ORALCE/run2",
                        help="Path to run2 dir containing X_test.npy / Y_test.npy")
    parser.add_argument("--weights", type=str, default="./best_model_v8DG.pth",
                        help="Path to best_model.pth (checkpoint or state_dict)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--save_conf", action="store_true",
                        help="Save confusion matrix to conf_run2.npy and conf_run2.txt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--strict", action="store_true",
                        help="Force strict=True only (no fallback partial load).")
    parser.add_argument("--dc_block", action="store_true", help="Subtract per-channel mean (DC block) before inference.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation using apply_receiver_dg and average logits.")
    parser.add_argument("--tta_n", type=int, default=8, help="Number of TTA samples per batch when --tta is set.")
    parser.add_argument("--tta_envs", type=int, default=8, help="Number of DG envs for env_id in apply_receiver_dg.")
    parser.add_argument("--tta_p_lowpass", type=float, default=0.7)
    parser.add_argument("--tta_p_multipath", type=float, default=0.7)
    args = parser.parse_args()

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 明确拼接 run2 的测试文件路径
    x_path = os.path.join(args.run2_dir, "X_test.npy")
    y_path = os.path.join(args.run2_dir, "Y_test.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Run2 test files not found. Expect:\n  {x_path}\n  {y_path}"
        )

    # ✅ 关键：绕开 WiFiRFFIDataset，强制读 run2 的 npy
    ds = NpyRFFIDataset(x_path, y_path)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # model
    model = CVSincNet(num_classes=args.num_classes).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    # ------------------ FIX: robust checkpoint loading ------------------
    sd = _load_state_dict(args.weights, map_location="cpu")
    for k, v in sd.items():
        if torch.is_tensor(v):
            sd[k] = v.to(device)

    if args.strict:
        model.load_state_dict(sd, strict=True)
        print("[OK] load_state_dict strict=True (forced).")
    else:
        _safe_load_into_model(model, sd, strict=True)
    # -------------------------------------------------------------------

    # eval
    acc, conf, per_class_acc = evaluate(model, loader, device, num_classes=args.num_classes, args=args)

    print(f"[RUN2 TEST] weights = {args.weights}")
    print(f"[RUN2 TEST] run2_dir = {args.run2_dir}")
    print(f"[RUN2 TEST] Accuracy: {acc:.2f}%")

    print("[RUN2 TEST] Per-class Acc (%):")
    for i, a in enumerate(per_class_acc):
        print(f"  class {i:02d}: {a:.2f}%   (n={conf[i].sum()})")

    if args.save_conf:
        np.save("conf_run2.npy", conf)
        with open("conf_run2.txt", "w", encoding="utf-8") as f:
            f.write(np.array2string(conf, separator=", "))
        print("[RUN2 TEST] Confusion saved: conf_run2.npy, conf_run2.txt")


if __name__ == "__main__":
    main()
