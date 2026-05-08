import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import CVSincNet


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


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    correct = 0
    total = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
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
    parser.add_argument("--weights", type=str, default="./best_model.pth",
                        help="Path to best_model.pth(state_dict)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--save_conf", action="store_true",
                        help="Save confusion matrix to conf_run2.npy and conf_run2.txt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
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

    # ✅ 防误读保护：如果你想更严格，就打开下面两行
    # assert "run2" in os.path.realpath(ds.X.filename), f"[FATAL] Actually loaded: {ds.X.filename}"

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

    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)

    # eval
    acc, conf, per_class_acc = evaluate(model, loader, device, num_classes=args.num_classes)

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
