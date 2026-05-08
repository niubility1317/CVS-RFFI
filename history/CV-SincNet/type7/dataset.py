import torch
from torch.utils.data import Dataset
import numpy as np
import os


class WiFiRFFIDataset(Dataset):
    def __init__(self, root_dir: str, mode: str = "train", run_name: str = None, mmap: bool = True):
        """
        root_dir: 数据集根目录，例如 './Dataset_ORALCE' 或 './Dataset_ORALCE/run1'
        mode: 'train' 或 'test'
        run_name: 例如 'run1'。若提供，则优先读取 root_dir/run_name（或 root_dir 本身就是 run_name）
        mmap: 是否对 X 使用 np.load(..., mmap_mode="r")
        """
        self.mode = mode

        root_dir = os.path.normpath(root_dir)

        # ---------- 1) 若显式给了 run_name：优先用它 ----------
        data_dir = None
        if run_name is not None:
            rn = str(run_name)
            base = os.path.basename(root_dir).lower()
            if base == rn.lower():
                cand = root_dir
            else:
                cand = os.path.join(root_dir, rn)

            x_cand = os.path.join(cand, f"X_{mode}.npy")
            y_cand = os.path.join(cand, f"Y_{mode}.npy")
            if os.path.isdir(cand) and os.path.exists(x_cand) and os.path.exists(y_cand):
                data_dir = cand
            else:
                raise FileNotFoundError(
                    f"run_name='{run_name}' specified but files not found:\n  {x_cand}\n  {y_cand}"
                )

        # ---------- 2) 否则：保持你原来的自动 run1 逻辑 ----------
        if data_dir is None:
            base_name = os.path.basename(root_dir).lower()
            if base_name == "run1":
                data_dir = root_dir
            else:
                run1_dir = os.path.join(root_dir, "run1")
                run1_x = os.path.join(run1_dir, f"X_{mode}.npy")
                run1_y = os.path.join(run1_dir, f"Y_{mode}.npy")
                if os.path.isdir(run1_dir) and os.path.exists(run1_x) and os.path.exists(run1_y):
                    data_dir = run1_dir
                else:
                    data_dir = root_dir

        x_path = os.path.join(data_dir, f"X_{mode}.npy")
        y_path = os.path.join(data_dir, f"Y_{mode}.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(
                f"Data files not found. Looking for:\n  {x_path}\n  {y_path}"
            )

        print(f"[{mode.upper()}] Loading data from {x_path} ...", flush=True)

        if mmap:
            self.X = np.load(x_path, mmap_mode="r")
        else:
            self.X = np.load(x_path)

        self.Y = np.load(y_path)
        print(f"[{mode.upper()}] Loaded. Shape: {self.X.shape}", flush=True)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # 形状: (2, 1024) 其中 2 = [I, Q]
        # copy() 保证可写/连续（后续增强可能会 in-place）
        signal = torch.from_numpy(self.X[idx].copy()).float()
        label = torch.tensor(self.Y[idx]).long()

        # 能量归一化（避免不同样本幅度差异导致训练不稳定）
        energy = torch.sqrt(torch.sum(signal ** 2))
        if energy > 1e-8:
            signal = signal / energy

        return signal, label
