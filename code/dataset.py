import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WiFiRFFIDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        run_name: str | None = "run1",
        mmap: bool = True,
    ):
        """
        ORALCE-style npy dataset loader.

        Parameters
        ----------
        root_dir : str
            Dataset root directory, e.g. './Dataset_ORALCE' or './Dataset_ORALCE/run1'
        mode : str
            'train' or 'test'
        run_name : str | None
            If provided, prefer root_dir/run_name (unless root_dir already ends with run_name).
            Default is 'run1' for backward compatibility.
            Set None to enable auto-detect (legacy behavior).
        mmap : bool
            Whether to mmap X via np.load(..., mmap_mode='r')
        """
        assert mode in ("train", "test"), f"mode must be 'train' or 'test', got {mode}"
        self.mode = mode

        root_dir = os.path.normpath(root_dir)

        # ---------- 1) If run_name is explicitly given: prefer it ----------
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

        # ---------- 2) Otherwise: legacy auto logic (prefer run1 if exists) ----------
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
            raise FileNotFoundError(f"Data files not found. Looking for:\n  {x_path}\n  {y_path}")

        print(f"[{mode.upper()}] Loading data from {x_path} ...", flush=True)

        self.X = np.load(x_path, mmap_mode="r") if mmap else np.load(x_path)
        self.Y = np.load(y_path)
        print(f"[{mode.upper()}] Loaded. Shape: {self.X.shape}", flush=True)

    def __len__(self):
        return int(len(self.Y))

    def __getitem__(self, idx: int):
        # X shape: (2, L) where 2=[I,Q]
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = int(self.Y[idx])

        # Energy normalization for stability
        energy = torch.sqrt(torch.sum(x * x))
        if energy > 1e-8:
            x = x / energy

        return x, torch.tensor(y).long()
