import torch
from torch.utils.data import Dataset
import numpy as np
import os

class WiFiRFFIDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        root_dir: 数据集根目录，例如 './Dataset_ORALCE'
                  或者你也可以直接传 './Dataset_ORALCE/run1'
        mode: 'train' 或 'test'
        """
        self.mode = mode

        # ====== 自动指向 run1（不改接口）======
        # 规则：
        # 1) 若 root_dir 本身已经是 .../run1，则直接用 root_dir
        # 2) 否则若 root_dir/run1 存在且包含 X_{mode}.npy，则优先使用 root_dir/run1
        # 3) 否则回退使用 root_dir
        
        base_name = os.path.basename(os.path.normpath(root_dir)).lower()
        root_dir = './Dataset_ORALCE/run1'
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
        # =====================================

        x_path = os.path.join(data_dir, f"X_{mode}.npy")
        y_path = os.path.join(data_dir, f"Y_{mode}.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(
                f"Data files not found. Looking for:\n  {x_path}\n  {y_path}"
            )

        print(f"[{mode.upper()}] Loading data from {x_path} ...")

        # mmap 防止内存溢出
        self.X = np.load(x_path, mmap_mode='r')
        self.Y = np.load(y_path)

        print(f"[{mode.upper()}] Loaded. Shape: {self.X.shape}")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # 形状: (2, 1024)
        signal = torch.from_numpy(self.X[idx].copy()).float()
        label = torch.tensor(self.Y[idx]).long()

        # 能量归一化
        energy = torch.sqrt(torch.sum(signal ** 2))
        if energy > 1e-8:
            signal = signal / energy

        return signal, label
