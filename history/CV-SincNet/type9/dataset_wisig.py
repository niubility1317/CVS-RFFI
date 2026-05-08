# wisig_loader.py
# ------------------------------------------------------------
# WiSig compact PKL loader (works for ManySig/ManyRx/ManyTx/SingleDay)
# Output: x (2,T), y (tx_id), d (domain_id), meta
# DG split helper: Leave-One-Day-Out
# ------------------------------------------------------------
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _rms_normalize_iq(x_2t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    x_2t: shape (2, T) float
    RMS over time of power = I^2 + Q^2
    """
    # power(t) = I^2 + Q^2
    p = np.mean(x_2t[0] * x_2t[0] + x_2t[1] * x_2t[1])
    s = np.sqrt(p + eps)
    return x_2t / s


def _pad_or_crop_2t(x_2t: np.ndarray, out_len: int, mode: str = "center") -> np.ndarray:
    """
    x_2t: (2, T)
    If T < out_len -> zero-pad
    If T > out_len -> crop
    mode: "center" | "left"
    """
    assert x_2t.ndim == 2 and x_2t.shape[0] == 2
    T = x_2t.shape[1]
    if T == out_len:
        return x_2t
    if T > out_len:
        if mode == "left":
            return x_2t[:, :out_len]
        # center crop
        start = (T - out_len) // 2
        return x_2t[:, start : start + out_len]

    # pad
    pad = out_len - T
    if mode == "left":
        out = np.zeros((2, out_len), dtype=x_2t.dtype)
        out[:, :T] = x_2t
        return out
    # center pad
    left = pad // 2
    out = np.zeros((2, out_len), dtype=x_2t.dtype)
    out[:, left : left + T] = x_2t
    return out


def load_wisig_compact_pkl(pkl_path: str) -> Dict[str, Any]:
    """
    Load a WiSig compact subset pickle (e.g., ManySig.pkl).
    Expected keys typically include:
      - tx_list, rx_list, capture_date_list, equalized_list, data
    Some helper-generated pkls may use node_list/data instead.
    """
    pkl_path = os.path.realpath(pkl_path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"WiSig PKL not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)

    if not isinstance(ds, dict):
        raise ValueError(f"Unexpected pkl content type: {type(ds)}")

    # Minimal sanity checks
    if "data" not in ds:
        raise KeyError("WiSig pkl missing key 'data'.")

    # Normalize naming if needed
    if "tx_list" not in ds and "node_list" in ds:
        ds["tx_list"] = ds["node_list"]
    if "capture_date_list" not in ds:
        # some merged formats may not include days; treat as single day
        ds["capture_date_list"] = [None]
    if "rx_list" not in ds:
        ds["rx_list"] = [None]
    if "equalized_list" not in ds:
        ds["equalized_list"] = [0]

    return ds


@dataclass(frozen=True)
class WiSigIndex:
    tx_i: int
    rx_i: int
    day_i: int
    eq_i: int
    sig_i: int


class WiSigCompactDataset(Dataset):
    """
    Build sample-level indexing over WiSig compact dataset:
      data[tx][rx][day][eq] -> (N, 256, 2) where last dim is (I,Q)

    Returns:
      x: FloatTensor (2, out_len)
      y: int (tx index)
      d: int (domain id)
      meta: dict with tx/rx/day/eq original labels + indices
    """

    def __init__(
        self,
        ds: Dict[str, Any],
        *,
        out_len: int = 256,
        crop_mode: str = "center",
        normalize: bool = True,
        equalized: Union[int, str] = 1,
        # selectors (None = all)
        tx_keep: Optional[Sequence[int]] = None,
        rx_keep: Optional[Sequence[int]] = None,
        day_keep: Optional[Sequence[int]] = None,
        # domain
        domain: str = "day",  # "day" | "rx" | "rx_day"
        # optional transform on torch tensor (2,T)
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_samples_per_combo: Optional[int] = None,
        seed: int = 0,
        build_index: bool = True,
    ):
        super().__init__()
        self.ds = ds
        self.data = ds["data"]
        self.tx_list = list(ds.get("tx_list", []))
        self.rx_list = list(ds.get("rx_list", []))
        self.day_list = list(ds.get("capture_date_list", []))
        self.eq_list = list(ds.get("equalized_list", [0]))

        self.out_len = int(out_len)
        self.crop_mode = crop_mode
        self.normalize = bool(normalize)
        self.domain = domain
        self.transform = transform
        self.max_samples_per_combo = max_samples_per_combo
        self.rng = np.random.default_rng(seed)

        # Which equalization(s) to use
        if isinstance(equalized, str) and equalized.lower() == "both":
            self.eq_keep = list(range(len(self.eq_list)))
        else:
            eq_val = int(equalized)
            if eq_val not in self.eq_list:
                raise ValueError(
                    f"equalized={eq_val} not in ds['equalized_list']={self.eq_list}. "
                    f"Try equalized='both' or use one of {self.eq_list}."
                )
            self.eq_keep = [self.eq_list.index(eq_val)]

        n_tx = len(self.tx_list)
        n_rx = len(self.rx_list)
        n_day = len(self.day_list)

        self.tx_keep = list(range(n_tx)) if tx_keep is None else list(tx_keep)
        self.rx_keep = list(range(n_rx)) if rx_keep is None else list(rx_keep)
        self.day_keep = list(range(n_day)) if day_keep is None else list(day_keep)

        self._domain_lut = self._build_domain_lut(domain)
        self.index: List[WiSigIndex] = []
        if build_index:
            self._build_sample_index()

    def _build_domain_lut(self, domain: str) -> Dict[Tuple[int, int], int]:
        """
        Map (rx_i, day_i) -> domain id according to domain setting.
        """
        domain = domain.lower()
        lut: Dict[Tuple[int, int], int] = {}
        if domain == "day":
            for rx_i in range(len(self.rx_list)):
                for day_i in range(len(self.day_list)):
                    lut[(rx_i, day_i)] = day_i
        elif domain == "rx":
            for rx_i in range(len(self.rx_list)):
                for day_i in range(len(self.day_list)):
                    lut[(rx_i, day_i)] = rx_i
        elif domain == "rx_day":
            did = 0
            for rx_i in range(len(self.rx_list)):
                for day_i in range(len(self.day_list)):
                    lut[(rx_i, day_i)] = did
                    did += 1
        else:
            raise ValueError("domain must be one of: 'day', 'rx', 'rx_day'")
        return lut

    def _build_sample_index(self) -> None:
        """
        Build a flat list of sample pointers (tx,rx,day,eq,sig).
        ManySig scale is manageable (typically < 1M samples).
        """
        idx: List[WiSigIndex] = []
        for tx_i in self.tx_keep:
            for rx_i in self.rx_keep:
                for day_i in self.day_keep:
                    for eq_i in self.eq_keep:
                        arr = self.data[tx_i][rx_i][day_i][eq_i]
                        if arr is None:
                            continue
                        n = int(arr.shape[0])
                        if n <= 0:
                            continue

                        if self.max_samples_per_combo is not None and n > self.max_samples_per_combo:
                            # random subset for speed
                            choose = self.rng.choice(n, size=self.max_samples_per_combo, replace=False)
                            for s in choose.tolist():
                                idx.append(WiSigIndex(tx_i, rx_i, day_i, eq_i, int(s)))
                        else:
                            # full
                            for s in range(n):
                                idx.append(WiSigIndex(tx_i, rx_i, day_i, eq_i, s))
        self.index = idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        it = self.index[k]
        x = self.data[it.tx_i][it.rx_i][it.day_i][it.eq_i][it.sig_i]  # (256,2)
        # convert to (2,T)
        x_2t = np.asarray(x, dtype=np.float32).T  # (2,256)

        if self.out_len != x_2t.shape[1]:
            x_2t = _pad_or_crop_2t(x_2t, self.out_len, mode=self.crop_mode)

        if self.normalize:
            x_2t = _rms_normalize_iq(x_2t)

        x_t = torch.from_numpy(x_2t)  # (2,T) float32
        if self.transform is not None:
            x_t = self.transform(x_t)

        y = int(it.tx_i)
        d = int(self._domain_lut[(it.rx_i, it.day_i)])

        meta = {
            "tx_i": it.tx_i,
            "rx_i": it.rx_i,
            "day_i": it.day_i,
            "eq_i": it.eq_i,
            "sig_i": it.sig_i,
            "tx": self.tx_list[it.tx_i] if it.tx_i < len(self.tx_list) else it.tx_i,
            "rx": self.rx_list[it.rx_i] if it.rx_i < len(self.rx_list) else it.rx_i,
            "day": self.day_list[it.day_i] if it.day_i < len(self.day_list) else it.day_i,
            "equalized": self.eq_list[it.eq_i] if it.eq_i < len(self.eq_list) else None,
        }
        return x_t, y, d, meta


def make_leave_one_day_out_split(
    ds: Dict[str, Any],
    *,
    heldout_day: Optional[Union[int, str]] = None,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "day",
    normalize: bool = True,
    crop_mode: str = "center",
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    # for speed
    max_samples_per_combo_train: Optional[int] = None,
    max_samples_per_combo_test: Optional[int] = None,
    seed: int = 0,
) -> Tuple[WiSigCompactDataset, WiSigCompactDataset]:
    """
    ManySig recommended DG protocol: train on 3 days, test on 1 unseen day.
    heldout_day:
      - None -> use last day in capture_date_list
      - int -> day index
      - str/int -> match value in capture_date_list
    """
    day_list = list(ds.get("capture_date_list", []))
    if len(day_list) == 0:
        raise ValueError("Dataset has empty capture_date_list; cannot do LODO split.")

    if heldout_day is None:
        heldout_day_i = len(day_list) - 1
    elif isinstance(heldout_day, int) and 0 <= heldout_day < len(day_list):
        heldout_day_i = int(heldout_day)
    else:
        # match by value
        try:
            heldout_day_i = day_list.index(heldout_day)
        except ValueError:
            # try string/int conversion
            heldout_day_str = str(heldout_day)
            found = None
            for i, v in enumerate(day_list):
                if str(v) == heldout_day_str:
                    found = i
                    break
            if found is None:
                raise ValueError(f"heldout_day={heldout_day} not found in capture_date_list={day_list}")
            heldout_day_i = found

    train_days = [i for i in range(len(day_list)) if i != heldout_day_i]
    test_days = [heldout_day_i]

    train_set = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=train_days,
        domain=domain,
        transform=transform,
        max_samples_per_combo=max_samples_per_combo_train,
        seed=seed,
        build_index=True,
    )
    test_set = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=test_days,
        domain=domain,
        transform=None,  # 通常测试不做随机增强
        max_samples_per_combo=max_samples_per_combo_test,
        seed=seed,
        build_index=True,
    )
    return train_set, test_set


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # Quick sanity check:
    # python wisig_loader.py /path/to/ManySig.pkl
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wisig_loader.py /path/to/ManySig.pkl")
        raise SystemExit(1)

    pkl = sys.argv[1]
    ds = load_wisig_compact_pkl(pkl)
    print("[WiSig] keys:", list(ds.keys()))
    print("[WiSig] tx/rx/day/eq lens:", len(ds["tx_list"]), len(ds["rx_list"]), len(ds["capture_date_list"]), len(ds["equalized_list"]))

    train_set, test_set = make_leave_one_day_out_split(ds, heldout_day=None, equalized=1, out_len=256, domain="day")
    print("[Split] train:", len(train_set), "test:", len(test_set))

    x, y, d, meta = train_set[0]
    print("[Sample] x:", tuple(x.shape), "y:", y, "d:", d, "meta:", meta)