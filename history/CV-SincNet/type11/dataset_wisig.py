from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def _rms_normalize_iq(x_2t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.mean(x_2t[0] * x_2t[0] + x_2t[1] * x_2t[1])
    s = np.sqrt(p + eps)
    return x_2t / s


def _pad_or_crop_2t(x_2t: np.ndarray, out_len: int, mode: str = "center") -> np.ndarray:
    assert x_2t.ndim == 2 and x_2t.shape[0] == 2
    T = x_2t.shape[1]
    if T == out_len:
        return x_2t
    if T > out_len:
        if mode == "left":
            return x_2t[:, :out_len]
        start = (T - out_len) // 2
        return x_2t[:, start : start + out_len]
    out = np.zeros((2, out_len), dtype=x_2t.dtype)
    if mode == "left":
        out[:, :T] = x_2t
    else:
        left = (out_len - T) // 2
        out[:, left : left + T] = x_2t
    return out


def load_wisig_compact_pkl(pkl_path: str) -> Dict[str, Any]:
    pkl_path = os.path.realpath(pkl_path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"WiSig PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)
    if not isinstance(ds, dict):
        raise ValueError(f"Unexpected pkl content type: {type(ds)}")
    if "data" not in ds:
        raise KeyError("WiSig pkl missing key 'data'.")
    if "tx_list" not in ds and "node_list" in ds:
        ds["tx_list"] = ds["node_list"]
    if "capture_date_list" not in ds:
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
    def __init__(
        self,
        ds: Dict[str, Any],
        *,
        out_len: int = 256,
        crop_mode: str = "center",
        normalize: bool = True,
        equalized: Union[int, str] = 1,
        tx_keep: Optional[Sequence[int]] = None,
        rx_keep: Optional[Sequence[int]] = None,
        day_keep: Optional[Sequence[int]] = None,
        domain: str = "day",
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
        self.crop_mode = str(crop_mode)
        self.normalize = bool(normalize)
        self.domain = str(domain)
        self.transform = transform
        self.max_samples_per_combo = max_samples_per_combo
        self.rng = np.random.default_rng(seed)

        if isinstance(equalized, str) and equalized.lower() == "both":
            self.eq_keep = list(range(len(self.eq_list)))
        else:
            eq_val = int(equalized)
            if eq_val not in self.eq_list:
                raise ValueError(f"equalized={eq_val} not in {self.eq_list}")
            self.eq_keep = [self.eq_list.index(eq_val)]

        n_tx = len(self.tx_list)
        n_rx = len(self.rx_list)
        n_day = len(self.day_list)
        self.tx_keep = list(range(n_tx)) if tx_keep is None else list(tx_keep)
        self.rx_keep = list(range(n_rx)) if rx_keep is None else list(rx_keep)
        self.day_keep = list(range(n_day)) if day_keep is None else list(day_keep)
        self._domain_lut = self._build_domain_lut(self.domain)
        self.index: List[WiSigIndex] = []
        if build_index:
            self._build_sample_index()

    def _build_domain_lut(self, domain: str) -> Dict[Tuple[int, int], int]:
        lut: Dict[Tuple[int, int], int] = {}
        domain = domain.lower()
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
                            choose = self.rng.choice(n, size=self.max_samples_per_combo, replace=False)
                            for s in choose.tolist():
                                idx.append(WiSigIndex(tx_i, rx_i, day_i, eq_i, int(s)))
                        else:
                            for s in range(n):
                                idx.append(WiSigIndex(tx_i, rx_i, day_i, eq_i, s))
        self.index = idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        it = self.index[k]
        x = self.data[it.tx_i][it.rx_i][it.day_i][it.eq_i][it.sig_i]
        x_2t = np.asarray(x, dtype=np.float32).T
        if self.out_len != x_2t.shape[1]:
            x_2t = _pad_or_crop_2t(x_2t, self.out_len, mode=self.crop_mode)
        if self.normalize:
            x_2t = _rms_normalize_iq(x_2t)
        x_t = torch.from_numpy(x_2t)
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


class WiSigSubsetDataset(Dataset):
    def __init__(self, base: WiSigCompactDataset, selected: Sequence[int], split_source: str):
        self.base = base
        self.selected = np.asarray(selected, dtype=np.int64)
        self.split_source = str(split_source)
        self.index = [base.index[int(i)] for i in self.selected.tolist()]
        self._domain_lut = getattr(base, "_domain_lut", None)
        self.tx_list = getattr(base, "tx_list", None)
        self.rx_list = getattr(base, "rx_list", None)
        self.day_list = getattr(base, "day_list", None)

    def __len__(self) -> int:
        return int(self.selected.shape[0])

    def __getitem__(self, k: int):
        x, y, d, meta = self.base[int(self.selected[k])]
        meta = dict(meta)
        meta["split_source"] = self.split_source
        return x, y, d, meta


class WiSigConcatDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        sizes = [len(ds) for ds in self.datasets]
        self.cumulative_sizes = np.cumsum(sizes).tolist()

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side="right"))
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def _resolve_days(day_list: Sequence[Any], days: Optional[Sequence[Union[int, str]]], default: Sequence[int]) -> List[int]:
    if days is None:
        return list(default)
    out: List[int] = []
    for d in days:
        if isinstance(d, int) and 0 <= d < len(day_list):
            out.append(int(d))
            continue
        ds = str(d)
        found = None
        for i, v in enumerate(day_list):
            if str(v) == ds:
                found = i
                break
        if found is None:
            raise ValueError(f"Cannot resolve day {d!r} from {list(day_list)}")
        out.append(found)
    return out


def make_day123_randomsplit_plus_day4_test(
    ds: Dict[str, Any],
    *,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "day",
    normalize: bool = True,
    crop_mode: str = "center",
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    train_ratio: float = 0.8,
    train_days: Optional[Sequence[Union[int, str]]] = None,
    full_test_days: Optional[Sequence[Union[int, str]]] = None,
    max_samples_per_combo_train: Optional[int] = None,
    max_samples_per_combo_test: Optional[int] = None,
    seed: int = 0,
):
    day_list = list(ds.get("capture_date_list", []))
    n_day = len(day_list)
    if n_day < 2:
        raise ValueError("WiSig split requires at least 2 days.")
    default_train_days = list(range(min(3, max(1, n_day - 1))))
    default_full_test_days = [min(3, n_day - 1)] if n_day >= 4 else [n_day - 1]
    train_day_idx = _resolve_days(day_list, train_days, default_train_days)
    full_test_day_idx = _resolve_days(day_list, full_test_days, default_full_test_days)
    train_day_idx = [d for d in train_day_idx if d not in full_test_day_idx]
    if len(train_day_idx) == 0:
        raise ValueError("No train days left after removing full test days.")

    base = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=train_day_idx,
        domain=domain,
        transform=transform,
        max_samples_per_combo=max_samples_per_combo_train,
        seed=seed,
        build_index=True,
    )

    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[int, int, int, int], List[int]] = {}
    for global_i, it in enumerate(base.index):
        key = (int(it.day_i), int(it.tx_i), int(it.rx_i), int(it.eq_i))
        groups.setdefault(key, []).append(global_i)

    train_sel: List[int] = []
    test_sel: List[int] = []
    train_ratio = min(0.99, max(0.01, float(train_ratio)))
    for _, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * train_ratio))
        n_tr = min(max(1, n_tr), max(1, n - 1)) if n > 1 else 1
        train_sel.extend(idxs[:n_tr].tolist())
        if n > 1:
            test_sel.extend(idxs[n_tr:].tolist())

    train_ds = WiSigSubsetDataset(base, sorted(train_sel), split_source="random80_from_train_days")
    test_day123 = WiSigSubsetDataset(base, sorted(test_sel), split_source="random20_from_train_days")

    test_parts: List[Dataset] = [test_day123]
    for d_idx in full_test_day_idx:
        full_day_ds = WiSigCompactDataset(
            ds,
            out_len=out_len,
            crop_mode=crop_mode,
            normalize=normalize,
            equalized=equalized,
            day_keep=[d_idx],
            domain=domain,
            transform=None,
            max_samples_per_combo=max_samples_per_combo_test,
            seed=seed,
            build_index=True,
        )
        test_parts.append(WiSigSubsetDataset(full_day_ds, list(range(len(full_day_ds))), split_source=f"full_test_day_{d_idx}"))

    test_ds = WiSigConcatDataset(test_parts)
    info = {
        "mode": "random80_trainDays_plus_fullTestDays",
        "train_days_idx": train_day_idx,
        "train_days_label": [day_list[i] for i in train_day_idx],
        "full_test_days_idx": full_test_day_idx,
        "full_test_days_label": [day_list[i] for i in full_test_day_idx],
        "train_ratio": train_ratio,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "test_day123_size": len(test_day123),
        "test_full_days_size": sum(len(ds_i) for ds_i in test_parts[1:]),
    }
    return train_ds, test_ds, info
