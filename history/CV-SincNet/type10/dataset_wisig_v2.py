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

                        # 注意：如果这里随机抽样，会破坏 sig_i 的时序连续性。
                        # 为了让后续 train/val 连续切分更稳，这里使用“按 sig_i 升序保留前 max_samples_per_combo 个”。
                        if self.max_samples_per_combo is not None and n > self.max_samples_per_combo:
                            keep = np.arange(int(self.max_samples_per_combo), dtype=np.int64)
                            for s in keep.tolist():
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
    def __init__(
        self,
        base: WiSigCompactDataset,
        selected: Sequence[int],
        split_source: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.base = base
        self.selected = np.asarray(selected, dtype=np.int64)
        self.split_source = str(split_source)
        self.transform = transform
        self.index = [base.index[int(i)] for i in self.selected.tolist()]
        self._domain_lut = getattr(base, "_domain_lut", None)
        self.tx_list = getattr(base, "tx_list", None)
        self.rx_list = getattr(base, "rx_list", None)
        self.day_list = getattr(base, "day_list", None)

    def __len__(self) -> int:
        return int(self.selected.shape[0])

    def __getitem__(self, k: int):
        x, y, d, meta = self.base[int(self.selected[k])]
        if self.transform is not None:
            x = self.transform(x)
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


def _resolve_days(
    day_list: Sequence[Any],
    days: Optional[Sequence[Union[int, str]]],
    default: Sequence[int],
) -> List[int]:
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


def _build_grouped_indices_sorted_by_sig(
    base: WiSigCompactDataset,
) -> Dict[Tuple[int, int, int, int], List[int]]:
    groups: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}
    for global_i, it in enumerate(base.index):
        key = (int(it.day_i), int(it.tx_i), int(it.rx_i), int(it.eq_i))
        groups.setdefault(key, []).append((int(it.sig_i), int(global_i)))

    out: Dict[Tuple[int, int, int, int], List[int]] = {}
    for key, pairs in groups.items():
        pairs = sorted(pairs, key=lambda z: z[0])  # 按 sig_i 时序排序
        out[key] = [gidx for _, gidx in pairs]
    return out


def _contiguous_train_val_split(
    idxs_sorted: Sequence[int],
    train_ratio: float,
    guard_gap: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    连续切分：
      train = 前 train_ratio 段
      gap   = 中间隔离带（丢弃不用）
      val   = 后剩余段

    返回: train_ids, val_ids, dropped_gap_ids
    """
    n = len(idxs_sorted)
    if n <= 0:
        return [], [], []
    if n == 1:
        return [int(idxs_sorted[0])], [], []

    train_ratio = float(train_ratio)
    train_ratio = min(0.99, max(0.01, train_ratio))
    n_tr = int(round(n * train_ratio))
    n_tr = min(max(1, n_tr), n - 1)

    gap = max(0, int(guard_gap))
    max_gap = max(0, n - n_tr - 1)  # 至少留一个给 val
    gap = min(gap, max_gap)

    train_ids = list(idxs_sorted[:n_tr])
    gap_ids = list(idxs_sorted[n_tr:n_tr + gap])
    val_ids = list(idxs_sorted[n_tr + gap:])

    # 极端情况下再保底
    if len(val_ids) == 0 and len(train_ids) >= 2:
        val_ids = [train_ids.pop()]
        gap_ids = []

    return train_ids, val_ids, gap_ids


def _cap_per_group_keep_order(
    selected: Sequence[int],
    base: WiSigCompactDataset,
    max_per_combo: Optional[int],
) -> List[int]:
    if max_per_combo is None or max_per_combo <= 0:
        return list(selected)

    selected_set = set(int(x) for x in selected)
    groups: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}
    for global_i, it in enumerate(base.index):
        if global_i not in selected_set:
            continue
        key = (int(it.day_i), int(it.tx_i), int(it.rx_i), int(it.eq_i))
        groups.setdefault(key, []).append((int(it.sig_i), int(global_i)))

    out: List[int] = []
    for key in groups:
        pairs = sorted(groups[key], key=lambda z: z[0])
        out.extend([gidx for _, gidx in pairs[:max_per_combo]])
    return sorted(out)


def make_day123_trainval_day4_test(
    ds: Dict[str, Any],
    *,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "day",
    normalize: bool = True,
    crop_mode: str = "center",
    transform_train: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    train_ratio: float = 0.8,
    guard_gap: int = 8,
    train_days: Optional[Sequence[Union[int, str]]] = None,
    test_days: Optional[Sequence[Union[int, str]]] = None,
    max_samples_per_combo_day123: Optional[int] = None,
    max_samples_per_combo_test: Optional[int] = None,
    max_samples_per_combo_train: Optional[int] = None,
    max_samples_per_combo_val: Optional[int] = None,
    seed: int = 0,
):
    """
    更稳妥的三分法：
      - train: day1-3 内每个 (day, tx, rx, eq) 组合按 sig_i 顺序取前 train_ratio
      - val  : day1-3 内同组合剩余数据（可选跳过 guard_gap）
      - test : day4 全量，单独返回，不再并入 val

    说明：
      1) 不再随机打乱 day1-3，减少相邻片段泄露风险
      2) val 与 test 完全分开
      3) 若 sig_i 保留采集顺序，这比随机 80/20 更合理
    """
    day_list = list(ds.get("capture_date_list", []))
    n_day = len(day_list)
    if n_day < 2:
        raise ValueError("WiSig split requires at least 2 days.")

    default_train_days = list(range(min(3, max(1, n_day - 1))))
    default_test_days = [min(3, n_day - 1)] if n_day >= 4 else [n_day - 1]

    train_day_idx = _resolve_days(day_list, train_days, default_train_days)
    test_day_idx = _resolve_days(day_list, test_days, default_test_days)
    train_day_idx = [d for d in train_day_idx if d not in test_day_idx]
    if len(train_day_idx) == 0:
        raise ValueError("No train days left after removing test days.")

    # day1-3 原始池，不做 transform，避免 train/val 共用同一 transform
    base_day123 = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=train_day_idx,
        domain=domain,
        transform=None,
        max_samples_per_combo=max_samples_per_combo_day123,
        seed=seed,
        build_index=True,
    )

    groups = _build_grouped_indices_sorted_by_sig(base_day123)

    train_sel: List[int] = []
    val_sel: List[int] = []
    dropped_gap_sel: List[int] = []

    for _, idxs_sorted in groups.items():
        tr_ids, va_ids, gap_ids = _contiguous_train_val_split(
            idxs_sorted=idxs_sorted,
            train_ratio=train_ratio,
            guard_gap=guard_gap,
        )
        train_sel.extend(tr_ids)
        val_sel.extend(va_ids)
        dropped_gap_sel.extend(gap_ids)

    train_sel = _cap_per_group_keep_order(train_sel, base_day123, max_samples_per_combo_train)
    val_sel = _cap_per_group_keep_order(val_sel, base_day123, max_samples_per_combo_val)

    train_ds = WiSigSubsetDataset(
        base_day123,
        sorted(train_sel),
        split_source="contiguous_front_from_day123",
        transform=transform_train,
    )
    val_ds = WiSigSubsetDataset(
        base_day123,
        sorted(val_sel),
        split_source="contiguous_tail_from_day123",
        transform=transform_eval,
    )

    test_parts: List[Dataset] = []
    for d_idx in test_day_idx:
        full_day_ds = WiSigCompactDataset(
            ds,
            out_len=out_len,
            crop_mode=crop_mode,
            normalize=normalize,
            equalized=equalized,
            day_keep=[d_idx],
            domain=domain,
            transform=transform_eval,
            max_samples_per_combo=max_samples_per_combo_test,
            seed=seed,
            build_index=True,
        )
        test_parts.append(
            WiSigSubsetDataset(
                full_day_ds,
                list(range(len(full_day_ds))),
                split_source=f"full_test_day_{d_idx}",
                transform=None,
            )
        )

    test_ds = WiSigConcatDataset(test_parts) if len(test_parts) > 1 else test_parts[0]

    info = {
        "mode": "contiguous_day123_train_val__day4_test_only",
        "train_days_idx": train_day_idx,
        "train_days_label": [day_list[i] for i in train_day_idx],
        "test_days_idx": test_day_idx,
        "test_days_label": [day_list[i] for i in test_day_idx],
        "train_ratio": float(train_ratio),
        "guard_gap": int(guard_gap),
        "day123_pool_size": len(base_day123),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "dropped_gap_size": len(dropped_gap_sel),
        "max_samples_per_combo_day123": max_samples_per_combo_day123,
        "max_samples_per_combo_train": max_samples_per_combo_train,
        "max_samples_per_combo_val": max_samples_per_combo_val,
        "max_samples_per_combo_test": max_samples_per_combo_test,
    }
    return train_ds, val_ds, test_ds, info