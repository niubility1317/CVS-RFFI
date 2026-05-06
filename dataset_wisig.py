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



_TORCH_FROM_NUMPY_OK: Optional[bool] = None

def _safe_to_torch_float_tensor(x: Any) -> torch.Tensor:
    """Convert numpy-like IQ array to torch.float32 robustly.

    This avoids crashes such as:
        TypeError: expected np.ndarray (got numpy.ndarray)
    caused by PyTorch/NumPy C-API incompatibility. The fast path is tested once;
    if it fails, later samples skip it and use copy/list fallbacks.
    """
    global _TORCH_FROM_NUMPY_OK
    arr = np.asarray(x, dtype=np.float32, order="C")

    if _TORCH_FROM_NUMPY_OK is not False:
        try:
            t = torch.from_numpy(arr)
            _TORCH_FROM_NUMPY_OK = True
            return t.float()
        except Exception:
            _TORCH_FROM_NUMPY_OK = False

    try:
        return torch.as_tensor(arr.copy(), dtype=torch.float32)
    except Exception:
        # Last resort: avoids NumPy <-> Torch C-API interop entirely.
        return torch.tensor(arr.tolist(), dtype=torch.float32)

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
        x_t = _safe_to_torch_float_tensor(x_2t)
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
        global_index = int(self.selected[k])
        x, y, d, meta = self.base[global_index]
        if self.transform is not None:
            x = self.transform(x)
        meta = dict(meta)
        meta["split_source"] = self.split_source
        meta["global_index"] = global_index
        meta["has_tx_label"] = True
        return x, y, d, meta


class WiSigUnlabeledSubsetDataset(Dataset):
    """WiSig view for SSDG/SSL: hide TX label, keep RX/day domain and truth meta for auditing."""

    def __init__(
        self,
        base: WiSigCompactDataset,
        selected: Sequence[int],
        split_source: str = "ssdg_unlabeled_pool",
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
        global_index = int(self.selected[k])
        x, y_true, d, meta = self.base[global_index]
        if self.transform is not None:
            x = self.transform(x)
        meta = dict(meta)
        meta["split_source"] = self.split_source
        meta["global_index"] = global_index
        meta["true_tx_i"] = int(y_true)
        meta["has_tx_label"] = False
        return x, -1, d, meta


def build_unlabeled_indices_from_splits(
    pool_size: int,
    train_selected: Sequence[int],
    val_selected: Sequence[int],
) -> List[int]:
    labeled = {int(i) for i in train_selected}
    validation = {int(i) for i in val_selected}
    return [int(i) for i in range(int(pool_size)) if i not in labeled and i not in validation]


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



def _resolve_rxs(
    rx_list: Sequence[Any],
    rxs: Optional[Sequence[Union[int, str]]],
    default: Sequence[int],
) -> List[int]:
    if rxs is None:
        return list(default)
    out: List[int] = []
    for r in rxs:
        if isinstance(r, int) and 0 <= r < len(rx_list):
            out.append(int(r))
            continue
        rs = str(r)
        found = None
        for i, v in enumerate(rx_list):
            if str(v) == rs:
                found = i
                break
        if found is None:
            raise ValueError(f"Cannot resolve rx {r!r} from {list(rx_list)}")
        out.append(found)
    return out


def _build_full_subset(
    ds: Dict[str, Any],
    *,
    out_len: int,
    crop_mode: str,
    normalize: bool,
    equalized: Union[int, str],
    day_keep: Sequence[int],
    rx_keep: Sequence[int],
    domain: str,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]],
    max_samples_per_combo_test: Optional[int],
    seed: int,
    split_source: str,
):
    base = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=list(day_keep),
        rx_keep=list(rx_keep),
        domain=domain,
        transform=transform_eval,
        max_samples_per_combo=max_samples_per_combo_test,
        seed=seed,
        build_index=True,
    )
    if len(base) <= 0:
        return None
    return WiSigSubsetDataset(
        base,
        list(range(len(base))),
        split_source=split_source,
        transform=None,
    )


def make_wisig_trainval_test_by_day_rx(
    ds: Dict[str, Any],
    *,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "day",
    normalize: bool = True,
    crop_mode: str = "center",
    transform_train: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    train_ratio: float = 0.1,
    guard_gap: int = 8,
    train_days: Optional[Sequence[Union[int, str]]] = None,
    test_days: Optional[Sequence[Union[int, str]]] = None,
    train_rxs: Optional[Sequence[Union[int, str]]] = None,
    test_rxs: Optional[Sequence[Union[int, str]]] = None,
    max_samples_per_combo_day123: Optional[int] = None,
    max_samples_per_combo_test: Optional[int] = None,
    max_samples_per_combo_train: Optional[int] = None,
    max_samples_per_combo_val: Optional[int] = None,
    seed: int = 0,
):
    """
    General WiSig split with explicit day and receiver isolation.

    Train/Val:
      - built only from the intersection of train_days x train_rxs
      - each (day, tx, rx, eq) combo is split contiguously by sig_i:
          train = front train_ratio
          val   = tail after guard_gap
    Test subsets (reported separately):
      - unseen_day_seen_rx   = test_days x train_rxs
      - seen_day_unseen_rx   = train_days x test_rxs
      - unseen_day_unseen_rx = test_days x test_rxs

    Notes:
      1) train_days and train_rxs are made exclusive against test_days / test_rxs.
      2) This allows isolating cross-day and cross-receiver performance separately.
    """
    day_list = list(ds.get("capture_date_list", []))
    rx_list = list(ds.get("rx_list", []))
    n_day = len(day_list)
    n_rx = len(rx_list)
    train_ratio = float(train_ratio)
    if not (0.01 <= train_ratio <= 0.99):
        raise ValueError(f"train_ratio must be in [0.01, 0.99], got {train_ratio}")

    if n_day < 1:
        raise ValueError("WiSig split requires at least 1 day.")
    if n_rx < 1:
        raise ValueError("WiSig split requires at least 1 receiver.")

    default_train_days = list(range(min(3, max(1, n_day - 1))))
    default_test_days = [min(3, n_day - 1)] if n_day >= 4 else [n_day - 1]
    default_train_rxs = list(range(n_rx))
    default_test_rxs: List[int] = []

    train_day_idx = _resolve_days(day_list, train_days, default_train_days)
    test_day_idx = _resolve_days(day_list, test_days, default_test_days)
    train_rx_idx = _resolve_rxs(rx_list, train_rxs, default_train_rxs)
    test_rx_idx = _resolve_rxs(rx_list, test_rxs, default_test_rxs)

    train_day_idx = [d for d in train_day_idx if d not in test_day_idx]
    train_rx_idx = [r for r in train_rx_idx if r not in test_rx_idx]

    if len(train_day_idx) == 0:
        raise ValueError("No train days left after removing test days.")
    if len(train_rx_idx) == 0:
        raise ValueError("No train receivers left after removing test receivers.")

    base_trainval = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=train_day_idx,
        rx_keep=train_rx_idx,
        domain=domain,
        transform=None,
        max_samples_per_combo=max_samples_per_combo_day123,
        seed=seed,
        build_index=True,
    )

    groups = _build_grouped_indices_sorted_by_sig(base_trainval)
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

    train_sel = _cap_per_group_keep_order(train_sel, base_trainval, max_samples_per_combo_train)
    val_sel = _cap_per_group_keep_order(val_sel, base_trainval, max_samples_per_combo_val)

    train_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(train_sel),
        split_source="contiguous_front_from_train_days_train_rxs",
        transform=transform_train,
    )
    val_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(val_sel),
        split_source="contiguous_tail_from_train_days_train_rxs",
        transform=transform_eval,
    )

    named_tests: Dict[str, Dataset] = {}
    named_test_meta: Dict[str, Dict[str, Any]] = {}

    if len(test_day_idx) > 0:
        ds_day = _build_full_subset(
            ds,
            out_len=out_len,
            crop_mode=crop_mode,
            normalize=normalize,
            equalized=equalized,
            day_keep=test_day_idx,
            rx_keep=train_rx_idx,
            domain=domain,
            transform_eval=transform_eval,
            max_samples_per_combo_test=max_samples_per_combo_test,
            seed=seed,
            split_source="full_test_unseen_day_seen_rx",
        )
        if ds_day is not None and len(ds_day) > 0:
            named_tests["test_unseen_day_seen_rx"] = ds_day
            named_test_meta["test_unseen_day_seen_rx"] = {
                "days_idx": list(test_day_idx),
                "days_label": [day_list[i] for i in test_day_idx],
                "rxs_idx": list(train_rx_idx),
                "rxs_label": [rx_list[i] for i in train_rx_idx],
                "size": len(ds_day),
            }

        for d_idx in test_day_idx:
            ds_one = _build_full_subset(
                ds,
                out_len=out_len,
                crop_mode=crop_mode,
                normalize=normalize,
                equalized=equalized,
                day_keep=[d_idx],
                rx_keep=train_rx_idx,
                domain=domain,
                transform_eval=transform_eval,
                max_samples_per_combo_test=max_samples_per_combo_test,
                seed=seed,
                split_source=f"full_test_day_{d_idx}_seen_rx",
            )
            if ds_one is not None and len(ds_one) > 0:
                key = f"test_day_{d_idx}"
                named_tests[key] = ds_one
                named_test_meta[key] = {
                    "days_idx": [d_idx],
                    "days_label": [day_list[d_idx]],
                    "rxs_idx": list(train_rx_idx),
                    "rxs_label": [rx_list[i] for i in train_rx_idx],
                    "size": len(ds_one),
                }

    if len(test_rx_idx) > 0:
        ds_rx = _build_full_subset(
            ds,
            out_len=out_len,
            crop_mode=crop_mode,
            normalize=normalize,
            equalized=equalized,
            day_keep=train_day_idx,
            rx_keep=test_rx_idx,
            domain=domain,
            transform_eval=transform_eval,
            max_samples_per_combo_test=max_samples_per_combo_test,
            seed=seed,
            split_source="full_test_seen_day_unseen_rx",
        )
        if ds_rx is not None and len(ds_rx) > 0:
            named_tests["test_seen_day_unseen_rx"] = ds_rx
            named_test_meta["test_seen_day_unseen_rx"] = {
                "days_idx": list(train_day_idx),
                "days_label": [day_list[i] for i in train_day_idx],
                "rxs_idx": list(test_rx_idx),
                "rxs_label": [rx_list[i] for i in test_rx_idx],
                "size": len(ds_rx),
            }

        for r_idx in test_rx_idx:
            ds_one = _build_full_subset(
                ds,
                out_len=out_len,
                crop_mode=crop_mode,
                normalize=normalize,
                equalized=equalized,
                day_keep=train_day_idx,
                rx_keep=[r_idx],
                domain=domain,
                transform_eval=transform_eval,
                max_samples_per_combo_test=max_samples_per_combo_test,
                seed=seed,
                split_source=f"full_test_train_day_rx_{r_idx}",
            )
            if ds_one is not None and len(ds_one) > 0:
                key = f"test_rx_{r_idx}"
                named_tests[key] = ds_one
                named_test_meta[key] = {
                    "days_idx": list(train_day_idx),
                    "days_label": [day_list[i] for i in train_day_idx],
                    "rxs_idx": [r_idx],
                    "rxs_label": [rx_list[r_idx]],
                    "size": len(ds_one),
                }

    if len(test_day_idx) > 0 and len(test_rx_idx) > 0:
        ds_day_rx = _build_full_subset(
            ds,
            out_len=out_len,
            crop_mode=crop_mode,
            normalize=normalize,
            equalized=equalized,
            day_keep=test_day_idx,
            rx_keep=test_rx_idx,
            domain=domain,
            transform_eval=transform_eval,
            max_samples_per_combo_test=max_samples_per_combo_test,
            seed=seed,
            split_source="full_test_unseen_day_unseen_rx",
        )
        if ds_day_rx is not None and len(ds_day_rx) > 0:
            named_tests["test_unseen_day_unseen_rx"] = ds_day_rx
            named_test_meta["test_unseen_day_unseen_rx"] = {
                "days_idx": list(test_day_idx),
                "days_label": [day_list[i] for i in test_day_idx],
                "rxs_idx": list(test_rx_idx),
                "rxs_label": [rx_list[i] for i in test_rx_idx],
                "size": len(ds_day_rx),
            }

    test_parts = [named_tests[k] for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_tests]
    if len(test_parts) == 0:
        raise ValueError("No test subsets were created. Please check test_days/test_rxs.")
    test_ds = WiSigConcatDataset(test_parts) if len(test_parts) > 1 else test_parts[0]

    info = {
        "mode": "contiguous_trainval_with_explicit_day_rx_isolation",
        "train_days_idx": train_day_idx,
        "train_days_label": [day_list[i] for i in train_day_idx],
        "test_days_idx": test_day_idx,
        "test_days_label": [day_list[i] for i in test_day_idx],
        "train_rxs_idx": train_rx_idx,
        "train_rxs_label": [rx_list[i] for i in train_rx_idx],
        "test_rxs_idx": test_rx_idx,
        "test_rxs_label": [rx_list[i] for i in test_rx_idx],
        "train_ratio": float(train_ratio),
        "requested_val_ratio": float(1.0 - train_ratio),
        "guard_gap": int(guard_gap),
        "trainval_pool_size": len(base_trainval),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "effective_train_ratio": float(len(train_ds) / max(1, len(base_trainval))),
        "effective_val_ratio": float(len(val_ds) / max(1, len(base_trainval))),
        "test_size": len(test_ds),
        "dropped_gap_size": len(dropped_gap_sel),
        "max_samples_per_combo_day123": max_samples_per_combo_day123,
        "max_samples_per_combo_train": max_samples_per_combo_train,
        "max_samples_per_combo_val": max_samples_per_combo_val,
        "max_samples_per_combo_test": max_samples_per_combo_test,
        "named_test_sizes": {k: len(v) for k, v in named_tests.items()},
    }
    return train_ds, val_ds, test_ds, named_tests, named_test_meta, info


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
    Backward-compatible wrapper around the more general day/rx split.
    Keeps the historical API returning only train/val/test/info.
    """
    train_ds, val_ds, test_ds, _, _, info = make_wisig_trainval_test_by_day_rx(
        ds,
        equalized=equalized,
        out_len=out_len,
        domain=domain,
        normalize=normalize,
        crop_mode=crop_mode,
        transform_train=transform_train,
        transform_eval=transform_eval,
        train_ratio=train_ratio,
        guard_gap=guard_gap,
        train_days=train_days,
        test_days=test_days,
        train_rxs=None,
        test_rxs=None,
        max_samples_per_combo_day123=max_samples_per_combo_day123,
        max_samples_per_combo_test=max_samples_per_combo_test,
        max_samples_per_combo_train=max_samples_per_combo_train,
        max_samples_per_combo_val=max_samples_per_combo_val,
        seed=seed,
    )
    return train_ds, val_ds, test_ds, info
