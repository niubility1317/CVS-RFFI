from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


MIN_WISIG_TRAIN_RATIO = 0.001
MAX_WISIG_TRAIN_RATIO = 0.99


def _normalize_strategy(value: str, *, name: str, allowed: Sequence[str]) -> str:
    strategy = str(value or "").strip().lower()
    if strategy not in set(allowed):
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}")
    return strategy


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
        sample_strategy: str = "front",
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
        self.sample_strategy = _normalize_strategy(
            sample_strategy,
            name="sample_strategy",
            allowed=("front", "random"),
        )
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
                            limit = int(self.max_samples_per_combo)
                            if self.sample_strategy == "random":
                                keep = np.sort(self.rng.permutation(n)[:limit].astype(np.int64))
                            else:
                                keep = np.arange(limit, dtype=np.int64)
                            for s in keep.tolist():
                                idx.append(WiSigIndex(tx_i, rx_i, day_i, eq_i, int(s)))
                            continue
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
            "base_index": int(k),
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


class WiSigMetaSslSubsetDataset(WiSigSubsetDataset):
    """WiSig subset with explicit Meta-SSL role and TX-label masking.

    The underlying sample metadata keeps the original transmitter id for audit,
    but unlabeled-source samples return y=-1 so ordinary TX CE paths cannot
    accidentally consume masked labels.
    """

    def __init__(
        self,
        base: WiSigCompactDataset,
        selected: Sequence[int],
        split_source: str,
        *,
        role: str,
        tx_label_visible: bool,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(base, selected, split_source=split_source, transform=transform)
        self.role = str(role)
        self.tx_label_visible = bool(tx_label_visible)

    def __getitem__(self, k: int):
        x, y, d, meta = super().__getitem__(k)
        meta = dict(meta)
        meta["meta_ssl_role"] = self.role
        meta["tx_label_visible"] = self.tx_label_visible
        if not self.tx_label_visible:
            meta["true_tx_i"] = int(y)
            y = -1
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


def _build_grouped_indices_by_tx_rx_eq_sorted(
    base: WiSigCompactDataset,
) -> Dict[Tuple[int, int, int], List[int]]:
    groups: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    for global_i, it in enumerate(base.index):
        key = (int(it.tx_i), int(it.rx_i), int(it.eq_i))
        groups.setdefault(key, []).append((int(it.day_i), int(it.sig_i), int(global_i)))

    out: Dict[Tuple[int, int, int], List[int]] = {}
    for key, triples in groups.items():
        triples = sorted(triples, key=lambda z: (z[0], z[1], z[2]))
        out[key] = [gidx for _, _, gidx in triples]
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
    train_ratio = min(MAX_WISIG_TRAIN_RATIO, max(MIN_WISIG_TRAIN_RATIO, train_ratio))
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


def _random_train_val_split(
    idxs_sorted: Sequence[int],
    train_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[int]]:
    n = len(idxs_sorted)
    if n <= 0:
        return [], [], []
    if n == 1:
        return [int(idxs_sorted[0])], [], []

    train_ratio = float(train_ratio)
    train_ratio = min(MAX_WISIG_TRAIN_RATIO, max(MIN_WISIG_TRAIN_RATIO, train_ratio))
    n_tr = int(round(n * train_ratio))
    n_tr = min(max(1, n_tr), n - 1)

    perm = rng.permutation(n).tolist()
    train_pos = set(int(i) for i in perm[:n_tr])
    val_pos = set(range(n)).difference(train_pos)
    train_ids = [int(idxs_sorted[i]) for i in sorted(train_pos)]
    val_ids = [int(idxs_sorted[i]) for i in sorted(val_pos)]

    if len(val_ids) == 0 and len(train_ids) >= 2:
        val_ids = [train_ids.pop()]
    return train_ids, val_ids, []


def _cap_per_group_keep_order(
    selected: Sequence[int],
    base: WiSigCompactDataset,
    max_per_combo: Optional[int],
    *,
    strategy: str = "front",
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if max_per_combo is None or max_per_combo <= 0:
        return list(selected)
    strategy = _normalize_strategy(strategy, name="cap_strategy", allowed=("front", "random"))
    if rng is None:
        rng = np.random.default_rng(0)

    selected_set = set(int(x) for x in selected)
    groups: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}
    for global_i, it in enumerate(base.index):
        if global_i not in selected_set:
            continue
        key = (int(it.day_i), int(it.tx_i), int(it.rx_i), int(it.eq_i))
        groups.setdefault(key, []).append((int(it.sig_i), int(global_i)))

    out: List[int] = []
    for key in sorted(groups):
        pairs = sorted(groups[key], key=lambda z: z[0])
        if strategy == "random" and len(pairs) > int(max_per_combo):
            pick = rng.permutation(len(pairs))[: int(max_per_combo)].tolist()
            chosen = sorted([pairs[int(i)] for i in pick], key=lambda z: z[0])
        else:
            chosen = pairs[: int(max_per_combo)]
        out.extend([gidx for _, gidx in chosen])
    return sorted(out)


def _cap_per_class_keep_order(
    selected: Sequence[int],
    base: WiSigCompactDataset,
    max_per_class: Optional[int],
    *,
    strategy: str = "domain_balanced",
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if max_per_class is None or max_per_class <= 0:
        return list(selected)
    strategy = _normalize_strategy(
        strategy,
        name="train_class_cap_strategy",
        allowed=("domain_balanced", "rx_day_balanced", "random", "front"),
    )
    if rng is None:
        rng = np.random.default_rng(0)

    selected_set = set(int(x) for x in selected)
    by_tx: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for global_i, it in enumerate(base.index):
        if global_i not in selected_set:
            continue
        by_tx.setdefault(int(it.tx_i), []).append(
            (
                int(it.rx_i),
                int(it.day_i),
                int(it.eq_i),
                int(it.sig_i),
                int(global_i),
                int(it.tx_i),
            )
        )

    out: List[int] = []
    limit = int(max_per_class)
    for tx_i in sorted(by_tx):
        entries = sorted(by_tx[tx_i], key=lambda z: (z[1], z[0], z[2], z[3], z[4]))
        if len(entries) <= limit:
            out.extend([entry[4] for entry in entries])
            continue

        if strategy == "random":
            pick = rng.permutation(len(entries))[:limit].tolist()
            chosen = [entries[int(i)] for i in pick]
        elif strategy == "front":
            chosen = entries[:limit]
        else:
            domain_groups: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int, int, int]]] = {}
            for entry in entries:
                rx_i, day_i, eq_i, *_ = entry
                domain_groups.setdefault((int(day_i), int(rx_i), int(eq_i)), []).append(entry)

            domain_keys = sorted(domain_groups)
            if strategy == "rx_day_balanced":
                days = sorted({key[0] for key in domain_keys})
                rxs = sorted({key[1] for key in domain_keys})
                eqs = sorted({key[2] for key in domain_keys})
                if len(days) > 1:
                    days = [days[int(i)] for i in rng.permutation(len(days)).tolist()]
                if len(rxs) > 1:
                    rxs = [rxs[int(i)] for i in rng.permutation(len(rxs)).tolist()]
                domain_order = []
                seen = set()
                max_rounds = max(1, len(domain_keys) * max(1, len(days)) * max(1, len(rxs)))
                step = 0
                while len(domain_order) < len(domain_keys) and step < max_rounds:
                    day_i = days[step % max(1, len(days))]
                    rx_i = rxs[step % max(1, len(rxs))]
                    for eq_i in eqs:
                        key = (int(day_i), int(rx_i), int(eq_i))
                        if key in domain_groups and key not in seen:
                            domain_order.append(key)
                            seen.add(key)
                    step += 1
                for key in domain_keys:
                    if key not in seen:
                        domain_order.append(key)
            else:
                domain_order = [domain_keys[int(i)] for i in rng.permutation(len(domain_keys)).tolist()]
            queues: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int, int, int]]] = {}
            for key in domain_order:
                vals = sorted(domain_groups[key], key=lambda z: z[3])
                if len(vals) > 1:
                    order = rng.permutation(len(vals)).tolist()
                    vals = [vals[int(i)] for i in order]
                queues[key] = vals

            chosen = []
            while len(chosen) < limit and any(queues[key] for key in domain_order):
                for key in domain_order:
                    if queues[key]:
                        chosen.append(queues[key].pop(0))
                        if len(chosen) >= limit:
                            break

        chosen = sorted(chosen, key=lambda z: (z[1], z[0], z[2], z[3], z[4]))
        out.extend([entry[4] for entry in chosen])
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


def _meta_ssl_partition_group(
    idxs_sorted: Sequence[int],
    *,
    labeled_ratio: float,
    unlabeled_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[int]]:
    n = len(idxs_sorted)
    if n <= 0:
        return [], [], []
    if n == 1:
        return [int(idxs_sorted[0])], [], []
    if n == 2:
        return [int(idxs_sorted[0])], [], [int(idxs_sorted[1])]

    ratios = np.asarray([labeled_ratio, unlabeled_ratio, val_ratio], dtype=np.float64)
    if np.any(ratios <= 0):
        raise ValueError("Meta-SSL split ratios must all be positive.")
    ratios = ratios / ratios.sum()

    counts = np.floor(ratios * n).astype(np.int64)
    counts = np.maximum(counts, 1)
    while int(counts.sum()) > n:
        candidates = np.where(counts > 1)[0]
        if candidates.size == 0:
            break
        largest = candidates[np.argmax(counts[candidates])]
        counts[largest] -= 1
    while int(counts.sum()) < n:
        largest_ratio = int(np.argmax(ratios))
        counts[largest_ratio] += 1

    perm = rng.permutation(n).tolist()
    ordered = [int(idxs_sorted[int(i)]) for i in perm]
    n_l, n_u, n_v = [int(x) for x in counts.tolist()]
    labeled = sorted(ordered[:n_l])
    unlabeled = sorted(ordered[n_l:n_l + n_u])
    source_val = sorted(ordered[n_l + n_u:n_l + n_u + n_v])
    return labeled, unlabeled, source_val


def make_wisig_meta_ssl_source_split(
    ds: Dict[str, Any],
    *,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "rx_day",
    normalize: bool = True,
    crop_mode: str = "center",
    transform_labeled: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_unlabeled: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_val: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    labeled_ratio: float = 0.1,
    unlabeled_ratio: float = 0.7,
    val_ratio: float = 0.2,
    train_days: Optional[Sequence[Union[int, str]]] = None,
    holdout_days: Optional[Sequence[Union[int, str]]] = None,
    train_rxs: Optional[Sequence[Union[int, str]]] = None,
    holdout_rxs: Optional[Sequence[Union[int, str]]] = None,
    max_samples_per_combo_source: Optional[int] = None,
    seed: int = 0,
    sample_strategy: str = "front",
):
    """Build the source-only 0.1L/0.7U/0.2Val Meta-SSL-CVS split.

    Only source train day/rx combinations are used. Optional holdout day/rx
    values are removed before the split so target/deployment samples cannot
    enter pseudo-label fitting, prototype warm start, or early stopping.
    """
    day_list = list(ds.get("capture_date_list", []))
    rx_list = list(ds.get("rx_list", []))
    n_day = len(day_list)
    n_rx = len(rx_list)
    if n_day < 1:
        raise ValueError("WiSig Meta-SSL split requires at least one day.")
    if n_rx < 1:
        raise ValueError("WiSig Meta-SSL split requires at least one receiver.")

    sample_strategy = _normalize_strategy(
        sample_strategy,
        name="meta_ssl_sample_strategy",
        allowed=("front", "random"),
    )
    default_train_days = list(range(min(3, max(1, n_day - 1))))
    default_holdout_days = [min(3, n_day - 1)] if n_day >= 4 else []
    default_train_rxs = list(range(n_rx))
    default_holdout_rxs: List[int] = []

    train_day_idx = _resolve_days(day_list, train_days, default_train_days)
    holdout_day_idx = _resolve_days(day_list, holdout_days, default_holdout_days)
    train_rx_idx = _resolve_rxs(rx_list, train_rxs if train_rxs is not None else default_train_rxs, [])
    holdout_rx_idx = _resolve_rxs(rx_list, holdout_rxs if holdout_rxs is not None else default_holdout_rxs, [])

    train_day_idx = [d for d in train_day_idx if d not in holdout_day_idx]
    train_rx_idx = [r for r in train_rx_idx if r not in holdout_rx_idx]
    if not train_day_idx:
        raise ValueError("No Meta-SSL source train days left after holdout removal.")
    if not train_rx_idx:
        raise ValueError("No Meta-SSL source train receivers left after holdout removal.")

    base = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=train_day_idx,
        rx_keep=train_rx_idx,
        domain=domain,
        transform=None,
        max_samples_per_combo=max_samples_per_combo_source,
        sample_strategy=sample_strategy,
        seed=seed,
        build_index=True,
    )
    groups = _build_grouped_indices_sorted_by_sig(base)
    rng = np.random.default_rng(int(seed))
    labeled_sel: List[int] = []
    unlabeled_sel: List[int] = []
    val_sel: List[int] = []
    for _, idxs_sorted in sorted(groups.items()):
        lab, unl, val = _meta_ssl_partition_group(
            idxs_sorted,
            labeled_ratio=float(labeled_ratio),
            unlabeled_ratio=float(unlabeled_ratio),
            val_ratio=float(val_ratio),
            rng=rng,
        )
        labeled_sel.extend(lab)
        unlabeled_sel.extend(unl)
        val_sel.extend(val)

    labeled_ds = WiSigMetaSslSubsetDataset(
        base,
        sorted(labeled_sel),
        split_source="meta_ssl_labeled_train_0p1_source",
        role="labeled_train",
        tx_label_visible=True,
        transform=transform_labeled,
    )
    unlabeled_ds = WiSigMetaSslSubsetDataset(
        base,
        sorted(unlabeled_sel),
        split_source="meta_ssl_unlabeled_source_0p7_tx_masked",
        role="unlabeled_source",
        tx_label_visible=False,
        transform=transform_unlabeled,
    )
    source_val_ds = WiSigMetaSslSubsetDataset(
        base,
        sorted(val_sel),
        split_source="meta_ssl_source_val_0p2_tx_visible_eval_only",
        role="source_val",
        tx_label_visible=True,
        transform=transform_val,
    )

    sets = [set(map(int, labeled_sel)), set(map(int, unlabeled_sel)), set(map(int, val_sel))]
    overlap_count = len((sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2]))
    total = len(base)
    info = {
        "mode": "meta_ssl_source_only_0p1L_0p7U_0p2Val",
        "source_ssl_split": "0.1L/0.7U/0.2Val",
        "ground_dg_claim_scope": "source_only",
        "labeled_ratio": float(labeled_ratio),
        "unlabeled_ratio": float(unlabeled_ratio),
        "val_ratio": float(val_ratio),
        "train_days_idx": train_day_idx,
        "train_days_label": [day_list[i] for i in train_day_idx],
        "holdout_days_idx": holdout_day_idx,
        "train_rxs_idx": train_rx_idx,
        "train_rxs_label": [rx_list[i] for i in train_rx_idx],
        "holdout_rxs_idx": holdout_rx_idx,
        "base_source_pool_size": int(total),
        "labeled_size": len(labeled_ds),
        "unlabeled_size": len(unlabeled_ds),
        "source_val_size": len(source_val_ds),
        "effective_labeled_ratio": float(len(labeled_ds) / max(1, total)),
        "effective_unlabeled_ratio": float(len(unlabeled_ds) / max(1, total)),
        "effective_val_ratio": float(len(source_val_ds) / max(1, total)),
        "group_count": len(groups),
        "overlap_count": int(overlap_count),
        "tx_label_policy": {
            "labeled_train": "visible",
            "unlabeled_source": "masked_y_minus_1_true_tx_in_meta_only",
            "source_val": "visible_eval_only",
        },
        "seed": int(seed),
    }
    return labeled_ds, unlabeled_ds, source_val_ds, info


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
    sample_strategy: str = "front",
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
        sample_strategy=sample_strategy,
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
    max_samples_per_class_train: Optional[int] = None,
    seed: int = 0,
    split_strategy: str = "random",
    cap_strategy: str = "random",
    train_class_cap_strategy: str = "domain_balanced",
):
    """
    General WiSig split with explicit day and receiver isolation.

    Train/Val:
      - built only from the intersection of train_days x train_rxs
      - each (day, tx, rx, eq) combo is split by sig_i using split_strategy:
          random     = seed-controlled random train/val membership
          contiguous = legacy front train_ratio / tail validation after guard_gap
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
    if not (MIN_WISIG_TRAIN_RATIO <= train_ratio <= MAX_WISIG_TRAIN_RATIO):
        raise ValueError(
            f"train_ratio must be in [{MIN_WISIG_TRAIN_RATIO}, {MAX_WISIG_TRAIN_RATIO}], got {train_ratio}"
        )
    split_strategy = _normalize_strategy(
        split_strategy,
        name="split_strategy",
        allowed=("random", "contiguous"),
    )
    cap_strategy = _normalize_strategy(
        cap_strategy,
        name="cap_strategy",
        allowed=("random", "front"),
    )
    train_class_cap_strategy = _normalize_strategy(
        train_class_cap_strategy,
        name="train_class_cap_strategy",
        allowed=("domain_balanced", "rx_day_balanced", "random", "front"),
    )

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
    train_rx_idx = _resolve_rxs(rx_list, train_rxs if train_rxs is not None else default_train_rxs, [])
    test_rx_idx = _resolve_rxs(rx_list, test_rxs if test_rxs is not None else default_test_rxs, [])

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
        sample_strategy=cap_strategy,
        seed=seed,
        build_index=True,
    )

    groups = _build_grouped_indices_sorted_by_sig(base_trainval)
    split_rng = np.random.default_rng(int(seed))
    train_cap_rng = np.random.default_rng(int(seed) + 104729)
    val_cap_rng = np.random.default_rng(int(seed) + 130363)
    train_class_cap_rng = np.random.default_rng(int(seed) + 15485863)
    train_sel: List[int] = []
    val_sel: List[int] = []
    dropped_gap_sel: List[int] = []

    for _, idxs_sorted in sorted(groups.items()):
        if split_strategy == "random":
            tr_ids, va_ids, gap_ids = _random_train_val_split(
                idxs_sorted=idxs_sorted,
                train_ratio=train_ratio,
                rng=split_rng,
            )
        else:
            tr_ids, va_ids, gap_ids = _contiguous_train_val_split(
                idxs_sorted=idxs_sorted,
                train_ratio=train_ratio,
                guard_gap=guard_gap,
            )
        train_sel.extend(tr_ids)
        val_sel.extend(va_ids)
        dropped_gap_sel.extend(gap_ids)

    train_sel = _cap_per_group_keep_order(
        train_sel,
        base_trainval,
        max_samples_per_combo_train,
        strategy=cap_strategy,
        rng=train_cap_rng,
    )
    train_sel = _cap_per_class_keep_order(
        train_sel,
        base_trainval,
        max_samples_per_class_train,
        strategy=train_class_cap_strategy,
        rng=train_class_cap_rng,
    )
    val_sel = _cap_per_group_keep_order(
        val_sel,
        base_trainval,
        max_samples_per_combo_val,
        strategy=cap_strategy,
        rng=val_cap_rng,
    )
    train_source = (
        "random_seeded_from_train_days_train_rxs"
        if split_strategy == "random"
        else "contiguous_front_from_train_days_train_rxs"
    )
    val_source = (
        "random_seeded_from_train_days_train_rxs_validation"
        if split_strategy == "random"
        else "contiguous_tail_from_train_days_train_rxs"
    )

    train_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(train_sel),
        split_source=train_source,
        transform=transform_train,
    )
    val_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(val_sel),
        split_source=val_source,
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
            sample_strategy=cap_strategy,
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
                sample_strategy=cap_strategy,
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
            sample_strategy=cap_strategy,
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
                sample_strategy=cap_strategy,
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
            sample_strategy=cap_strategy,
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

        for r_idx in test_rx_idx:
            ds_one = _build_full_subset(
                ds,
                out_len=out_len,
                crop_mode=crop_mode,
                normalize=normalize,
                equalized=equalized,
                day_keep=test_day_idx,
                rx_keep=[r_idx],
                domain=domain,
                transform_eval=transform_eval,
                max_samples_per_combo_test=max_samples_per_combo_test,
                seed=seed,
                split_source=f"full_test_unseen_day_rx_{r_idx}",
                sample_strategy=cap_strategy,
            )
            if ds_one is not None and len(ds_one) > 0:
                key = f"test_unseen_day_rx_{r_idx}"
                named_tests[key] = ds_one
                named_test_meta[key] = {
                    "days_idx": list(test_day_idx),
                    "days_label": [day_list[i] for i in test_day_idx],
                    "rxs_idx": [r_idx],
                    "rxs_label": [rx_list[r_idx]],
                    "size": len(ds_one),
                }

    test_parts = [named_tests[k] for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_tests]
    if len(test_parts) == 0:
        raise ValueError("No test subsets were created. Please check test_days/test_rxs.")
    test_ds = WiSigConcatDataset(test_parts) if len(test_parts) > 1 else test_parts[0]

    info = {
        "mode": f"{split_strategy}_trainval_with_explicit_day_rx_isolation",
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
        "effective_guard_gap": int(guard_gap) if split_strategy == "contiguous" else 0,
        "guard_gap_active": bool(split_strategy == "contiguous"),
        "seed": int(seed),
        "split_strategy": split_strategy,
        "cap_strategy": cap_strategy,
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
        "max_samples_per_class_train": max_samples_per_class_train,
        "train_class_cap_strategy": train_class_cap_strategy,
        "named_test_sizes": {k: len(v) for k, v in named_tests.items()},
    }
    return train_ds, val_ds, test_ds, named_tests, named_test_meta, info


def make_wisig_drift_day1_split(
    ds: Dict[str, Any],
    *,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "rx",
    normalize: bool = True,
    crop_mode: str = "center",
    transform_train: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    day: Union[int, str] = 0,
    train_rxs: Optional[Sequence[Union[int, str]]] = None,
    test_rxs: Optional[Sequence[Union[int, str]]] = None,
    train_samples_per_combo: int = 800,
    val_samples_per_combo: int = 200,
    test_samples_per_combo: int = 200,
    seed: int = 0,
):
    """DRIFT paper WiSig Day1 receiver-disjoint protocol.

    The paper protocol uses WiSig/ManySig Day 1, training receivers
    {1-1, 14-7, 7-7}, test receivers
    {1-19, 19-2, 2-1, 2-19, 20-1, 7-14, 8-8}, 800 train samples per
    transmitter/receiver pair, and 200 test samples per pair.
    """

    day_list = list(ds.get("capture_date_list", []))
    rx_list = list(ds.get("rx_list", []))
    if not day_list:
        raise ValueError("WiSig DRIFT split requires at least one day.")
    if not rx_list:
        raise ValueError("WiSig DRIFT split requires receiver labels.")

    default_train_rxs = ["1-1", "14-7", "7-7"]
    default_test_rxs = ["1-19", "19-2", "2-1", "2-19", "20-1", "7-14", "8-8"]
    day_idx = _resolve_days(day_list, [day], [0])
    train_rx_idx = _resolve_rxs(rx_list, train_rxs if train_rxs is not None else default_train_rxs, [])
    test_rx_idx = _resolve_rxs(rx_list, test_rxs if test_rxs is not None else default_test_rxs, [])

    overlap = sorted(set(train_rx_idx).intersection(test_rx_idx))
    if overlap:
        raise ValueError(f"DRIFT train/test receiver sets must be disjoint, overlap={overlap}")

    base_trainval = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=day_idx,
        rx_keep=train_rx_idx,
        domain=domain,
        transform=None,
        max_samples_per_combo=None,
        seed=seed,
        build_index=True,
    )
    groups = _build_grouped_indices_sorted_by_sig(base_trainval)
    train_n = max(1, int(train_samples_per_combo))
    val_n = max(0, int(val_samples_per_combo))
    train_sel: List[int] = []
    val_sel: List[int] = []
    for _, idxs_sorted in groups.items():
        train_sel.extend(idxs_sorted[:train_n])
        if val_n > 0:
            val_sel.extend(idxs_sorted[train_n:train_n + val_n])

    if not train_sel:
        raise ValueError("DRIFT split produced an empty train set.")
    if not val_sel:
        # The trainers need a validation loader; reserve a tiny deterministic
        # tail only when the dataset has fewer samples than the paper subset.
        fallback = []
        for _, idxs_sorted in groups.items():
            if len(idxs_sorted) > 1:
                fallback.append(idxs_sorted[-1])
        val_sel = fallback
        train_sel = [idx for idx in train_sel if idx not in set(val_sel)]
    if not val_sel:
        raise ValueError("DRIFT split produced an empty validation set.")

    train_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(train_sel),
        split_source="drift_day1_first_800_train_rx",
        transform=transform_train,
    )
    val_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(val_sel),
        split_source="drift_day1_tail_train_rx_validation",
        transform=transform_eval,
    )

    test_ds = _build_full_subset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=day_idx,
        rx_keep=test_rx_idx,
        domain=domain,
        transform_eval=transform_eval,
        max_samples_per_combo_test=max(1, int(test_samples_per_combo)),
        seed=seed,
        split_source="drift_day1_first_200_test_rx",
    )
    if test_ds is None or len(test_ds) == 0:
        raise ValueError("DRIFT split produced an empty test set.")

    named_tests: Dict[str, Dataset] = {
        "test_seen_day_unseen_rx": test_ds,
        "test_unseen_rx_day1": test_ds,
    }
    named_test_meta: Dict[str, Dict[str, Any]] = {
        "test_seen_day_unseen_rx": {
            "days_idx": list(day_idx),
            "days_label": [day_list[i] for i in day_idx],
            "rxs_idx": list(test_rx_idx),
            "rxs_label": [rx_list[i] for i in test_rx_idx],
            "size": len(test_ds),
            "paper_protocol_alias": "drift_day1_unseen_receiver",
        },
        "test_unseen_rx_day1": {
            "days_idx": list(day_idx),
            "days_label": [day_list[i] for i in day_idx],
            "rxs_idx": list(test_rx_idx),
            "rxs_label": [rx_list[i] for i in test_rx_idx],
            "size": len(test_ds),
            "paper_protocol_alias": "drift_day1_unseen_receiver",
        },
    }
    for r_idx in test_rx_idx:
        ds_one = _build_full_subset(
            ds,
            out_len=out_len,
            crop_mode=crop_mode,
            normalize=normalize,
            equalized=equalized,
            day_keep=day_idx,
            rx_keep=[r_idx],
            domain=domain,
            transform_eval=transform_eval,
            max_samples_per_combo_test=max(1, int(test_samples_per_combo)),
            seed=seed,
            split_source=f"drift_day1_first_200_test_rx_{r_idx}",
        )
        if ds_one is not None and len(ds_one) > 0:
            key = f"test_rx_{r_idx}"
            named_tests[key] = ds_one
            named_test_meta[key] = {
                "days_idx": list(day_idx),
                "days_label": [day_list[i] for i in day_idx],
                "rxs_idx": [r_idx],
                "rxs_label": [rx_list[r_idx]],
                "size": len(ds_one),
            }

    info = {
        "mode": "wisig_drift_day1_receiver_disjoint_800_200",
        "paper": "DRIFT",
        "paper_protocol": "WiSig ManySig Day1 train_rx/test_rx disjoint",
        "day_idx": list(day_idx),
        "day_label": [day_list[i] for i in day_idx],
        "train_days_idx": list(day_idx),
        "train_days_label": [day_list[i] for i in day_idx],
        "test_days_idx": list(day_idx),
        "test_days_label": [day_list[i] for i in day_idx],
        "train_rxs_idx": list(train_rx_idx),
        "train_rxs_label": [rx_list[i] for i in train_rx_idx],
        "test_rxs_idx": list(test_rx_idx),
        "test_rxs_label": [rx_list[i] for i in test_rx_idx],
        "train_samples_per_combo": int(train_samples_per_combo),
        "val_samples_per_combo": int(val_samples_per_combo),
        "test_samples_per_combo": int(test_samples_per_combo),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "named_test_sizes": {k: len(v) for k, v in named_tests.items()},
        "primary_named_test": "test_seen_day_unseen_rx",
        "strict_named_test": "test_seen_day_unseen_rx",
        "aggregate_test_keys": ["test_seen_day_unseen_rx"],
    }
    return train_ds, val_ds, test_ds, named_tests, named_test_meta, info


def make_wisig_riei_receiver_holdout_split(
    ds: Dict[str, Any],
    *,
    equalized: Union[int, str] = 1,
    out_len: int = 256,
    domain: str = "rx",
    normalize: bool = True,
    crop_mode: str = "center",
    transform_train: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    train_rxs: Optional[Sequence[Union[int, str]]] = None,
    test_rxs: Optional[Sequence[Union[int, str]]] = None,
    train_samples_per_combo: int = 2400,
    val_samples_per_combo: int = 800,
    test_samples_per_combo: int = 800,
    seed: int = 0,
):
    """RIEI paper WiSig two-source-receiver holdout protocol.

    The RIEI WiSig table trains on two receivers and tests on a disjoint
    receiver. With six transmitters, the reported 14400 training samples per
    receiver and 4800 testing samples per receiver correspond to 2400 and 800
    samples per transmitter/receiver pair, respectively.
    """

    day_list = list(ds.get("capture_date_list", []))
    rx_list = list(ds.get("rx_list", []))
    if not day_list:
        raise ValueError("WiSig RIEI split requires at least one day.")
    if not rx_list:
        raise ValueError("WiSig RIEI split requires receiver labels.")

    default_train_rxs = ["1-1", "7-7"]
    default_test_rxs = ["1-19"]
    train_rx_idx = _resolve_rxs(rx_list, train_rxs if train_rxs is not None else default_train_rxs, [])
    test_rx_idx = _resolve_rxs(rx_list, test_rxs if test_rxs is not None else default_test_rxs, [])

    overlap = sorted(set(train_rx_idx).intersection(test_rx_idx))
    if overlap:
        raise ValueError(f"RIEI train/test receiver sets must be disjoint, overlap={overlap}")
    if len(train_rx_idx) != 2:
        raise ValueError(f"RIEI paper protocol expects exactly two training receivers, got {train_rx_idx}")
    if len(test_rx_idx) < 1:
        raise ValueError("RIEI paper protocol requires at least one target receiver.")

    day_idx = list(range(len(day_list)))
    base_trainval = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=day_idx,
        rx_keep=train_rx_idx,
        domain=domain,
        transform=None,
        max_samples_per_combo=None,
        seed=seed,
        build_index=True,
    )
    groups = _build_grouped_indices_by_tx_rx_eq_sorted(base_trainval)
    rng = np.random.default_rng(int(seed))
    train_n = max(1, int(train_samples_per_combo))
    val_n = max(0, int(val_samples_per_combo))
    train_sel: List[int] = []
    val_sel: List[int] = []
    for _, idxs_sorted in sorted(groups.items()):
        idxs = list(idxs_sorted)
        if len(idxs) > 1:
            idxs = [idxs[int(i)] for i in rng.permutation(len(idxs)).tolist()]
        train_sel.extend(idxs[:train_n])
        if val_n > 0:
            val_sel.extend(idxs[train_n:train_n + val_n])

    if not train_sel:
        raise ValueError("RIEI split produced an empty train set.")
    if not val_sel:
        fallback = []
        for _, idxs_sorted in sorted(groups.items()):
            if len(idxs_sorted) > 1:
                fallback.append(idxs_sorted[-1])
        val_sel = fallback
        train_sel = [idx for idx in train_sel if idx not in set(val_sel)]
    if not val_sel:
        raise ValueError("RIEI split produced an empty validation set.")

    train_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(train_sel),
        split_source="riei_original_random_source_rx_train",
        transform=transform_train,
    )
    val_ds = WiSigSubsetDataset(
        base_trainval,
        sorted(val_sel),
        split_source="riei_original_random_source_rx_validation",
        transform=transform_eval,
    )

    base_test = WiSigCompactDataset(
        ds,
        out_len=out_len,
        crop_mode=crop_mode,
        normalize=normalize,
        equalized=equalized,
        day_keep=day_idx,
        rx_keep=test_rx_idx,
        domain=domain,
        transform=None,
        max_samples_per_combo=None,
        seed=seed,
        build_index=True,
    )
    test_groups = _build_grouped_indices_by_tx_rx_eq_sorted(base_test)
    rng_test = np.random.default_rng(int(seed) + 7919)
    test_sel: List[int] = []
    test_n = max(1, int(test_samples_per_combo))
    for _, idxs_sorted in sorted(test_groups.items()):
        idxs = list(idxs_sorted)
        if len(idxs) > 1:
            idxs = [idxs[int(i)] for i in rng_test.permutation(len(idxs)).tolist()]
        test_sel.extend(idxs[:test_n])
    if not test_sel:
        raise ValueError("RIEI split produced an empty test set.")
    test_ds = WiSigSubsetDataset(
        base_test,
        sorted(test_sel),
        split_source="riei_original_random_target_rx_test",
        transform=transform_eval,
    )

    named_tests: Dict[str, Dataset] = {
        "test_seen_day_unseen_rx": test_ds,
        "test_riei_target_rx": test_ds,
    }
    named_test_meta: Dict[str, Dict[str, Any]] = {
        "test_seen_day_unseen_rx": {
            "days_idx": list(day_idx),
            "days_label": [day_list[i] for i in day_idx],
            "rxs_idx": list(test_rx_idx),
            "rxs_label": [rx_list[i] for i in test_rx_idx],
            "size": len(test_ds),
            "paper_protocol_alias": "riei_two_source_receiver_holdout",
        },
        "test_riei_target_rx": {
            "days_idx": list(day_idx),
            "days_label": [day_list[i] for i in day_idx],
            "rxs_idx": list(test_rx_idx),
            "rxs_label": [rx_list[i] for i in test_rx_idx],
            "size": len(test_ds),
            "paper_protocol_alias": "riei_two_source_receiver_holdout",
        },
    }
    for r_idx in test_rx_idx:
        one_indices = [i for i, it in enumerate(base_test.index) if int(it.rx_i) == int(r_idx)]
        one_set = set(one_indices)
        one_selected = [idx for idx in sorted(test_sel) if idx in one_set]
        if not one_selected:
            continue
        ds_one = WiSigSubsetDataset(
            base_test,
            one_selected,
            split_source=f"riei_original_random_target_rx_test_{r_idx}",
            transform=transform_eval,
        )
        key = f"test_rx_{r_idx}"
        named_tests[key] = ds_one
        named_test_meta[key] = {
            "days_idx": list(day_idx),
            "days_label": [day_list[i] for i in day_idx],
            "rxs_idx": [r_idx],
            "rxs_label": [rx_list[r_idx]],
            "size": len(ds_one),
            "paper_protocol_alias": "riei_two_source_receiver_holdout",
        }

    info = {
        "mode": "wisig_riei_two_source_receiver_holdout_14400_4800",
        "paper": "RIEI",
        "paper_protocol": "WiSig train two source receivers, test one disjoint receiver",
        "day_idx": list(day_idx),
        "day_label": [day_list[i] for i in day_idx],
        "train_days_idx": list(day_idx),
        "train_days_label": [day_list[i] for i in day_idx],
        "test_days_idx": list(day_idx),
        "test_days_label": [day_list[i] for i in day_idx],
        "train_rxs_idx": list(train_rx_idx),
        "train_rxs_label": [rx_list[i] for i in train_rx_idx],
        "test_rxs_idx": list(test_rx_idx),
        "test_rxs_label": [rx_list[i] for i in test_rx_idx],
        "train_samples_per_combo": int(train_samples_per_combo),
        "val_samples_per_combo": int(val_samples_per_combo),
        "test_samples_per_combo": int(test_samples_per_combo),
        "train_samples_per_receiver": int(train_samples_per_combo) * int(len(ds.get("tx_list", []))),
        "test_samples_per_receiver": int(test_samples_per_combo) * int(len(ds.get("tx_list", []))),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "named_test_sizes": {k: len(v) for k, v in named_tests.items()},
        "primary_named_test": "test_seen_day_unseen_rx",
        "strict_named_test": "test_seen_day_unseen_rx",
        "aggregate_test_keys": ["test_seen_day_unseen_rx"],
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
    max_samples_per_class_train: Optional[int] = None,
    seed: int = 0,
    split_strategy: str = "random",
    cap_strategy: str = "random",
    train_class_cap_strategy: str = "domain_balanced",
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
        max_samples_per_class_train=max_samples_per_class_train,
        seed=seed,
        split_strategy=split_strategy,
        cap_strategy=cap_strategy,
        train_class_cap_strategy=train_class_cap_strategy,
    )
    return train_ds, val_ds, test_ds, info
