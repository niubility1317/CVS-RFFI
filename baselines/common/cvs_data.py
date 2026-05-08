from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from dataset_wisig import load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx


def parse_csv_indices(value: str | Sequence[int] | None) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if text == "":
        return None
    return [int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip() != ""]


def add_cvs_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_equalized", type=str, default="1")
    parser.add_argument("--wisig_domain", type=str, default="rx_day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_train_ratio", type=float, default=0.2)
    parser.add_argument("--wisig_val_ratio", type=float, default=-1.0)
    parser.add_argument("--wisig_guard_gap", type=int, default=8)
    parser.add_argument("--wisig_train_days", type=str, default="0,1")
    parser.add_argument("--wisig_test_days", type=str, default="2,3")
    parser.add_argument("--wisig_train_rxs", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--wisig_test_rxs", type=str, default="7,8,9,10,11")
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    return parser


class CVSIdentityTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CVSDictDataset(Dataset):
    """Wrap CVS/WiSig tuple samples as dictionaries used by baseline code."""

    def __init__(self, dataset: Dataset, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[int(idx)]
        if isinstance(sample, dict):
            item = dict(sample)
            if self.transform is not None:
                item["iq"] = self.transform(item["iq"])
            return item
        x, y, d, meta = sample
        if self.transform is not None:
            x = self.transform(x)
        meta = dict(meta)
        return {
            "iq": x,
            "label": int(y),
            "domain": int(d),
            "receiver": int(meta.get("rx_i", d)),
            "day": int(meta.get("day_i", -1)),
            "sig_i": int(meta.get("sig_i", -1)),
            "meta": meta,
        }


class FewShotPerClassDataset(Dataset):
    def __init__(self, dataset: Dataset, shots_per_class: int, seed: int = 0):
        self.dataset = dataset
        shots = max(1, int(shots_per_class))
        gen = torch.Generator().manual_seed(int(seed))
        by_label: Dict[int, List[int]] = {}
        for idx in range(len(dataset)):
            label = int(dataset[idx]["label"])
            by_label.setdefault(label, []).append(idx)
        selected: List[int] = []
        for label in sorted(by_label):
            idxs = by_label[label]
            if len(idxs) > shots:
                perm = torch.randperm(len(idxs), generator=gen)[:shots].tolist()
                selected.extend([idxs[int(i)] for i in perm])
            else:
                selected.extend(idxs)
        self.selected = sorted(selected)

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[int(self.selected[int(idx)])]


def collate_cvs_dict(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "iq": torch.stack([b["iq"] for b in batch], dim=0),
        "label": torch.tensor([int(b["label"]) for b in batch], dtype=torch.long),
        "domain": torch.tensor([int(b.get("domain", b.get("receiver", -1))) for b in batch], dtype=torch.long),
        "receiver": torch.tensor([int(b["receiver"]) for b in batch], dtype=torch.long),
        "day": torch.tensor([int(b.get("day", -1)) for b in batch], dtype=torch.long),
        "sig_i": torch.tensor([int(b.get("sig_i", -1)) for b in batch], dtype=torch.long),
        "meta": [b.get("meta", {}) for b in batch],
    }


@dataclass
class CVSSplit:
    train: Dataset
    val: Dataset
    test: Dataset
    named_tests: Dict[str, Dataset]
    named_test_meta: Dict[str, Dict[str, Any]]
    split_info: Dict[str, Any]
    num_classes: int
    num_receivers: int
    input_len: int


def _cap_arg(value: int) -> Optional[int]:
    return None if int(value) <= 0 else int(value)


def build_cvs_split(
    args,
    *,
    transform_train: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> CVSSplit:
    ds_w = load_wisig_compact_pkl(args.wisig_pkl)
    if float(getattr(args, "wisig_val_ratio", -1.0)) > 0.0:
        args.wisig_train_ratio = 1.0 - float(args.wisig_val_ratio)
    eq = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
    train_ds, val_ds, test_ds, named_tests, named_test_meta, split_info = make_wisig_trainval_test_by_day_rx(
        ds_w,
        equalized=eq,
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        normalize=True,
        crop_mode="center",
        transform_train=None,
        transform_eval=None,
        train_ratio=float(args.wisig_train_ratio),
        guard_gap=int(args.wisig_guard_gap),
        train_days=parse_csv_indices(args.wisig_train_days),
        test_days=parse_csv_indices(args.wisig_test_days),
        train_rxs=parse_csv_indices(args.wisig_train_rxs),
        test_rxs=parse_csv_indices(args.wisig_test_rxs),
        max_samples_per_combo_day123=_cap_arg(args.wisig_max_day123_per_combo),
        max_samples_per_combo_train=_cap_arg(args.wisig_max_train_per_combo),
        max_samples_per_combo_val=_cap_arg(args.wisig_max_val_per_combo),
        max_samples_per_combo_test=_cap_arg(args.wisig_max_test_per_combo),
        seed=int(getattr(args, "seed", 1337)),
    )
    return CVSSplit(
        train=CVSDictDataset(train_ds, transform=transform_train),
        val=CVSDictDataset(val_ds, transform=transform_eval),
        test=CVSDictDataset(test_ds, transform=transform_eval),
        named_tests={name: CVSDictDataset(ds, transform=transform_eval) for name, ds in named_tests.items()},
        named_test_meta=named_test_meta,
        split_info=split_info,
        num_classes=int(len(ds_w.get("tx_list", []))),
        num_receivers=int(len(ds_w.get("rx_list", []))),
        input_len=int(args.wisig_out_len),
    )


def make_cvs_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int, device, drop_last: bool = False, prefetch_factor: int = 2) -> DataLoader:
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": getattr(device, "type", "cpu") == "cuda",
        "drop_last": bool(drop_last),
        "collate_fn": collate_cvs_dict,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(dataset, **kwargs)


@dataclass
class CVSLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    named_tests: Dict[str, DataLoader]
    split: CVSSplit


def build_cvs_loaders(
    args,
    device,
    *,
    transform_train: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_eval: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> CVSLoaders:
    split = build_cvs_split(args, transform_train=transform_train, transform_eval=transform_eval)
    train_loader = make_cvs_loader(
        split.train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        device=device,
        drop_last=True,
        prefetch_factor=int(args.prefetch_factor),
    )
    val_loader = make_cvs_loader(
        split.val,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        device=device,
        drop_last=False,
        prefetch_factor=int(args.prefetch_factor),
    )
    test_loader = make_cvs_loader(
        split.test,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        device=device,
        drop_last=False,
        prefetch_factor=int(args.prefetch_factor),
    )
    named = {
        name: make_cvs_loader(
            ds,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            device=device,
            drop_last=False,
            prefetch_factor=int(args.prefetch_factor),
        )
        for name, ds in split.named_tests.items()
    }
    return CVSLoaders(train=train_loader, val=val_loader, test=test_loader, named_tests=named, split=split)
