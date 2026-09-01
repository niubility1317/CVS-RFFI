from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIR = _REPO_ROOT / "code"
if _CODE_DIR.is_dir():
    _code_dir_str = str(_CODE_DIR)
    if sys.path[0] != _code_dir_str:
        sys.path = [p for p in sys.path if p != _code_dir_str]
        sys.path.insert(0, _code_dir_str)

from dataset_wisig import (
    load_wisig_compact_pkl,
    make_wisig_drift_day1_split,
    make_wisig_meta_ssl_source_split,
    make_wisig_riei_receiver_holdout_split,
    make_wisig_trainval_test_by_day_rx,
)


def parse_csv_values(value: str | Sequence[int] | None) -> Optional[List[int | str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) if isinstance(v, int) or str(v).lstrip("-").isdigit() else str(v) for v in value]
    text = str(value).strip()
    if text == "":
        return None
    out: List[int | str] = []
    for raw in text.replace(";", ",").split(","):
        token = raw.strip()
        if token == "":
            continue
        out.append(int(token) if token.lstrip("-").isdigit() else token)
    return out


def parse_csv_indices(value: str | Sequence[int] | None) -> Optional[List[int | str]]:
    return parse_csv_values(value)


def add_cvs_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument(
        "--wisig_protocol",
        type=str,
        default="cvs_day_rx",
        choices=["cvs_day_rx", "drift_day1", "riei_original"],
    )
    parser.add_argument("--wisig_equalized", type=str, default="1")
    parser.add_argument("--wisig_domain", type=str, default="rx_day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_train_ratio", type=float, default=0.2)
    parser.add_argument("--wisig_val_ratio", type=float, default=-1.0)
    parser.add_argument(
        "--use_source_ssl_split",
        action="store_true",
        help="Use a source-only labeled/unlabeled/validation split for pseudo-label training.",
    )
    parser.add_argument("--wisig_labeled_ratio", type=float, default=0.07)
    parser.add_argument("--wisig_unlabeled_ratio", type=float, default=0.63)
    parser.add_argument("--wisig_source_val_ratio", type=float, default=0.3)
    parser.add_argument("--wisig_guard_gap", type=int, default=8)
    parser.add_argument("--wisig_train_days", type=str, default="0,1")
    parser.add_argument("--wisig_test_days", type=str, default="2,3")
    parser.add_argument("--wisig_train_rxs", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--wisig_test_rxs", type=str, default="7,8,9,10,11")
    parser.add_argument(
        "--wisig_source_holdout_rxs",
        type=str,
        default="",
        help="Source receivers removed from L/U/V without changing the target-test receiver set.",
    )
    parser.add_argument("--wisig_split_strategy", type=str, default="random", choices=["random", "contiguous"])
    parser.add_argument(
        "--wisig_split_seed",
        type=int,
        default=-1,
        help="Dataset-partition seed; negative values reuse --seed for backward compatibility.",
    )
    parser.add_argument("--wisig_cap_strategy", type=str, default="random", choices=["random", "front"])
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument(
        "--wisig_train_shots_per_class",
        "--wisig_max_train_per_class_total",
        dest="wisig_train_shots_per_class",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--wisig_train_shot_strategy",
        type=str,
        default="domain_balanced",
        choices=["domain_balanced", "rx_day_balanced", "random", "front"],
    )
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)
    parser.add_argument("--wisig_paper_day", type=str, default="0")
    parser.add_argument("--wisig_paper_train_samples_per_combo", type=int, default=800)
    parser.add_argument("--wisig_paper_val_samples_per_combo", type=int, default=200)
    parser.add_argument("--wisig_paper_test_samples_per_combo", type=int, default=200)
    parser.add_argument(
        "--wisig_paper_sample_strategy",
        type=str,
        default="front",
        choices=["front", "random"],
        help="Paper-protocol per-TX/RX sample selection. DRIFT v2 requires random.",
    )
    parser.add_argument(
        "--wisig_rms_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-packet RMS normalization. DRIFT v2 disables this because equalization is its only signal preprocessing.",
    )
    parser.add_argument("--paper_eval_last_n", type=int, default=0)
    parser.add_argument("--paper_eval_name", type=str, default="")
    parser.add_argument("--test_eval_interval", type=int, default=0)
    parser.add_argument("--test_eval_start_epoch", type=int, default=1)
    parser.add_argument("--test_on_val_improve", dest="test_on_val_improve", action="store_true", default=True)
    parser.add_argument("--no_test_on_val_improve", dest="test_on_val_improve", action="store_false")
    parser.add_argument("--final_test_best_by_val", action="store_true")
    parser.add_argument("--final_test_target_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--train_drop_last", dest="train_drop_last", action="store_true", default=True)
    parser.add_argument("--no_train_drop_last", dest="train_drop_last", action="store_false")
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
            "true_label": int(meta.get("true_tx_i", y)),
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
        "true_label": torch.tensor([int(b.get("true_label", b["label"])) for b in batch], dtype=torch.long),
        "domain": torch.tensor([int(b.get("domain", b.get("receiver", -1))) for b in batch], dtype=torch.long),
        "receiver": torch.tensor([int(b["receiver"]) for b in batch], dtype=torch.long),
        "day": torch.tensor([int(b.get("day", -1)) for b in batch], dtype=torch.long),
        "sig_i": torch.tensor([int(b.get("sig_i", -1)) for b in batch], dtype=torch.long),
        "meta": [b.get("meta", {}) for b in batch],
    }


@dataclass
class CVSSplit:
    train: Dataset
    unlabeled: Optional[Dataset]
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
    eq = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
    protocol = str(getattr(args, "wisig_protocol", "cvs_day_rx")).lower()
    split_seed = int(getattr(args, "wisig_split_seed", -1))
    if split_seed < 0:
        split_seed = int(getattr(args, "seed", 1337))
    unlabeled_ds = None
    if bool(getattr(args, "use_source_ssl_split", False)):
        if protocol != "cvs_day_rx":
            raise ValueError("Source SSL split is only valid with wisig_protocol=cvs_day_rx.")
        source_holdout_raw = str(getattr(args, "wisig_source_holdout_rxs", "") or "").strip()
        source_holdout_rxs = (
            parse_csv_indices(source_holdout_raw)
            if source_holdout_raw
            else parse_csv_indices(args.wisig_test_rxs)
        )
        labeled_ds, unlabeled_ds, val_ds, ssl_info = make_wisig_meta_ssl_source_split(
            ds_w,
            equalized=eq,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform_labeled=None,
            transform_unlabeled=None,
            transform_val=None,
            labeled_ratio=float(args.wisig_labeled_ratio),
            unlabeled_ratio=float(args.wisig_unlabeled_ratio),
            val_ratio=float(args.wisig_source_val_ratio),
            train_days=parse_csv_indices(args.wisig_train_days),
            holdout_days=parse_csv_indices(args.wisig_test_days),
            train_rxs=parse_csv_indices(args.wisig_train_rxs),
            holdout_rxs=source_holdout_rxs,
            max_samples_per_combo_source=_cap_arg(args.wisig_max_day123_per_combo),
            seed=split_seed,
            sample_strategy=str(getattr(args, "wisig_cap_strategy", "random")),
        )
        _, _, test_ds, named_tests, named_test_meta, test_info = make_wisig_trainval_test_by_day_rx(
            ds_w,
            equalized=eq,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform_train=None,
            transform_eval=None,
            train_ratio=float(args.wisig_labeled_ratio),
            guard_gap=int(args.wisig_guard_gap),
            train_days=parse_csv_indices(args.wisig_train_days),
            test_days=parse_csv_indices(args.wisig_test_days),
            train_rxs=parse_csv_indices(args.wisig_train_rxs),
            test_rxs=parse_csv_indices(args.wisig_test_rxs),
            max_samples_per_combo_day123=_cap_arg(args.wisig_max_day123_per_combo),
            max_samples_per_combo_train=_cap_arg(args.wisig_max_train_per_combo),
            max_samples_per_combo_val=_cap_arg(args.wisig_max_val_per_combo),
            max_samples_per_combo_test=_cap_arg(args.wisig_max_test_per_combo),
            max_samples_per_class_train=_cap_arg(getattr(args, "wisig_train_shots_per_class", 0)),
            seed=split_seed,
            split_strategy=str(getattr(args, "wisig_split_strategy", "random")),
            cap_strategy=str(getattr(args, "wisig_cap_strategy", "random")),
            train_class_cap_strategy=str(getattr(args, "wisig_train_shot_strategy", "domain_balanced")),
        )
        train_ds = labeled_ds
        split_info = {
            **test_info,
            **ssl_info,
            "mode": "source_ssl_cvs_day_rx_with_explicit_holdout_tests",
            "test_days_idx": test_info.get("test_days_idx", []),
            "test_days_label": test_info.get("test_days_label", []),
            "test_rxs_idx": test_info.get("test_rxs_idx", []),
            "test_rxs_label": test_info.get("test_rxs_label", []),
            "named_test_sizes": test_info.get("named_test_sizes", {}),
            "test_size": test_info.get("test_size", len(test_ds)),
        }
    elif protocol == "drift_day1":
        train_ds, val_ds, test_ds, named_tests, named_test_meta, split_info = make_wisig_drift_day1_split(
            ds_w,
            equalized=eq,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=bool(getattr(args, "wisig_rms_normalize", True)),
            crop_mode="center",
            transform_train=None,
            transform_eval=None,
            day=parse_csv_values(getattr(args, "wisig_paper_day", "0"))[0],
            train_rxs=parse_csv_values(args.wisig_train_rxs),
            test_rxs=parse_csv_values(args.wisig_test_rxs),
            train_samples_per_combo=int(args.wisig_paper_train_samples_per_combo),
            val_samples_per_combo=int(args.wisig_paper_val_samples_per_combo),
            test_samples_per_combo=int(args.wisig_paper_test_samples_per_combo),
            seed=split_seed,
            sample_strategy=str(getattr(args, "wisig_paper_sample_strategy", "front")),
        )
    elif protocol == "riei_original":
        train_ds, val_ds, test_ds, named_tests, named_test_meta, split_info = make_wisig_riei_receiver_holdout_split(
            ds_w,
            equalized=eq,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=bool(getattr(args, "wisig_rms_normalize", True)),
            crop_mode="center",
            transform_train=None,
            transform_eval=None,
            train_rxs=parse_csv_values(args.wisig_train_rxs),
            test_rxs=parse_csv_values(args.wisig_test_rxs),
            train_samples_per_combo=int(args.wisig_paper_train_samples_per_combo),
            val_samples_per_combo=int(args.wisig_paper_val_samples_per_combo),
            test_samples_per_combo=int(args.wisig_paper_test_samples_per_combo),
            seed=split_seed,
        )
    else:
        if float(getattr(args, "wisig_val_ratio", -1.0)) > 0.0:
            args.wisig_train_ratio = 1.0 - float(args.wisig_val_ratio)
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
            max_samples_per_class_train=_cap_arg(getattr(args, "wisig_train_shots_per_class", 0)),
            seed=split_seed,
            split_strategy=str(getattr(args, "wisig_split_strategy", "random")),
            cap_strategy=str(getattr(args, "wisig_cap_strategy", "random")),
            train_class_cap_strategy=str(getattr(args, "wisig_train_shot_strategy", "domain_balanced")),
        )
        unlabeled_ds = None
    split_info = {
        **split_info,
        "model_seed": int(getattr(args, "seed", 1337)),
        "split_seed": split_seed,
    }
    return CVSSplit(
        train=CVSDictDataset(train_ds, transform=transform_train),
        unlabeled=CVSDictDataset(unlabeled_ds, transform=transform_train) if unlabeled_ds is not None else None,
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
    unlabeled: Optional[DataLoader]
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
        drop_last=bool(getattr(args, "train_drop_last", True)),
        prefetch_factor=int(args.prefetch_factor),
    )
    unlabeled_loader = (
        make_cvs_loader(
            split.unlabeled,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            device=device,
            drop_last=bool(getattr(args, "train_drop_last", True)),
            prefetch_factor=int(args.prefetch_factor),
        )
        if split.unlabeled is not None
        else None
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
    return CVSLoaders(train=train_loader, unlabeled=unlabeled_loader, val=val_loader, test=test_loader, named_tests=named, split=split)
