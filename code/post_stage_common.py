from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from dataset import WiFiRFFIDataset
from dataset_wisig import load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx
from model_dual_cvsincnet import build_dual_model
from cvsrffi.eval import evaluate_loader, evaluate_named_loaders, make_loader
from cvsrffi.tensors import (
    build_domain_label_map,
    extract_domain_from_extra,
    parse_csv_indices,
    remap_domain_tensor,
    set_seed,
    unpack_batch,
)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def load_yaml_or_json(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        try:
            data = json.loads(text)
        except Exception:
            data = _parse_simple_yaml(text)
    return data or {}


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Small YAML subset parser for the repository's flat experiment configs."""

    def parse_scalar(raw: str):
        value = raw.strip()
        if value == "":
            return {}
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return value.strip("\"'")

    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = parse_scalar(raw)
        parent[key] = value
        if isinstance(value, dict):
            stack.append((indent, value))
    return root


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def as_namespace(data: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(data))


def default_data_args() -> Dict[str, Any]:
    return {
        "dataset": "wisig",
        "dataset_dir": "./Dataset_ORALCE",
        "run_name": "post_stage",
        "wisig_pkl": "./Dataset_WigSig/ManySig.pkl",
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
        "wisig_out_len": 256,
        "wisig_train_ratio": 0.2,
        "wisig_val_ratio": -1.0,
        "wisig_guard_gap": 8,
        "wisig_train_days": "0,1",
        "wisig_test_days": "2,3",
        "wisig_train_rxs": "0,1,2,3,4,5,6",
        "wisig_test_rxs": "7,8,9,10,11",
        "wisig_split_strategy": "random",
        "wisig_cap_strategy": "random",
        "wisig_max_day123_per_combo": 0,
        "wisig_max_train_per_combo": 0,
        "wisig_max_val_per_combo": 0,
        "wisig_max_test_per_combo": 0,
        "num_classes": 16,
        "batch_size": 128,
        "eval_batch_size": 256,
        "num_workers": 4,
        "prefetch_factor": 2,
        "device": "cuda:0",
        "seed": 1337,
        "eval_max_batches": 0,
        "sample_rate_hz": 0.0,
    }


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    defaults = default_data_args()
    parser.add_argument("--dataset", type=str, default=defaults["dataset"], choices=["wisig", "oralce"])
    parser.add_argument("--dataset_dir", type=str, default=defaults["dataset_dir"])
    parser.add_argument("--run_name", type=str, default=defaults["run_name"])
    parser.add_argument("--wisig_pkl", type=str, default=defaults["wisig_pkl"])
    parser.add_argument("--wisig_equalized", type=str, default=defaults["wisig_equalized"])
    parser.add_argument("--wisig_domain", type=str, default=defaults["wisig_domain"], choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=defaults["wisig_out_len"])
    parser.add_argument("--wisig_train_ratio", type=float, default=defaults["wisig_train_ratio"])
    parser.add_argument("--wisig_val_ratio", type=float, default=defaults["wisig_val_ratio"])
    parser.add_argument("--wisig_guard_gap", type=int, default=defaults["wisig_guard_gap"])
    parser.add_argument("--wisig_train_days", type=str, default=defaults["wisig_train_days"])
    parser.add_argument("--wisig_test_days", type=str, default=defaults["wisig_test_days"])
    parser.add_argument("--wisig_train_rxs", type=str, default=defaults["wisig_train_rxs"])
    parser.add_argument("--wisig_test_rxs", type=str, default=defaults["wisig_test_rxs"])
    parser.add_argument("--wisig_split_strategy", type=str, default=defaults["wisig_split_strategy"], choices=["random", "contiguous"])
    parser.add_argument("--wisig_cap_strategy", type=str, default=defaults["wisig_cap_strategy"], choices=["random", "front"])
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=defaults["wisig_max_day123_per_combo"])
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=defaults["wisig_max_train_per_combo"])
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=defaults["wisig_max_val_per_combo"])
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=defaults["wisig_max_test_per_combo"])
    parser.add_argument("--num_classes", type=int, default=defaults["num_classes"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--prefetch_factor", type=int, default=defaults["prefetch_factor"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--eval_max_batches", type=int, default=defaults["eval_max_batches"])
    parser.add_argument("--sample_rate_hz", type=float, default=defaults["sample_rate_hz"])


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def _positive_or_none(value: Any) -> Optional[int]:
    ivalue = int(value)
    return None if ivalue <= 0 else ivalue


def build_standard_data(args, device: torch.device):
    if str(args.dataset).lower() == "wisig":
        if float(getattr(args, "wisig_val_ratio", -1.0)) > 0.0:
            args.wisig_train_ratio = 1.0 - float(args.wisig_val_ratio)
        ds_w = load_wisig_compact_pkl(args.wisig_pkl)
        infer_nc = len(ds_w.get("tx_list", []))
        if infer_nc > 0:
            args.num_classes = infer_nc
        eq = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
        train_ds, val_ds, test_ds, named_tests, named_meta, split_info = make_wisig_trainval_test_by_day_rx(
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
            max_samples_per_combo_day123=_positive_or_none(args.wisig_max_day123_per_combo),
            max_samples_per_combo_train=_positive_or_none(args.wisig_max_train_per_combo),
            max_samples_per_combo_val=_positive_or_none(args.wisig_max_val_per_combo),
            max_samples_per_combo_test=_positive_or_none(args.wisig_max_test_per_combo),
            seed=int(args.seed),
            split_strategy=str(args.wisig_split_strategy),
            cap_strategy=str(args.wisig_cap_strategy),
        )
        input_len = int(args.wisig_out_len)
    else:
        train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train", run_name=args.run_name)
        val_ds = WiFiRFFIDataset(args.dataset_dir, mode="test", run_name=args.run_name)
        test_ds = val_ds
        named_tests = {"test_default": test_ds}
        named_meta = {"test_default": {"size": len(test_ds)}}
        split_info = {"mode": "oralce_train_test"}
        try:
            x0, _ = train_ds[0]
            input_len = int(x0.shape[-1])
        except Exception:
            input_len = 1024

    train_loader = make_loader(
        train_ds, int(args.batch_size), True, int(args.num_workers), device, True, int(args.prefetch_factor)
    )
    val_loader = make_loader(
        val_ds, int(args.eval_batch_size), False, int(args.num_workers), device, False, int(args.prefetch_factor)
    )
    named_test_loaders = {
        name: make_loader(ds, int(args.eval_batch_size), False, int(args.num_workers), device, False, int(args.prefetch_factor))
        for name, ds in named_tests.items()
    }
    domain_label_map = build_domain_label_map(train_ds)
    num_domains = max(1, len(domain_label_map))
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "named_test_loaders": named_test_loaders,
        "named_test_meta": named_meta,
        "split_info": split_info,
        "domain_label_map": domain_label_map,
        "num_domains": num_domains,
        "input_len": input_len,
    }


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, Mapping) and "model" in ckpt:
        return dict(ckpt)
    if isinstance(ckpt, Mapping):
        return {"model": ckpt, "args": {}, "stats": {}, "split_info": None}
    raise TypeError(f"Unsupported checkpoint payload in {path}")


def merge_checkpoint_args(ckpt: Mapping[str, Any], cli_args, *, input_len: int, num_domains: int) -> SimpleNamespace:
    merged = default_data_args()
    merged.update(dict(ckpt.get("args") or {}))
    for key, value in vars(cli_args).items():
        if key in merged:
            merged[key] = value
    if float(merged.get("sample_rate_hz", 0.0)) <= 0.0:
        merged["sample_rate_hz"] = 25e6 if str(merged.get("dataset", "wisig")) == "wisig" else 5e6
    merged["input_len"] = int(input_len)
    merged["num_domains"] = int(max(1, num_domains))
    return as_namespace(merged)


def build_baseline_model(model_args, device: torch.device) -> nn.Module:
    return build_dual_model(
        int(model_args.num_classes),
        int(model_args.num_domains),
        model_size=str(getattr(model_args, "model_size", "M")),
        dataset=str(getattr(model_args, "dataset", "wisig")),
        input_len=int(getattr(model_args, "input_len", 256)),
        sample_rate_hz=float(getattr(model_args, "sample_rate_hz", 25e6)),
        id_feature_key=str(getattr(model_args, "id_feature_key", "feat_joint")),
        dom_feature_key=str(getattr(model_args, "dom_feature_key", "feat_imp")),
        model_variant=str(getattr(model_args, "model_variant", "lite_c")),
        branch_ablation=str(getattr(model_args, "branch_ablation", "none")),
        mixstyle_on=bool(getattr(model_args, "use_mixstyle", False)),
        mixstyle_p=float(getattr(model_args, "mixstyle_p", 0.3)),
        mixstyle_alpha=float(getattr(model_args, "mixstyle_alpha", 0.1)),
        mixstyle_eps=float(getattr(model_args, "mixstyle_eps", 1e-6)),
        mixstyle_layers=str(getattr(model_args, "mixstyle_layers", "time_down,t1")),
        mixstyle_use_domain_label=bool(getattr(model_args, "mixstyle_use_domain_label", True)),
        mixstyle_mix=str(getattr(model_args, "mixstyle_mix", "crossdomain")),
        mixstyle_strength=float(getattr(model_args, "mixstyle_strength", 1.0)),
        mixstyle_fallback=str(getattr(model_args, "mixstyle_fallback", "random")),
        domain_branch_ablation=str(getattr(model_args, "domain_branch_ablation", "same")),
        domain_enhancer=str(getattr(model_args, "domain_enhancer", "rcn_stats")),
        domain_enhancer_strength=float(getattr(model_args, "domain_enhancer_strength", 0.35)),
        id_time_stability_mode=str(getattr(model_args, "id_time_stability_mode", "off")),
        id_freq_stability_mode=str(getattr(model_args, "id_freq_stability_mode", "off")),
        domain_time_stability_mode=str(getattr(model_args, "domain_time_stability_mode", "off")),
        domain_freq_stability_mode=str(getattr(model_args, "domain_freq_stability_mode", "off")),
        time_stability_channels=int(getattr(model_args, "time_stability_channels", 8)),
        freq_stability_channels=int(getattr(model_args, "freq_stability_channels", 4)),
        fast_infer_when_no_aux=bool(getattr(model_args, "fast_infer_when_no_aux", True)),
        arch_family=str(getattr(model_args, "arch_family", "cvsincnet")),
    ).to(device)


def load_baseline_from_checkpoint(
    ckpt_path: str,
    cli_args,
    data_ctx: Mapping[str, Any],
    device: torch.device,
    *,
    freeze: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], SimpleNamespace]:
    ckpt = load_checkpoint(ckpt_path, device)
    model_args = merge_checkpoint_args(
        ckpt,
        cli_args,
        input_len=int(data_ctx["input_len"]),
        num_domains=int(data_ctx["num_domains"]),
    )
    model = build_baseline_model(model_args, device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[CKPT-WARN] missing keys while loading baseline: {len(missing)}", flush=True)
    if unexpected:
        print(f"[CKPT-WARN] unexpected keys while loading baseline: {len(unexpected)}", flush=True)
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model, ckpt, model_args


def move_batch(batch, device: torch.device):
    x, y, extra = unpack_batch(batch)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), extra


def mean_logs(log_items: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    count: Dict[str, int] = {}
    for logs in log_items:
        for key, value in logs.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                val = float(value.detach().cpu())
            else:
                try:
                    val = float(value)
                except Exception:
                    continue
            acc[key] = acc.get(key, 0.0) + val
            count[key] = count.get(key, 0) + 1
    return {key: acc[key] / max(1, count[key]) for key in sorted(acc)}


def save_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), str(path))


class SGCWrappedTeacher(nn.Module):
    def __init__(self, sgc: nn.Module, teacher: nn.Module):
        super().__init__()
        self.sgc = sgc
        self.teacher = teacher

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        x_hat = self.sgc(x)
        return self.teacher(
            x_hat,
            y_tx=y_tx,
            grl_lambda=grl_lambda,
            return_aux=return_aux,
            domain_labels=domain_labels,
        )


def evaluate_post_model(model, data_ctx: Mapping[str, Any], device: torch.device, max_batches: int = 0):
    val = evaluate_loader(
        model,
        data_ctx["val_loader"],
        device,
        domain_label_map=data_ctx["domain_label_map"],
        max_batches=max_batches,
    )
    named = evaluate_named_loaders(
        model,
        data_ctx["named_test_loaders"],
        device,
        domain_label_map=data_ctx["domain_label_map"],
        max_batches=max_batches,
    )
    return val, named


def domain_from_extra(extra, domain_label_map: Mapping[int, int], device: torch.device) -> Optional[torch.Tensor]:
    raw = extract_domain_from_extra(extra, device)
    return remap_domain_tensor(raw, dict(domain_label_map), device) if raw is not None else None
