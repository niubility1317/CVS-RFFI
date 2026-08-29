from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

PROJECT_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_CODE_DIR, os.pardir))
if PROJECT_CODE_DIR not in sys.path:
    sys.path.insert(0, PROJECT_CODE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch

import train as train_mod
from cvsrffi.eval import (
    MAIN_SAT_EVAL_ON_NAMES,
    apply_sat_channel_for_scenario,
    evaluate_loader,
    evaluate_loader_sat_channel,
    make_loader,
)
from cvsrffi.tensors import make_torch_generator, unpack_batch
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx
from model_dual_cvsincnet import build_dual_model
from training_controls import parse_sat_scenarios

try:
    from SSDG import train_ssdg as ssdg_mod
except Exception:  # pragma: no cover - fallback keeps the script usable outside SSDG checkouts
    ssdg_mod = None


MODEL_DEFAULTS: Dict[str, Any] = {
    "model_size": "small",
    "dataset": "wisig",
    "model_variant": "dual",
    "branch_ablation": "none",
    "domain_branch_ablation": "none",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    "id_time_stability_mode": "off",
    "id_freq_stability_mode": "off",
    "domain_time_stability_mode": "off",
    "domain_freq_stability_mode": "off",
    "time_stability_channels": 8,
    "freq_stability_channels": 8,
    "use_circularity": True,
    "use_freq_stats": False,
    "use_pa_stats": False,
    "use_freq_band_gate": False,
    "freq_feature_source": "raw_fft",
    "pa_feature_source": "raw_iq",
    "pa_orders": "",
    "use_aux_spectral_stats": False,
    "channel_trim_scale": 1.0,
    "fast_infer_when_no_aux": True,
    "use_mixstyle": False,
    "mixstyle_p": 0.0,
    "mixstyle_alpha": 0.1,
    "mixstyle_eps": 1e-6,
    "mixstyle_layers": "",
    "mixstyle_use_domain_label": False,
    "mixstyle_mix": "random",
    "mixstyle_strength": 1.0,
    "mixstyle_fallback": "none",
    "train_mode": "",
    "arch_family": "cvsincnet",
}


DATA_DEFAULTS: Dict[str, Any] = {
    "wisig_pkl": "./Dataset_WigSig/ManySig.pkl",
    "wisig_equalized": "1",
    "wisig_domain": "rx_day",
    "wisig_out_len": 256,
    "wisig_train_ratio": 0.2,
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
    "wisig_train_shots_per_class": 0,
    "seed": 1337,
    "num_classes": 16,
}


def _parse_csv_indices(text: str):
    return train_mod.parse_csv_indices(text)


def _explicit_target_requested(args: argparse.Namespace) -> bool:
    days = str(getattr(args, "explicit_test_days", "") or "").strip()
    receivers = str(getattr(args, "explicit_test_rxs", "") or "").strip()
    if bool(days) != bool(receivers):
        raise ValueError("--explicit_test_days and --explicit_test_rxs must be provided together")
    return bool(days)


def _build_explicit_target_loader(args: SimpleNamespace, overrides: argparse.Namespace, device: torch.device):
    days = _parse_csv_indices(overrides.explicit_test_days)
    receivers = _parse_csv_indices(overrides.explicit_test_rxs)
    ds_w = load_wisig_compact_pkl(str(args.wisig_pkl))
    day_labels = list(ds_w.get("capture_date_list", []))
    receiver_labels = list(ds_w.get("rx_list", []))
    if not days or any(index < 0 or index >= len(day_labels) for index in days):
        raise ValueError(f"invalid explicit target days: {days}")
    if not receivers or any(index < 0 or index >= len(receiver_labels) for index in receivers):
        raise ValueError(f"invalid explicit target receivers: {receivers}")
    source_receivers = set(_parse_csv_indices(str(getattr(args, "wisig_train_rxs", ""))))
    overlap = sorted(source_receivers.intersection(receivers))
    if overlap:
        raise ValueError(f"explicit target receivers overlap source receivers: {overlap}")
    equalized = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
    cap = int(getattr(args, "wisig_max_test_per_combo", 0) or 0)
    target_ds = WiSigCompactDataset(
        ds_w,
        out_len=int(args.wisig_out_len),
        crop_mode="center",
        normalize=True,
        equalized=equalized,
        day_keep=days,
        rx_keep=receivers,
        domain=str(args.wisig_domain),
        max_samples_per_combo=None if cap <= 0 else cap,
        sample_strategy=str(getattr(args, "wisig_cap_strategy", "random")),
        seed=int(args.seed),
        build_index=True,
    )
    loader = make_loader(
        target_ds,
        int(overrides.eval_batch_size),
        False,
        int(overrides.num_workers),
        device,
        False,
        int(overrides.prefetch_factor),
    )
    meta = {
        "days_idx": list(days),
        "days_label": [day_labels[index] for index in days],
        "rxs_idx": list(receivers),
        "rxs_label": [receiver_labels[index] for index in receivers],
        "size": len(target_ds),
        "target_access": True,
        "state_updates": False,
    }
    return loader, meta


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _args_from_checkpoint(ckpt: Dict[str, Any], overrides: argparse.Namespace) -> SimpleNamespace:
    saved = {}
    saved.update(DATA_DEFAULTS)
    saved.update(MODEL_DEFAULTS)
    saved.update(dict(ckpt.get("args") or {}))
    saved["device"] = overrides.device
    saved["eval_batch_size"] = int(overrides.eval_batch_size or saved.get("eval_batch_size", 256))
    saved["num_workers"] = int(overrides.num_workers)
    saved["prefetch_factor"] = int(overrides.prefetch_factor)
    saved["eval_sat_scenarios"] = str(overrides.scenarios)
    saved["sat_eval_max_batches"] = int(overrides.max_batches)
    saved["sat_seed"] = int(overrides.sat_seed)
    saved["sample_rate_hz"] = float(saved.get("sample_rate_hz") or 25e6)
    return SimpleNamespace(**saved)


def _build_named_loaders(args: SimpleNamespace, device: torch.device):
    ds_w = load_wisig_compact_pkl(str(args.wisig_pkl))
    infer_nc = len(ds_w.get("tx_list", []))
    if infer_nc > 0:
        args.num_classes = int(infer_nc)
    eq = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
    max_day123 = None if int(args.wisig_max_day123_per_combo) <= 0 else int(args.wisig_max_day123_per_combo)
    max_tr = None if int(args.wisig_max_train_per_combo) <= 0 else int(args.wisig_max_train_per_combo)
    max_tr_class = None if int(getattr(args, "wisig_train_shots_per_class", 0)) <= 0 else int(args.wisig_train_shots_per_class)
    max_va = None if int(args.wisig_max_val_per_combo) <= 0 else int(args.wisig_max_val_per_combo)
    max_te = None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo)

    train_ds, _val_ds, _test_ds, named_tests, named_test_meta, split_info = make_wisig_trainval_test_by_day_rx(
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
        train_days=_parse_csv_indices(args.wisig_train_days),
        test_days=_parse_csv_indices(args.wisig_test_days),
        train_rxs=_parse_csv_indices(args.wisig_train_rxs),
        test_rxs=_parse_csv_indices(args.wisig_test_rxs),
        max_samples_per_combo_day123=max_day123,
        max_samples_per_combo_test=max_te,
        max_samples_per_combo_train=max_tr,
        max_samples_per_combo_val=max_va,
        max_samples_per_class_train=max_tr_class,
        seed=int(args.seed),
        split_strategy=str(args.wisig_split_strategy),
        cap_strategy=str(args.wisig_cap_strategy),
        train_class_cap_strategy=str(getattr(args, "wisig_train_shot_strategy", "random")),
    )
    named_loaders = {
        name: make_loader(ds, int(args.eval_batch_size), False, int(args.num_workers), device, False, int(args.prefetch_factor))
        for name, ds in named_tests.items()
    }
    domain_label_map = train_mod.build_domain_label_map(train_ds)
    return named_loaders, named_test_meta, domain_label_map, split_info


def _build_model(args: SimpleNamespace, num_domains: int, input_len: int, device: torch.device):
    parsed_pa_orders = train_mod.parse_pa_orders_arg(getattr(args, "pa_orders", ""))
    return build_dual_model(
        int(args.num_classes),
        int(num_domains),
        model_size=str(getattr(args, "model_size", "small")),
        dataset=str(getattr(args, "dataset", "wisig")),
        input_len=int(input_len),
        sample_rate_hz=float(args.sample_rate_hz),
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        model_variant=str(getattr(args, "model_variant", "dual")),
        branch_ablation=str(getattr(args, "branch_ablation", "none")),
        mixstyle_on=_bool(getattr(args, "use_mixstyle", False)),
        mixstyle_p=float(getattr(args, "mixstyle_p", 0.0)),
        mixstyle_alpha=float(getattr(args, "mixstyle_alpha", 0.1)),
        mixstyle_eps=float(getattr(args, "mixstyle_eps", 1e-6)),
        mixstyle_layers=str(getattr(args, "mixstyle_layers", "")),
        mixstyle_use_domain_label=_bool(getattr(args, "mixstyle_use_domain_label", False)),
        mixstyle_mix=str(getattr(args, "mixstyle_mix", "random")),
        mixstyle_strength=float(getattr(args, "mixstyle_strength", 1.0)),
        mixstyle_fallback=str(getattr(args, "mixstyle_fallback", "none")),
        domain_branch_ablation=str(getattr(args, "domain_branch_ablation", "none")),
        domain_enhancer=str(getattr(args, "domain_enhancer", "rcn_stats")),
        domain_enhancer_strength=float(getattr(args, "domain_enhancer_strength", 0.35)),
        id_time_stability_mode=str(getattr(args, "id_time_stability_mode", "off")),
        id_freq_stability_mode=str(getattr(args, "id_freq_stability_mode", "off")),
        domain_time_stability_mode=str(getattr(args, "domain_time_stability_mode", "off")),
        domain_freq_stability_mode=str(getattr(args, "domain_freq_stability_mode", "off")),
        time_stability_channels=int(getattr(args, "time_stability_channels", 8)),
        freq_stability_channels=int(getattr(args, "freq_stability_channels", 8)),
        use_circularity=_bool(getattr(args, "use_circularity", True)),
        use_freq_stats=_bool(getattr(args, "use_freq_stats", False)),
        use_pa_stats=_bool(getattr(args, "use_pa_stats", False)),
        use_freq_band_gate=_bool(getattr(args, "use_freq_band_gate", False)),
        freq_feature_source=str(getattr(args, "freq_feature_source", "raw_fft")),
        pa_feature_source=str(getattr(args, "pa_feature_source", "raw_iq")),
        pa_orders=(parsed_pa_orders or None),
        use_aux_spectral_stats=_bool(getattr(args, "use_aux_spectral_stats", False)),
        channel_trim_scale=float(getattr(args, "channel_trim_scale", 1.0)),
        fast_infer_when_no_aux=_bool(getattr(args, "fast_infer_when_no_aux", True)),
        use_tx_adv_on_zdom=_bool(getattr(args, "use_tx_adv_on_zdom", False)) or str(getattr(args, "train_mode", "")).lower() == "fedcvs_vmb",
        arch_family=str(getattr(args, "arch_family", "cvsincnet")),
    ).to(device)


def _load_checkpoint_state(
    model,
    state: Dict[str, Any],
    *,
    strict_reconstruction: bool,
) -> Dict[str, Any]:
    incompatible = model.load_state_dict(state, strict=bool(strict_reconstruction))
    missing = list(getattr(incompatible, "missing_keys", incompatible[0]))
    unexpected = list(getattr(incompatible, "unexpected_keys", incompatible[1]))
    return {
        "checkpoint_load_strict": bool(strict_reconstruction),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "shape_mismatches": 0,
    }


def _build_exact_ssdg_context(
    ckpt: Dict[str, Any],
    overrides: argparse.Namespace,
    device: torch.device,
    *,
    strict_reconstruction: bool = False,
):
    if ssdg_mod is None:
        raise ImportError("SSDG.train_ssdg could not be imported for exact checkpoint reconstruction")
    parser = ssdg_mod.build_arg_parser()
    args = parser.parse_args(["--output_dir", os.path.join(PROJECT_ROOT, ".tmp_eval_ssdg_sat_per_rx")])
    for key, value in dict(ckpt.get("args") or {}).items():
        setattr(args, key, value)
    args.device = overrides.device
    args.eval_batch_size = int(overrides.eval_batch_size)
    args.num_workers = int(overrides.num_workers)
    args.prefetch_factor = int(overrides.prefetch_factor)
    args.eval_sat_channel = True
    args.eval_sat_scenarios = str(overrides.scenarios)
    args.sat_eval_max_batches = int(overrides.max_batches)
    args.sat_seed = int(overrides.sat_seed)
    args.sample_rate_hz = float(getattr(args, "sample_rate_hz", 0.0) or 25e6)
    if hasattr(ssdg_mod, "set_seed"):
        ssdg_mod.set_seed(int(getattr(args, "seed", 1337)))
    data_ctx = ssdg_mod._build_ssdg_wisig_data(args, device)
    model_args = ssdg_mod.merge_checkpoint_args(
        ckpt,
        args,
        input_len=int(data_ctx["input_len"]),
        num_domains=int(data_ctx["num_domains"]),
    )
    model_args = ssdg_mod._apply_model_cli_args(model_args, args)
    model = ssdg_mod.build_baseline_model(model_args, device)
    load_audit = _load_checkpoint_state(
        model,
        ckpt["model"],
        strict_reconstruction=bool(strict_reconstruction),
    )
    return (
        model,
        data_ctx["named_test_loaders"],
        data_ctx["split_info"].get("named_test_meta", {}),
        data_ctx["domain_label_map"],
        data_ctx["split_info"],
        args,
        load_audit,
    )


def _build_direct_context(
    ckpt: Dict[str, Any],
    overrides: argparse.Namespace,
    device: torch.device,
):
    args = _args_from_checkpoint(ckpt, overrides)
    named_loaders, named_meta, domain_label_map, split_info = _build_named_loaders(args, device)
    model = _build_model(args, len(domain_label_map), int(args.wisig_out_len), device)
    load_audit = _load_checkpoint_state(
        model,
        ckpt["model"],
        strict_reconstruction=False,
    )
    return (
        model,
        named_loaders,
        named_meta,
        domain_label_map,
        split_info,
        args,
        load_audit,
    )


def _build_evaluation_context(
    ckpt: Dict[str, Any],
    overrides: argparse.Namespace,
    device: torch.device,
    *,
    strict_reconstruction: bool,
):
    try:
        context = _build_exact_ssdg_context(
            ckpt,
            overrides,
            device,
            strict_reconstruction=bool(strict_reconstruction),
        )
        reconstruction = "SSDG.train_ssdg"
        fallback_used = False
    except Exception as exc:
        if strict_reconstruction:
            raise RuntimeError(f"strict SSDG reconstruction failed: {exc}") from exc
        print(f"[WARN] exact SSDG reconstruction failed, falling back to direct builder: {exc}", flush=True)
        context = _build_direct_context(ckpt, overrides, device)
        reconstruction = "direct_builder_fallback"
        fallback_used = True
    *base_context, load_audit = context
    reconstruction_audit = {
        "strict_requested": bool(strict_reconstruction),
        **dict(load_audit),
        "fallback_used": bool(fallback_used),
    }
    return (*base_context, reconstruction, reconstruction_audit)


def _select_names(named_loaders: Dict[str, Any], spec: str) -> List[str]:
    raw = str(spec or "unseen_rx").strip()
    if raw.lower() in {"unseen_rx", "unseen_day_rx", "target_rx"}:
        names = [name for name in named_loaders if name.startswith("test_unseen_day_rx_")]
        return sorted(names, key=lambda item: int(item.rsplit("_", 1)[-1]))
    if raw.lower() in {"seen_rx", "seen_day_rx", "train_day_rx"}:
        names = [name for name in named_loaders if name.startswith("test_rx_")]
        return sorted(names, key=lambda item: int(item.rsplit("_", 1)[-1]))
    if raw.lower() in {"all_rx", "rx"}:
        names = [name for name in named_loaders if name.startswith("test_rx_") or name.startswith("test_unseen_day_rx_")]
        return sorted(names, key=lambda item: (0 if item.startswith("test_rx_") else 1, int(item.rsplit("_", 1)[-1])))
    names = []
    for part in raw.replace(";", ",").split(","):
        name = part.strip()
        if name in named_loaders and name not in names:
            names.append(name)
    if not names:
        raise ValueError(f"No named loaders matched --eval_on={spec!r}")
    return names


def _aggregate(rows: List[Dict[str, Any]], scenario: str) -> Dict[str, Any]:
    total = sum(int(row["sat_total"]) for row in rows if row["scenario"] == scenario)
    correct = sum(int(row["sat_correct"]) for row in rows if row["scenario"] == scenario)
    return {
        "scenario": scenario,
        "tx_acc": 100.0 * correct / max(1, total),
        "tx_correct": int(correct),
        "tx_total": int(total),
    }


def _group_value(extra: Any, group_key: str, device: torch.device) -> torch.Tensor:
    if not isinstance(extra, dict) or group_key not in extra:
        raise KeyError(f"Batch extra does not contain group key {group_key!r}")
    value = extra[group_key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(device=device, non_blocking=True).long()


def _group_values(extra: Any, group_key: str, device: torch.device) -> torch.Tensor:
    keys = tuple(part.strip() for part in str(group_key).split(",") if part.strip())
    if not keys:
        raise ValueError("--group_key must contain at least one metadata key")
    return torch.stack([_group_value(extra, key, device) for key in keys], dim=1)


def _group_identity(group_id: Any, group_key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    keys = tuple(part.strip() for part in str(group_key).split(",") if part.strip())
    values = group_id if isinstance(group_id, tuple) else (int(group_id),)
    if len(keys) != len(values):
        raise ValueError(f"group identity mismatch: keys={keys} values={values}")
    identity: Dict[str, Any] = {}
    rx_labels = dict(zip(meta.get("rxs_idx", []), meta.get("rxs_label", [])))
    day_labels = dict(zip(meta.get("days_idx", []), meta.get("days_label", [])))
    for key, value in zip(keys, values):
        if key == "rx_i":
            identity["rx_idx"] = int(value)
            identity["rx_label"] = rx_labels.get(int(value), "")
        elif key == "day_i":
            identity["day_idx"] = int(value)
            identity["day_label"] = day_labels.get(int(value), "")
        else:
            identity[key] = int(value)
    return identity


def _evaluate_loader_grouped(model, loader, device, *, group_key: str, args=None, scenario: str | None = None, seed: int = 0):
    model.eval()
    stats: Dict[Any, Dict[str, int]] = {}
    gen = make_torch_generator(device, int(seed)) if scenario is not None else None
    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            y = batch[1]
            extra = batch[3] if isinstance(batch, (tuple, list)) and len(batch) > 3 and isinstance(batch[3], dict) else None
            if extra is None:
                _x, _y, unpacked_extra = unpack_batch(batch)
                x, y = _x, _y
                extra = unpacked_extra
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            groups = _group_values(extra, group_key, device)
            if scenario is not None:
                x, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
            out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
            pred = out["tx_logits"].argmax(dim=1)
            for group_values in torch.unique(groups, dim=0).detach().cpu().tolist():
                group_tuple = tuple(int(value) for value in group_values)
                group_id: Any = group_tuple[0] if len(group_tuple) == 1 else group_tuple
                expected = torch.as_tensor(group_values, device=device, dtype=groups.dtype)
                mask = (groups == expected).all(dim=1)
                correct = int((pred[mask] == y[mask]).sum().item())
                total = int(mask.sum().item())
                slot = stats.setdefault(int(group_id), {"tx_correct": 0, "tx_total": 0})
                slot["tx_correct"] += correct
                slot["tx_total"] += total
    return {
        group_id: {
            "tx_correct": values["tx_correct"],
            "tx_total": values["tx_total"],
            "tx_acc": 100.0 * values["tx_correct"] / max(1, values["tx_total"]),
        }
        for group_id, values in sorted(stats.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSDG checkpoint satellite-channel accuracy by receiver domain.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--eval_on", default="unseen_rx")
    parser.add_argument("--group_loader", default="", help="Optional named loader to evaluate once and group by --group_key.")
    parser.add_argument("--group_key", default="rx_i")
    parser.add_argument("--scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--sat_seed", type=int, default=2027)
    parser.add_argument("--strict_reconstruction", action="store_true")
    parser.add_argument("--explicit_test_days", default="")
    parser.add_argument("--explicit_test_rxs", default="")
    parser.add_argument("--explicit_group_name", default="explicit_target")
    args_cli = parser.parse_args()

    device = torch.device(args_cli.device if torch.cuda.is_available() or not str(args_cli.device).startswith("cuda") else "cpu")
    ckpt = torch.load(args_cli.ckpt, map_location=device)
    (
        model,
        named_loaders,
        named_meta,
        domain_label_map,
        split_info,
        args,
        reconstruction,
        reconstruction_audit,
    ) = _build_evaluation_context(
        ckpt,
        args_cli,
        device,
        strict_reconstruction=bool(args_cli.strict_reconstruction),
    )
    explicit_target = _explicit_target_requested(args_cli)
    if explicit_target:
        explicit_name = str(args_cli.explicit_group_name or "explicit_target").strip()
        if not explicit_name or explicit_name in named_loaders:
            raise ValueError(f"invalid or colliding --explicit_group_name={explicit_name!r}")
        explicit_loader, explicit_meta = _build_explicit_target_loader(args, args_cli, device)
        named_loaders[explicit_name] = explicit_loader
        named_meta[explicit_name] = explicit_meta
        split_info = dict(split_info)
        split_info["explicit_target_eval"] = dict(explicit_meta)
    group_loader = str(args_cli.group_loader or "").strip()
    if explicit_target and not group_loader:
        group_loader = str(args_cli.explicit_group_name).strip()
    if not group_loader and str(args_cli.eval_on).strip().lower() in {"unseen_rx", "unseen_day_rx", "target_rx"}:
        group_loader = "test_unseen_day_unseen_rx"
    if group_loader:
        if group_loader not in named_loaders:
            raise ValueError(f"--group_loader={group_loader!r} not found in named loaders")
        selected_names = [group_loader]
    else:
        selected_names = _select_names(named_loaders, args_cli.eval_on)
    if reconstruction_audit["missing_keys"] or reconstruction_audit["unexpected_keys"]:
        print(
            "[WARN] checkpoint load "
            f"missing={reconstruction_audit['missing_keys']} "
            f"unexpected={reconstruction_audit['unexpected_keys']}",
            flush=True,
        )

    max_batches = int(args_cli.max_batches)
    scenarios = parse_sat_scenarios(args_cli.scenarios)
    rows: List[Dict[str, Any]] = []
    if group_loader:
        if max_batches > 0:
            raise ValueError("Grouped exact evaluation requires full loader; use --max_batches -1 or 0")
        meta = named_meta.get(group_loader, {})
        clean_by_group = _evaluate_loader_grouped(model, named_loaders[group_loader], device, group_key=args_cli.group_key)
        try:
            loader_seed_index = MAIN_SAT_EVAL_ON_NAMES.index(group_loader)
        except ValueError:
            loader_seed_index = 0
        for si, scenario in enumerate(scenarios):
            sat_by_group = _evaluate_loader_grouped(
                model,
                named_loaders[group_loader],
                device,
                group_key=args_cli.group_key,
                args=args,
                scenario=scenario,
                seed=int(args_cli.sat_seed) + si * 1009 + loader_seed_index * 97,
            )
            for group_id in sorted(clean_by_group):
                if group_id not in sat_by_group:
                    continue
                clean = clean_by_group[group_id]
                sat = sat_by_group[group_id]
                identity = _group_identity(group_id, args_cli.group_key, meta)
                row = {
                    "name": f"{group_loader}:{args_cli.group_key}_{group_id}",
                    **identity,
                    "days_label": identity.get("day_label", ",".join(str(v) for v in meta.get("days_label", []))),
                    "scenario": scenario,
                    "clean_acc": float(clean["tx_acc"]),
                    "clean_correct": int(clean["tx_correct"]),
                    "clean_total": int(clean["tx_total"]),
                    "sat_acc": float(sat["tx_acc"]),
                    "sat_correct": int(sat["tx_correct"]),
                    "sat_total": int(sat["tx_total"]),
                    "delta_pp": float(sat["tx_acc"]) - float(clean["tx_acc"]),
                }
                rows.append(row)
                print(
                    f"[PER-RX-SAT] scenario={scenario} rx={row.get('rx_idx', '')} label={row.get('rx_label', '')} "
                    f"day={row.get('day_idx', '')} day_label={row.get('day_label', '')} "
                    f"clean={row['clean_acc']:.2f}% sat={row['sat_acc']:.2f}% "
                    f"delta={row['delta_pp']:.2f}pp ({row['sat_correct']}/{row['sat_total']})",
                    flush=True,
                )
    else:
        clean_by_name = {
            name: evaluate_loader(model, named_loaders[name], device, domain_label_map=domain_label_map, max_batches=max_batches)
            for name in selected_names
        }
        for si, scenario in enumerate(scenarios):
            for li, name in enumerate(selected_names):
                sat = evaluate_loader_sat_channel(
                    model,
                    named_loaders[name],
                    device,
                    domain_label_map=domain_label_map,
                    scenario=scenario,
                    args=args,
                    max_batches=max_batches,
                    seed=int(args_cli.sat_seed) + si * 1009 + li * 97,
                )
                clean = clean_by_name[name]
                meta = named_meta.get(name, {})
                row = {
                    "name": name,
                    "rx_idx": (meta.get("rxs_idx") or [""])[0],
                    "rx_label": (meta.get("rxs_label") or [""])[0],
                    "days_label": ",".join(str(v) for v in meta.get("days_label", [])),
                    "scenario": scenario,
                    "clean_acc": float(clean["tx_acc"]),
                    "clean_correct": int(clean["tx_correct"]),
                    "clean_total": int(clean["tx_total"]),
                    "sat_acc": float(sat["tx_acc"]),
                    "sat_correct": int(sat["tx_correct"]),
                    "sat_total": int(sat["tx_total"]),
                    "delta_pp": float(sat["tx_acc"]) - float(clean["tx_acc"]),
                }
                rows.append(row)
                print(
                    f"[PER-RX-SAT] scenario={scenario} rx={row['rx_idx']} label={row['rx_label']} "
                    f"clean={row['clean_acc']:.2f}% sat={row['sat_acc']:.2f}% "
                    f"delta={row['delta_pp']:.2f}pp ({row['sat_correct']}/{row['sat_total']})",
                    flush=True,
                )

    aggregates = [_aggregate(rows, scenario) for scenario in scenarios]
    for agg in aggregates:
        print(
            f"[PER-RX-SAT-AGG] scenario={agg['scenario']} tx_acc={agg['tx_acc']:.2f}% "
            f"({agg['tx_correct']}/{agg['tx_total']})",
            flush=True,
        )

    payload = {
        "schema": "ssdg_sat_per_rx_eval_v1",
        "checkpoint": os.path.abspath(args_cli.ckpt),
        "checkpoint_epoch": ckpt.get("epoch"),
        "run_name": str(getattr(args, "run_name", "")),
        "reconstruction": reconstruction,
        "reconstruction_audit": reconstruction_audit,
        "eval_on": str(args_cli.eval_on),
        "group_loader": group_loader,
        "group_key": str(args_cli.group_key),
        "selected_names": selected_names,
        "scenarios": scenarios,
        "max_batches": max_batches,
        "sat_seed": int(args_cli.sat_seed),
        "split": split_info,
        "rows": rows,
        "aggregates": aggregates,
    }
    output_json = os.path.abspath(args_cli.output_json)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    output_csv = str(args_cli.output_csv or "").strip()
    if output_csv:
        output_csv = os.path.abspath(output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    print(f"[OUTPUT] json={output_json} csv={output_csv}", flush=True)


if __name__ == "__main__":
    main()
