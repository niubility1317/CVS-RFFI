from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

PROJECT_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_CODE_DIR, os.pardir))
if PROJECT_CODE_DIR not in sys.path:
    sys.path.insert(0, PROJECT_CODE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch

import train as train_mod
from cvsrffi.eval import evaluate_loader, evaluate_loader_sat_channel, make_loader
from dataset_wisig import load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx
from model_dual_cvsincnet import build_dual_model
from training_controls import parse_sat_scenarios


def _parse_csv_indices(text: str):
    return train_mod.parse_csv_indices(text)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _args_from_checkpoint(ckpt: Dict[str, Any], overrides: argparse.Namespace) -> SimpleNamespace:
    saved = dict(ckpt.get("args") or {})
    saved["device"] = overrides.device
    saved["eval_batch_size"] = int(overrides.eval_batch_size or saved.get("eval_batch_size", 256))
    saved["num_workers"] = int(overrides.num_workers)
    saved["prefetch_factor"] = int(overrides.prefetch_factor)
    saved["eval_sat_scenarios"] = str(overrides.scenarios)
    saved["sat_eval_max_batches"] = int(overrides.max_batches)
    saved["sat_seed"] = int(overrides.sat_seed)
    saved["sample_rate_hz"] = float(saved.get("sample_rate_hz") or 25e6)
    return SimpleNamespace(**saved)


def _build_val_loader(args: SimpleNamespace, device: torch.device):
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

    train_ds, val_ds, _test_ds, _named_tests, _named_test_meta, split_info = make_wisig_trainval_test_by_day_rx(
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
    val_loader = make_loader(
        val_ds,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        device,
        False,
        int(args.prefetch_factor),
    )
    domain_label_map = train_mod.build_domain_label_map(train_ds)
    return train_ds, val_ds, val_loader, domain_label_map, split_info


def _build_model(args: SimpleNamespace, num_domains: int, input_len: int, device: torch.device):
    parsed_pa_orders = train_mod.parse_pa_orders_arg(getattr(args, "pa_orders", ""))
    model = build_dual_model(
        int(args.num_classes),
        int(num_domains),
        model_size=str(args.model_size),
        dataset=str(args.dataset),
        input_len=int(input_len),
        sample_rate_hz=float(args.sample_rate_hz),
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        model_variant=str(args.model_variant),
        branch_ablation=str(args.branch_ablation),
        mixstyle_on=_bool(args.use_mixstyle),
        mixstyle_p=float(args.mixstyle_p),
        mixstyle_alpha=float(args.mixstyle_alpha),
        mixstyle_eps=float(args.mixstyle_eps),
        mixstyle_layers=str(args.mixstyle_layers),
        mixstyle_use_domain_label=_bool(args.mixstyle_use_domain_label),
        mixstyle_mix=str(args.mixstyle_mix),
        mixstyle_strength=float(args.mixstyle_strength),
        mixstyle_fallback=str(args.mixstyle_fallback),
        domain_branch_ablation=str(args.domain_branch_ablation),
        domain_enhancer=str(args.domain_enhancer),
        domain_enhancer_strength=float(args.domain_enhancer_strength),
        id_time_stability_mode=str(args.id_time_stability_mode),
        id_freq_stability_mode=str(args.id_freq_stability_mode),
        domain_time_stability_mode=str(args.domain_time_stability_mode),
        domain_freq_stability_mode=str(args.domain_freq_stability_mode),
        time_stability_channels=int(args.time_stability_channels),
        freq_stability_channels=int(args.freq_stability_channels),
        use_circularity=_bool(args.use_circularity),
        use_freq_stats=_bool(args.use_freq_stats),
        use_pa_stats=_bool(args.use_pa_stats),
        use_freq_band_gate=_bool(args.use_freq_band_gate),
        freq_feature_source=str(args.freq_feature_source),
        pa_feature_source=str(args.pa_feature_source),
        pa_orders=(parsed_pa_orders or None),
        use_aux_spectral_stats=_bool(args.use_aux_spectral_stats),
        channel_trim_scale=float(args.channel_trim_scale),
        fast_infer_when_no_aux=_bool(args.fast_infer_when_no_aux),
        use_tx_adv_on_zdom=_bool(args.use_tx_adv_on_zdom) or str(args.train_mode).lower() == "fedcvs_vmb",
        arch_family=str(args.arch_family),
    ).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simplified LEO satellite channel on the H06 source validation split.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--sat_seed", type=int, default=2027)
    args_cli = parser.parse_args()

    device = torch.device(args_cli.device if torch.cuda.is_available() or not str(args_cli.device).startswith("cuda") else "cpu")
    ckpt = torch.load(args_cli.ckpt, map_location=device)
    args = _args_from_checkpoint(ckpt, args_cli)
    train_ds, val_ds, val_loader, domain_label_map, split_info = _build_val_loader(args, device)
    model = _build_model(args, len(domain_label_map), int(args.wisig_out_len), device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[WARN] checkpoint load missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    max_batches = int(args_cli.max_batches)
    clean = evaluate_loader(model, val_loader, device, domain_label_map=domain_label_map, max_batches=max_batches)
    scenarios = parse_sat_scenarios(args_cli.scenarios)
    sat = {}
    for i, scenario in enumerate(scenarios):
        stats = evaluate_loader_sat_channel(
            model,
            val_loader,
            device,
            domain_label_map=domain_label_map,
            scenario=scenario,
            args=args,
            max_batches=max_batches,
            seed=int(args_cli.sat_seed) + i * 1009,
        )
        sat[scenario] = stats
        print(
            f"[VAL-SAT] scenario={scenario} tx_acc={stats['tx_acc']:.2f}% "
            f"({stats['tx_correct']}/{stats['tx_total']})",
            flush=True,
        )
    mean = sum(float(v["tx_acc"]) for v in sat.values()) / max(1, len(sat))
    payload = {
        "schema": "h06_val_simplified_leo_eval_v1",
        "checkpoint": os.path.abspath(args_cli.ckpt),
        "checkpoint_epoch": ckpt.get("epoch"),
        "run_name": str(getattr(args, "run_name", "")),
        "split": {
            "role": "source_validation",
            "val_size": len(val_ds),
            "train_size": len(train_ds),
            "train_days": split_info.get("train_days_label") if isinstance(split_info, dict) else None,
            "train_rxs": split_info.get("train_rxs_idx") if isinstance(split_info, dict) else None,
            "train_ratio": split_info.get("train_ratio") if isinstance(split_info, dict) else None,
        },
        "clean_val": clean,
        "sat_val": sat,
        "sat_val_mean_tx_acc": mean,
        "scenarios": scenarios,
        "max_batches": max_batches,
        "sat_seed": int(args_cli.sat_seed),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_json)), exist_ok=True)
    with open(args_cli.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[VAL-CLEAN] tx_acc={clean['tx_acc']:.2f}% ({clean['tx_correct']}/{clean['tx_total']})", flush=True)
    print(f"[VAL-SAT-MEAN] tx_acc={mean:.2f}% scenarios={','.join(scenarios)}", flush=True)
    print(f"[OUTPUT] {args_cli.output_json}", flush=True)


if __name__ == "__main__":
    main()
