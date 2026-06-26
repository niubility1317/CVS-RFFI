from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from dataset_wisig import load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx
from federated.reliability_fusion import collaborative_probability_fusion, normalize_probabilities
from model_dual_cvsincnet import build_dual_model
from cvsrffi.eval import MAIN_SAT_EVAL_ON, apply_sat_channel_for_scenario, make_loader, resolve_sat_eval_loader_names
from cvsrffi.tensors import build_domain_label_map, make_torch_generator, parse_csv_indices, unpack_batch
from training_controls import parse_sat_scenarios


def parse_collab_counts(spec: str | Sequence[int] | None, *, receiver_count: int) -> list[int]:
    receiver_count = int(receiver_count)
    if receiver_count < 1:
        raise ValueError(f"receiver_count must be >= 1, got {receiver_count}")
    if spec is None or str(spec).strip().lower() in {"", "all", "*", "1..n"}:
        return list(range(1, receiver_count + 1))
    if isinstance(spec, str):
        raw_items = [item.strip() for item in spec.replace(";", ",").split(",") if item.strip()]
    else:
        raw_items = [str(item) for item in spec]
    counts: list[int] = []
    for item in raw_items:
        k = int(item)
        if k < 1 or k > receiver_count:
            raise ValueError(f"collaborative receiver count {k} is outside valid range 1..{receiver_count}")
        if k not in counts:
            counts.append(k)
    if not counts:
        raise ValueError("no collaborative receiver counts were requested")
    return counts


def _meta_value(meta: Mapping[str, Any], key: str, index: int) -> Any:
    value = meta[key]
    if torch.is_tensor(value):
        return value[index].detach().cpu().item()
    if isinstance(value, (list, tuple)):
        return value[index]
    try:
        return value[index]
    except Exception:
        return value


def _forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    try:
        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
    except TypeError:
        out = model(x)
    if isinstance(out, Mapping):
        logits = out.get("tx_logits", out.get("logits"))
    else:
        logits = out
    if not torch.is_tensor(logits):
        raise TypeError("model forward must return logits or a mapping with tx_logits/logits")
    return logits.float()


def _finalize_counts(raw: Mapping[str, int]) -> dict[str, Any]:
    out = {
        key: int(raw.get(key, 0))
        for key in ("total", "base_correct", "fused_correct", "rescue", "harm", "excluded_incomplete_groups")
    }
    out["net_gain"] = int(out["rescue"] - out["harm"])
    total = max(1, int(out["total"]))
    out["base_tx_acc"] = 100.0 * float(out["base_correct"]) / float(total)
    out["fused_tx_acc"] = 100.0 * float(out["fused_correct"]) / float(total)
    return out


def _extract_meta(extra: Sequence[Any]) -> Mapping[str, Any]:
    if len(extra) >= 2 and isinstance(extra[1], Mapping):
        return extra[1]
    if len(extra) >= 1 and isinstance(extra[0], Mapping):
        return extra[0]
    raise ValueError("collaborative receiver evaluation requires WiSig metadata with tx_i/rx_i/day_i/eq_i/sig_i")


def evaluate_collaborative_receiver_fusion(
    model,
    loader,
    device: torch.device,
    *,
    collab_counts: Sequence[int] | None = None,
    fusion: str = "soft",
    max_batches: int = 0,
    input_transform=None,
) -> dict[str, Any]:
    """Evaluate receiver-aligned probability fusion.

    A collaborative group is keyed by `(tx_i, day_i, eq_i, sig_i)`. `K` means
    the total number of receiver observations used from that group. Therefore
    `K=1` is the group-level single-receiver baseline.
    """

    model.eval()
    groups: "OrderedDict[tuple[int, int, int, int], list[dict[str, Any]]]" = OrderedDict()
    receiver_ids: set[int] = set()
    num_classes = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= int(max_batches):
                break
            x, y, extra = unpack_batch(batch)
            meta = _extract_meta(extra)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1).long()
            if input_transform is not None:
                x = input_transform(x, batch_idx)
            probs = normalize_probabilities(F.softmax(_forward_logits(model, x), dim=1)).detach().cpu()
            y_cpu = y.detach().cpu()
            num_classes = int(probs.size(1))
            for i in range(int(y_cpu.numel())):
                tx_i = int(_meta_value(meta, "tx_i", i))
                rx_i = int(_meta_value(meta, "rx_i", i))
                day_i = int(_meta_value(meta, "day_i", i))
                eq_i = int(_meta_value(meta, "eq_i", i))
                sig_i = int(_meta_value(meta, "sig_i", i))
                key = (tx_i, day_i, eq_i, sig_i)
                receiver_ids.add(rx_i)
                groups.setdefault(key, []).append(
                    {
                        "rx_i": rx_i,
                        "prob": probs[i],
                        "target": int(y_cpu[i].item()),
                    }
                )

    if not groups:
        raise ValueError("no collaborative groups were collected from the loader")

    unique_group_sizes = []
    deduped_groups = []
    for key, rows in groups.items():
        by_rx: OrderedDict[int, dict[str, Any]] = OrderedDict()
        for row in sorted(rows, key=lambda item: int(item["rx_i"])):
            by_rx.setdefault(int(row["rx_i"]), row)
        values = list(by_rx.values())
        unique_group_sizes.append(len(values))
        deduped_groups.append((key, values))

    receiver_count = len(receiver_ids)
    counts = parse_collab_counts(collab_counts, receiver_count=receiver_count)
    max_requested = max(int(k) for k in counts)
    eligible_groups = [(key, rows) for key, rows in deduped_groups if len(rows) >= max_requested]
    excluded_incomplete = int(len(deduped_groups) - len(eligible_groups))
    if not eligible_groups:
        raise ValueError(
            f"no collaborative groups contain {max_requested} receiver observations; "
            f"observed receiver_count={receiver_count}"
        )
    fusion = str(fusion or "soft").lower()
    results: dict[str, Any] = {}
    for k in counts:
        raw = {
            "total": 0,
            "base_correct": 0,
            "fused_correct": 0,
            "rescue": 0,
            "harm": 0,
            "excluded_incomplete_groups": excluded_incomplete,
        }
        for _, rows in eligible_groups:
            selected = rows[: int(k)]
            target = torch.tensor([int(selected[0]["target"])], dtype=torch.long)
            p_base = selected[0]["prob"].view(1, -1)
            if int(k) == 1:
                p_fused = p_base
            else:
                aux = torch.stack([row["prob"] for row in selected[1:]], dim=0).unsqueeze(1)
                p_fused = collaborative_probability_fusion(p_base, aux, mode=fusion)
            base_ok = bool(p_base.argmax(dim=1).eq(target).item())
            fused_ok = bool(p_fused.argmax(dim=1).eq(target).item())
            raw["total"] += 1
            raw["base_correct"] += int(base_ok)
            raw["fused_correct"] += int(fused_ok)
            raw["rescue"] += int((not base_ok) and fused_ok)
            raw["harm"] += int(base_ok and (not fused_ok))
        finalized = _finalize_counts(raw)
        finalized["requested_receivers"] = int(k)
        results[str(k)] = finalized

    return {
        "enabled": True,
        "protocol": "receiver_group_probability_fusion",
        "fusion": fusion,
        "receiver_count": int(receiver_count),
        "observed_receiver_ids": sorted(int(v) for v in receiver_ids),
        "group_count": int(len(groups)),
        "eligible_group_count": int(len(eligible_groups)),
        "excluded_incomplete_groups": excluded_incomplete,
        "num_classes": int(num_classes),
        "group_key": ["tx_i", "day_i", "eq_i", "sig_i"],
        "counts": results,
    }


def _get_arg(args: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = args.get(name, default)
    return default if value is None else value


def _none_if_nonpositive(value: Any) -> int | None:
    value = int(value or 0)
    return None if value <= 0 else value


def _load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    ckpt_path = Path(path)
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping, got {type(payload)}")
    return payload


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    expected_run_name: str | None = None,
    checkpoint_sha256: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    ckpt_args = dict(payload.get("args", {}) or {})
    run_name = str(ckpt_args.get("run_name", "") or "")
    output_dir = str(ckpt_args.get("output_dir", "") or "")
    checkpoint_path = str(checkpoint_path)
    identity_text = "\n".join([run_name, output_dir, checkpoint_path])
    expected_run_name = str(expected_run_name or "").strip()
    if expected_run_name and expected_run_name not in identity_text:
        raise ValueError(
            f"checkpoint identity mismatch: expected run token {expected_run_name!r} "
            f"not found in run_name/output_dir/path"
        )
    checkpoint_sha256 = str(checkpoint_sha256 or "").strip().lower()
    expected_sha256 = str(expected_sha256 or "").strip().lower()
    if expected_sha256 and checkpoint_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {expected_sha256}, got {checkpoint_sha256 or '<not computed>'}"
        )
    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "expected_run_name": expected_run_name,
        "expected_sha256": expected_sha256,
    }


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state dict")
    cleaned = OrderedDict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]
        cleaned[name] = value
    return cleaned


def build_wisig_context_from_checkpoint(payload: Mapping[str, Any], overrides: argparse.Namespace):
    ckpt_args = dict(payload.get("args", {}) or {})
    if str(_get_arg(ckpt_args, "dataset", "wisig")).lower() != "wisig":
        raise ValueError("collaborative receiver evaluation currently supports WiSig checkpoints only")
    pkl_path = str(overrides.wisig_pkl or _get_arg(ckpt_args, "wisig_pkl", "./Dataset_WigSig/ManySig.pkl"))
    ds_w = load_wisig_compact_pkl(pkl_path)
    infer_classes = len(ds_w.get("tx_list", []))
    num_classes = int(overrides.num_classes or _get_arg(ckpt_args, "num_classes", infer_classes or 16))
    if infer_classes > 0:
        num_classes = infer_classes
    equalized_raw = str(_get_arg(ckpt_args, "wisig_equalized", "1"))
    equalized = "both" if equalized_raw.lower() == "both" else int(equalized_raw)
    train_ratio = float(_get_arg(ckpt_args, "wisig_train_ratio", 0.1))
    train_ds, val_ds, test_ds, named_tests, named_test_meta, split_info = make_wisig_trainval_test_by_day_rx(
        ds_w,
        equalized=equalized,
        out_len=int(overrides.wisig_out_len or _get_arg(ckpt_args, "wisig_out_len", 256)),
        domain=str(overrides.wisig_domain or _get_arg(ckpt_args, "wisig_domain", "day")),
        normalize=True,
        crop_mode="center",
        transform_train=None,
        transform_eval=None,
        train_ratio=train_ratio,
        guard_gap=int(_get_arg(ckpt_args, "wisig_guard_gap", 8)),
        train_days=parse_csv_indices(_get_arg(ckpt_args, "wisig_train_days", "")),
        test_days=parse_csv_indices(_get_arg(ckpt_args, "wisig_test_days", "")),
        train_rxs=parse_csv_indices(_get_arg(ckpt_args, "wisig_train_rxs", "")),
        test_rxs=parse_csv_indices(_get_arg(ckpt_args, "wisig_test_rxs", "")),
        max_samples_per_combo_day123=_none_if_nonpositive(_get_arg(ckpt_args, "wisig_max_day123_per_combo", 0)),
        max_samples_per_combo_test=_none_if_nonpositive(
            overrides.max_samples_per_combo_test or _get_arg(ckpt_args, "wisig_max_test_per_combo", 0)
        ),
        max_samples_per_combo_train=_none_if_nonpositive(_get_arg(ckpt_args, "wisig_max_train_per_combo", 0)),
        max_samples_per_combo_val=_none_if_nonpositive(_get_arg(ckpt_args, "wisig_max_val_per_combo", 0)),
        seed=int(_get_arg(ckpt_args, "seed", 1337)),
    )
    domain_label_map = build_domain_label_map(train_ds)
    input_len = int(overrides.wisig_out_len or _get_arg(ckpt_args, "wisig_out_len", 256))
    return SimpleNamespace(
        ckpt_args=ckpt_args,
        num_classes=num_classes,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        named_tests=named_tests,
        named_test_meta=named_test_meta,
        split_info=split_info,
        domain_label_map=domain_label_map,
        num_domains=max(1, len(domain_label_map)),
        input_len=input_len,
    )


def build_model_from_checkpoint_args(context, overrides: argparse.Namespace, device: torch.device):
    args = context.ckpt_args
    sample_rate_hz = float(overrides.sample_rate_hz or _get_arg(args, "sample_rate_hz", 25e6))
    if sample_rate_hz <= 0.0:
        sample_rate_hz = 25e6
    model = build_dual_model(
        int(context.num_classes),
        int(context.num_domains),
        model_size=str(overrides.model_size or _get_arg(args, "model_size", "M")),
        dataset="wisig",
        input_len=int(context.input_len),
        sample_rate_hz=sample_rate_hz,
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        model_variant=str(overrides.model_variant or _get_arg(args, "model_variant", "lite_c")),
        branch_ablation=str(overrides.branch_ablation or _get_arg(args, "branch_ablation", "none")),
        mixstyle_on=bool(_get_arg(args, "use_mixstyle", False)),
        mixstyle_p=float(_get_arg(args, "mixstyle_p", 0.3)),
        mixstyle_alpha=float(_get_arg(args, "mixstyle_alpha", 0.1)),
        mixstyle_eps=float(_get_arg(args, "mixstyle_eps", 1e-6)),
        mixstyle_layers=str(_get_arg(args, "mixstyle_layers", "time_down,t1")),
        mixstyle_use_domain_label=bool(_get_arg(args, "mixstyle_use_domain_label", True)),
        mixstyle_mix=str(_get_arg(args, "mixstyle_mix", "crossdomain")),
        mixstyle_strength=float(_get_arg(args, "mixstyle_strength", 1.0)),
        mixstyle_fallback=str(_get_arg(args, "mixstyle_fallback", "random")),
        domain_branch_ablation=str(overrides.domain_branch_ablation or _get_arg(args, "domain_branch_ablation", "same")),
        domain_enhancer=str(_get_arg(args, "domain_enhancer", "rcn_stats")),
        domain_enhancer_strength=float(_get_arg(args, "domain_enhancer_strength", 0.35)),
        id_time_stability_mode=str(_get_arg(args, "id_time_stability_mode", "off")),
        id_freq_stability_mode=str(_get_arg(args, "id_freq_stability_mode", "off")),
        domain_time_stability_mode=str(_get_arg(args, "domain_time_stability_mode", "off")),
        domain_freq_stability_mode=str(_get_arg(args, "domain_freq_stability_mode", "off")),
        time_stability_channels=int(_get_arg(args, "time_stability_channels", 8)),
        freq_stability_channels=int(_get_arg(args, "freq_stability_channels", 4)),
        fast_infer_when_no_aux=bool(_get_arg(args, "fast_infer_when_no_aux", True)),
        use_tx_adv_on_zdom=bool(_get_arg(args, "use_tx_adv_on_zdom", False)),
    ).to(device)
    return model


def load_model_state(
    model,
    payload: Mapping[str, Any],
    *,
    max_missing: int = 0,
    max_unexpected: int = 0,
) -> dict[str, Any]:
    load_result = model.load_state_dict(_checkpoint_state(payload), strict=False)
    report = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "missing_count": int(len(load_result.missing_keys)),
        "unexpected_count": int(len(load_result.unexpected_keys)),
    }
    if int(report["missing_count"]) > int(max_missing) or int(report["unexpected_count"]) > int(max_unexpected):
        raise ValueError(
            f"checkpoint load key mismatch: missing={report['missing_count']} unexpected={report['unexpected_count']} "
            f"(allowed missing={int(max_missing)}, unexpected={int(max_unexpected)})"
        )
    return report


def _selected_splits(named_tests: Mapping[str, Any], spec: str) -> list[str]:
    raw = str(spec or "test_unseen_day_unseen_rx").strip()
    if raw.lower() in {"all", "*"}:
        return list(named_tests.keys())
    names = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    missing = [name for name in names if name not in named_tests]
    if missing:
        raise ValueError(f"unknown eval split(s): {missing}; available={list(named_tests)}")
    return names


def _requested_collab_counts(args: argparse.Namespace) -> list[int] | None:
    if str(args.collab_counts).strip().lower() in {"all", "*", "1..n"}:
        return None
    return [int(item.strip()) for item in str(args.collab_counts).replace(";", ",").split(",") if item.strip()]


def _aggregate_results(split_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for result in split_results.values():
        for k, row in (result.get("counts") or {}).items():
            acc = totals.setdefault(
                k,
                {
                    "total": 0,
                    "base_correct": 0,
                    "fused_correct": 0,
                    "rescue": 0,
                    "harm": 0,
                    "excluded_incomplete_groups": 0,
                },
            )
            for field in acc:
                acc[field] += int(row.get(field, 0))
    return {k: _finalize_counts(v) for k, v in sorted(totals.items(), key=lambda item: int(item[0]))}


def run_checkpoint_collaborative_eval(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    payload = _load_checkpoint(args.ckpt)
    checkpoint_sha256 = sha256_file(args.ckpt)
    identity_report = validate_checkpoint_identity(
        payload,
        checkpoint_path=args.ckpt,
        expected_run_name=args.expect_run_name,
        checkpoint_sha256=checkpoint_sha256,
        expected_sha256=args.expect_sha256,
    )
    context = build_wisig_context_from_checkpoint(payload, args)
    model = build_model_from_checkpoint_args(context, args, device)
    load_report = load_model_state(
        model,
        payload,
        max_missing=int(args.max_missing_keys),
        max_unexpected=int(args.max_unexpected_keys),
    )
    names = _selected_splits(context.named_tests, args.eval_on)
    requested_counts = _requested_collab_counts(args)
    split_results = OrderedDict()
    for name in names:
        dataset = context.named_tests[name]
        loader = make_loader(
            dataset,
            int(args.eval_batch_size),
            False,
            int(args.num_workers),
            device,
            False,
            int(args.prefetch_factor),
        )
        split_results[name] = evaluate_collaborative_receiver_fusion(
            model,
            loader,
            device,
            collab_counts=requested_counts,
            fusion=str(args.collab_fusion),
            max_batches=int(args.max_batches),
        )
    satellite_results = OrderedDict()
    if bool(args.eval_sat_channel):
        scenario_names = parse_sat_scenarios(args.eval_sat_scenarios)
        sat_names = resolve_sat_eval_loader_names(context.named_tests, args.eval_sat_on)
        sat_max_batches = int(args.sat_eval_max_batches)
        for si, scenario in enumerate(scenario_names):
            scenario_splits = OrderedDict()
            for li, name in enumerate(sat_names):
                dataset = context.named_tests[name]
                loader = make_loader(
                    dataset,
                    int(args.eval_batch_size),
                    False,
                    int(args.num_workers),
                    device,
                    False,
                    int(args.prefetch_factor),
                )
                seed = int(args.sat_seed) + si * 1009 + li * 97
                gen = make_torch_generator(device, seed)

                def _sat_transform(x, _batch_idx, *, _scenario=scenario, _gen=gen):
                    x_sat, _ = apply_sat_channel_for_scenario(x, _scenario, args, gen=_gen, return_meta=False)
                    return x_sat

                scenario_splits[name] = evaluate_collaborative_receiver_fusion(
                    model,
                    loader,
                    device,
                    collab_counts=requested_counts,
                    fusion=str(args.collab_fusion),
                    max_batches=sat_max_batches,
                    input_transform=_sat_transform,
                )
            satellite_results[scenario] = {
                "selected_names": list(sat_names),
                "splits": scenario_splits,
                "aggregate": _aggregate_results(scenario_splits),
            }
    return {
        "checkpoint": str(args.ckpt),
        "checkpoint_identity": identity_report,
        "checkpoint_epoch": int(payload.get("epoch", -1)) if isinstance(payload.get("epoch", -1), int) else payload.get("epoch", -1),
        "load_report": load_report,
        "eval_on": list(names),
        "collab_counts": str(args.collab_counts),
        "fusion": str(args.collab_fusion),
        "max_batches": int(args.max_batches),
        "split_info": context.split_info,
        "splits": split_results,
        "aggregate": _aggregate_results(split_results),
        "satellite_enabled": bool(args.eval_sat_channel),
        "satellite": satellite_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate central CVS-RFFI checkpoints with receiver collaborative inference.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path saved by train.py.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--expect_run_name", default="", help="Fail unless this token appears in checkpoint run_name/output_dir/path.")
    parser.add_argument("--expect_sha256", default="", help="Fail unless checkpoint SHA256 matches this value.")
    parser.add_argument("--max_missing_keys", type=int, default=0, help="Maximum missing model keys allowed when loading checkpoint.")
    parser.add_argument("--max_unexpected_keys", type=int, default=0, help="Maximum unexpected model keys allowed when loading checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_on", default="test_unseen_day_unseen_rx", help="Named test split, comma list, or all.")
    parser.add_argument("--collab_counts", default="all", help="Comma-separated K values or all for 1..receiver_count.")
    parser.add_argument("--collab_fusion", default="soft", choices=["soft", "adaptive", "conservative"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--eval_sat_channel", action="store_true", help="Also evaluate receiver collaborative inference after satellite-channel transforms.")
    parser.add_argument("--eval_sat_scenarios", default="clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit")
    parser.add_argument("--eval_sat_on", default=MAIN_SAT_EVAL_ON, help="Satellite eval split spec: main, all, or comma-separated named tests.")
    parser.add_argument("--sat_eval_max_batches", type=int, default=0, help="Max batches for satellite collaborative eval; 0 means full.")
    parser.add_argument("--sat_seed", type=int, default=2027)
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--wisig_pkl", default="")
    parser.add_argument("--wisig_out_len", type=int, default=0)
    parser.add_argument("--wisig_domain", default="")
    parser.add_argument("--max_samples_per_combo_test", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--model_size", default="")
    parser.add_argument("--model_variant", default="")
    parser.add_argument("--branch_ablation", default="")
    parser.add_argument("--domain_branch_ablation", default="")
    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    args = parser.parse_args(argv)

    result = run_checkpoint_collaborative_eval(args)
    print(
        f"[COLLAB-EVAL-CONFIG] ckpt={args.ckpt} eval_on={','.join(result['eval_on'])} "
        f"counts={args.collab_counts} fusion={args.collab_fusion} max_batches={args.max_batches}",
        flush=True,
    )
    identity = result["checkpoint_identity"]
    print(
        f"[COLLAB-EVAL-CKPT] run_name={identity.get('run_name')} sha256={identity.get('checkpoint_sha256')}",
        flush=True,
    )
    for split, split_result in result["splits"].items():
        for k, row in (split_result.get("counts") or {}).items():
            print(
                f"[COLLAB-EVAL] split={split} k={k} total={row['total']} "
                f"base={row['base_tx_acc']:.2f} fused={row['fused_tx_acc']:.2f} "
                f"rescue={row['rescue']} harm={row['harm']} net={row['net_gain']} "
                f"receiver_count={split_result['receiver_count']} eligible_groups={split_result['eligible_group_count']} "
                f"excluded_incomplete={split_result['excluded_incomplete_groups']}",
                flush=True,
            )
    for scenario, scenario_result in (result.get("satellite") or {}).items():
        for split, split_result in (scenario_result.get("splits") or {}).items():
            for k, row in (split_result.get("counts") or {}).items():
                print(
                    f"[COLLAB-SAT-EVAL] scenario={scenario} split={split} k={k} total={row['total']} "
                    f"base={row['base_tx_acc']:.2f} fused={row['fused_tx_acc']:.2f} "
                    f"rescue={row['rescue']} harm={row['harm']} net={row['net_gain']} "
                    f"receiver_count={split_result['receiver_count']} eligible_groups={split_result['eligible_group_count']} "
                    f"excluded_incomplete={split_result['excluded_incomplete_groups']}",
                    flush=True,
                )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[COLLAB-EVAL-OUTPUT] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
