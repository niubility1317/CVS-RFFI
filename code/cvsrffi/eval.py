from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from training_controls import sat_channel_config_for_scenario
from training_test_eval import aggregate_named_stats, format_named_test_lines
from baseline_origin_sat_view import normalize_crra_nuisance_meta
from cvsrffi.crra_evaluation import CRRATelemetryAccumulator
from cvsrffi.ntrs_evaluation import (
    NTRSTelemetryAccumulator,
    ntrs_prototypes_from_model,
    ntrs_unknown_rescue_from_model,
)
from cvsrffi.tensors import (
    extract_domain_from_extra,
    make_torch_generator,
    remap_domain_tensor,
    safe_iq_tensor,
    unpack_batch,
)

try:
    from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
except Exception:  # pragma: no cover - optional satellite simulation dependency
    SatSimConfig = None
    apply_sat_gnd_channel_batch = None


MAIN_SAT_EVAL_ON_NAMES = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
MAIN_SAT_EVAL_ON = ",".join(MAIN_SAT_EVAL_ON_NAMES)


def _safe_nan(v: float) -> str:
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"


safe_nan = _safe_nan


def _model_uses_ntrs(model: Any) -> bool:
    for candidate in (model, getattr(model, "_orig_mod", None), getattr(model, "module", None)):
        if candidate is not None and bool(getattr(candidate, "use_ntrs", False)):
            return True
    return False


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def metric_or_neg_inf(stats: Dict[str, Any], key: str = "tx_acc") -> float:
    """Read a metric safely for best-checkpoint comparisons."""
    try:
        v = float(stats.get(key, float("-inf")))
    except Exception:
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def compute_primary_ood_score(test_overall: float, unseen_day_unseen_rx: float, udu_weight: float) -> float:
    if not math.isfinite(float(test_overall)):
        test_overall = float("-inf")
    if not math.isfinite(float(unseen_day_unseen_rx)):
        unseen_day_unseen_rx = float("-inf")
    w = max(0.0, min(1.0, float(udu_weight)))
    if not math.isfinite(test_overall) or not math.isfinite(unseen_day_unseen_rx):
        return max(float(test_overall), float(unseen_day_unseen_rx))
    return (1.0 - w) * float(test_overall) + w * float(unseen_day_unseen_rx)


def compute_worst_unseen_rx_score(named_test_stats: Dict[str, Dict[str, Any]]) -> Tuple[float, str]:
    rx_scores: List[Tuple[str, float]] = []
    for name, stats in named_test_stats.items():
        if not str(name).startswith("test_rx_"):
            continue
        score = metric_or_neg_inf(stats, "tx_acc")
        if math.isfinite(score):
            rx_scores.append((str(name), float(score)))
    if not rx_scores:
        return float("-inf"), ""
    worst_name, worst_score = min(rx_scores, key=lambda x: x[1])
    return float(worst_score), worst_name


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device, drop_last: bool, prefetch_factor: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "drop_last": drop_last,
        "persistent_workers": (int(num_workers) > 0),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(dataset, **kwargs)


def make_sat_config(scenario: str, args):
    if SatSimConfig is None:
        raise ImportError("sat_channel.py is required for satellite channel evaluation/training.")
    kwargs = sat_channel_config_for_scenario(scenario)
    kwargs["fs_hz"] = float(getattr(args, "sat_fs_hz", 25e6))
    kwargs["fc_hz"] = float(getattr(args, "sat_fc_hz", 2.462e9))
    return SatSimConfig(**kwargs)


def apply_sat_channel_for_scenario(
    x: torch.Tensor,
    scenario: str,
    args,
    *,
    gen=None,
    return_meta: bool = False,
):
    if apply_sat_gnd_channel_batch is None:
        raise ImportError("sat_channel.py is required for satellite channel evaluation/training.")
    cfg = make_sat_config(scenario, args)
    y, meta, _ = apply_sat_gnd_channel_batch(safe_iq_tensor(x), cfg, gen=gen, return_meta=return_meta)
    return y.to(device=x.device, dtype=x.dtype), meta


def resolve_sat_eval_loader_names(named_loaders: Dict[str, DataLoader], spec: str) -> List[str]:
    raw = str(spec or MAIN_SAT_EVAL_ON).strip().lower()
    if raw in ("all", "all_named", "*"):
        return list(named_loaders.keys())
    if raw in ("main", "main_ood", "ood"):
        return [k for k in MAIN_SAT_EVAL_ON_NAMES if k in named_loaders]
    names = []
    for item in raw.replace(";", ",").replace("+", ",").split(","):
        name = item.strip()
        if name and name in named_loaders and name not in names:
            names.append(name)
    if not names:
        names = [k for k in MAIN_SAT_EVAL_ON_NAMES if k in named_loaders]
    return names


def evaluate_loader(model, loader, device, domain_label_map: Dict[int, int], max_batches: int = 0):
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d_raw = extract_domain_from_extra(extra, device)
        d = remap_domain_tensor(d_raw, domain_label_map, device) if d_raw is not None else None

        out = model(
            x,
            y_tx=None,
            grl_lambda=1.0,
            return_aux=True,
            domain_labels=d,
        )
        tx_logits = out["tx_logits"]
        tx_pred = tx_logits.argmax(dim=1)
        tx_correct += int((tx_pred == y).sum().item())
        tx_total += int(y.numel())

        if d is not None:
            valid = d >= 0
            if valid.any():
                dom_y = d[valid]
                dom_correct += int((out["dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                dom_total += int(dom_y.numel())

        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    return {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": float("nan"),
        "tx_correct": int(tx_correct),
        "tx_total": int(tx_total),
    }


def evaluate_named_loaders(model, named_loaders: Dict[str, DataLoader], device, domain_label_map: Dict[int, int], max_batches: int = 0):
    out = {}
    for name, loader in named_loaders.items():
        out[name] = evaluate_loader(model, loader, device, domain_label_map=domain_label_map, max_batches=max_batches)
    return out


def evaluate_loader_sat_channel(
    model,
    loader,
    device,
    domain_label_map: Dict[int, int],
    scenario: str,
    args,
    max_batches: int = 0,
    seed: int = 0,
):
    was_training = bool(getattr(model, "training", False))
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    gen = make_torch_generator(device, int(seed))
    crra_telemetry = CRRATelemetryAccumulator() if bool(getattr(args, "eval_crra_telemetry", False)) else None
    ntrs_telemetry = (
        NTRSTelemetryAccumulator(
            prototypes=ntrs_prototypes_from_model(model),
            unknown_rescue=ntrs_unknown_rescue_from_model(model),
        )
        if bool(getattr(args, "eval_ntrs_telemetry", False))
        else None
    )
    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x, y, extra = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                use_ntrs = _model_uses_ntrs(model)
                x_sat, raw_sat_meta = apply_sat_channel_for_scenario(
                    x,
                    scenario,
                    args,
                    gen=gen,
                    return_meta=use_ntrs,
                )
                d_raw = extract_domain_from_extra(extra, device)
                d = remap_domain_tensor(d_raw, domain_label_map, device) if d_raw is not None else None
                ntrs_metadata = None
                ntrs_metadata_valid = None
                if use_ntrs:
                    _, ntrs_metadata, ntrs_metadata_valid, _ = normalize_crra_nuisance_meta(
                        raw_sat_meta,
                        scenario=str(scenario),
                        batch_size=int(x.size(0)),
                        device=x.device,
                    )

                clean_out = (
                    model(
                        x,
                        y_tx=None,
                        grl_lambda=1.0,
                        return_aux=True,
                        domain_labels=d,
                    )
                    if crra_telemetry is not None or ntrs_telemetry is not None
                    else None
                )
                out = model(
                    x_sat,
                    y_tx=None,
                    grl_lambda=1.0,
                    return_aux=True,
                    domain_labels=d,
                    ntrs_metadata=ntrs_metadata,
                    ntrs_metadata_valid=ntrs_metadata_valid,
                )
                tx_logits = out["tx_logits"]
                tx_pred = tx_logits.argmax(dim=1)
                tx_correct += int((tx_pred == y).sum().item())
                tx_total += int(y.numel())

                if crra_telemetry is not None and clean_out is not None:
                    crra_telemetry.update(clean_out, out, y)
                if ntrs_telemetry is not None and clean_out is not None:
                    ntrs_telemetry.update(clean_out, out, y)

                if d is not None:
                    valid = d >= 0
                    if valid.any():
                        dom_y = d[valid]
                        dom_correct += int((out["dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                        dom_total += int(dom_y.numel())

                if max_batches > 0 and (bi + 1) >= max_batches:
                    break
    finally:
        model.train(was_training)

    result = {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": float("nan"),
        "tx_correct": int(tx_correct),
        "tx_total": int(tx_total),
    }
    if crra_telemetry is not None:
        result["crra_telemetry"] = crra_telemetry.summary()
        # Internal only: evaluate_sat_scenarios merges it before serialising named rows.
        result["_crra_telemetry_state"] = crra_telemetry
    if ntrs_telemetry is not None:
        result["ntrs_telemetry"] = ntrs_telemetry.summary()
        result["_ntrs_telemetry_state"] = ntrs_telemetry
    return result


def evaluate_sat_scenarios(
    model,
    named_loaders: Dict[str, DataLoader],
    device,
    domain_label_map: Dict[int, int],
    scenario_names: List[str],
    args,
    max_batches: int = 0,
):
    selected_names = resolve_sat_eval_loader_names(named_loaders, getattr(args, "eval_sat_on", MAIN_SAT_EVAL_ON))
    out = {}
    seed_base = int(getattr(args, "sat_seed", 2027))
    collect_crra_telemetry = bool(getattr(args, "eval_crra_telemetry", False))
    collect_ntrs_telemetry = bool(getattr(args, "eval_ntrs_telemetry", False))
    for si, scenario in enumerate(scenario_names):
        named_stats = {}
        named_seeds = {}
        scenario_telemetry = CRRATelemetryAccumulator() if collect_crra_telemetry else None
        scenario_ntrs_telemetry = (
            NTRSTelemetryAccumulator(
                prototypes=ntrs_prototypes_from_model(model),
                unknown_rescue=ntrs_unknown_rescue_from_model(model),
            )
            if collect_ntrs_telemetry
            else None
        )
        for li, name in enumerate(selected_names):
            eval_seed = seed_base + si * 1009 + li * 97
            stats = evaluate_loader_sat_channel(
                model,
                named_loaders[name],
                device,
                domain_label_map=domain_label_map,
                scenario=scenario,
                args=args,
                max_batches=max_batches,
                seed=eval_seed,
            )
            telemetry_state = stats.pop("_crra_telemetry_state", None)
            if scenario_telemetry is not None and telemetry_state is not None:
                scenario_telemetry.merge(telemetry_state)
            ntrs_telemetry_state = stats.pop("_ntrs_telemetry_state", None)
            if scenario_ntrs_telemetry is not None and ntrs_telemetry_state is not None:
                scenario_ntrs_telemetry.merge(ntrs_telemetry_state)
            named_stats[name] = dict(stats, sat_seed=int(eval_seed))
            named_seeds[name] = int(eval_seed)
        main_keys = [k for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_stats]
        if not main_keys:
            main_keys = list(named_stats.keys())
        aggregate = aggregate_named_stats(named_stats, main_keys)
        strict = named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan"))
        receiver_seen_day = {
            name: stats for name, stats in named_stats.items() if str(name).startswith("test_rx_")
        }
        receiver_strict = {
            name: stats
            for name, stats in named_stats.items()
            if str(name).startswith("test_unseen_day_rx_")
        }
        receiver_named = {**receiver_seen_day, **receiver_strict}

        def _receiver_floor(rows: Dict[str, Dict[str, Any]]) -> float:
            values = []
            for stats in rows.values():
                try:
                    value = float(stats.get("tx_acc", float("nan")))
                except Exception:
                    continue
                if math.isfinite(value):
                    values.append(value)
            return min(values) if values else float("nan")

        out[scenario] = {
            "aggregate": aggregate,
            "strict_udu": strict,
            "named": named_stats,
            "selected_names": list(selected_names),
            "evaluation_seed": {
                "base": int(seed_base),
                "scenario_index": int(si),
                "scenario_offset": int(si * 1009),
                "loader_stride": 97,
                "named": named_seeds,
            },
            "receiver_named": receiver_named,
            "receiver_seen_day_named": receiver_seen_day,
            "receiver_strict_named": receiver_strict,
            "receiver_floor": _receiver_floor(receiver_named),
            "receiver_seen_day_floor": _receiver_floor(receiver_seen_day),
            "receiver_strict_floor": _receiver_floor(receiver_strict),
        }
        if scenario_telemetry is not None:
            out[scenario]["crra_telemetry"] = scenario_telemetry.summary()
        if scenario_ntrs_telemetry is not None:
            out[scenario]["ntrs_telemetry"] = scenario_ntrs_telemetry.summary()
    return out


def format_sat_test_lines(sat_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = []
    for scenario, stats in sat_stats.items():
        agg = stats.get("aggregate", {})
        strict = stats.get("strict_udu", float("nan"))
        selected = ",".join(stats.get("selected_names", []))
        seed_base = (stats.get("evaluation_seed", {}) or {}).get("base", "")
        lines.append(
            f"[SAT-TEST] scenario={scenario} selected={selected} "
            f"seed_base={seed_base} "
            f"overall_tx={agg.get('tx_acc', float('nan')):.2f}% "
            f"strict_udu={safe_nan(strict)}% "
            f"receiver_floor={safe_nan(stats.get('receiver_floor', float('nan')))}% "
            f"receiver_seen_day_floor={safe_nan(stats.get('receiver_seen_day_floor', float('nan')))}% "
            f"receiver_strict_floor={safe_nan(stats.get('receiver_strict_floor', float('nan')))}% "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})"
        )
    return lines

