from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from baselines.common.cvs_trainer import aggregate_named_stats, accuracy_counts, logits_from_output
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario

try:
    from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
except Exception:
    SatSimConfig = None
    apply_sat_gnd_channel_batch = None


def add_cvs_sat_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--eval_sat_channel", action="store_true")
    parser.add_argument(
        "--eval_sat_scenarios",
        type=str,
        default="simple_leo",
    )
    parser.add_argument("--eval_sat_on", type=str, default="all")
    parser.add_argument("--sat_eval_max_batches", type=int, default=0)
    parser.add_argument("--sat_seed", type=int, default=2027)
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--sat_eval_log_interval", type=int, default=100)
    return parser


def make_torch_generator(device, seed: int):
    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def safe_iq_tensor(x: torch.Tensor, clamp: float = 8.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(-float(clamp), float(clamp))


def make_sat_config(scenario: str, args):
    if SatSimConfig is None:
        raise ImportError("sat_channel.py is required for --eval_sat_channel.")
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
):
    if apply_sat_gnd_channel_batch is None:
        raise ImportError("sat_channel.py is required for --eval_sat_channel.")
    cfg = make_sat_config(scenario, args)
    y, _, _ = apply_sat_gnd_channel_batch(safe_iq_tensor(x), cfg, gen=gen, return_meta=False)
    return y.to(device=x.device, dtype=x.dtype)


def resolve_sat_eval_loader_names(named_loaders: Dict[str, Any], spec: str) -> List[str]:
    raw = str(spec or "all").strip().lower()
    if raw in ("all", "all_named", "*"):
        return list(named_loaders.keys())
    if raw in ("main", "main_ood", "ood", "target", "targets", "target_ood"):
        wanted = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
        return [k for k in wanted if k in named_loaders]
    if raw in ("strict", "target_strict", "strict_target", "udu", "unseen_day_unseen_rx"):
        return ["test_unseen_day_unseen_rx"] if "test_unseen_day_unseen_rx" in named_loaders else []
    names = []
    for item in raw.replace(";", ",").replace("+", ",").split(","):
        name = item.strip()
        if name and name in named_loaders and name not in names:
            names.append(name)
    if not names:
        names = list(named_loaders.keys())
    return names


def _transform_sat_batch(x_sat: torch.Tensor, transform: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> torch.Tensor:
    if transform is None:
        return x_sat
    return torch.stack([transform(x.cpu()).to(device=x_sat.device) for x in x_sat], dim=0)


@torch.no_grad()
def evaluate_loader_sat_channel(
    model,
    loader,
    device,
    *,
    scenario: str,
    args,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_batches: int = 0,
    seed: int = 0,
    progress_prefix: str = "",
    log_interval: int = 0,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    gen = make_torch_generator(device, int(seed))
    try:
        total_batches = len(loader)
    except Exception:
        total_batches = 0
    if progress_prefix:
        limit = int(max_batches) if max_batches else total_batches
        limit_text = str(limit) if limit else "unknown"
        print(f"{progress_prefix}[START] batches={limit_text}", flush=True)
    for batch_i, batch in enumerate(loader):
        if max_batches and batch_i >= int(max_batches):
            break
        y = batch["label"].to(device)
        x = batch["iq"].to(device)
        x_sat = apply_sat_channel_for_scenario(x, scenario, args, gen=gen)
        batch_sat = dict(batch)
        batch_sat["iq"] = _transform_sat_batch(x_sat, input_transform)
        if forward_fn is None:
            out = model(batch_sat["iq"].to(device))
        else:
            out = forward_fn(model, batch_sat, device)
        logits = logits_from_output(out)
        counts = accuracy_counts(logits, y)
        correct += int(counts["tx_correct"])
        total += int(counts["tx_total"])
        if progress_prefix and int(log_interval) > 0 and (batch_i + 1) % int(log_interval) == 0:
            denom = int(max_batches) if max_batches else total_batches
            denom_text = str(denom) if denom else "?"
            print(
                f"{progress_prefix}[BATCH {batch_i + 1}/{denom_text}] "
                f"tx={100.0 * correct / max(1, total):.2f}% n={total}",
                flush=True,
            )
    if progress_prefix:
        print(
            f"{progress_prefix}[DONE] tx={100.0 * correct / max(1, total):.2f}% "
            f"({correct}/{total})",
            flush=True,
        )
    return {"tx_acc": 100.0 * correct / max(1, total), "tx_correct": correct, "tx_total": total}


@torch.no_grad()
def evaluate_sat_scenarios(
    model,
    named_loaders: Dict[str, Any],
    device,
    *,
    scenario_names: Iterable[str],
    args,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_batches: int = 0,
) -> Dict[str, Dict[str, Any]]:
    scenario_list = list(scenario_names)
    selected_names = resolve_sat_eval_loader_names(named_loaders, getattr(args, "eval_sat_on", "all"))
    out: Dict[str, Dict[str, Any]] = {}
    log_interval = max(0, int(getattr(args, "sat_eval_log_interval", 100)))
    print(
        f"[SAT-TEST-START] scenarios={','.join(scenario_list)} "
        f"selected={','.join(selected_names)} max_batches={int(max_batches)}",
        flush=True,
    )
    for si, scenario in enumerate(scenario_list):
        named_stats = {}
        print(f"[SAT-TEST-SCENARIO-START] scenario={scenario}", flush=True)
        for li, name in enumerate(selected_names):
            named_stats[name] = evaluate_loader_sat_channel(
                model,
                named_loaders[name],
                device,
                scenario=scenario,
                args=args,
                forward_fn=forward_fn,
                input_transform=input_transform,
                max_batches=max_batches,
                seed=int(getattr(args, "sat_seed", 2027)) + si * 1009 + li * 97,
                progress_prefix=f"[SAT-TEST-LOADER] scenario={scenario} split={name} ",
                log_interval=log_interval,
            )
        main_keys = [k for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_stats]
        if not main_keys:
            main_keys = list(named_stats.keys())
        aggregate = aggregate_named_stats(named_stats, main_keys)
        all_named_aggregate = aggregate_named_stats(named_stats, list(named_stats.keys()))
        out[scenario] = {
            "aggregate": aggregate,
            "all_named_aggregate": all_named_aggregate,
            "strict_udu": named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan")),
            "named": named_stats,
            "selected_names": list(selected_names),
        }
        print(f"[SAT-TEST-SCENARIO-DONE] scenario={scenario}", flush=True)
    return out


def format_sat_test_lines(sat_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = []
    for scenario, stats in sat_stats.items():
        agg = stats.get("aggregate", {})
        all_agg = stats.get("all_named_aggregate", {})
        selected = ",".join(stats.get("selected_names", []))
        strict = stats.get("strict_udu", float("nan"))
        lines.append(
            f"[SAT-TEST] scenario={scenario} selected={selected} "
            f"overall_tx={agg.get('tx_acc', float('nan')):.2f}% "
            f"all_named_tx={all_agg.get('tx_acc', float('nan')):.2f}% "
            f"strict_udu={float(strict):.2f}% "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})"
        )
        named = stats.get("named", {})
        if isinstance(named, dict):
            priority = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
            ordered = [k for k in priority if k in named] + [k for k in named if k not in priority]
            for name in ordered:
                cur = named[name]
                lines.append(
                    f"[SAT-TEST-SPLIT] scenario={scenario} {name}: "
                    f"tx={cur.get('tx_acc', float('nan')):.2f}% "
                    f"({int(cur.get('tx_correct', 0))}/{int(cur.get('tx_total', 0))})"
                )
    return lines


def parse_and_validate_sat_scenarios(args) -> List[str]:
    scenarios = parse_sat_scenarios(getattr(args, "eval_sat_scenarios", ""))
    for scenario in scenarios:
        sat_channel_config_for_scenario(scenario)
    return scenarios
