from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from training_controls import sat_channel_config_for_scenario
from training_test_eval import aggregate_named_stats, format_named_test_lines
from cvsrffi.checkpoint import FCR_FEATURE_SCHEMA, FCR_V2_FEATURE_SCHEMA
from cvsrffi.tensors import (
    extract_batch_meta,
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
FCR_PREDICTION_SCENARIOS = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


def _safe_nan(v: float) -> str:
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"


safe_nan = _safe_nan


def select_identity_logits(
    outputs: Mapping[str, Any],
    *,
    model=None,
    use_fcr: bool | None = None,
) -> torch.Tensor:
    """Select the explicit formal identity route without mutating legacy outputs."""

    if use_fcr is None:
        raw_model = getattr(model, "_orig_mod", model)
        use_fcr = bool(getattr(raw_model, "use_fcr", False))
    key = "fcr_tx_logits" if bool(use_fcr) else "tx_logits"
    logits = outputs.get(key)
    if not torch.is_tensor(logits):
        raise KeyError(f"formal identity output {key!r} is unavailable")
    if bool(use_fcr):
        if model is None:
            expected_schemas = (FCR_FEATURE_SCHEMA, FCR_V2_FEATURE_SCHEMA)
        else:
            raw_model = getattr(model, "_orig_mod", model)
            expected_schemas = (
                FCR_V2_FEATURE_SCHEMA
                if str(getattr(raw_model, "fcr_version", "v1")) == "v2"
                else FCR_FEATURE_SCHEMA,
            )
        actual_schema = outputs.get("feature_schema")
        if actual_schema not in expected_schemas:
            raise ValueError(
                "formal FCR identity output has an incompatible feature schema: "
                f"expected={expected_schemas} actual={actual_schema!r}"
            )
    return logits


def validate_fcr_prediction_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_samples_per_scenario: int,
    run_id: str,
    row_id: str,
) -> dict[str, Any]:
    """Fail closed on the four-scenario, prediction-first FCR artifact."""

    expected = int(expected_samples_per_scenario)
    if expected <= 0:
        raise ValueError("expected_samples_per_scenario must be positive")
    expected_count = expected * len(FCR_PREDICTION_SCENARIOS)
    if len(records) != expected_count:
        raise ValueError(f"FCR prediction record count must be {expected_count}, got {len(records)}")
    expected_run = str(run_id)
    expected_row = str(row_id)
    if not expected_run or expected_row not in {f"R{index}" for index in range(9)}:
        raise ValueError("FCR prediction run_id/row_id binding is invalid")

    ids_by_scenario: dict[str, list[str]] = {
        scenario: [] for scenario in FCR_PREDICTION_SCENARIOS
    }
    for record in records:
        scenario = str(record.get("scenario", ""))
        if scenario not in ids_by_scenario:
            raise ValueError(f"unexpected FCR prediction scenario {scenario!r}")
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError("FCR prediction sample_id is missing")
        if str(record.get("run_id", "")) != expected_run:
            raise ValueError("FCR prediction run_id mismatch")
        if str(record.get("row_id", "")) != expected_row:
            raise ValueError("FCR prediction row_id mismatch")
        if record.get("feature_schema") != FCR_FEATURE_SCHEMA:
            raise ValueError("FCR prediction feature schema mismatch")
        if record.get("logit_route") != "fcr_tx_logits":
            raise ValueError("FCR prediction logit route mismatch")
        predicted = record.get("predicted_class")
        if isinstance(predicted, bool) or not isinstance(predicted, int) or predicted < 0:
            raise ValueError("FCR prediction class must be a non-negative integer")
        ids_by_scenario[scenario].append(sample_id)

    reference_ids: set[str] | None = None
    for scenario, sample_ids in ids_by_scenario.items():
        if len(sample_ids) != expected:
            raise ValueError(f"FCR prediction scenario {scenario} has an incomplete row count")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"FCR prediction scenario {scenario} contains duplicate sample IDs")
        scenario_ids = set(sample_ids)
        if reference_ids is None:
            reference_ids = scenario_ids
        elif scenario_ids != reference_ids:
            raise ValueError("FCR prediction scenarios do not bind the same sample ID set")
    return {
        "schema": "adv3b02_fcr_prediction_validation:v1",
        "run_id": expected_run,
        "row_id": expected_row,
        "feature_schema": FCR_FEATURE_SCHEMA,
        "logit_route": "fcr_tx_logits",
        "scenarios": list(FCR_PREDICTION_SCENARIOS),
        "samples_per_scenario": expected,
        "record_count": expected_count,
    }


def _prediction_sample_ids(extra: Any, batch_size: int) -> list[str]:
    meta = extract_batch_meta(extra)
    if not isinstance(meta, Mapping):
        raise ValueError("FCR prediction export requires collated Phase1 sample metadata")
    raw_ids = meta.get("physical_sample_id")
    if torch.is_tensor(raw_ids):
        raw_ids = raw_ids.detach().cpu().reshape(-1).tolist()
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, (tuple, list)) or len(raw_ids) != int(batch_size):
        raise ValueError("FCR prediction sample identifiers do not align with the batch")
    sample_ids = [str(value) for value in raw_ids]
    if any(not value for value in sample_ids):
        raise ValueError("FCR prediction sample identifier is empty")
    return sample_ids


@torch.no_grad()
def export_fcr_predictions(
    model,
    loader,
    device,
    *,
    args,
    output_path: str | Path,
    run_id: str,
    row_id: str,
) -> dict[str, Any]:
    """Export complete four-scenario FCR predictions without consuming labels."""

    raw_model = getattr(model, "_orig_mod", model)
    if not bool(getattr(raw_model, "use_fcr", False)):
        raise ValueError("FCR prediction export requires use_fcr=True")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite FCR predictions: {destination}")
    try:
        expected_samples = int(len(loader.dataset))
    except (AttributeError, TypeError):
        raise ValueError("FCR prediction loader must expose a finite dataset length") from None
    if expected_samples <= 0:
        raise ValueError("FCR prediction loader is empty")

    records: list[dict[str, Any]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for scenario_index, scenario in enumerate(FCR_PREDICTION_SCENARIOS):
            generator = make_torch_generator(
                device,
                int(getattr(args, "sat_seed", 2027)) + scenario_index * 1009,
            )
            for batch in loader:
                x = batch[0].to(device=device, non_blocking=True)
                extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
                sample_ids = _prediction_sample_ids(extra, int(x.size(0)))
                if scenario == "clean":
                    received = x
                else:
                    received, _ = apply_sat_channel_for_scenario(
                        x,
                        scenario,
                        args,
                        gen=generator,
                        return_meta=False,
                    )
                outputs = model(received, y_tx=None, grl_lambda=1.0, return_aux=True)
                logits = select_identity_logits(outputs, model=model)
                predicted = logits.argmax(dim=1).detach().cpu().tolist()
                for sample_id, predicted_class in zip(sample_ids, predicted):
                    records.append(
                        {
                            "sample_id": sample_id,
                            "scenario": scenario,
                            "predicted_class": int(predicted_class),
                            "feature_schema": FCR_FEATURE_SCHEMA,
                            "row_id": str(row_id),
                            "run_id": str(run_id),
                            "logit_route": "fcr_tx_logits",
                        }
                    )
    finally:
        model.train(was_training)

    validation = validate_fcr_prediction_records(
        records,
        expected_samples_per_scenario=expected_samples,
        run_id=str(run_id),
        row_id=str(row_id),
    )
    payload = {
        **validation,
        "schema": "adv3b02_fcr_predictions:v1",
        "validation_schema": validation["schema"],
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return validation


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

        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        tx_logits = select_identity_logits(out, model=model)
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
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    gen = make_torch_generator(device, int(seed))
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_sat, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
        d_raw = extract_domain_from_extra(extra, device)
        d = remap_domain_tensor(d_raw, domain_label_map, device) if d_raw is not None else None

        out = model(x_sat, y_tx=None, grl_lambda=1.0, return_aux=True)
        tx_logits = select_identity_logits(out, model=model)
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
    for si, scenario in enumerate(scenario_names):
        named_stats = {}
        named_seeds = {}
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

