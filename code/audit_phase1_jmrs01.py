#!/usr/bin/env python3
"""Run the source-only JMRS01 S0 mechanism screen with frozen Core90."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.jmrs01 import (
    ALLOWED_S0_ROWS,
    JMRS01Config,
    MechanismOutput,
    build_mechanism,
    mechanism_loss,
    validate_s0_rows,
)
from train_phase1_ccoi_pa import (
    _limited,
    _meta_value,
    _move_batch,
    _prepare_ssdg_args,
    _satellite_view,
    _torch_load,
    freeze_base_model,
    validate_source_roles,
)


REQUIRED_SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SOURCE_ROLES = ("L_s", "V_select", "V_cal")


@dataclass(frozen=True)
class LOROPartition:
    held_receiver: int
    train: Tensor
    select: Tensor
    cal: Tensor
    audit: Tensor


@dataclass
class RoleCache:
    iq: dict[str, Tensor]
    z_id: dict[str, Tensor]
    base_logits: dict[str, Tensor]
    tx: Tensor
    domain: Tensor
    receiver: Tensor
    day: Tensor
    base_index: Tensor

    def __len__(self) -> int:
        return int(self.tx.numel())


def validate_source_only_args(args: argparse.Namespace) -> tuple[str, ...]:
    if bool(args.target_or_query_access):
        raise ValueError("target/query access is forbidden for JMRS01 source-only screening")
    actual_roles = (args.train_role, args.select_role, args.cal_role, args.audit_role)
    expected_roles = ("L_s", "V_select", "V_cal", "V_select")
    if actual_roles != expected_roles:
        raise ValueError(f"JMRS01 role contract must be {expected_roles}, got {actual_roles}")
    rows = [item.strip().upper() for item in str(args.rows).split(",") if item.strip()]
    validated = validate_s0_rows(rows)
    if validated != ALLOWED_S0_ROWS:
        raise ValueError(f"formal JMRS01 S0 matrix must be exactly {ALLOWED_S0_ROWS}")
    return validated


def validate_output_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite immutable output root: {resolved}")
    return resolved


def build_loro_partition(
    train_receiver: Tensor,
    select_receiver: Tensor,
    cal_receiver: Tensor,
    *,
    held_receiver: int,
) -> LOROPartition:
    held = int(held_receiver)
    partition = LOROPartition(
        held_receiver=held,
        train=train_receiver.ne(held),
        select=select_receiver.ne(held),
        cal=cal_receiver.ne(held),
        audit=select_receiver.eq(held),
    )
    if not bool(partition.train.any()) or not bool(partition.select.any()) or not bool(partition.cal.any()):
        raise ValueError(f"held receiver {held} leaves an empty train/select/cal role")
    if not bool(partition.audit.any()):
        raise ValueError(f"held receiver {held} has no V_select audit samples")
    if bool(partition.select.logical_and(partition.audit).any()):
        raise RuntimeError("held receiver leaked into inner V_select")
    return partition


def write_closed_prediction_truth_streams(
    output_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    truth_path = output / "truth.jsonl"
    if prediction_path.exists() or truth_path.exists():
        raise FileExistsError(f"refusing to overwrite prediction/truth streams in {output}")
    prediction_ids: list[str] = []
    with prediction_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            sample_id = str(row["sample_id"])
            prediction_ids.append(sample_id)
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    truth_ids = [str(row["sample_id"]) for row in truths]
    if len(set(prediction_ids)) != len(prediction_ids) or len(set(truth_ids)) != len(truth_ids):
        raise RuntimeError("prediction/truth stream contains duplicate sample IDs")
    if prediction_ids != truth_ids:
        raise RuntimeError("prediction/truth closure mismatch before truth publication")
    with truth_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in truths:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return {
        "prediction_path": str(prediction_path),
        "truth_path": str(truth_path),
        "prediction_count": len(prediction_ids),
        "truth_written_after_prediction_close": True,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _metadata_tensor(extra: Any, key: str, count: int) -> Tensor:
    values = []
    for index in range(int(count)):
        value = _meta_value(extra, key, index, None)
        try:
            values.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing source metadata {key} at batch position {index}") from exc
    return torch.tensor(values, dtype=torch.long)


def _base_forward(base: nn.Module, iq: Tensor, domain: Tensor) -> tuple[Tensor, Tensor]:
    output = base(iq, y_tx=None, return_aux=True, domain_labels=domain)
    if not isinstance(output, Mapping) or "tx_logits" not in output or "z_id" not in output:
        raise TypeError("real Core90 checkpoint must expose tx_logits and z_id")
    logits = output["tx_logits"]
    z_id = output["z_id"]
    if logits.ndim != 2 or z_id.ndim != 2 or logits.size(0) != z_id.size(0):
        raise ValueError("Core90 tx_logits/z_id batch geometry is invalid")
    if not torch.isfinite(logits).all() or not torch.isfinite(z_id).all():
        raise FloatingPointError("Core90 returned non-finite tx_logits or z_id")
    return logits, z_id


def _collect_role_cache(
    loader,
    *,
    base: nn.Module,
    ssdg,
    data_ctx,
    data_args,
    device: torch.device,
    max_batches: int,
    sat_seed: int,
) -> RoleCache:
    views: dict[str, dict[str, list[Tensor]]] = {
        scenario: {"iq": [], "z_id": [], "base_logits": []} for scenario in REQUIRED_SCENARIOS
    }
    metadata: dict[str, list[Tensor]] = {
        name: [] for name in ("tx", "domain", "receiver", "day", "base_index")
    }
    generators = {
        scenario: ssdg.make_torch_generator(device, int(sat_seed) + 1009 * index)
        for index, scenario in enumerate(REQUIRED_SCENARIOS[1:], start=1)
    }
    base.eval()
    with torch.no_grad():
        for _, batch in _limited(loader, int(max_batches)):
            clean, tx, domain, extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            count = int(clean.size(0))
            for scenario in REQUIRED_SCENARIOS:
                iq = clean if scenario == "clean" else _satellite_view(
                    ssdg, clean, scenario, data_args, generators[scenario]
                )
                logits, z_id = _base_forward(base, iq, domain)
                views[scenario]["iq"].append(iq.detach().float().cpu())
                views[scenario]["z_id"].append(z_id.detach().float().cpu())
                views[scenario]["base_logits"].append(logits.detach().float().cpu())
            metadata["tx"].append(tx.detach().long().cpu())
            metadata["domain"].append(domain.detach().long().cpu())
            metadata["receiver"].append(_metadata_tensor(extra, "rx_i", count))
            metadata["day"].append(_metadata_tensor(extra, "day_i", count))
            metadata["base_index"].append(_metadata_tensor(extra, "base_index", count))
    if not metadata["tx"]:
        raise RuntimeError("source role produced zero samples")
    result = RoleCache(
        iq={scenario: torch.cat(value["iq"]) for scenario, value in views.items()},
        z_id={scenario: torch.cat(value["z_id"]) for scenario, value in views.items()},
        base_logits={scenario: torch.cat(value["base_logits"]) for scenario, value in views.items()},
        **{name: torch.cat(value) for name, value in metadata.items()},
    )
    if len(set(result.base_index.tolist())) != len(result):
        raise ValueError("base_index must be unique within each source role")
    return result


def _mask_indices(mask: Tensor) -> Tensor:
    return torch.nonzero(mask, as_tuple=False).view(-1).cpu()


def _forward_cached(
    model: nn.Module,
    cache: RoleCache,
    scenario: str,
    indices: Tensor,
    device: torch.device,
) -> MechanismOutput:
    return model(
        iq=cache.iq[scenario][indices].to(device, non_blocking=True),
        z_id=cache.z_id[scenario][indices].to(device, non_blocking=True),
    )


def _accuracy_on_mask(
    model: nn.Module,
    cache: RoleCache,
    mask: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    indices = _mask_indices(mask)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, int(indices.numel()), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            output = _forward_cached(model, cache, "clean", batch_indices, device)
            truth = cache.tx[batch_indices].to(device)
            correct += int(output.logits.argmax(1).eq(truth).sum().item())
            total += int(truth.numel())
    if total == 0:
        raise RuntimeError("selection mask produced zero samples")
    return correct / total


def _train_fold(
    row: str,
    model: nn.Module,
    train_cache: RoleCache,
    select_cache: RoleCache,
    partition: LOROPartition,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    train_indices = _mask_indices(partition.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    generator = torch.Generator().manual_seed(int(seed))
    best_accuracy = -1.0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    satellite_scenarios = REQUIRED_SCENARIOS[1:]
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        permutation = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        sums = {
            name: 0.0
            for name in (
                "total",
                "ce",
                "clean_sat",
                "class_cond_rx",
                "tx_margin",
                "preserve",
                "quality",
                "mechanism_regularization",
            )
        }
        steps = 0
        for start in range(0, int(permutation.numel()), int(args.batch_size)):
            if int(args.max_train_batches) > 0 and steps >= int(args.max_train_batches):
                break
            indices = permutation[start : start + int(args.batch_size)]
            scenario = satellite_scenarios[(epoch + steps - 1) % len(satellite_scenarios)]
            clean = _forward_cached(model, train_cache, "clean", indices, device)
            satellite = _forward_cached(model, train_cache, scenario, indices, device)
            labels = train_cache.tx[indices].to(device)
            receivers = train_cache.receiver[indices].to(device)
            losses = mechanism_loss(
                clean,
                satellite,
                labels,
                receivers,
                base_logits=(
                    train_cache.base_logits["clean"][indices].to(device) if row == "R1" else None
                ),
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite JMRS01 loss at epoch={epoch} step={steps + 1}")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for name in sums:
                sums[name] += float(losses[name].detach().item())
            steps += 1
        if steps == 0:
            raise RuntimeError("JMRS01 training produced zero steps")
        row = {"epoch": epoch, "steps": steps, **{name: value / steps for name, value in sums.items()}}
        if epoch == 1 or epoch == int(args.epochs) or epoch % int(args.selection_interval) == 0:
            accuracy = _accuracy_on_mask(
                model,
                select_cache,
                partition.select,
                device=device,
                batch_size=int(args.eval_batch_size),
            )
            row["inner_v_select_accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
    if best_state is None:
        raise RuntimeError("JMRS01 fold did not select a model state")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    return history


def _safe_fuse(base_logits: Tensor, mechanism_logits: Tensor, alpha: float) -> Tensor:
    if base_logits.shape != mechanism_logits.shape:
        raise ValueError("safe fusion requires matching base/mechanism logits")
    centered = mechanism_logits - mechanism_logits.mean(dim=1, keepdim=True)
    centered = centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    base_scale = base_logits.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    residual = centered * base_scale
    limit = 0.25 * base_logits.norm(dim=1, keepdim=True).clamp_min(1e-6)
    residual = residual * torch.clamp(limit / residual.norm(dim=1, keepdim=True).clamp_min(1e-6), max=1.0)
    return base_logits + float(alpha) * residual


def _calibrate_safe_alpha(
    model: nn.Module,
    cache: RoleCache,
    mask: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    indices = _mask_indices(mask)
    base_rows = []
    mechanism_rows = []
    truth_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(indices.numel()), int(batch_size)):
            chosen = indices[start : start + int(batch_size)]
            mechanism = _forward_cached(model, cache, "clean", chosen, device)
            base_rows.append(cache.base_logits["clean"][chosen])
            mechanism_rows.append(mechanism.logits.detach().cpu())
            truth_rows.append(cache.tx[chosen])
    base = torch.cat(base_rows)
    mechanism = torch.cat(mechanism_rows)
    truth = torch.cat(truth_rows)
    base_accuracy = float(base.argmax(1).eq(truth).float().mean().item())
    grid = {}
    eligible = []
    for alpha in (0.0, 0.02, 0.05, 0.10):
        accuracy = float(_safe_fuse(base, mechanism, alpha).argmax(1).eq(truth).float().mean().item())
        grid[f"{alpha:.2f}"] = accuracy
        if 100.0 * (base_accuracy - accuracy) <= 0.30 + 1e-12:
            eligible.append((accuracy, -alpha, alpha))
    selected = max(eligible)[2] if eligible else 0.0
    return {
        "alpha": selected,
        "grid": grid,
        "base_accuracy": base_accuracy,
        "scope": "inner_nonheld_V_cal_only",
        "clean_drop_constraint_pp": 0.30,
    }


def _sample_probe_indices(cache: RoleCache, mask: Tensor, per_receiver: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    chosen = []
    for receiver in torch.unique(cache.receiver[mask]).tolist():
        candidates = _mask_indices(mask & cache.receiver.eq(int(receiver)))
        order = torch.randperm(candidates.numel(), generator=generator)
        chosen.append(candidates[order[: min(int(per_receiver), int(candidates.numel()))]])
    return torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long)


def _records_for_indices(
    *,
    row: str,
    held_receiver: int,
    scope: str,
    scenarios: Sequence[str],
    indices: Tensor,
    cache: RoleCache,
    model: nn.Module | None,
    safe_alpha: float,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    parameter_count = 0 if model is None else sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count > 50_000:
        raise RuntimeError(f"row {row} exceeds 50k parameter budget: {parameter_count}")
    for scenario in scenarios:
        for start in range(0, int(indices.numel()), int(batch_size)):
            chosen = indices[start : start + int(batch_size)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                if model is None:
                    embedding = cache.z_id[scenario][chosen].to(device)
                    logits = cache.base_logits[scenario][chosen].to(device)
                    reliability = torch.ones(chosen.numel(), device=device)
                else:
                    output = _forward_cached(model, cache, scenario, chosen, device)
                    embedding, logits, reliability = output.embedding, output.logits, output.reliability
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = 1000.0 * (time.perf_counter() - started) / max(1, int(chosen.numel()))
            base_logits = cache.base_logits[scenario][chosen].to(device)
            safe_logits = _safe_fuse(base_logits, logits, safe_alpha) if model is not None else base_logits
            for position, cache_index in enumerate(chosen.tolist()):
                sample_id = (
                    f"{row}:rx{held_receiver}:{scope}:{scenario}:"
                    f"{int(cache.base_index[cache_index])}"
                )
                prediction = {
                    "sample_id": sample_id,
                    "row": row,
                    "scenario": scenario,
                    "scope": scope,
                    "held_receiver": int(held_receiver),
                    "receiver": int(cache.receiver[cache_index]),
                    "day": int(cache.day[cache_index]),
                    "base_index": int(cache.base_index[cache_index]),
                    "predicted_class": int(logits[position].argmax().item()),
                    "safe_predicted_class": int(safe_logits[position].argmax().item()),
                    "base_predicted_class": int(base_logits[position].argmax().item()),
                    "embedding": embedding[position].detach().float().cpu().tolist(),
                    "reliability": float(reliability[position].detach().item()),
                    "parameter_count": parameter_count,
                    "runtime_ms_per_sample": elapsed_ms,
                    "safe_alpha": float(safe_alpha),
                }
                predictions.append(prediction)
                truths.append({"sample_id": sample_id, "true_class": int(cache.tx[cache_index])})
    return predictions, truths


def _real_checkpoint_smoke(
    *,
    base: nn.Module,
    data_ctx,
    ssdg,
    device: torch.device,
    rows: Sequence[str],
    seed: int,
) -> tuple[JMRS01Config, dict[str, Any]]:
    _, batch = next(iter(_limited(data_ctx["train_loader"], 1)))
    iq, _tx, domain, _extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
    with torch.no_grad():
        base_logits, z_id = _base_forward(base, iq, domain)
    cfg = JMRS01Config(z_dim=int(z_id.size(1)), num_classes=int(base_logits.size(1)), seed=int(seed))
    mechanisms = {}
    for row in rows:
        if row == "M0":
            continue
        model = build_mechanism(row, cfg).to(device).eval()
        with torch.no_grad():
            output = model(iq=iq, z_id=z_id)
        mechanisms[row] = {
            "embedding_shape": list(output.embedding.shape),
            "logit_shape": list(output.logits.shape),
            "finite": bool(torch.isfinite(output.embedding).all() and torch.isfinite(output.logits).all()),
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    if not all(value["finite"] and value["parameter_count"] <= 50_000 for value in mechanisms.values()):
        raise RuntimeError("real checkpoint JMRS01 smoke failed finite/budget contract")
    return cfg, {
        "batch_size": int(iq.size(0)),
        "input_length": int(iq.size(-1)),
        "z_dim": int(z_id.size(1)),
        "num_classes": int(base_logits.size(1)),
        "base_finite": bool(torch.isfinite(base_logits).all() and torch.isfinite(z_id).all()),
        "mechanisms": mechanisms,
        "target_or_query_access": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PHASE1 JMRS01 source-only S0 mechanism screen")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--rows", default=",".join(ALLOWED_S0_ROWS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--sat_seed", type=int, default=20260824)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--selection_interval", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--probe_samples_per_receiver", type=int, default=128)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--eval_sat_on", default="main")
    parser.add_argument("--train_role", default="L_s", choices=("L_s",))
    parser.add_argument("--select_role", default="V_select", choices=("V_select",))
    parser.add_argument("--cal_role", default="V_cal", choices=("V_cal",))
    parser.add_argument("--audit_role", default="V_select", choices=("V_select",))
    parser.add_argument("--target_or_query_access", action="store_true", default=False)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    rows = validate_source_only_args(args)
    output = validate_output_root(Path(args.output_dir))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "rows": rows,
                    "scenarios": REQUIRED_SCENARIOS,
                    "target_or_query_access": False,
                }
            ),
            flush=True,
        )
        return 0
    checkpoint_path = Path(args.checkpoint).resolve()
    wisig_path = Path(args.wisig_pkl).resolve()
    if not checkpoint_path.is_file() or not wisig_path.is_file():
        raise FileNotFoundError("checkpoint and wisig_pkl must both exist")
    output.mkdir(parents=True, exist_ok=False)
    _seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = _torch_load(checkpoint_path, device)
    ssdg, data_args = _prepare_ssdg_args(args, checkpoint)
    data_ctx = ssdg._build_ssdg_wisig_data(data_args, device)
    validate_source_roles(data_args, data_ctx["split_info"])
    base, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(data_ctx["input_len"]),
        device=device,
        ssdg_module=ssdg,
    )
    base = freeze_base_model(base)
    cfg, smoke = _real_checkpoint_smoke(
        base=base,
        data_ctx=data_ctx,
        ssdg=ssdg,
        device=device,
        rows=rows,
        seed=int(args.seed),
    )
    protocol = {
        "status": "REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS",
        "protocol": "Phase1_source_only_nested_LORO",
        "source_roles": data_ctx["split_info"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_audit": checkpoint_audit,
        "smoke": smoke,
        "rows": rows,
        "scenarios": REQUIRED_SCENARIOS,
        "removed_rows": {"D2": "NO_KNOWN_TRANSMITTED_SYMBOLS"},
        "target_or_query_access": False,
    }
    _write_json(output / "protocol_and_smoke.json", protocol)
    if args.smoke_only:
        print(f"[JMRS01-SMOKE] PASS output={output}", flush=True)
        return 0

    print("[JMRS01] caching source roles and four fixed views", flush=True)
    train_cache = _collect_role_cache(
        data_ctx["train_loader"],
        base=base,
        ssdg=ssdg,
        data_ctx=data_ctx,
        data_args=data_args,
        device=device,
        max_batches=int(args.max_eval_batches),
        sat_seed=int(args.sat_seed) + 10000,
    )
    select_cache = _collect_role_cache(
        data_ctx["val_loader"],
        base=base,
        ssdg=ssdg,
        data_ctx=data_ctx,
        data_args=data_args,
        device=device,
        max_batches=int(args.max_eval_batches),
        sat_seed=int(args.sat_seed) + 20000,
    )
    cal_cache = _collect_role_cache(
        data_ctx["source_calibration_loader"],
        base=base,
        ssdg=ssdg,
        data_ctx=data_ctx,
        data_args=data_args,
        device=device,
        max_batches=int(args.max_eval_batches),
        sat_seed=int(args.sat_seed) + 30000,
    )
    source_receivers = sorted(
        set(train_cache.receiver.tolist())
        | set(select_cache.receiver.tolist())
        | set(cal_cache.receiver.tolist())
    )
    if len(source_receivers) != 7:
        raise ValueError(f"JMRS01 preregistered seven source receivers, observed {source_receivers}")
    predictions: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "status": "PREDICTIONS_PENDING",
        "source_receivers": source_receivers,
        "rows": {},
        "folds": {},
        "scenarios": REQUIRED_SCENARIOS,
        "target_or_query_access": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=False)
    history_dir = output / "training_history"
    history_dir.mkdir(exist_ok=False)
    for fold_index, held_receiver in enumerate(source_receivers):
        partition = build_loro_partition(
            train_cache.receiver,
            select_cache.receiver,
            cal_cache.receiver,
            held_receiver=held_receiver,
        )
        fold_name = f"rx{held_receiver}"
        manifest["folds"][fold_name] = {
            "train_count": int(partition.train.sum()),
            "select_count": int(partition.select.sum()),
            "cal_count": int(partition.cal.sum()),
            "audit_count": int(partition.audit.sum()),
            "held_receiver_absent_from_inner_roles": True,
        }
        probe_fit = _sample_probe_indices(
            cal_cache,
            partition.cal,
            int(args.probe_samples_per_receiver),
            int(args.seed) + fold_index * 17,
        )
        probe_eval = _sample_probe_indices(
            select_cache,
            partition.select,
            int(args.probe_samples_per_receiver),
            int(args.seed) + fold_index * 17 + 1,
        )
        audit_indices = _mask_indices(partition.audit)
        for row in rows:
            model: nn.Module | None = None
            safe_alpha = 0.0
            row_history: list[dict[str, Any]] = []
            calibration: dict[str, Any] = {"alpha": 0.0, "scope": "M0_NOT_APPLICABLE"}
            if row != "M0":
                model = build_mechanism(row, cfg).to(device)
                row_history = _train_fold(
                    row,
                    model,
                    train_cache,
                    select_cache,
                    partition,
                    args=args,
                    device=device,
                    seed=int(args.seed) + fold_index * 1009 + ALLOWED_S0_ROWS.index(row) * 101,
                )
                calibration = _calibrate_safe_alpha(
                    model,
                    cal_cache,
                    partition.cal,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                )
                safe_alpha = float(calibration["alpha"])
                torch.save(
                    {
                        "schema": "cvs.phase1.jmrs01.s0.v1",
                        "row": row,
                        "held_receiver": int(held_receiver),
                        "config": asdict(cfg),
                        "safe_calibration": calibration,
                        "base_checkpoint": str(checkpoint_path),
                        "state_dict": model.state_dict(),
                        "target_or_query_access": False,
                    },
                    model_dir / f"{row}_{fold_name}.pth",
                )
            _write_json(
                history_dir / f"{row}_{fold_name}.json",
                {"history": row_history, "safe_calibration": calibration},
            )
            for scope, cache, indices, scenarios in (
                ("probe_fit", cal_cache, probe_fit, ("clean",)),
                ("probe_eval", select_cache, probe_eval, ("clean",)),
                ("held_audit", select_cache, audit_indices, REQUIRED_SCENARIOS),
            ):
                pred, truth = _records_for_indices(
                    row=row,
                    held_receiver=held_receiver,
                    scope=scope,
                    scenarios=scenarios,
                    indices=indices,
                    cache=cache,
                    model=model,
                    safe_alpha=safe_alpha,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                )
                predictions.extend(pred)
                truths.extend(truth)
            manifest["rows"].setdefault(row, {})[fold_name] = {
                "training_complete": row != "M0",
                "safe_alpha": safe_alpha,
                "model": None if row == "M0" else str(model_dir / f"{row}_{fold_name}.pth"),
                "held_audit_scenarios": REQUIRED_SCENARIOS,
            }
            _write_json(output / "run_manifest.json", manifest)
            print(
                f"[JMRS01] row={row} held_receiver={held_receiver} records={len(predictions)}",
                flush=True,
            )
    closure = write_closed_prediction_truth_streams(output, predictions, truths)
    manifest["status"] = "PREDICTIONS_COMPLETE_TRUTH_NOT_SCORED"
    manifest["prediction_closure"] = closure
    _write_json(output / "run_manifest.json", manifest)
    print(f"[JMRS01-PREDICTIONS] COMPLETE output={output}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
