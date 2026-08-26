#!/usr/bin/env python3
"""Run the role-correct JMRS02 J1 source-receiver LORO screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from audit_phase1_jmrs01 import (
    REQUIRED_SCENARIOS,
    _base_forward,
    _collect_role_cache,
    _mask_indices,
    _seed_all,
    _write_json,
    build_loro_partition,
    validate_output_root,
    write_closed_prediction_truth_streams,
)
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.jmrs02_j1 import J1Config, J1Output, J1_ROWS, build_j1_module, validate_j1_rows
from cvsrffi.jmrs02_rx2 import (
    RX2_ROWS,
    build_rx2_module,
    real_core_backward_probe,
    require_finite_gradients,
)
from train_phase1_ccoi_pa import (
    _limited,
    _move_batch,
    _prepare_ssdg_args,
    _torch_load,
    freeze_base_model,
    validate_source_roles,
)


def validate_j1_args(args: argparse.Namespace) -> tuple[str, ...]:
    if bool(args.target_or_query_access):
        raise ValueError("target/query access is forbidden for JMRS02 J1")
    roles = (args.train_role, args.select_role, args.cal_role, args.audit_role)
    if roles != ("L_s", "V_select", "V_cal", "V_select"):
        raise ValueError("JMRS02 J1 requires L_s/V_select/V_cal/V_select source roles")
    rows = tuple(x.strip().upper() for x in str(args.rows).split(",") if x.strip())
    if bool(getattr(args, "focused_rx2", False)):
        if rows != RX2_ROWS:
            raise ValueError(f"focused JMRS02 RX2 matrix must be exactly {RX2_ROWS}")
    elif validate_j1_rows(rows) != J1_ROWS:
        raise ValueError(f"formal JMRS02 J1 matrix must be exactly {J1_ROWS}")
    return rows


def _build_module(row: str, cfg: J1Config) -> nn.Module:
    return build_rx2_module(row, cfg) if row in ("RX0", "RX2") else build_j1_module(row, cfg)


def _row_seed_offset(row: str) -> int:
    return J1_ROWS.index(row) if row in J1_ROWS else len(J1_ROWS) + RX2_ROWS.index(row)


def smoke_bypass_audit(row: str, candidate_logits: Tensor, base_logits: Tensor) -> dict[str, Any]:
    max_delta = float((candidate_logits - base_logits).abs().max().detach().item())
    agreement = float(candidate_logits.argmax(1).eq(base_logits.argmax(1)).float().mean().item())
    numeric_parity = bool(torch.allclose(candidate_logits, base_logits, atol=1e-5, rtol=1e-5))
    return {
        "epoch0_bypass_pass": bool(agreement == 1.0) if row in ("RX0", "RX1", "RX2") else numeric_parity,
        "prediction_agreement": agreement,
        "numeric_logit_parity": numeric_parity,
        "max_abs_logit_delta": max_delta,
        "criterion": "decision_parity" if row in ("RX0", "RX1", "RX2") else "numeric_logit_parity",
    }


def sanitize_nonfinite_gradients(model: nn.Module) -> int:
    """Zero only non-finite gradient elements before norm clipping.

    Frozen Core90 can have finite logits but singular input derivatives.  RX1
    must not let those derivatives poison its identity-initialized estimator.
    The returned count is persisted in training history for auditability.
    """
    count = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        invalid = ~torch.isfinite(gradient)
        count += int(invalid.sum().item())
        if bool(invalid.any()):
            gradient.masked_fill_(invalid, 0.0)
    return count


def _phase_proxy(clean_iq: Tensor, changed_iq: Tensor) -> Tensor:
    def stats(iq: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.complex(iq[:, 0], iq[:, 1])
        step = x[:, 1:] * x[:, :-1].conj()
        unit = step / step.abs().clamp_min(1e-6)
        mean = unit.mean(1)
        mean_phase = torch.atan2(mean.imag, mean.real)
        concentration = mean.abs()
        log_power = iq.square().mean((1, 2)).clamp_min(1e-8).log()
        spectrum = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1).abs()
        axis = torch.linspace(-1.0, 1.0, spectrum.size(1), device=iq.device)
        centroid = (spectrum * axis).sum(1) / spectrum.sum(1).clamp_min(1e-6)
        return mean_phase, concentration, log_power, centroid

    c = stats(clean_iq)
    s = stats(changed_iq)
    phase_delta = torch.atan2(torch.sin(s[0] - c[0]), torch.cos(s[0] - c[0]))
    return torch.stack((phase_delta, s[1] - c[1], s[2] - c[2], s[3] - c[3]), 1)


def _forward_candidate(
    row: str,
    model: nn.Module,
    *,
    iq: Tensor,
    z_id: Tensor,
    base_logits: Tensor,
    domain: Tensor,
    base: nn.Module,
) -> J1Output:
    output = model(iq=iq, z_id=z_id, base_logits=base_logits, domain=domain)
    if row in ("RX0", "RX1", "RX2"):
        corrected_logits, corrected_z = _base_forward(base, output.corrected_iq, domain)
        output.final_logits = corrected_logits
        output.residual_logits = corrected_logits - base_logits
        output.embedding = corrected_z
    return output


def _group_objective(logits: Tensor, base_logits: Tensor, labels: Tensor, receivers: Tensor) -> dict[str, float]:
    prediction = logits.argmax(1)
    base_prediction = base_logits.argmax(1)
    by_receiver = []
    for receiver in receivers.unique():
        mask = receivers.eq(receiver)
        by_receiver.append(float(prediction[mask].eq(labels[mask]).float().mean().item()))
    rescue = base_prediction.ne(labels) & prediction.eq(labels)
    harm = base_prediction.eq(labels) & prediction.ne(labels)
    mean = float(prediction.eq(labels).float().mean().item())
    floor = min(by_receiver)
    harm_rate = float(harm.float().mean().item())
    return {
        "mean_accuracy": mean,
        "receiver_floor": floor,
        "rescue_rate": float(rescue.float().mean().item()),
        "harm_rate": harm_rate,
        "objective": mean + 0.25 * floor - 2.0 * harm_rate,
    }


def _train_model(
    row: str,
    model: nn.Module,
    train_cache,
    train_mask: Tensor,
    *,
    base: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    seed: int,
) -> list[dict[str, float]]:
    indices = _mask_indices(train_mask)
    generator = torch.Generator().manual_seed(int(seed))
    effective_lr = float(args.learning_rate) * (0.10 if row in ("RX0", "RX1", "RX2") else 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    satellites = REQUIRED_SCENARIOS[1:]
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = indices[torch.randperm(indices.numel(), generator=generator)]
        total = ce_total = preserve_total = gate_total = nuisance_total = 0.0
        steps = 0
        sanitized_gradient_elements = 0
        for start in range(0, int(order.numel()), int(args.batch_size)):
            chosen = order[start : start + int(args.batch_size)]
            scenario = satellites[(epoch + steps - 1) % len(satellites)]
            clean_iq = train_cache.iq["clean"][chosen].to(device)
            sat_iq = train_cache.iq[scenario][chosen].to(device)
            labels = train_cache.tx[chosen].to(device)
            receivers = train_cache.receiver[chosen].to(device)
            domain = train_cache.domain[chosen].to(device)
            clean_base = train_cache.base_logits["clean"][chosen].to(device)
            sat_base = train_cache.base_logits[scenario][chosen].to(device)
            clean_z = train_cache.z_id["clean"][chosen].to(device)
            sat_z = train_cache.z_id[scenario][chosen].to(device)
            clean_out = _forward_candidate(
                row, model, iq=clean_iq, z_id=clean_z, base_logits=clean_base, domain=domain, base=base
            )
            sat_out = _forward_candidate(
                row, model, iq=sat_iq, z_id=sat_z, base_logits=sat_base, domain=domain, base=base
            )
            zero = clean_base.sum() * 0.0
            ce = preserve = gate = nuisance = zero
            if row == "P0":
                target = _phase_proxy(clean_iq, sat_iq).detach()
                nuisance = F.smooth_l1_loss(clean_out.nuisance_prediction, torch.zeros_like(target))
                nuisance = nuisance + F.smooth_l1_loss(sat_out.nuisance_prediction, target)
                loss = nuisance
            else:
                base_hardness = F.cross_entropy(sat_base.detach(), labels, reduction="none").clamp(max=4.0)
                ce_per = F.cross_entropy(sat_out.final_logits, labels, reduction="none")
                ce = (ce_per * (1.0 + 0.25 * base_hardness)).mean()
                ce = ce + 0.5 * F.cross_entropy(clean_out.final_logits, labels)
                protected = clean_base.argmax(1).eq(labels) & F.softmax(clean_base, 1).max(1).values.ge(0.80)
                if bool(protected.any()):
                    preserve = F.kl_div(
                        F.log_softmax(clean_out.final_logits[protected], 1),
                        F.softmax(clean_base[protected].detach(), 1),
                        reduction="batchmean",
                    )
                receiver_losses = []
                for receiver in receivers.unique():
                    mask = receivers.eq(receiver)
                    receiver_losses.append(F.cross_entropy(sat_out.final_logits[mask], labels[mask]))
                worst = torch.logsumexp(torch.stack(receiver_losses) / 0.25, 0) * 0.25
                candidate = sat_out.final_logits.detach().argmax(1)
                base_pred = sat_base.argmax(1)
                rescue = (base_pred.ne(labels) & candidate.eq(labels)).float()
                harm = (base_pred.eq(labels) & candidate.ne(labels)).float()
                gate = F.binary_cross_entropy_with_logits(sat_out.gate_logits[:, 0], rescue)
                gate = gate + 2.0 * F.binary_cross_entropy_with_logits(sat_out.gate_logits[:, 1], harm)
                correction = sat_out.residual_logits.square().mean()
                if "correction_norm" in sat_out.diagnostics:
                    correction = correction + sat_out.diagnostics["correction_norm"].square().mean()
                loss = ce + 0.20 * preserve + 0.10 * worst + 0.05 * gate + 0.01 * correction
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite J1 loss row={row} epoch={epoch} step={steps + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if row in ("RX0", "RX2"):
                health = require_finite_gradients(model)
                sanitized_gradient_elements += int(health["nonfinite_elements"])
            else:
                sanitized_gradient_elements += sanitize_nonfinite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item())
            ce_total += float(ce.detach().item())
            preserve_total += float(preserve.detach().item())
            gate_total += float(gate.detach().item())
            nuisance_total += float(nuisance.detach().item())
            steps += 1
        history.append({
            "epoch": float(epoch), "loss": total / steps, "ce": ce_total / steps,
            "preserve": preserve_total / steps, "gate": gate_total / steps,
            "nuisance": nuisance_total / steps,
            "learning_rate": effective_lr,
            "sanitized_gradient_elements": float(sanitized_gradient_elements),
        })
    return history


@torch.no_grad()
def _evaluate_arrays(row: str, model: nn.Module, cache, mask: Tensor, scenario: str, *, base: nn.Module, device: torch.device, batch_size: int):
    indices = _mask_indices(mask)
    outputs = []
    for start in range(0, int(indices.numel()), int(batch_size)):
        chosen = indices[start : start + int(batch_size)]
        outputs.append((chosen, _forward_candidate(
            row,
            model,
            iq=cache.iq[scenario][chosen].to(device),
            z_id=cache.z_id[scenario][chosen].to(device),
            base_logits=cache.base_logits[scenario][chosen].to(device),
            domain=cache.domain[chosen].to(device),
            base=base,
        )))
    return indices, outputs


@torch.no_grad()
def _calibrate_gate(row: str, model: nn.Module, cache, mask: Tensor, *, base: nn.Module, device: torch.device, batch_size: int) -> dict[str, Any]:
    if row == "P0":
        return {"threshold": None, "coverage": 0.0, "status": "NUISANCE_ONLY_NO_TX_GATE"}
    utility_rows, candidate_rows, base_rows, truth_rows = [], [], [], []
    for scenario in REQUIRED_SCENARIOS:
        _, batches = _evaluate_arrays(row, model, cache, mask, scenario, base=base, device=device, batch_size=batch_size)
        for chosen, output in batches:
            probs = torch.sigmoid(output.gate_logits)
            utility_rows.append((probs[:, 0] - 2.0 * probs[:, 1]).cpu())
            candidate_rows.append(output.final_logits.cpu())
            base_rows.append(cache.base_logits[scenario][chosen])
            truth_rows.append(cache.tx[chosen])
    utility = torch.cat(utility_rows)
    candidate = torch.cat(candidate_rows)
    base_logits = torch.cat(base_rows)
    truth = torch.cat(truth_rows)
    base_acc = float(base_logits.argmax(1).eq(truth).float().mean())
    quantiles = torch.quantile(utility, torch.tensor((0.50, 0.65, 0.75, 0.85, 0.90, 0.95))).unique()
    choices = []
    for threshold in quantiles.tolist():
        selected = utility > float(threshold)
        fused = torch.where(selected[:, None], candidate, base_logits)
        accuracy = float(fused.argmax(1).eq(truth).float().mean())
        coverage = float(selected.float().mean())
        if coverage > 0.0 and 100.0 * (base_acc - accuracy) <= 0.30 + 1e-12:
            choices.append((accuracy, coverage, float(threshold)))
    if not choices:
        return {"threshold": None, "coverage": 0.0, "base_accuracy": base_acc, "status": "NO_NONZERO_SAFE_GATE"}
    accuracy, coverage, threshold = max(choices)
    return {"threshold": threshold, "coverage": coverage, "base_accuracy": base_acc, "selected_accuracy": accuracy, "status": "NONZERO_GATE_SELECTED"}


def _inner_loro_audit(row: str, cfg: J1Config, train_cache, select_cache, outer_held: int, source_receivers: Sequence[int], *, base: nn.Module, device: torch.device, args: argparse.Namespace) -> list[dict[str, Any]]:
    if row == "B0":
        return []
    results = []
    for index, inner_held in enumerate(r for r in source_receivers if r != outer_held):
        model = _build_module(row, cfg).to(device)
        mask = train_cache.receiver.ne(outer_held) & train_cache.receiver.ne(inner_held)
        _train_model(row, model, train_cache, mask, base=base, device=device, args=args, epochs=int(args.inner_epochs), seed=int(args.seed) + outer_held * 1000 + inner_held * 17)
        audit_mask = select_cache.receiver.eq(inner_held)
        if row == "P0":
            _, batches = _evaluate_arrays(row, model, select_cache, audit_mask, "leo_low_elev_weak", base=base, device=device, batch_size=int(args.eval_batch_size))
            prediction = torch.cat([out.nuisance_prediction.cpu() for _, out in batches])
            target = _phase_proxy(select_cache.iq["clean"][audit_mask].to(device), select_cache.iq["leo_low_elev_weak"][audit_mask].to(device)).cpu()
            results.append({"inner_held_receiver": int(inner_held), "nuisance_mae": float((prediction - target).abs().mean())})
        else:
            _, batches = _evaluate_arrays(row, model, select_cache, audit_mask, "leo_low_elev_weak", base=base, device=device, batch_size=int(args.eval_batch_size))
            logits = torch.cat([out.final_logits.cpu() for _, out in batches])
            indices = _mask_indices(audit_mask)
            results.append({"inner_held_receiver": int(inner_held), **_group_objective(logits, select_cache.base_logits["leo_low_elev_weak"][indices], select_cache.tx[indices], select_cache.receiver[indices])})
    return results


@torch.no_grad()
def _append_records(row: str, held_receiver: int, model: nn.Module | None, cache, mask: Tensor, *, gate: Mapping[str, Any], base: nn.Module, device: torch.device, batch_size: int, predictions: list[dict[str, Any]], truths: list[dict[str, Any]]) -> None:
    indices = _mask_indices(mask)
    for scenario in REQUIRED_SCENARIOS:
        for start in range(0, int(indices.numel()), int(batch_size)):
            chosen = indices[start : start + int(batch_size)]
            base_logits = cache.base_logits[scenario][chosen].to(device)
            if row == "B0":
                candidate = base_logits
                utility = torch.full((chosen.numel(),), float("-inf"), device=device)
                selected = torch.zeros(chosen.numel(), dtype=torch.bool, device=device)
                nuisance = None
                diagnostics = {}
                parameter_count = 0
            else:
                output = _forward_candidate(row, model, iq=cache.iq[scenario][chosen].to(device), z_id=cache.z_id[scenario][chosen].to(device), base_logits=base_logits, domain=cache.domain[chosen].to(device), base=base)
                candidate = output.final_logits
                probability = torch.sigmoid(output.gate_logits)
                utility = probability[:, 0] - 2.0 * probability[:, 1]
                threshold = gate.get("threshold")
                selected = utility > float(threshold) if threshold is not None else torch.zeros_like(utility, dtype=torch.bool)
                nuisance = output.nuisance_prediction
                diagnostics = output.diagnostics
                parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            final = torch.where(selected[:, None], candidate, base_logits)
            for position, cache_index in enumerate(chosen.tolist()):
                sample_id = f"{row}:rx{held_receiver}:V_select:{scenario}:{int(cache.base_index[cache_index])}"
                record = {
                    "sample_id": sample_id, "row": row, "scenario": scenario,
                    "held_receiver": int(held_receiver), "receiver": int(cache.receiver[cache_index]),
                    "day": int(cache.day[cache_index]), "base_index": int(cache.base_index[cache_index]),
                    "base_predicted_class": int(base_logits[position].argmax()),
                    "candidate_predicted_class": int(candidate[position].argmax()),
                    "final_predicted_class": int(final[position].argmax()),
                    "gate_selected": bool(selected[position]), "gate_utility": float(utility[position]),
                    "parameter_count": int(parameter_count), "target_or_query_access": False,
                }
                if nuisance is not None:
                    record["nuisance_prediction"] = nuisance[position].detach().cpu().tolist()
                    target = _phase_proxy(cache.iq["clean"][cache_index:cache_index+1].to(device), cache.iq[scenario][cache_index:cache_index+1].to(device))
                    record["nuisance_target_proxy"] = target[0].cpu().tolist()
                if "valid_bin_fraction" in diagnostics and torch.is_tensor(diagnostics["valid_bin_fraction"]):
                    record["valid_bin_fraction"] = float(diagnostics["valid_bin_fraction"][position])
                predictions.append(record)
                truths.append({"sample_id": sample_id, "true_class": int(cache.tx[cache_index])})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JMRS02 J1 role-correct single-module source-LORO screen")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--rows", default=",".join(J1_ROWS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--sat_seed", type=int, default=20260824)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--inner_epochs", type=int, default=10)
    parser.add_argument("--outer_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--eval_sat_on", default="main")
    parser.add_argument("--train_role", default="L_s", choices=("L_s",))
    parser.add_argument("--select_role", default="V_select", choices=("V_select",))
    parser.add_argument("--cal_role", default="V_cal", choices=("V_cal",))
    parser.add_argument("--audit_role", default="V_select", choices=("V_select",))
    parser.add_argument("--target_or_query_access", action="store_true", default=False)
    parser.add_argument("--focused_rx2", action="store_true", default=False)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    rows = validate_j1_args(args)
    output = validate_output_root(Path(args.output_dir))
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "rows": rows, "target_or_query_access": False}), flush=True)
        return 0
    checkpoint_path = Path(args.checkpoint).resolve()
    wisig_path = Path(args.wisig_pkl).resolve()
    if not checkpoint_path.is_file() or not wisig_path.is_file():
        raise FileNotFoundError("checkpoint and wisig_pkl must exist")
    output.mkdir(parents=True, exist_ok=False)
    _seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = _torch_load(checkpoint_path, device)
    ssdg, data_args = _prepare_ssdg_args(args, checkpoint)
    data_ctx = ssdg._build_ssdg_wisig_data(data_args, device)
    validate_source_roles(data_args, data_ctx["split_info"])
    base, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(checkpoint, input_len=int(data_ctx["input_len"]), device=device, ssdg_module=ssdg)
    base = freeze_base_model(base)
    _, smoke_batch = next(iter(_limited(data_ctx["train_loader"], 1)))
    smoke_iq, smoke_tx, smoke_domain, _ = _move_batch(ssdg, smoke_batch, device, data_ctx["domain_label_map"])
    smoke_base, smoke_z = _base_forward(base, smoke_iq, smoke_domain)
    cfg = J1Config(z_dim=int(smoke_z.size(1)), num_classes=int(smoke_base.size(1)), seed=int(args.seed))
    smoke = {}
    for row in rows[1:]:
        model = _build_module(row, cfg).to(device)
        out = _forward_candidate(row, model, iq=smoke_iq, z_id=smoke_z, base_logits=smoke_base, domain=smoke_domain, base=base)
        smoke[row] = {
            "finite": bool(torch.isfinite(out.final_logits).all()),
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **smoke_bypass_audit(row, out.final_logits, smoke_base),
        }
        if row in ("RX0", "RX2"):
            smoke[row]["real_core_backward"] = real_core_backward_probe(
                model, out.final_logits, smoke_tx, out.diagnostics["correction_norm"]
            )
    if not all(x["finite"] and x["epoch0_bypass_pass"] and x["parameter_count"] <= 50_000 for x in smoke.values()):
        raise RuntimeError("JMRS02 J1 real-checkpoint smoke failed")
    _write_json(output / "protocol_and_smoke.json", {"status": "REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS", "rows": rows, "scenarios": REQUIRED_SCENARIOS, "checkpoint_audit": checkpoint_audit, "source_roles": data_ctx["split_info"], "smoke": smoke, "target_or_query_access": False, "spectral_ratio_removed": True})
    if args.smoke_only:
        print(f"[JMRS02-J1-SMOKE] PASS output={output}", flush=True)
        return 0
    print("[JMRS02-J1] caching source roles", flush=True)
    train_cache = _collect_role_cache(data_ctx["train_loader"], base=base, ssdg=ssdg, data_ctx=data_ctx, data_args=data_args, device=device, max_batches=int(args.max_eval_batches), sat_seed=int(args.sat_seed) + 10000)
    select_cache = _collect_role_cache(data_ctx["val_loader"], base=base, ssdg=ssdg, data_ctx=data_ctx, data_args=data_args, device=device, max_batches=int(args.max_eval_batches), sat_seed=int(args.sat_seed) + 20000)
    cal_cache = _collect_role_cache(data_ctx["source_calibration_loader"], base=base, ssdg=ssdg, data_ctx=data_ctx, data_args=data_args, device=device, max_batches=int(args.max_eval_batches), sat_seed=int(args.sat_seed) + 30000)
    source_receivers = sorted(set(train_cache.receiver.tolist()) | set(select_cache.receiver.tolist()) | set(cal_cache.receiver.tolist()))
    if len(source_receivers) != 7:
        raise ValueError(f"expected seven source receivers, got {source_receivers}")
    predictions: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"status": "PREDICTIONS_PENDING", "rows": rows, "source_receivers": source_receivers, "folds": {}, "target_or_query_access": False}
    _write_json(output / "run_manifest.json", manifest)
    (output / "models").mkdir()
    (output / "training_history").mkdir()
    for held_receiver in source_receivers:
        partition = build_loro_partition(train_cache.receiver, select_cache.receiver, cal_cache.receiver, held_receiver=held_receiver)
        fold = f"rx{held_receiver}"
        manifest["folds"][fold] = {}
        for row in rows:
            if row == "B0":
                model = None
                gate = {"threshold": None, "coverage": 0.0, "status": "BASE_ONLY"}
                inner = []
            else:
                print(f"[JMRS02-J1] fold={fold} row={row} inner-LORO", flush=True)
                inner = [] if bool(args.focused_rx2) else _inner_loro_audit(row, cfg, train_cache, select_cache, held_receiver, source_receivers, base=base, device=device, args=args)
                model = _build_module(row, cfg).to(device)
                history = _train_model(row, model, train_cache, partition.train, base=base, device=device, args=args, epochs=int(args.outer_epochs), seed=int(args.seed) + held_receiver * 101 + _row_seed_offset(row))
                gate = _calibrate_gate(row, model, cal_cache, partition.cal, base=base, device=device, batch_size=int(args.eval_batch_size))
                torch.save({"row": row, "held_receiver": held_receiver, "config": asdict(cfg), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "gate": gate}, output / "models" / f"{fold}_{row}.pt")
                _write_json(output / "training_history" / f"{fold}_{row}.json", {"inner_loro": inner, "outer_history": history})
            manifest["folds"][fold][row] = {"gate": gate, "inner_loro": inner}
            _append_records(row, held_receiver, model, select_cache, partition.audit, gate=gate, base=base, device=device, batch_size=int(args.eval_batch_size), predictions=predictions, truths=truths)
            _write_json(output / "run_manifest.json", manifest)
    closure = write_closed_prediction_truth_streams(output, predictions, truths)
    manifest["status"] = "PREDICTIONS_COMPLETE"
    manifest["closure"] = closure
    _write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": "PREDICTIONS_COMPLETE", "count": len(predictions), "target_or_query_access": False}), flush=True)
    return 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
