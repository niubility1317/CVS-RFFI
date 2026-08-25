"""PA-M2.1 source-only theta-transfer audit and conditional expert gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from audit_phase1_ccoi_pa_v2 import build_sidecar_from_state, validate_sidecar_payload
from cvsrffi.ccoi_pa import CCOIPASidecar, raw_support_holdout_masks
from cvsrffi.ccoi_pa_m21 import (
    FoldRecords,
    SidecarArchitectureConfig,
    bounded_residual_fusion,
    build_fold_records,
    build_relation_indices,
    build_sidecar_v3_payload,
    compose_factor_rows,
    conditional_q_probe,
    duplicate_audit,
    evaluate_stage_a,
    evaluate_stage_b,
    fit_truth_blind_gate,
    m0_exact_pair_retrieval,
    predict_truth_blind_gate,
    run_factor_matrix,
    run_loto_residual,
    split_v_select_retro,
)
from cvsrffi.ccoi_causal_audit import group_paired_bootstrap, token_code_audit
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from train_phase1_ccoi_pa import (
    FrozenCore90CCOI,
    SCENARIOS,
    _infer_base_dimensions,
    _limited,
    _meta_value,
    _move_batch,
    _prepare_ssdg_args,
    _satellite_view,
    _torch_load,
    _train_sidecar,
    freeze_base_model,
    validate_source_roles,
)


SCHEMA = "cvs.phase1.ccoi_pa_m21_audit.v1"
AGGREGATE_ARTIFACTS = (
    "split_manifest.json",
    "sidecar_architecture_c1p.json",
    "sidecar_architecture_c4p.json",
    "sidecar_training_summary.json",
    "duplicate_audit.json",
    "q_conditional_probe.json",
    "m0_exact_pair_retrieval.json",
    "factor_matrix_c1p.json",
    "factor_matrix_c4p.json",
    "loto_residual_audit.json",
    "gate_calibration_summary.json",
    "gate_audit_summary.json",
    "decision_manifest.json",
    "final_report.md",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PA-M2.1 independent theta-transfer audit")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sat_seed", type=int, default=20260824)
    parser.add_argument("--eval_sat_on", default="main")
    parser.add_argument("--train_role", default="L_s", choices=("L_s",))
    parser.add_argument("--gate_role", default="V_cal", choices=("V_cal",))
    parser.add_argument("--fit_role", default="V_select_fit", choices=("V_select_fit",))
    parser.add_argument("--audit_role", default="V_audit_retro", choices=("V_audit_retro",))
    parser.add_argument("--stage_a", default="M2.1A_THETA_TRANSFER_AUDIT", choices=("M2.1A_THETA_TRANSFER_AUDIT",))
    parser.add_argument("--stage_b", default="M2.1B_TRUTH_BLIND_EXPERT_GATE", choices=("M2.1B_TRUTH_BLIND_EXPERT_GATE",))
    parser.add_argument("--target_or_query_access", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--head_epochs", type=int, default=60)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--min_match_cosine", type=float, default=0.70)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--factor_steps", type=int, default=800)
    parser.add_argument("--probe_steps", type=int, default=400)
    parser.add_argument("--loto_steps", type=int, default=800)
    parser.add_argument("--gate_steps", type=int, default=200)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--block_candidates", default="10,20,25")
    parser.add_argument("--fit_ratio", type=float, default=0.65)
    parser.add_argument("--gate_coverage_min", type=float, default=0.05)
    parser.add_argument("--major_cell_minimum", type=int, default=10)
    parser.add_argument("--legacy_migration_mode", action="store_true", default=False)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--synthetic_smoke", action="store_true")
    return parser


def validate_output_root(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing PA-M2.1 output: {output}")
    return output


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_not_run_gate_payload(stage_a_status: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN_A_GATE",
        "stage_a_status": str(stage_a_status),
        "technical_failure": False,
        "target_or_query_access": False,
        "reason": "stage B is authorized only by A_PASS",
        "sample_level_state_persisted": False,
    }


def _synthetic_smoke(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    base = {
        "status": "SYNTHETIC_SMOKE",
        "schema": SCHEMA,
        "target_or_query_access": False,
        "sample_level_state_persisted": False,
    }
    for name in AGGREGATE_ARTIFACTS:
        if name.endswith(".json"):
            payload = dict(base)
            if name == "decision_manifest.json":
                payload["artifact_count"] = len(AGGREGATE_ARTIFACTS)
            _json_write(output / name, payload)
        else:
            (output / name).write_text(
                "# PA-M2.1 synthetic smoke\n\n状态：SYNTHETIC_SMOKE。\n",
                encoding="utf-8",
                newline="\n",
            )
    return 0


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


def _collect_packets(loader, ssdg, data_ctx, device: torch.device, max_batches: int) -> dict[str, Tensor]:
    rows: dict[str, list[Tensor]] = {
        key: [] for key in ("iq", "tx", "receiver", "day", "eq", "sig_i", "base_index")
    }
    with torch.no_grad():
        for _, batch in _limited(loader, int(max_batches)):
            x, y, _domain, extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            count = int(x.size(0))
            values = {
                "iq": x.detach().float().cpu(),
                "tx": y.detach().long().cpu(),
                "receiver": _metadata_tensor(extra, "rx_i", count),
                "day": _metadata_tensor(extra, "day_i", count),
                "eq": _metadata_tensor(extra, "eq_i", count),
                "sig_i": _metadata_tensor(extra, "sig_i", count),
                "base_index": _metadata_tensor(extra, "base_index", count),
            }
            for name, value in values.items():
                rows[name].append(value)
    if not rows["iq"]:
        raise RuntimeError("source role produced zero packets")
    result = {name: torch.cat(values, dim=0) for name, values in rows.items()}
    if len(set(result["base_index"].tolist())) != int(result["base_index"].numel()):
        raise ValueError("source role base_index is not unique")
    return result


def _metadata_view(packets: Mapping[str, Tensor], indices: Sequence[int] | None = None) -> dict[str, Tensor]:
    names = ("tx", "receiver", "day", "eq", "sig_i", "base_index")
    if indices is None:
        return {name: packets[name] for name in names}
    selected = torch.tensor(tuple(int(index) for index in indices), dtype=torch.long)
    return {name: packets[name][selected] for name in names}


def _packet_view(packets: Mapping[str, Tensor], indices: Sequence[int]) -> dict[str, Tensor]:
    selected = torch.tensor(tuple(int(index) for index in indices), dtype=torch.long)
    return {name: value[selected] for name, value in packets.items()}


def _subset_loader(loader, indices: Sequence[int], args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = Subset(loader.dataset, tuple(int(index) for index in indices))
    kwargs: dict[str, Any] = {
        "batch_size": int(args.eval_batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    if int(args.num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    return DataLoader(dataset, **kwargs)


def _fixed_physical_features(iq: Tensor) -> Tensor:
    packets = torch.as_tensor(iq).detach().float().cpu()
    complex_iq = torch.complex(packets[:, 0], packets[:, 1])
    amplitude = complex_iq.abs().clamp_min(1e-8)
    power = amplitude.square()
    rms = power.mean(dim=1).sqrt()
    papr = power.max(dim=1).values / power.mean(dim=1).clamp_min(1e-8)
    normalized = amplitude / rms[:, None].clamp_min(1e-8)
    moment4 = normalized.pow(4).mean(dim=1)
    moment6 = normalized.pow(6).mean(dim=1)
    envelope_diff = (normalized[:, 1:] - normalized[:, :-1]).abs().mean(dim=1)
    autocorr = (normalized[:, 1:] * normalized[:, :-1]).mean(dim=1)
    memory_terms = []
    for delay in (0, 1):
        delayed = torch.roll(complex_iq, delay, dims=1)
        delayed_amplitude = delayed.abs()
        for order in (1, 3, 5):
            memory_terms.append(delayed * delayed_amplitude.pow(order - 1))
    design = torch.stack(memory_terms, dim=-1)
    gram = torch.einsum("nli,nlj->nij", design.conj(), design) / float(design.size(1))
    regularizer = 1e-4 * gram.diagonal(dim1=1, dim2=2).real.mean(dim=1).clamp_min(1e-8)
    eye = torch.eye(design.size(-1), dtype=gram.dtype).unsqueeze(0)
    eigen = torch.linalg.eigvalsh(gram + regularizer[:, None, None] * eye).real
    condition = eigen[:, -1] / eigen[:, 0].clamp_min(1e-8)
    phase_step = torch.angle(complex_iq[:, 1:] * complex_iq[:, :-1].conj())
    phase_instability = phase_step.std(dim=1, unbiased=False)
    return torch.stack(
        (rms, papr, moment4, moment6, envelope_diff, autocorr, condition, phase_instability),
        dim=1,
    )


def _base_forward(base, x: Tensor, domain: Tensor) -> tuple[Tensor, Tensor]:
    output = base(x, return_aux=True, domain_labels=domain)
    pa_map = (output.get("aux_id", {}) or {}).get("pa_token_map")
    if not torch.is_tensor(pa_map):
        raise KeyError("Core90 auxiliary output is missing pa_token_map")
    return output["tx_logits"], pa_map


def _collect_fold_records(
    base,
    sidecar: CCOIPASidecar,
    loader,
    *,
    conditioned: bool,
    ssdg,
    data_ctx,
    device: torch.device,
    max_batches: int,
    scenario: str = "clean",
    data_args=None,
    sat_seed: int = 0,
) -> tuple[FoldRecords, dict[str, Tensor]]:
    fold_rows: dict[str, list[Tensor]] = {
        name: []
        for name in ("base_index", "fold_id", "theta", "q_holdout", "target", "support_raw_mask", "holdout_raw_mask")
    }
    packet_rows: dict[str, list[Tensor]] = {
        name: [] for name in ("q", "code_prob", "base_logits", "operator_logits", "coverage", "iq", "tx", "receiver", "day", "eq", "sig_i", "base_index")
    }
    sidecar.eval()
    base.eval()
    for fold in range(4):
        generator = None if scenario == "clean" else ssdg.make_torch_generator(device, int(sat_seed))
        for _, batch in _limited(loader, int(max_batches)):
            x, y, domain, extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            view = x if scenario == "clean" else _satellite_view(ssdg, x, scenario, data_args, generator)
            count = int(view.size(0))
            base_index = _metadata_tensor(extra, "base_index", count)
            with torch.no_grad():
                base_logits, pa_map = _base_forward(base, view, domain)
                token_count = 1 + (
                    int(view.size(-1)) - int(sidecar.challenge_encoder.token_length)
                ) // int(sidecar.challenge_encoder.stride)
                support_raw, holdout_raw = raw_support_holdout_masks(
                    int(view.size(-1)),
                    token_count,
                    token_length=sidecar.challenge_encoder.token_length,
                    stride=sidecar.challenge_encoder.stride,
                    fold=fold,
                )
                support_view = view * support_raw.to(device=device, dtype=view.dtype).view(1, 1, -1)
                holdout_view = view * holdout_raw.to(device=device, dtype=view.dtype).view(1, 1, -1)
                _support_logits, support_pa = _base_forward(base, support_view, domain)
                _target_logits, target_pa = _base_forward(base, holdout_view, domain)
                side = sidecar(
                    view,
                    pa_map.detach(),
                    conditioned=conditioned,
                    holdout_fold=fold,
                    holdout_support_pa_map=support_pa.detach(),
                    holdout_target_pa_map=target_pa.detach(),
                )
            fold_rows["base_index"].append(base_index)
            fold_rows["fold_id"].append(torch.full((count,), fold, dtype=torch.long))
            fold_rows["theta"].append(side["support_theta"].detach().float().cpu())
            fold_rows["q_holdout"].append(side["q_holdout"].detach().float().cpu())
            fold_rows["target"].append(side["heldout_target"].detach().float().cpu())
            fold_rows["support_raw_mask"].append(support_raw.unsqueeze(0).expand(count, -1))
            fold_rows["holdout_raw_mask"].append(holdout_raw.unsqueeze(0).expand(count, -1))
            if fold == 0:
                values = {
                    "q": side["q"],
                    "code_prob": side["code_prob"],
                    "base_logits": base_logits,
                    "operator_logits": side["logit_correction"],
                    "coverage": side["coverage"],
                    "iq": view,
                    "tx": y,
                    "receiver": _metadata_tensor(extra, "rx_i", count),
                    "day": _metadata_tensor(extra, "day_i", count),
                    "eq": _metadata_tensor(extra, "eq_i", count),
                    "sig_i": _metadata_tensor(extra, "sig_i", count),
                    "base_index": base_index,
                }
                for name, value in values.items():
                    packet_rows[name].append(
                        value.detach().float().cpu() if value.is_floating_point() else value.detach().long().cpu()
                    )
    records = FoldRecords(
        base_index=torch.cat(fold_rows["base_index"]),
        fold_id=torch.cat(fold_rows["fold_id"]),
        theta=torch.cat(fold_rows["theta"]),
        q_holdout=torch.cat(fold_rows["q_holdout"]),
        target=torch.cat(fold_rows["target"]),
        support_raw_mask=torch.cat(fold_rows["support_raw_mask"]),
        holdout_raw_mask=torch.cat(fold_rows["holdout_raw_mask"]),
        fold_count=4,
    )
    packets = {name: torch.cat(values) for name, values in packet_rows.items()}
    for fold in range(4):
        if bool((records.support_raw_mask[records.fold_id == fold] & records.holdout_raw_mask[records.fold_id == fold]).any()):
            raise RuntimeError(f"fold {fold} support and holdout overlap")
    return records, packets


def _relation_mappings(
    audit_packets: Mapping[str, Tensor],
    bank_packets: Mapping[str, Tensor],
    seed: int,
) -> dict[str, Any]:
    audit_metadata = _metadata_view(audit_packets)
    bank_metadata = _metadata_view(bank_packets)
    audit_physical = _fixed_physical_features(audit_packets["iq"])
    bank_physical = _fixed_physical_features(bank_packets["iq"])
    return {
        relation: build_relation_indices(
            audit_metadata,
            bank_metadata,
            relation,
            seed=int(seed),
            audit_physical_features=audit_physical if relation == "F6" else None,
            bank_physical_features=bank_physical if relation == "F6" else None,
        )
        for relation in ("F2", "F3", "F4", "F5", "F6", "F7")
    }


def _new_capacity_matched_sidecar(
    legacy: CCOIPASidecar,
    *,
    pa_channels: int,
    num_classes: int,
    device: torch.device,
) -> CCOIPASidecar:
    q_dim = int(legacy.challenge_encoder.q_dim)
    response_dim = int(legacy.response_head.pa_proj.out_features)
    operator_dim = int(legacy.operator_pool.value.out_features)
    return CCOIPASidecar(
        pa_channels=int(pa_channels),
        num_classes=int(num_classes),
        challenge_encoder=deepcopy(legacy.challenge_encoder),
        q_dim=q_dim,
        response_dim=response_dim,
        operator_dim=operator_dim,
    ).to(device)


def _architecture_config(
    sidecar: CCOIPASidecar,
    *,
    input_length: int,
    pa_channels: int,
    num_classes: int,
    num_domains: int,
    conditioned: bool,
) -> SidecarArchitectureConfig:
    encoder = sidecar.challenge_encoder
    return SidecarArchitectureConfig(
        input_length=int(input_length),
        token_length=int(encoder.token_length),
        stride=int(encoder.stride),
        q_dim=int(encoder.q_dim),
        challenge_hidden_dim=int(encoder.q_head.in_features),
        codebook_size=int(encoder.codebook_size),
        response_dim=int(sidecar.response_head.pa_proj.out_features),
        operator_dim=int(sidecar.operator_pool.value.out_features),
        pa_channels=int(pa_channels),
        num_classes=int(num_classes),
        num_domains=int(num_domains),
        holdout_anchor_policy="all_nonoverlap_folds",
        conditioned=bool(conditioned),
        pa_map_contract="core90_pa_token_map_v1",
    )


def _logit_margin_entropy(logits: Tensor) -> tuple[Tensor, Tensor]:
    probability = F.softmax(torch.as_tensor(logits).detach().float().cpu(), dim=1)
    top = probability.topk(2, dim=1).values
    margin = top[:, 0] - top[:, 1]
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
    return margin, entropy


def _gate_features(scene: Mapping[str, Tensor]) -> dict[str, Tensor]:
    base = scene["base_logits"].float()
    operator = scene["operator_logits"].float()
    iq = scene["iq"].float()
    physical = _fixed_physical_features(iq)
    base_margin, base_entropy = _logit_margin_entropy(base)
    operator_margin, operator_entropy = _logit_margin_entropy(operator)
    p = F.softmax(base, dim=1).clamp_min(1e-8)
    q = F.softmax(operator, dim=1).clamp_min(1e-8)
    mean = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.log() - mean.log())).sum(dim=1)
        + (q * (q.log() - mean.log())).sum(dim=1)
    )
    complex_iq = torch.complex(iq[:, 0], iq[:, 1])
    amplitude = complex_iq.abs()
    spectrum = torch.fft.fft(complex_iq, dim=1).abs().square()
    spectral_null = (spectrum < spectrum.median(dim=1).values[:, None] * 0.01).float().mean(dim=1)
    clipping = (amplitude >= amplitude.max(dim=1).values[:, None] * 0.995).float().mean(dim=1)
    smooth = F.avg_pool1d(iq, kernel_size=5, stride=1, padding=2)
    signal_power = iq.square().mean(dim=(1, 2))
    residual_power = (iq - smooth).square().mean(dim=(1, 2)).clamp_min(1e-8)
    snr_proxy = 10.0 * torch.log10(signal_power.clamp_min(1e-8) / residual_power)
    phase_step = torch.angle(complex_iq[:, 1:] * complex_iq[:, :-1].conj())
    return {
        "base_margin": base_margin,
        "base_entropy": base_entropy,
        "operator_margin": operator_margin,
        "operator_entropy": operator_entropy,
        "js_divergence": js,
        "top1_disagreement": base.argmax(dim=1).ne(operator.argmax(dim=1)).float(),
        "rms": physical[:, 0],
        "papr": physical[:, 1],
        "pa_condition_number": physical[:, 6],
        "spectral_null_ratio": spectral_null,
        "clipping_ratio": clipping,
        "snr_proxy": snr_proxy,
        "residual_cfo": phase_step.mean(dim=1).abs(),
        "phase_instability": physical[:, 7],
        "challenge_coverage": scene["coverage"].float().view(-1),
    }


def _collect_gate_scene(
    base,
    sidecar: CCOIPASidecar,
    loader,
    *,
    scenario: str,
    ssdg,
    data_ctx,
    data_args,
    device: torch.device,
    max_batches: int,
    sat_seed: int,
) -> dict[str, Tensor]:
    names = (
        "base_logits", "operator_logits", "coverage", "iq", "truth",
        "receiver", "day", "eq", "sig_i", "base_index",
    )
    rows: dict[str, list[Tensor]] = {name: [] for name in names}
    generator = None if scenario == "clean" else ssdg.make_torch_generator(device, int(sat_seed))
    base.eval()
    sidecar.eval()
    with torch.no_grad():
        for _, batch in _limited(loader, int(max_batches)):
            x, y, domain, extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            view = x if scenario == "clean" else _satellite_view(ssdg, x, scenario, data_args, generator)
            base_logits, pa_map = _base_forward(base, view, domain)
            side = sidecar(view, pa_map.detach(), conditioned=True)
            count = int(view.size(0))
            values = {
                "base_logits": base_logits,
                "operator_logits": side["logit_correction"],
                "coverage": side["coverage"],
                "iq": view,
                "truth": y,
                "receiver": _metadata_tensor(extra, "rx_i", count),
                "day": _metadata_tensor(extra, "day_i", count),
                "eq": _metadata_tensor(extra, "eq_i", count),
                "sig_i": _metadata_tensor(extra, "sig_i", count),
                "base_index": _metadata_tensor(extra, "base_index", count),
            }
            for name, value in values.items():
                rows[name].append(
                    value.detach().float().cpu() if value.is_floating_point() else value.detach().long().cpu()
                )
    if not rows["truth"]:
        raise RuntimeError(f"gate role produced zero samples for {scenario}")
    return {name: torch.cat(values) for name, values in rows.items()}


def _concat_feature_maps(items: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("gate feature maps do not share a schema")
    return {name: torch.cat([item[name] for item in items]) for name in sorted(keys)}


def _fusion_outcomes(scene: Mapping[str, Tensor], eta: float, clip_norm: float) -> dict[str, Tensor]:
    base = scene["base_logits"]
    final = bounded_residual_fusion(
        base,
        scene["operator_logits"],
        gate=torch.ones(base.size(0)),
        eta=float(eta),
        scale=1.0,
        clip_norm=float(clip_norm),
    )
    truth = scene["truth"].long()
    base_correct = base.argmax(dim=1).eq(truth)
    final_correct = final.argmax(dim=1).eq(truth)
    return {"rescue": ~base_correct & final_correct, "harm": base_correct & ~final_correct}


def _accuracy_gain_pp(base_logits: Tensor, final_logits: Tensor, truth: Tensor) -> float:
    base = base_logits.argmax(dim=1).eq(truth).float().mean()
    final = final_logits.argmax(dim=1).eq(truth).float().mean()
    return 100.0 * float((final - base).item())


def _accuracy_group_bootstrap(
    base_correct: Tensor,
    final_correct: Tensor,
    groups: Tensor,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    base = torch.as_tensor(base_correct).detach().float().cpu().view(-1)
    final = torch.as_tensor(final_correct).detach().float().cpu().view(-1)
    group = torch.as_tensor(groups).detach().cpu()
    if group.ndim == 1:
        group = group[:, None]
    if base.numel() == 0 or final.numel() != base.numel() or group.size(0) != base.numel():
        raise ValueError("accuracy bootstrap inputs must align")
    _, inverse = torch.unique(group, dim=0, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    delta = 100.0 * (final - base)
    sums = torch.zeros(group_count).scatter_add_(0, inverse, delta)
    sizes = torch.zeros(group_count).scatter_add_(0, inverse, torch.ones_like(delta))
    point = float(sums.sum().item() / sizes.sum().clamp_min(1.0).item())
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    for _ in range(max(1, int(resamples))):
        selected = torch.randint(group_count, (group_count,), generator=generator)
        draws.append(sums[selected].sum() / sizes[selected].sum().clamp_min(1.0))
    distribution = torch.stack(draws)
    return {
        "gain_pp": point,
        "ci95_low_pp": float(torch.quantile(distribution, 0.025).item()),
        "ci95_high_pp": float(torch.quantile(distribution, 0.975).item()),
        "group_count": group_count,
        "sample_count": int(base.numel()),
    }


def _gate_fit_payload(fitted) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "feature_names": list(fitted.feature_names),
        "eta": fitted.eta,
        "clip_norm": fitted.clip_norm,
        "tau": fitted.tau,
        "lambda_h": fitted.lambda_h,
        "oof_sample_count": fitted.oof_sample_count,
        "oof_coverage": fitted.oof_coverage,
        "oof_weighted_utility": fitted.oof_weighted_utility,
        "group_overlap_count": fitted.group_overlap_count,
        "positive_receiver_cv_count": fitted.positive_receiver_cv_count,
        "receiver_cv_count": fitted.receiver_cv_count,
        "audit_labels_consumed": fitted.audit_labels_consumed,
        "rescue_model": {
            "mean": fitted.rescue_mean.tolist(),
            "scale": fitted.rescue_scale.tolist(),
            "weight": fitted.rescue_weight.tolist(),
        },
        "harm_model": {
            "mean": fitted.harm_mean.tolist(),
            "scale": fitted.harm_scale.tolist(),
            "weight": fitted.harm_weight.tolist(),
        },
        "sample_level_state_persisted": False,
        "target_or_query_access": False,
    }


def _write_final_report(output: Path, decision: Mapping[str, Any]) -> None:
    stage_a = decision["stage_a_verdict"]
    stage_b = decision["stage_b_verdict"]
    text = f"""# PA-M2.1独立theta迁移审计最终报告

## 结论

阶段A状态：`{stage_a['status']}`；阶段B状态：`{stage_b['status']}`；最终路线：`{decision['next_route']}`。

本实验冻结Core90和旧C4 challenge encoder，从同一随机模板独立训练C1′与C4′。V_select按TX×RX×day×eq×capture block拆成权重选择子集、回溯审计子集和guard block；审计覆盖4个raw-disjoint fold，并只从独立support bank构造F2–F6。F7因没有已验证的跨receiver同步物理事件ID而保持`UNAVAILABLE`。

阶段B只有在`A_PASS`后才允许拟合。gate不读取true TX、receiver、day或审计标签，只使用预登记的部署时可得特征，并通过有界残差修正保护Core90。

## 证据边界

`V_audit_retro`只对本轮C1′/C4′新权重独立，不是研究历史完全未见集。当前q仍是已知受TX/RX/day/位置捷径污染的received-waveform excitation proxy；即使阶段A通过，也只能进入连续challenge重设计，不能直接晋级当前q。

全部正式JSON均为聚合结果；没有保存样本级q、theta、embedding、IQ或逐样本prediction stream。`target_or_query_access=false`。
"""
    (output / "final_report.md").write_text(text, encoding="utf-8", newline="\n")


def _real_run(args: argparse.Namespace, output: Path) -> int:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    sidecar_path = Path(args.sidecar).expanduser().resolve()
    wisig_path = Path(args.wisig_pkl).expanduser().resolve()
    for path in (checkpoint_path, sidecar_path, wisig_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not bool(args.legacy_migration_mode):
        raise ValueError("legacy_migration_mode=true is required to read the V2 challenge encoder")
    output.mkdir(parents=True, exist_ok=False)
    _seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = _torch_load(checkpoint_path, device)
    legacy_payload = _torch_load(sidecar_path, device)
    validate_sidecar_payload(legacy_payload)
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
    pa_channels, num_classes, base_smoke = _infer_base_dimensions(base, data_ctx, ssdg, device)
    legacy = build_sidecar_from_state(
        pa_channels=pa_channels,
        num_classes=num_classes,
        num_domains=int(data_ctx["num_domains"]),
        state_dict=legacy_payload["state_dict"],
        device=device,
    )
    legacy.freeze_challenge_encoder()

    _seed_all(args.seed + 101)
    template = _new_capacity_matched_sidecar(
        legacy, pa_channels=pa_channels, num_classes=num_classes, device=device
    )
    template.freeze_challenge_encoder()
    template_state = deepcopy(template.state_dict())
    c1p = _new_capacity_matched_sidecar(legacy, pa_channels=pa_channels, num_classes=num_classes, device=device)
    c4p = _new_capacity_matched_sidecar(legacy, pa_channels=pa_channels, num_classes=num_classes, device=device)
    c1p.load_state_dict(template_state, strict=True)
    c4p.load_state_dict(template_state, strict=True)
    c1p.freeze_challenge_encoder()
    c4p.freeze_challenge_encoder()
    if any(
        not torch.equal(c1p.state_dict()[name], c4p.state_dict()[name])
        for name in c1p.state_dict()
    ):
        raise RuntimeError("C1p and C4p did not start from the same template")

    _, smoke_batch = next(iter(_limited(data_ctx["val_loader"], 1)))
    smoke_x, _smoke_y, smoke_domain, _smoke_extra = _move_batch(
        ssdg, smoke_batch, device, data_ctx["domain_label_map"]
    )
    with torch.no_grad():
        _smoke_logits, smoke_pa = _base_forward(base, smoke_x, smoke_domain)
        smoke_c1 = c1p(smoke_x, smoke_pa, conditioned=False)
        smoke_c4 = c4p(smoke_x, smoke_pa, conditioned=True)
    smoke = {
        "status": "PASS",
        "base": base_smoke,
        "c1p_finite": bool(torch.isfinite(smoke_c1["logit_correction"]).all()),
        "c4p_finite": bool(torch.isfinite(smoke_c4["logit_correction"]).all()),
        "target_or_query_access": False,
        "checkpoint_audit": checkpoint_audit,
    }
    if not smoke["c1p_finite"] or not smoke["c4p_finite"]:
        raise FloatingPointError("real checkpoint no-query smoke produced non-finite logits")
    if bool(args.smoke_only):
        _json_write(output / "decision_manifest.json", smoke)
        print(f"[PA-M21-SMOKE] PASS output={output}", flush=True)
        return 0

    all_select_packets = _collect_packets(
        data_ctx["val_loader"], ssdg, data_ctx, device, int(args.max_eval_batches)
    )
    block_candidates = tuple(
        int(value.strip()) for value in str(args.block_candidates).split(",") if value.strip()
    )
    retro = split_v_select_retro(
        _metadata_view(all_select_packets),
        seed=int(args.seed),
        block_candidates=block_candidates,
        fit_ratio=float(args.fit_ratio),
    )
    fit_loader = _subset_loader(data_ctx["val_loader"], retro.fit_indices, args, device)
    audit_loader = _subset_loader(data_ctx["val_loader"], retro.audit_indices, args, device)
    fit_packets_raw = _packet_view(all_select_packets, retro.fit_indices)
    audit_packets_raw = _packet_view(all_select_packets, retro.audit_indices)
    split_manifest = {
        "status": "COMPLETE",
        "source_roles": data_ctx["split_info"],
        "block_size": retro.block_size,
        "fit_ratio": retro.fit_ratio,
        "fit_count": len(retro.fit_indices),
        "audit_count": len(retro.audit_indices),
        "guard_count": len(retro.guard_indices),
        "cell_count": retro.cell_count,
        "min_blocks_per_cell": retro.min_blocks_per_cell,
        "base_index_overlap_count": retro.base_index_overlap_count,
        "weight_independence_scope": "V_audit_retro_independent_of_new_C1p_C4p_weights_only",
        "historically_unseen": False,
        "target_or_query_access": False,
        "sample_level_state_persisted": False,
    }
    _json_write(output / "split_manifest.json", split_manifest)
    _json_write(
        output / "duplicate_audit.json",
        duplicate_audit(
            all_select_packets["iq"],
            _metadata_view(all_select_packets),
            retro,
            seed=int(args.seed) + 313,
        ),
    )

    train_ctx = dict(data_ctx)
    train_ctx["val_loader"] = fit_loader
    c1_model = FrozenCore90CCOI(base, c1p, row="C1", fusion_alpha=0.0).to(device)
    c4_model = FrozenCore90CCOI(base, c4p, row="C4", fusion_alpha=0.0).to(device)
    c1_history = _train_sidecar(c1_model, train_ctx, ssdg, data_args, args, device)
    c4_history = _train_sidecar(c4_model, train_ctx, ssdg, data_args, args, device)
    config_c1 = _architecture_config(
        c1p,
        input_length=int(data_ctx["input_len"]),
        pa_channels=pa_channels,
        num_classes=num_classes,
        num_domains=int(data_ctx["num_domains"]),
        conditioned=False,
    )
    config_c4 = _architecture_config(
        c4p,
        input_length=int(data_ctx["input_len"]),
        pa_channels=pa_channels,
        num_classes=num_classes,
        num_domains=int(data_ctx["num_domains"]),
        conditioned=True,
    )
    c1_payload = build_sidecar_v3_payload(
        c1p,
        row="C1p",
        base_checkpoint=str(checkpoint_path),
        architecture_config=config_c1,
        fusion_alpha=0.0,
        fusion_scale=1.0,
    )
    c4_payload = build_sidecar_v3_payload(
        c4p,
        row="C4p",
        base_checkpoint=str(checkpoint_path),
        architecture_config=config_c4,
        fusion_alpha=0.0,
        fusion_scale=1.0,
    )
    model_dir = output / "models"
    model_dir.mkdir()
    torch.save(c1_payload, model_dir / "c1p_sidecar_v3.pth")
    torch.save(c4_payload, model_dir / "c4p_sidecar_v3.pth")
    _json_write(output / "sidecar_architecture_c1p.json", {"schema": c1_payload["schema"], "row": "C1p", "architecture_config": asdict(config_c1)})
    _json_write(output / "sidecar_architecture_c4p.json", {"schema": c4_payload["schema"], "row": "C4p", "architecture_config": asdict(config_c4)})
    _json_write(
        output / "sidecar_training_summary.json",
        {
            "status": "COMPLETE",
            "same_initial_template": True,
            "same_parameter_count": sum(value.numel() for value in c1p.parameters()) == sum(value.numel() for value in c4p.parameters()),
            "selection_role": "V_select_fit",
            "training_role": "L_s",
            "c1p_history": c1_history,
            "c4p_history": c4_history,
            "audit_role_consumed_for_training_or_selection": False,
            "sample_level_state_persisted": False,
        },
    )

    c1_fit_records, c1_fit_packets = _collect_fold_records(
        base, c1p, fit_loader, conditioned=False, ssdg=ssdg, data_ctx=data_ctx,
        device=device, max_batches=int(args.max_eval_batches),
    )
    c1_audit_records, c1_audit_packets = _collect_fold_records(
        base, c1p, audit_loader, conditioned=False, ssdg=ssdg, data_ctx=data_ctx,
        device=device, max_batches=int(args.max_eval_batches),
    )
    c4_fit_records, c4_fit_packets = _collect_fold_records(
        base, c4p, fit_loader, conditioned=True, ssdg=ssdg, data_ctx=data_ctx,
        device=device, max_batches=int(args.max_eval_batches),
    )
    c4_audit_records, c4_audit_packets = _collect_fold_records(
        base, c4p, audit_loader, conditioned=True, ssdg=ssdg, data_ctx=data_ctx,
        device=device, max_batches=int(args.max_eval_batches),
    )
    mapping_seeds = (int(args.seed) + 501, int(args.seed) + 502, int(args.seed) + 503)
    head_seeds = (int(args.seed) + 601, int(args.seed) + 602, int(args.seed) + 603)
    eval_groups = torch.stack(
        (
            c4_audit_packets["tx"].repeat(4),
            c4_audit_packets["receiver"].repeat(4),
            c4_audit_packets["day"].repeat(4),
            c4_audit_records.fold_id,
        ),
        dim=1,
    )

    factor_results: dict[str, dict[int, Any]] = {"c1p": {}, "c4p": {}}
    mapping_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for mapping_seed in mapping_seeds:
        fit_mapping = _relation_mappings(c4_fit_packets, c4_fit_packets, mapping_seed)
        audit_mapping = _relation_mappings(c4_audit_packets, c4_fit_packets, mapping_seed)
        mapping_cache[mapping_seed] = (fit_mapping, audit_mapping)
        for name, fit_records, audit_records in (
            ("c1p", c1_fit_records, c1_audit_records),
            ("c4p", c4_fit_records, c4_audit_records),
        ):
            result = run_factor_matrix(
                compose_factor_rows(fit_records, fit_records, fit_mapping),
                compose_factor_rows(audit_records, fit_records, audit_mapping),
                eval_groups=eval_groups,
                head_seeds=head_seeds,
                steps=int(args.factor_steps),
                bootstrap_resamples=int(args.bootstrap_resamples),
                device=device,
            )
            factor_results[name][mapping_seed] = result
    for name, filename in (("c1p", "factor_matrix_c1p.json"), ("c4p", "factor_matrix_c4p.json")):
        _json_write(
            output / filename,
            {
                "status": "COMPLETE",
                "mapping_seeds": list(mapping_seeds),
                "candidate_selection_uses_learned_q": False,
                "fallback_count": 0,
                "F7": "UNAVAILABLE_NO_VERIFIED_SYNCHRONIZED_CROSS_RX_EVENT_ID",
                "runs": {str(seed): factor_results[name][seed].payload for seed in mapping_seeds},
                "sample_level_state_persisted": False,
            },
        )

    primary_seed = mapping_seeds[0]
    primary_c1 = factor_results["c1p"][primary_seed]
    primary_c4 = factor_results["c4p"][primary_seed]
    primary_eval_rows = compose_factor_rows(c4_audit_records, c4_fit_records, mapping_cache[primary_seed][1])
    c1_error = torch.stack([primary_c1.squared_errors[seed]["F3"] for seed in head_seeds]).mean(dim=0)
    c4_error = torch.stack([primary_c4.squared_errors[seed]["F3"] for seed in head_seeds]).mean(dim=0)
    common = (
        primary_c1.valid_masks["F3"]
        & primary_c4.valid_masks["F3"]
        & primary_eval_rows["F3"].common_anchor
    )
    if bool(common.any()):
        conditioning = group_paired_bootstrap(
            c1_error[common], c4_error[common], eval_groups[common],
            resamples=int(args.bootstrap_resamples), seed=int(args.seed) + 701,
        )
        conditioning_comparison = {
            "status": "COMPLETE",
            "c4_vs_c1_f3_relative_gain": conditioning["relative_gain"],
            "c4_vs_c1_f3_ci_low": conditioning["ci95_low"],
        }
    else:
        conditioning_comparison = {
            "status": "UNAVAILABLE_EMPTY_COMMON_ANCHOR",
            "c4_vs_c1_f3_relative_gain": -1.0,
            "c4_vs_c1_f3_ci_low": -1.0,
        }
    primary_mapping = mapping_cache[primary_seed][1]
    f3_valid = torch.tensor(primary_mapping["F3"].valid, dtype=torch.bool)
    selected_bank = primary_mapping["F3"].index
    cross_rx_by_tx: dict[int, set[int]] = {}
    cell_counts: dict[tuple[int, int, int], int] = {}
    for index, valid in enumerate(f3_valid.tolist()):
        if not valid:
            continue
        tx = int(c4_audit_packets["tx"][index])
        day = int(c4_audit_packets["day"][index])
        audit_rx = int(c4_audit_packets["receiver"][index])
        eligible_rx = c4_fit_packets["receiver"][
            c4_fit_packets["tx"].eq(tx)
            & c4_fit_packets["day"].eq(day)
            & c4_fit_packets["receiver"].ne(audit_rx)
        ]
        cross_rx_by_tx.setdefault(tx, set()).update(int(value) for value in eligible_rx.tolist())
        cell = (tx, int(c4_audit_packets["receiver"][index]), int(c4_audit_packets["day"][index]))
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    coverage = {
        "f3": float(f3_valid.float().mean().item()),
        "each_tx_two_cross_receiver_relations": all(len(values) >= 2 for values in cross_rx_by_tx.values()) and len(cross_rx_by_tx) == num_classes,
        "major_cell_minimum_pass": bool(cell_counts) and min(cell_counts.values()) >= int(args.major_cell_minimum),
        "major_cell_minimum": int(args.major_cell_minimum),
    }
    mapping_direction_count = sum(
        factor_results["c4p"][seed].payload["summary"]["f3_vs_f0_relative_gain"] > 0.0
        and factor_results["c4p"][seed].payload["summary"]["f3_vs_f5_relative_gain"] > 0.0
        for seed in mapping_seeds
    )

    satellite_signs = []
    satellite_records_main = None
    for offset in (0, 1):
        sat_records, sat_packets = _collect_fold_records(
            base, c4p, audit_loader, conditioned=True, ssdg=ssdg, data_ctx=data_ctx,
            device=device, max_batches=int(args.max_eval_batches), scenario="leo_clear_weak",
            data_args=data_args, sat_seed=int(args.sat_seed) + offset,
        )
        if offset == 0:
            satellite_records_main = sat_records
        sat_mapping = _relation_mappings(sat_packets, c4_fit_packets, primary_seed)
        sat_result = run_factor_matrix(
            compose_factor_rows(c4_fit_records, c4_fit_records, mapping_cache[primary_seed][0]),
            compose_factor_rows(sat_records, c4_fit_records, sat_mapping),
            eval_groups=eval_groups,
            head_seeds=head_seeds,
            steps=int(args.factor_steps),
            bootstrap_resamples=int(args.bootstrap_resamples),
            device=device,
        )
        satellite_signs.append(
            sat_result.payload["summary"]["f3_vs_f0_relative_gain"] > 0.0
            and sat_result.payload["summary"]["f3_vs_f5_relative_gain"] > 0.0
        )
    clean_sign = (
        primary_c4.payload["summary"]["f3_vs_f0_relative_gain"] > 0.0
        and primary_c4.payload["summary"]["f3_vs_f5_relative_gain"] > 0.0
    )
    sensitivity = {
        "head_seed_direction_count": primary_c4.payload["summary"]["head_seed_direction_count"],
        "candidate_seed_direction_count": mapping_direction_count,
        "satellite_seed_conclusion_reversal": any(value != clean_sign for value in satellite_signs),
        "satellite_seed_count": 2,
    }
    stage_a = evaluate_stage_a(
        factor_results["c1p"][primary_seed].payload["summary"],
        primary_c4.payload["summary"],
        conditioning_comparison,
        coverage,
        sensitivity,
    )

    q_probe = conditional_q_probe(
            train_q=c4_fit_packets["q"], eval_q=c4_audit_packets["q"],
            train_labels={name: c4_fit_packets[name] for name in ("tx", "receiver", "day")},
            eval_labels={name: c4_audit_packets[name] for name in ("tx", "receiver", "day")},
            seed=int(args.seed) + 801, steps=int(args.probe_steps), hidden_dim=64, device=device,
        )
    q_probe["codebook_diagnostic_only"] = token_code_audit(c4_audit_packets["code_prob"])
    q_probe["codebook_optimization_or_gate_use"] = False
    _json_write(output / "q_conditional_probe.json", q_probe)
    assert satellite_records_main is not None
    audit_fold_metadata = {
        name: c4_audit_packets[name] for name in ("tx", "receiver", "day", "eq", "sig_i", "base_index")
    }
    _json_write(
        output / "m0_exact_pair_retrieval.json",
        m0_exact_pair_retrieval(
            clean_q=c4_audit_records.q_holdout,
            satellite_q=satellite_records_main.q_holdout,
            clean_theta=c4_audit_records.theta,
            satellite_theta=satellite_records_main.theta,
            base_index=c4_audit_records.base_index,
            fold_id=c4_audit_records.fold_id,
            sample_metadata=audit_fold_metadata,
        ),
    )
    primary_rows = primary_eval_rows
    repeated_tx = c4_audit_packets["tx"].repeat(4)
    repeated_rx = c4_audit_packets["receiver"].repeat(4)
    repeated_day = c4_audit_packets["day"].repeat(4)
    loto_valid = primary_rows["F3"].valid & primary_rows["F3"].common_anchor
    loto_tx = repeated_tx[loto_valid]
    if bool(loto_valid.any()) and torch.unique(loto_tx).numel() >= 2:
        loto_payload = run_loto_residual(
                common_inputs=primary_rows["F0"].inputs[loto_valid],
                operator_inputs=primary_rows["F3"].inputs[loto_valid],
                target=primary_rows["F3"].target[loto_valid],
                tx=loto_tx,
                receiver=repeated_rx[loto_valid],
                day=repeated_day[loto_valid],
                fold_id=primary_rows["F3"].fold_id[loto_valid],
                seed=int(args.seed) + 901,
                steps=int(args.loto_steps),
                hidden_dim=64,
                device=device,
            )
    else:
        loto_payload = {
            "status": "UNAVAILABLE_INSUFFICIENT_COMMON_ANCHOR",
            "valid_count": int(loto_valid.sum()),
            "technical_failure": False,
            "sample_level_state_persisted": False,
        }
    _json_write(output / "loto_residual_audit.json", loto_payload)

    stage_b_payload: dict[str, Any]
    if not stage_a.stage_b_allowed:
        gate_calibration = build_not_run_gate_payload(stage_a.status)
        stage_b_payload = build_not_run_gate_payload(stage_a.status)
        stage_b = evaluate_stage_b({}, stage_a_status=stage_a.status)
    else:
        cal_scenes = [
            _collect_gate_scene(
                base, c4p, data_ctx["source_calibration_loader"], scenario=scenario,
                ssdg=ssdg, data_ctx=data_ctx, data_args=data_args, device=device,
                max_batches=int(args.max_eval_batches), sat_seed=int(args.sat_seed) + 1100 + index,
            )
            for index, scenario in enumerate(SCENARIOS)
        ]
        cal_features = _concat_feature_maps([_gate_features(scene) for scene in cal_scenes])
        candidate_outcomes: dict[tuple[float, float], dict[str, Tensor]] = {}
        for eta in (0.05, 0.10, 0.20):
            for clip_norm in (0.5, 1.0):
                rows = [_fusion_outcomes(scene, eta, clip_norm) for scene in cal_scenes]
                candidate_outcomes[(eta, clip_norm)] = {
                    "rescue": torch.cat([row["rescue"] for row in rows]),
                    "harm": torch.cat([row["harm"] for row in rows]),
                }
        groups = []
        receivers = []
        for scene in cal_scenes:
            for index in range(scene["truth"].numel()):
                groups.append((
                    int(scene["truth"][index]), int(scene["receiver"][index]),
                    int(scene["day"][index]), int(scene["eq"][index]),
                    int(scene["sig_i"][index]) // int(retro.block_size),
                ))
                receivers.append(int(scene["receiver"][index]))
        fitted_gate = fit_truth_blind_gate(
            cal_features,
            outcomes=candidate_outcomes,
            groups=groups,
            receivers=torch.tensor(receivers),
            folds=5,
            steps=int(args.gate_steps),
            seed=int(args.seed) + 1001,
        )
        gate_calibration = _gate_fit_payload(fitted_gate)
        audit_scenes = [
            _collect_gate_scene(
                base, c4p, audit_loader, scenario=scenario,
                ssdg=ssdg, data_ctx=data_ctx, data_args=data_args, device=device,
                max_batches=int(args.max_eval_batches), sat_seed=int(args.sat_seed) + 2100 + index,
            )
            for index, scenario in enumerate(SCENARIOS)
        ]
        scene_results = {}
        leo_base_correct = []
        leo_final_correct = []
        leo_groups = []
        leo_gain = []
        receiver_gains = []
        selected_rescue = selected_harm = selected_count = total_count = 0
        for index, (scenario, scene) in enumerate(zip(SCENARIOS, audit_scenes)):
            gate = predict_truth_blind_gate(fitted_gate, _gate_features(scene))
            final = bounded_residual_fusion(
                scene["base_logits"], scene["operator_logits"], gate=gate,
                eta=fitted_gate.eta, scale=1.0, clip_norm=fitted_gate.clip_norm,
            )
            truth = scene["truth"].long()
            base_correct = scene["base_logits"].argmax(dim=1).eq(truth)
            final_correct = final.argmax(dim=1).eq(truth)
            rescue = ~base_correct & final_correct
            harm = base_correct & ~final_correct
            gain = _accuracy_gain_pp(scene["base_logits"], final, truth)
            scene_results[scenario] = {
                "gain_pp": gain,
                "base_accuracy": float(base_correct.float().mean()),
                "final_accuracy": float(final_correct.float().mean()),
                "gate_coverage": float(gate.mean()),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
            }
            if scenario != "clean":
                selected_rescue += int((gate.bool() & rescue).sum())
                selected_harm += int((gate.bool() & harm).sum())
                selected_count += int(gate.sum())
                total_count += int(gate.numel())
                leo_gain.append(gain)
                leo_base_correct.append(base_correct)
                leo_final_correct.append(final_correct)
                group = torch.stack((
                    truth, scene["receiver"], scene["day"],
                    torch.full_like(truth, index),
                    scene["sig_i"].div(int(retro.block_size), rounding_mode="floor"),
                ), dim=1)
                leo_groups.append(group)
                for rx in torch.unique(scene["receiver"]):
                    mask = scene["receiver"].eq(rx)
                    receiver_gains.append(100.0 * float((final_correct[mask].float().mean() - base_correct[mask].float().mean()).item()))
        bootstrap = _accuracy_group_bootstrap(
            torch.cat(leo_base_correct), torch.cat(leo_final_correct), torch.cat(leo_groups),
            resamples=int(args.bootstrap_resamples), seed=int(args.seed) + 1201,
        )
        b_metrics = {
            "leo_mean_gain_pp": float(sum(leo_gain) / len(leo_gain)),
            "leo_gain_ci_low_pp": bootstrap["ci95_low_pp"],
            "clean_gain_pp": scene_results["clean"]["gain_pp"],
            "worst_receiver_gain_pp": min(receiver_gains),
            "selected_weighted_utility": selected_rescue - fitted_gate.lambda_h * selected_harm,
            "gate_coverage": selected_count / max(1, total_count),
            "gate_coverage_min": float(args.gate_coverage_min),
            "positive_receiver_cv_count": fitted_gate.positive_receiver_cv_count,
            "receiver_cv_count": fitted_gate.receiver_cv_count,
        }
        stage_b = evaluate_stage_b(b_metrics, stage_a_status=stage_a.status)
        stage_b_payload = {
            "status": stage_b.status,
            "metrics": b_metrics,
            "bootstrap": bootstrap,
            "scenarios": scene_results,
            "verdict": asdict(stage_b),
            "audit_labels_used_for_fit_or_threshold_selection": False,
            "target_or_query_access": False,
            "sample_level_state_persisted": False,
        }
    _json_write(output / "gate_calibration_summary.json", gate_calibration)
    _json_write(output / "gate_audit_summary.json", stage_b_payload)
    next_route = stage_b.next_route if stage_a.stage_b_allowed else stage_a.next_route
    decision = {
        "status": "ANALYZED",
        "schema": SCHEMA,
        "stage_a_verdict": asdict(stage_a),
        "stage_b_verdict": asdict(stage_b),
        "conditioning_comparison": conditioning_comparison,
        "coverage": coverage,
        "sensitivity": sensitivity,
        "next_route": next_route,
        "artifact_count": len(AGGREGATE_ARTIFACTS),
        "aggregate_artifacts": list(AGGREGATE_ARTIFACTS),
        "technical_model_artifacts": ["models/c1p_sidecar_v3.pth", "models/c4p_sidecar_v3.pth"],
        "target_or_query_access": False,
        "sample_level_state_persisted": False,
    }
    _json_write(output / "decision_manifest.json", decision)
    _write_final_report(output, decision)
    missing = [name for name in AGGREGATE_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"aggregate artifact closure failed: {missing}")
    print(f"[PA-M21] ANALYZED output={output} next={next_route}", flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    if bool(args.target_or_query_access):
        raise ValueError("target/query access is forbidden for this source-only audit")
    output = validate_output_root(args)
    if bool(args.synthetic_smoke):
        return _synthetic_smoke(output)
    return _real_run(args, output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
