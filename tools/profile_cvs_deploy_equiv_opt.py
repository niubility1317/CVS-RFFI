from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
for path in (ROOT, CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn as nn

from model_dual_cvsincnet import build_dual_model


def count_params(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        ident = id(p)
        if ident in seen:
            continue
        seen.add(ident)
        total += int(p.numel())
    return total


def build_cvs_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = build_dual_model(
        num_classes=int(args.num_classes),
        num_domains=int(args.num_domains),
        model_size="M",
        dataset="wisig",
        input_len=int(args.input_len),
        sample_rate_hz=25e6,
        model_variant=str(args.model_variant),
        branch_ablation=str(args.branch_ablation),
        domain_branch_ablation=str(args.domain_branch_ablation),
        domain_enhancer=str(args.domain_enhancer),
        domain_enhancer_strength=float(args.domain_enhancer_strength),
        domain_freq_stability_mode="dsq",
        freq_stability_channels=2,
        mixstyle_on=True,
        mixstyle_layers="time_down,t1",
        mixstyle_mix="same_tx_crossdomain",
        mixstyle_fallback="skip",
        mixstyle_strength=0.70,
        mixstyle_p=0.18,
        fast_infer_when_no_aux=True,
        arch_family="cvsincnet",
    )
    return model.to(device).eval()


def legacy_pa_lift_forward(lift: nn.Module, x: torch.Tensor) -> torch.Tensor:
    xr = torch.clamp(x[:, 0:1, :], -float(lift.clip), float(lift.clip))
    xi = torch.clamp(x[:, 1:2, :], -float(lift.clip), float(lift.clip))
    outs = []
    for delay in range(int(lift.memory_depth)):
        if delay <= 0:
            ar = xr
            ai = xi
        else:
            ar = torch.cat([xr.new_zeros(xr.size(0), xr.size(1), delay), xr[..., :-delay]], dim=-1)
            ai = torch.cat([xi.new_zeros(xi.size(0), xi.size(1), delay), xi[..., :-delay]], dim=-1)
        mag2 = ar * ar + ai * ai
        mag2_safe = torch.clamp(mag2, min=1e-8)
        for order in lift.orders:
            if int(order) == 1:
                scale = torch.ones_like(mag2)
            else:
                scale = mag2_safe
                for _ in range(1, (int(order) - 1) // 2):
                    scale = scale * mag2_safe
            outs.append(ar * scale)
            outs.append(ai * scale)
    return torch.cat(outs, dim=1)


def patch_legacy_pa_lift(model: nn.Module) -> None:
    backbone = getattr(model, "id_backbone", model)
    lift = getattr(backbone, "pa_lift", None)
    if lift is None:
        return
    lift.forward = lambda x: legacy_pa_lift_forward(lift, x)  # type: ignore[method-assign]


def forward_deploy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x, y_tx=None, domain_labels=None, return_aux=False)


def measure_latency_ms(model: nn.Module, x: torch.Tensor, warmup: int, iters: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            forward_deploy(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            forward_deploy(model, x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def profile_batch(args: argparse.Namespace, batch_size: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    current = build_cvs_model(args, device)
    legacy_pa = deepcopy(current).to(device).eval()
    patch_legacy_pa_lift(legacy_pa)

    x = torch.randn(int(batch_size), 2, int(args.input_len), device=device)
    with torch.inference_mode():
        current_logits = forward_deploy(current, x)
        legacy_logits = forward_deploy(legacy_pa, x)
    max_abs_diff = float((current_logits - legacy_logits).abs().max().item())

    current_samples = []
    legacy_samples = []
    for repeat in range(max(1, int(args.repeats))):
        if repeat % 2 == 0:
            current_samples.append(measure_latency_ms(current, x, int(args.warmup), int(args.iters), device))
            legacy_samples.append(measure_latency_ms(legacy_pa, x, int(args.warmup), int(args.iters), device))
        else:
            legacy_samples.append(measure_latency_ms(legacy_pa, x, int(args.warmup), int(args.iters), device))
            current_samples.append(measure_latency_ms(current, x, int(args.warmup), int(args.iters), device))
    current_ms = float(statistics.median(current_samples))
    legacy_ms = float(statistics.median(legacy_samples))
    deploy_module = getattr(current, "id_backbone", current)

    return {
        "batch_size": int(batch_size),
        "input_len": int(args.input_len),
        "params_train": count_params(current),
        "params_deploy": count_params(deploy_module),
        "current_latency_ms": current_ms,
        "current_latency_ms_repeats": current_samples,
        "current_ms_per_sample": current_ms / float(batch_size),
        "legacy_pa_latency_ms": legacy_ms,
        "legacy_pa_latency_ms_repeats": legacy_samples,
        "legacy_pa_ms_per_sample": legacy_ms / float(batch_size),
        "delta_ms_current_minus_legacy_pa": current_ms - legacy_ms,
        "speedup_vs_legacy_pa": (legacy_ms / current_ms) if current_ms > 0 else None,
        "logits_max_abs_diff_current_vs_legacy_pa": max_abs_diff,
    }


def parse_batch_sizes(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).replace(";", ",").split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile equivalent CVS deploy optimizations without retraining.")
    parser.add_argument("--model_variant", default="lite_d")
    parser.add_argument("--branch_ablation", default="no_dac")
    parser.add_argument("--domain_branch_ablation", default="no_stats")
    parser.add_argument("--domain_enhancer", default="rcn_stats")
    parser.add_argument("--domain_enhancer_strength", type=float, default=0.35)
    parser.add_argument("--batch_sizes", default="1,256")
    parser.add_argument("--input_len", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--num_domains", type=int, default=21)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    rows = [profile_batch(args, batch_size, device) for batch_size in parse_batch_sizes(args.batch_sizes)]
    result = {
        "device": str(device),
        "mode": "eval_return_aux_false_inference_mode_no_labels",
        "model_variant": str(args.model_variant),
        "branch_ablation": str(args.branch_ablation),
        "domain_branch_ablation": str(args.domain_branch_ablation),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
