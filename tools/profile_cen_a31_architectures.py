from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
for path in (ROOT, CODE):
    path_str = str(path)
    while path_str in sys.path:
        sys.path.remove(path_str)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))

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


class MacCounter:
    def __init__(self) -> None:
        self.macs = 0
        self.handles = []

    def add_hooks(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                self.handles.append(module.register_forward_hook(self._conv1d_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif module.__class__.__name__ == "SincConv1d":
                self.handles.append(module.register_forward_hook(self._sinc_hook))

    def clear(self) -> None:
        self.macs = 0

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv1d_hook(self, module: nn.Conv1d, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        del inputs
        if not torch.is_tensor(output):
            return
        batch = int(output.shape[0])
        out_ch = int(output.shape[1])
        out_len = int(output.shape[2])
        kernel = int(module.kernel_size[0])
        in_per_group = int(module.in_channels // module.groups)
        self.macs += batch * out_ch * out_len * in_per_group * kernel
        if module.bias is not None:
            self.macs += batch * out_ch * out_len

    def _linear_hook(self, module: nn.Linear, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        del inputs
        if not torch.is_tensor(output):
            return
        out_elements = int(output.numel())
        self.macs += out_elements * int(module.in_features)
        if module.bias is not None:
            self.macs += out_elements

    def _sinc_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        del inputs
        if not torch.is_tensor(output):
            return
        batch = int(output.shape[0])
        out_ch = int(output.shape[1])
        out_len = int(output.shape[2])
        kernel = int(getattr(module, "kernel_size", 1))
        self.macs += batch * out_ch * out_len * kernel


class ForwardCallCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.handles = []

    def add_hooks(self, model: nn.Module) -> None:
        for module in model.modules():
            if module is model:
                continue
            if len(list(module.children())) == 0:
                self.handles.append(module.register_forward_hook(self._hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs, output
        self.calls += 1


def parse_pa_orders(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None
        values = [int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    else:
        values = [int(x) for x in raw]
    orders = tuple(values)
    if any(order < 1 or order % 2 == 0 for order in orders):
        raise ValueError(f"pa_orders must contain positive odd integers, got {raw!r}")
    return orders


def make_model(
    arch_family: str,
    input_len: int,
    num_classes: int,
    num_domains: int,
    *,
    model_variant: str = "lite_d",
    branch_ablation: str = "no_dac",
    domain_branch_ablation: str = "no_stats",
    domain_enhancer: str = "rcn_stats",
    domain_enhancer_strength: float = 0.35,
    domain_freq_stability_mode: str = "dsq",
    use_circularity: bool = True,
    use_freq_stats: bool = True,
    use_pa_stats: bool = True,
    use_freq_band_gate: bool = True,
    freq_feature_source: str = "raw_fft",
    pa_feature_source: str = "raw_iq",
    pa_orders: Any = None,
    use_aux_spectral_stats: bool = True,
    channel_trim_scale: float = 1.0,
) -> nn.Module:
    return build_dual_model(
        num_classes=int(num_classes),
        num_domains=int(num_domains),
        model_size="M",
        dataset="wisig",
        input_len=int(input_len),
        sample_rate_hz=25e6,
        model_variant=str(model_variant),
        branch_ablation=str(branch_ablation),
        domain_branch_ablation=str(domain_branch_ablation),
        domain_enhancer=str(domain_enhancer),
        domain_enhancer_strength=float(domain_enhancer_strength),
        use_circularity=bool(use_circularity),
        use_freq_stats=bool(use_freq_stats),
        use_pa_stats=bool(use_pa_stats),
        use_freq_band_gate=bool(use_freq_band_gate),
        freq_feature_source=str(freq_feature_source),
        pa_feature_source=str(pa_feature_source),
        pa_orders=parse_pa_orders(pa_orders),
        use_aux_spectral_stats=bool(use_aux_spectral_stats),
        channel_trim_scale=float(channel_trim_scale),
        domain_freq_stability_mode=str(domain_freq_stability_mode),
        freq_stability_channels=2,
        mixstyle_on=True,
        mixstyle_layers="time_down,t1",
        mixstyle_mix="same_tx_crossdomain",
        mixstyle_fallback="skip",
        mixstyle_strength=0.70,
        mixstyle_p=0.18,
        fast_infer_when_no_aux=True,
        arch_family=arch_family,
    )


def measure_macs(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    *,
    return_aux: bool,
) -> int:
    counter = MacCounter()
    counter.add_hooks(model)
    counter.clear()
    with torch.no_grad():
        model(x, y_tx=y, domain_labels=d, return_aux=return_aux)
    macs = int(counter.macs)
    counter.close()
    return macs


def measure_module_forwards(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    *,
    return_aux: bool,
) -> int:
    counter = ForwardCallCounter()
    counter.add_hooks(model)
    with torch.no_grad():
        model(x, y_tx=y, domain_labels=d, return_aux=return_aux)
    calls = int(counter.calls)
    counter.close()
    return calls


def measure_latency_ms(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    *,
    return_aux: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(x, y_tx=y, domain_labels=d, return_aux=return_aux)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            model(x, y_tx=y, domain_labels=d, return_aux=return_aux)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def profile_architecture(
    arch_family: str,
    batch_size: int,
    input_len: int,
    num_classes: int,
    num_domains: int,
    device: torch.device,
    warmup: int,
    iters: int,
    *,
    model_variant: str = "lite_d",
    branch_ablation: str = "no_dac",
    domain_branch_ablation: str = "no_stats",
    domain_enhancer: str = "rcn_stats",
    domain_enhancer_strength: float = 0.35,
    domain_freq_stability_mode: str = "dsq",
    use_circularity: bool = True,
    use_freq_stats: bool = True,
    use_pa_stats: bool = True,
    use_freq_band_gate: bool = True,
    freq_feature_source: str = "raw_fft",
    pa_feature_source: str = "raw_iq",
    pa_orders: Any = None,
    use_aux_spectral_stats: bool = True,
    channel_trim_scale: float = 1.0,
) -> dict[str, Any]:
    torch.manual_seed(1337)
    model = make_model(
        arch_family,
        input_len,
        num_classes,
        num_domains,
        model_variant=model_variant,
        branch_ablation=branch_ablation,
        domain_branch_ablation=domain_branch_ablation,
        domain_enhancer=domain_enhancer,
        domain_enhancer_strength=domain_enhancer_strength,
        domain_freq_stability_mode=domain_freq_stability_mode,
        use_circularity=use_circularity,
        use_freq_stats=use_freq_stats,
        use_pa_stats=use_pa_stats,
        use_freq_band_gate=use_freq_band_gate,
        freq_feature_source=freq_feature_source,
        pa_feature_source=pa_feature_source,
        pa_orders=pa_orders,
        use_aux_spectral_stats=use_aux_spectral_stats,
        channel_trim_scale=channel_trim_scale,
    ).to(device).eval()
    x = torch.randn(batch_size, 2, input_len, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    d = torch.randint(0, num_domains, (batch_size,), device=device)
    deploy_module = getattr(model, "id_backbone", model)
    full_macs = measure_macs(model, x, y, d, return_aux=True)
    deploy_macs = measure_macs(model, x, y, d, return_aux=False)
    full_forwards = measure_module_forwards(model, x, y, d, return_aux=True)
    deploy_forwards = measure_module_forwards(model, x, y, d, return_aux=False)
    full_ms = measure_latency_ms(model, x, y, d, return_aux=True, warmup=warmup, iters=iters, device=device)
    deploy_ms = measure_latency_ms(model, x, y, d, return_aux=False, warmup=warmup, iters=iters, device=device)
    parsed_orders = parse_pa_orders(pa_orders)
    return {
        "arch_family": arch_family,
        "model_variant": str(model_variant),
        "branch_ablation": str(branch_ablation),
        "domain_branch_ablation": str(domain_branch_ablation),
        "domain_enhancer": str(domain_enhancer),
        "domain_freq_stability_mode": str(domain_freq_stability_mode),
        "use_circularity": int(bool(use_circularity)),
        "use_freq_stats": int(bool(use_freq_stats)),
        "use_pa_stats": int(bool(use_pa_stats)),
        "use_freq_band_gate": int(bool(use_freq_band_gate)),
        "freq_feature_source": str(freq_feature_source),
        "pa_feature_source": str(pa_feature_source),
        "pa_orders": ",".join(str(x) for x in parsed_orders) if parsed_orders is not None else "<variant-default>",
        "use_aux_spectral_stats": int(bool(use_aux_spectral_stats)),
        "channel_trim_scale": float(channel_trim_scale),
        "batch_size": int(batch_size),
        "input_len": int(input_len),
        "params_train": count_params(model),
        "params_deploy": count_params(deploy_module),
        "macs_train_full": int(full_macs),
        "macs_deploy": int(deploy_macs),
        "flops2_train_full": int(full_macs * 2),
        "flops2_deploy": int(deploy_macs * 2),
        "module_forwards_train_full": int(full_forwards),
        "module_forwards_deploy": int(deploy_forwards),
        "latency_ms_train_full": full_ms,
        "latency_ms_deploy": deploy_ms,
        "latency_ms_deploy_per_sample": deploy_ms / float(batch_size),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "arch",
        "variant",
        "branch",
        "freq src",
        "pa src",
        "pa orders",
        "B",
        "train params",
        "deploy params",
        "deploy MACs",
        "deploy FLOPs",
        "deploy fwd calls",
        "deploy ms",
        "ms/sample",
        "full-train ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["arch_family"]),
                    str(row["model_variant"]),
                    str(row["branch_ablation"]),
                    str(row["freq_feature_source"]),
                    str(row["pa_feature_source"]),
                    str(row["pa_orders"]),
                    str(row["batch_size"]),
                    f"{row['params_train']:,}",
                    f"{row['params_deploy']:,}",
                    f"{row['macs_deploy']:,}",
                    f"{row['flops2_deploy']:,}",
                    f"{row['module_forwards_deploy']:,}",
                    f"{row['latency_ms_deploy']:.3f}",
                    f"{row['latency_ms_deploy_per_sample']:.4f}",
                    f"{row['latency_ms_train_full']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).replace(";", ",").split(",") if x.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]


def expand_to_length(values: list[str], n: int, name: str) -> list[str]:
    if len(values) == 1:
        return values * int(n)
    if len(values) != int(n):
        raise ValueError(f"{name} must provide either one value or {n} values; got {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CEN_A31 architecture-comparison model costs.")
    parser.add_argument("--arch_families", default="cvsincnet,resnet18_1d,cvcnn,sinc_cvcnn")
    parser.add_argument("--model_variants", default="lite_d")
    parser.add_argument("--branch_ablations", default="no_dac")
    parser.add_argument("--domain_branch_ablations", default="no_stats")
    parser.add_argument("--domain_enhancers", default="rcn_stats")
    parser.add_argument("--domain_enhancer_strengths", default="0.35")
    parser.add_argument("--domain_freq_stability_mode", default="dsq", choices=["off", "same", "dsq"])
    parser.add_argument("--freq_feature_source", default="raw_fft", choices=["raw_fft", "sinc_energy", "sinc_phase_asym"])
    parser.add_argument("--pa_feature_source", default="raw_iq", choices=["raw_iq", "sinc_lowrank"])
    parser.add_argument("--pa_orders", default="")
    parser.add_argument("--use_circularity", type=int, default=1)
    parser.add_argument("--use_freq_stats", type=int, default=1)
    parser.add_argument("--use_pa_stats", type=int, default=1)
    parser.add_argument("--use_freq_band_gate", type=int, default=1)
    parser.add_argument("--use_aux_spectral_stats", type=int, default=1)
    parser.add_argument("--channel_trim_scale", type=float, default=1.0)
    parser.add_argument("--batch_sizes", default="1,256")
    parser.add_argument("--input_len", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--num_domains", type=int, default=21)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    arch_list = parse_csv_strings(args.arch_families)
    variants = expand_to_length(parse_csv_strings(args.model_variants), len(arch_list), "model_variants")
    branches = expand_to_length(parse_csv_strings(args.branch_ablations), len(arch_list), "branch_ablations")
    domain_branches = expand_to_length(parse_csv_strings(args.domain_branch_ablations), len(arch_list), "domain_branch_ablations")
    domain_enhancers = expand_to_length(parse_csv_strings(args.domain_enhancers), len(arch_list), "domain_enhancers")
    domain_strengths = [float(x) for x in expand_to_length(parse_csv_strings(args.domain_enhancer_strengths), len(arch_list), "domain_enhancer_strengths")]

    rows = []
    for batch_size in parse_csv_ints(args.batch_sizes):
        for idx, arch in enumerate(arch_list):
            rows.append(
                profile_architecture(
                    arch,
                    batch_size=batch_size,
                    input_len=int(args.input_len),
                    num_classes=int(args.num_classes),
                    num_domains=int(args.num_domains),
                    device=device,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    model_variant=variants[idx],
                    branch_ablation=branches[idx],
                    domain_branch_ablation=domain_branches[idx],
                    domain_enhancer=domain_enhancers[idx],
                    domain_enhancer_strength=domain_strengths[idx],
                    domain_freq_stability_mode=str(args.domain_freq_stability_mode),
                    use_circularity=bool(int(args.use_circularity)),
                    use_freq_stats=bool(int(args.use_freq_stats)),
                    use_pa_stats=bool(int(args.use_pa_stats)),
                    use_freq_band_gate=bool(int(args.use_freq_band_gate)),
                    freq_feature_source=str(args.freq_feature_source),
                    pa_feature_source=str(args.pa_feature_source),
                    pa_orders=str(args.pa_orders),
                    use_aux_spectral_stats=bool(int(args.use_aux_spectral_stats)),
                    channel_trim_scale=float(args.channel_trim_scale),
                )
            )

    payload = {
        "device": str(device),
        "note": "MAC/FLOP counts include Conv1d, Linear, and SincConv1d hooks; FFT and elementwise ops are not included.",
        "rows": rows,
    }
    print(render_markdown(rows))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
