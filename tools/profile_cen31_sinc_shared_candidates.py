from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
TOOLS = ROOT / "tools"
for path in (ROOT, CODE, TOOLS):
    path_str = str(path)
    while path_str in sys.path:
        sys.path.remove(path_str)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(CODE))

import torch

from profile_cen_a31_architectures import parse_csv_ints, profile_architecture


CANDIDATES: list[dict[str, Any]] = [
    {"idx": 1, "name": "sinc_shared_baseline_api"},
    {"idx": 2, "name": "freq_from_sinc_energy", "freq_feature_source": "sinc_energy"},
    {"idx": 3, "name": "freq_from_sinc_phase_asym", "freq_feature_source": "sinc_phase_asym"},
    {"idx": 4, "name": "pa_from_sinc_lowrank_135", "pa_feature_source": "sinc_lowrank", "pa_orders": "1,3,5"},
    {"idx": 5, "name": "pa_from_sinc_lowrank_15", "pa_feature_source": "sinc_lowrank", "pa_orders": "1,5"},
    {"idx": 6, "name": "time_pa_no_overlap_3rd", "pa_orders": "1,5"},
    {
        "idx": 7,
        "name": "fftless_freq_pa_joint",
        "freq_feature_source": "sinc_phase_asym",
        "pa_feature_source": "sinc_lowrank",
        "pa_orders": "1,5",
    },
    {"idx": 8, "name": "sinc_shared_channel_trim", "channel_trim_scale": 0.75},
    {"idx": 9, "name": "no_rho_circularity", "use_circularity": False},
    {"idx": 10, "name": "no_freq_stats_proj", "use_freq_stats": False},
    {"idx": 11, "name": "no_pa_stats_proj", "use_pa_stats": False},
    {
        "idx": 12,
        "name": "no_spectral_aux_stats_all",
        "use_circularity": False,
        "use_freq_stats": False,
        "use_pa_stats": False,
        "use_aux_spectral_stats": False,
    },
    {"idx": 13, "name": "no_dsq_freq_stability", "domain_freq_stability_mode": "off"},
    {"idx": 14, "name": "no_freq_band_gate", "use_freq_band_gate": False},
    {"idx": 15, "name": "domain_enhancer_off", "domain_enhancer": "off"},
    {"idx": 16, "name": "rcn_minimal_6stats", "domain_enhancer": "rcn_minimal_6stats"},
]


def render_candidate_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "candidate",
        "B",
        "train params",
        "deploy params",
        "deploy FLOPs",
        "deploy fwd calls",
        "deploy ms",
        "full-train ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row["batch_size"]),
                    f"{row['params_train']:,}",
                    f"{row['params_deploy']:,}",
                    f"{row['flops2_deploy']:,}",
                    f"{row['module_forwards_deploy']:,}",
                    f"{row['latency_ms_deploy']:.3f}",
                    f"{row['latency_ms_train_full']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the 16 CEN31 Sinc-shared/stat-pruning candidates.")
    parser.add_argument("--batch_sizes", default="1,256")
    parser.add_argument("--input_len", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--num_domains", type=int, default=21)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json_out", default="")
    parser.add_argument("--markdown_out", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        options = dict(candidate)
        name = str(options.pop("name"))
        idx = int(options.pop("idx"))
        for batch_size in parse_csv_ints(args.batch_sizes):
            row = profile_architecture(
                "cvsincnet",
                batch_size=batch_size,
                input_len=int(args.input_len),
                num_classes=int(args.num_classes),
                num_domains=int(args.num_domains),
                device=device,
                warmup=int(args.warmup),
                iters=int(args.iters),
                model_variant="lite_d",
                branch_ablation="no_dac",
                domain_branch_ablation="no_stats",
                **options,
            )
            row["candidate_idx"] = idx
            row["candidate"] = name
            rows.append(row)

    payload = {
        "device": str(device),
        "candidate_count": len(CANDIDATES),
        "batch_sizes": parse_csv_ints(args.batch_sizes),
        "note": "FLOPs are hooked Conv1d/Linear/SincConv estimates; module_forwards_deploy is a small-kernel proxy, not CUDA kernel count.",
        "rows": rows,
    }
    md = render_candidate_markdown(rows)
    print(md)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
