#!/usr/bin/env python
"""Real ADV3B02 checkpoint smoke for the source-only SID-FFT96 path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from SSDG import train_ssdg as ssdg  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--sid_mask_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=392002)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser


def _data_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = ssdg.build_arg_parser().parse_args(["--output_dir", str(Path(cli.output_json).parent / "unused")])
    args.wisig_pkl = str(cli.wisig_pkl)
    args.device = str(cli.device)
    args.seed = int(cli.seed)
    args.eval_batch_size = int(cli.batch_size)
    args.batch_size = int(cli.batch_size)
    args.num_workers = 0
    args.prefetch_factor = 2
    args.split_mode = "tx_rx_day_1_7_2"
    args.phase1_source_role_protocol = "l_s_u_s_v_cal_v_select"
    args.labeled_ratio = 0.07
    args.unlabeled_ratio = 0.63
    args.source_cal_ratio = 0.15
    args.source_select_ratio = 0.15
    args.source_val_ratio = 0.30
    return args


def main(argv: list[str] | None = None) -> int:
    cli = _parser().parse_args(argv)
    ssdg.set_seed(int(cli.seed))
    device = torch.device(str(cli.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    data_args = _data_args(cli)
    data_context = ssdg._build_ssdg_wisig_data(data_args, device)
    checkpoint = ssdg.load_checkpoint(str(cli.checkpoint), device)
    model_args = ssdg.merge_checkpoint_args(
        checkpoint,
        argparse.Namespace(),
        input_len=int(data_context["input_len"]),
        num_domains=int(data_context["num_domains"]),
    )
    model_args.sid_fft96_mode = "sid"
    model_args.sid_mask_path = str(cli.sid_mask_path)
    model_args.sid_residual_scale = 1.0
    model_args.use_crra = False
    model_args.use_ntrs = False
    model = ssdg.build_baseline_model(model_args, device)
    sid_args = argparse.Namespace(
        sid_fft96_mode="sid",
        sid_adapter_only=True,
        ntrs_variant="v1",
    )
    load_report = ssdg._load_training_checkpoint_state(model, checkpoint, sid_args)
    trainable_report = ssdg.configure_sid_trainable_parameters(model, sid_args)

    batch = next(iter(data_context["probe_train_loader"]))
    x, y, extra = ssdg.move_batch(batch, device)
    split_sources = []
    if len(extra) >= 2 and isinstance(extra[1], dict):
        value = extra[1].get("split_source", [])
        split_sources = list(value) if isinstance(value, (list, tuple)) else [value]
    if not split_sources or any(str(value) != "ssdg_labeled_tx_visible" for value in split_sources):
        raise ValueError(f"smoke batch is not exclusively L_s: {split_sources}")

    model.train()
    output = model(x, y_tx=y, return_aux=True)
    for key in ("tx_logits", "z_id", "logits_raw", "logits_sid", "z_id_raw", "z_id_sid", "sid_fft96"):
        value = output.get(key)
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"non-finite or missing smoke output: {key}")
    raw_sid_logit_max_abs = float((output["logits_raw"] - output["logits_sid"]).abs().max().item())
    raw_sid_z_max_abs = float((output["z_id_raw"] - output["z_id_sid"]).abs().max().item())
    if raw_sid_logit_max_abs > 1e-7 or raw_sid_z_max_abs > 1e-7:
        raise ValueError("zero-initialized SID path is not identical to raw ADV3B02")

    F.cross_entropy(output["tx_logits"].float(), y).backward()
    gradient_names = sorted(name for name, parameter in model.named_parameters() if parameter.grad is not None)
    if not gradient_names or any(not name.startswith("sid_fft96.") for name in gradient_names):
        raise ValueError(f"gradient escaped the SID whitelist: {gradient_names}")

    summary = {
        "status": "VERIFIED",
        "checkpoint": str(cli.checkpoint),
        "sid_mask_path": str(cli.sid_mask_path),
        "batch_role": "L_s",
        "batch_size": int(y.numel()),
        "source_sample_count": int(y.numel()),
        "query_input_count": 0,
        "target_input_count": 0,
        "missing_keys": list(load_report["missing_keys"]),
        "unexpected_keys": list(load_report["unexpected_keys"]),
        "raw_sid_logit_max_abs": raw_sid_logit_max_abs,
        "raw_sid_z_max_abs": raw_sid_z_max_abs,
        "all_outputs_finite": True,
        "gradient_parameter_names": gradient_names,
        **trainable_report,
    }
    output_path = Path(cli.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
