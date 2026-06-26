from __future__ import annotations

import argparse

import torch

from post_stage_common import build_standard_data, load_baseline_from_checkpoint, move_batch, resolve_device, set_seed

from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an SGC v3 checkpoint on a small loader slice.")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--sgc_v3_ckpt", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=10)
    from post_stage_cli import add_common_data_args

    add_common_data_args(parser)
    return parser


@torch.no_grad()
def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    data_ctx = build_standard_data(args, device)
    teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=True)
    ckpt = torch.load(args.sgc_v3_ckpt, map_location=device)
    raw_cfg = ckpt.get("config")
    if isinstance(raw_cfg, SGCv3Config):
        cfg = raw_cfg
    elif isinstance(raw_cfg, dict):
        cfg = SGCv3Config.from_mapping(raw_cfg)
    else:
        raise ValueError("Checkpoint must store an SGCv3Config or config mapping.")
    model = SGCv3Model(teacher, cfg).to(device)
    model.load_state_dict(ckpt["sgc_v3"], strict=False)
    model.eval()
    correct = total = 0
    for idx, batch in enumerate(data_ctx["val_loader"]):
        if int(args.max_batches) > 0 and idx >= int(args.max_batches):
            break
        x, y, _ = move_batch(batch, device)
        out = model(x)
        correct += int(out["logits_final"].argmax(dim=-1).eq(y).sum())
        total += int(y.numel())
    print(f"[SGC-V3-EVAL] val_tx={100.0 * correct / max(1, total):.2f}% ({correct}/{total})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
