from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from baselines.common.datasets import SyntheticIQDataset
from baselines.common.io import default_arg_parser, ensure_dir, get_nested, load_config, save_json, set_seed
from baselines.tifs2025_channel_receiver_rffi.data import SpectrogramTransform
from baselines.tifs2025_channel_receiver_rffi.models import ResNetRFF


def main() -> None:
    parser = default_arg_parser("TIFS2025 single-branch evaluation")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "outputs/tifs2025/eval"))
    num_classes = int(get_nested(cfg, "model.num_classes", 4))
    spec = SpectrogramTransform(n_fft=64, hop_length=32, win_length=64)
    ds = SyntheticIQDataset(num_tx=num_classes, num_rx=3, samples_per_pair=4, length=256, seed=seed, transform=spec)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: (torch.stack([z["iq"] for z in b]), torch.tensor([z["label"] for z in b])))
    model = ResNetRFF(num_classes=num_classes, feature_dim=int(get_nested(cfg, "model.feature_dim", 128))).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    total = correct = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(device))
            pred = logits.argmax(dim=1).cpu()
            total += int(y.numel())
            correct += int((pred == y).sum().item())
    elapsed = time.time() - t0
    save_json(
        {"overall_accuracy": correct / max(1, total), "avg_inference_ms_per_sample": 1000 * elapsed / max(1, total)},
        os.path.join(output_dir, "metrics.json"),
    )


if __name__ == "__main__":
    main()
