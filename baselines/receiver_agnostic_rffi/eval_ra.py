from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from baselines.common.datasets import SyntheticIQDataset
from baselines.common.io import default_arg_parser, ensure_dir, get_nested, load_config, save_json, set_seed
from baselines.receiver_agnostic_rffi.model import ReceiverAgnosticRFFI
from baselines.tifs2025_channel_receiver_rffi.data import SpectrogramTransform


def main() -> None:
    parser = default_arg_parser("Receiver-agnostic RFFI evaluation")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "outputs/receiver_agnostic_rffi_eval"))
    num_tx = int(get_nested(cfg, "data.num_tx", 4))
    num_rx = int(get_nested(cfg, "data.num_rx", 3))
    spec = SpectrogramTransform(n_fft=64, hop_length=32, win_length=64)
    ds = SyntheticIQDataset(num_tx=num_tx, num_rx=num_rx, samples_per_pair=4, length=256, seed=seed, transform=spec)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: (torch.stack([z["iq"] for z in b]), torch.tensor([z["label"] for z in b])))
    model = ReceiverAgnosticRFFI(num_tx=num_tx, num_rx=num_rx, feature_dim=int(get_nested(cfg, "model.feature_dim", 128))).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device), return_rx=False)["tx_logits"].argmax(dim=1).cpu()
            total += int(y.numel())
            correct += int((pred == y).sum().item())
    save_json({"overall_accuracy": correct / max(1, total)}, os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
