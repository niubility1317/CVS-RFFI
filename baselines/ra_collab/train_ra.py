from __future__ import annotations

import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from baselines.common.io import default_arg_parser, ensure_dir, get_nested, load_config, save_json, set_seed
from baselines.common.datasets import SyntheticIQDataset
from baselines.common.augmentation import OnlineRFChannelAugment
from baselines.ra_collab.losses import ra_collab_adversarial_loss
from baselines.ra_collab.model import RACollabRFFI
from baselines.ra_collab.spectrogram import SpectrogramTransform


def _collate_spec(batch):
    return {
        "spec": torch.stack([b["iq"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "receiver": torch.tensor([b["receiver"] for b in batch], dtype=torch.long),
    }


def main() -> None:
    parser = default_arg_parser("RA-Collab RFFI training")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "outputs/ra_collab"))
    num_tx = int(get_nested(cfg, "data.num_tx", 4))
    num_rx = int(get_nested(cfg, "data.num_rx", 3))
    spec = SpectrogramTransform(n_fft=64, hop_length=32, win_length=64)
    aug = OnlineRFChannelAugment(float(get_nested(cfg, "data.sample_rate", 1_000_000)), max_taps=4)
    ds = SyntheticIQDataset(num_tx=num_tx, num_rx=num_rx, samples_per_pair=4, length=256, seed=seed, transform=lambda x: spec(aug(x)))
    loader = DataLoader(ds, batch_size=int(get_nested(cfg, "train.batch_size", 16)), shuffle=True, collate_fn=_collate_spec)
    model = RACollabRFFI(num_tx=num_tx, num_rx=num_rx, feature_dim=int(get_nested(cfg, "model.feature_dim", 128))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(get_nested(cfg, "train.lr", 1e-4)))
    epochs = args.epochs if args.epochs is not None else int(get_nested(cfg, "train.epochs", 1))
    last = {}
    for _ in range(epochs):
        for batch in loader:
            out = model(batch["spec"].to(device), grl_lambda=float(get_nested(cfg, "loss.grl_lambda", 1.0)))
            losses = ra_collab_adversarial_loss(out, batch["label"].to(device), batch["receiver"].to(device), rx_weight=float(get_nested(cfg, "loss.rx_weight", 1.0)))
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            last = {k: float(v.detach().cpu()) for k, v in losses.items()}
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(output_dir, "best_model.pt"))
    save_json(last, os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()

