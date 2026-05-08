from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from baselines.common.augmentation import OnlineRFChannelAugment
from baselines.common.datasets import SyntheticIQDataset
from baselines.common.io import default_arg_parser, ensure_dir, get_nested, load_config, save_json, set_seed
from baselines.tifs2025_channel_receiver_rffi.data import PretrainDataset, SpectrogramTransform
from baselines.tifs2025_channel_receiver_rffi.losses import NTXentLoss
from baselines.tifs2025_channel_receiver_rffi.models import ProjectionHead, ResNetRFF


def main() -> None:
    parser = default_arg_parser("TIFS2025 contrastive pretraining")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "outputs/tifs2025/pretrain"))
    num_classes = int(get_nested(cfg, "model.num_classes", 4))

    spec = SpectrogramTransform(
        n_fft=int(get_nested(cfg, "spectrogram.n_fft", 64)),
        hop_length=int(get_nested(cfg, "spectrogram.hop_length", 32)),
        win_length=int(get_nested(cfg, "spectrogram.win_length", 64)),
        normalize=str(get_nested(cfg, "spectrogram.normalize", "zscore")),
    )
    augment = OnlineRFChannelAugment(
        sample_rate=float(get_nested(cfg, "data.sample_rate", 1_000_000)),
        max_taps=int(get_nested(cfg, "augmentation.max_taps", 4)),
        p=1.0 if bool(get_nested(cfg, "augmentation.enabled", True)) else 0.0,
    )
    base = SyntheticIQDataset(
        num_tx=num_classes,
        num_rx=int(get_nested(cfg, "data.num_rx", 3)),
        samples_per_pair=int(get_nested(cfg, "data.synthetic_samples_per_pair", 4)),
        length=int(get_nested(cfg, "data.input_length", 256)),
        seed=seed,
    )
    ds = PretrainDataset(base, augment=augment, spec_transform=spec)
    loader = DataLoader(ds, batch_size=int(get_nested(cfg, "train.batch_size", 16)), shuffle=True)

    model = ResNetRFF(num_classes=num_classes, feature_dim=int(get_nested(cfg, "model.feature_dim", 128))).to(device)
    projector = ProjectionHead(
        in_dim=int(get_nested(cfg, "model.feature_dim", 128)),
        projection_dim=int(get_nested(cfg, "model.projection_dim", 64)),
    ).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=float(get_nested(cfg, "train.lr", 1e-3)))
    criterion = NTXentLoss(temperature=float(get_nested(cfg, "train.temperature", 0.05)))
    epochs = args.epochs if args.epochs is not None else int(get_nested(cfg, "train.epochs", 1))
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        projector.train()
        for v1, v2 in loader:
            x = torch.cat([v1, v2], dim=0).to(device)
            _, z = model(x)
            loss = criterion(projector(z))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
    ckpt = {"model": model.state_dict(), "projector": projector.state_dict(), "config": cfg}
    torch.save(ckpt, os.path.join(output_dir, "last.pt"))
    torch.save(ckpt, os.path.join(output_dir, "best.pt"))
    save_json({"last_loss": last_loss, "seed": seed}, os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
