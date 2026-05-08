from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from baselines.common.augmentation import OnlineRFChannelAugment
from baselines.common.datasets import SyntheticIQDataset
from baselines.common.io import default_arg_parser, ensure_dir, get_nested, load_config, save_json, set_seed
from baselines.tifs2025_channel_receiver_rffi.data import SiamesePairDataset, SpectrogramTransform
from baselines.tifs2025_channel_receiver_rffi.losses import siamese_contrastive_ce_loss
from baselines.tifs2025_channel_receiver_rffi.models import ResNetRFF, SiameseRFF


def main() -> None:
    parser = default_arg_parser("TIFS2025 Siamese fine-tuning")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "outputs/tifs2025/finetune_siamese"))
    num_classes = int(get_nested(cfg, "model.num_classes", 4))
    feature_dim = int(get_nested(cfg, "model.feature_dim", 128))
    spec = SpectrogramTransform(
        n_fft=int(get_nested(cfg, "spectrogram.n_fft", 64)),
        hop_length=int(get_nested(cfg, "spectrogram.hop_length", 32)),
        win_length=int(get_nested(cfg, "spectrogram.win_length", 64)),
    )
    augment = OnlineRFChannelAugment(float(get_nested(cfg, "data.sample_rate", 1_000_000)), max_taps=4)
    base = SyntheticIQDataset(num_tx=num_classes, num_rx=3, samples_per_pair=4, length=256, seed=seed)
    ds = SiamesePairDataset(base, augment=augment, spec_transform=spec, seed=seed)
    loader = DataLoader(ds, batch_size=int(get_nested(cfg, "train.batch_size", 16)), shuffle=True)
    backbone = ResNetRFF(num_classes=num_classes, feature_dim=feature_dim).to(device)
    pretrained = cfg.get("pretrained_ckpt")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        backbone.load_state_dict(state.get("model", state), strict=False)
    model = SiameseRFF(backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(get_nested(cfg, "train.lr", 3e-4)))
    epochs = args.epochs if args.epochs is not None else int(get_nested(cfg, "train.epochs", 1))
    last_loss = 0.0
    for _ in range(epochs):
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            losses = siamese_contrastive_ce_loss(*model(x1, x2), y, temperature=float(get_nested(cfg, "loss.temperature", 0.05)))
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            last_loss = float(losses["loss"].detach().cpu())
    torch.save({"model": backbone.state_dict(), "config": cfg}, os.path.join(output_dir, "best.pt"))
    save_json({"last_loss": last_loss, "seed": seed}, os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
