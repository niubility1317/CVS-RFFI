from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from baselines.common.cvs_data import FewShotPerClassDataset, add_cvs_data_args, build_cvs_loaders, make_cvs_loader
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.io import set_seed
from baselines.receiver_agnostic_rffi.model import ReceiverAgnosticRFFI
from baselines.tifs2025_channel_receiver_rffi.data import SpectrogramTransform


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot target-receiver fine-tuning for receiver-agnostic RFFI")
    add_cvs_data_args(parser)
    parser.set_defaults(batch_size=32, eval_batch_size=256)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--shots_per_class", type=int, default=20)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_fft", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/receiver_agnostic_rffi_finetune")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    spec = SpectrogramTransform(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.n_fft)
    loaders = build_cvs_loaders(args, device, transform_train=spec, transform_eval=spec)
    target_name = "test_seen_day_unseen_rx"
    if target_name not in loaders.split.named_tests:
        raise ValueError(f"Few-shot fine-tuning requires named split {target_name!r}.")
    fine_ds = FewShotPerClassDataset(loaders.split.named_tests[target_name], args.shots_per_class, seed=args.seed)
    fine_loader = make_cvs_loader(
        fine_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )
    model = ReceiverAgnosticRFFI(
        loaders.split.num_classes,
        loaders.split.num_receivers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    def train_step(model, batch, device, epoch, step):
        out = model(batch["iq"].to(device), grl_lambda=0.0, return_rx=False)
        loss = F.cross_entropy(out["tx_logits"], batch["label"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device), grl_lambda=0.0, return_rx=False)

    run_validation_gated_training(
        model=model,
        train_loader=fine_loader,
        val_loader=loaders.val,
        named_test_loaders=loaders.named_tests,
        device=device,
        epochs=args.epochs,
        optimizer=optimizer,
        train_step_fn=train_step,
        forward_eval_fn=forward_eval,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
