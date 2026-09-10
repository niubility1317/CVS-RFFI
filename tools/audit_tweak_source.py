"""Bounded source-only Tweak runtime diagnostic; never reads the held-out split."""
import argparse
import json
from pathlib import Path
import torch
from paper_reproduction.gaskin_tweak_2023.model import TweakEncoder
from paper_reproduction.gaskin_tweak_2023.official_lora import load_configuration_records, split_record_frames
from paper_reproduction.gaskin_tweak_2023.official_lora_experiment import _source_training_batches, _source_monitor
from paper_reproduction.gaskin_tweak_2023.triplet import all_strict_hard_triplet_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.001)
    args = parser.parse_args()
    torch.manual_seed(20260908)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = load_configuration_records(args.data_root, 'Config2', device_ids=range(1, 11))
    split = split_record_frames(total_samples=records[0].total_samples)
    model = TweakEncoder().to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict'])
        print(json.dumps({'event': 'frozen_source_monitor', **_source_monitor(model, records, split, device=device)}), flush=True)
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
    for step, (iq, labels) in enumerate(_source_training_batches(records, split, seed=20260909, max_batches=args.batches), 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(iq.to(device))
        loss, active = all_strict_hard_triplet_loss(embeddings, labels.to(device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'nonfinite loss at {step}')
        if active:
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f'nonfinite gradient at {step}')
            optimizer.step()
        if step in (1, 100, 1000, 3000, args.batches):
            print(json.dumps({'step': step, 'loss': float(loss.detach()), 'shape': list(embeddings.shape),
                              'gradient_tensors': sum(p.grad is not None for p in model.parameters()),
                              **_source_monitor(model, records, split, device=device)}), flush=True)


if __name__ == '__main__':
    main()
