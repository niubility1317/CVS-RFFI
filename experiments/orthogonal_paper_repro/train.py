"""Train/freeze source and all incremental states, without query or truth input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.orthogonal_paper_repro.base import train_base
from experiments.orthogonal_paper_repro.increment import register_session
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.orthogonal_incremental_sei.model import SixBlockConv1DEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True, help='Training-only NPZ; must contain no query data')
    parser.add_argument('--output', required=True, help='New immutable output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026092701)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {'num_base_classes': 60, 'sessions': 7, 'new_classes_per_session': 10,
              'shots_per_class': 5, 'num_targets': 200, 'feature_dim': 256,
              'base_epochs_max': 100, 'increment_epochs_max': 50, 'batch_size': 128,
              'base_lr': .01, 'increment_lr': .08, 'patience': 4, 'seed': args.seed,
              'validation_fraction_of_base_training': .1,
              'unspecified_by_paper': {'encoder_channels': [32, 64, 96, 128, 160, 192],
                  'kernel_size': 5, 'feature_dim': 256, 'noise_range': .01,
                  'contrast_temperature': .1, 'center_temperature': .1,
                  'top_k': 60, 'margin': .2, 'tau_fuse': .01,
                  'early_stop_monitor': 'base validation loss; increment support loss'},
              'paper_split_exact': False, 'training_data': str(Path(args.training).resolve())}
    (output / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    with np.load(args.training, allow_pickle=False) as data:
        train_x = torch.from_numpy(data['base_train_x']).transpose(1, 2).contiguous()
        train_y = torch.from_numpy(data['base_train_y']).long()
        val_x = torch.from_numpy(data['base_validation_x']).transpose(1, 2).contiguous()
        val_y = torch.from_numpy(data['base_validation_y']).long()
        encoder = SixBlockConv1DEncoder(embedding_dim=256).to(device)
        torch.save({'encoder': encoder.state_dict(), 'config': config}, output / 'initial.pt')
        initial = torch.load(output / 'initial.pt', map_location=device, weights_only=False)
        encoder.load_state_dict(initial['encoder'], strict=True)
        encoder.eval()
        with torch.no_grad():
            features = encoder(train_x[:2].to(device))
        if features.shape != (2, 256) or not bool(torch.isfinite(features).all()):
            raise RuntimeError('real training-record checkpoint smoke failed')
        (output / 'source_smoke.json').write_text(json.dumps({'status': 'PASS', 'training_records': 2,
            'query_records': 0, 'checkpoint': 'initial.pt'}), encoding='utf8')
        with (output / 'epochs.jsonl').open('x', encoding='utf8') as log:
            def on_epoch(row):
                log.write(json.dumps(row) + '\n')
                log.flush()
            base, history = train_base(encoder, train_x, train_y, val_x, val_y,
                seed=args.seed, callback=on_epoch)
        weights = base['base_weights']
        torch.save({'encoder': encoder.state_dict(), 'weights': weights, 'stage': 0,
                    'best_epoch': base['best_epoch'], 'config': config}, output / 'stage_0.pt')
        for session in range(1, 8):
            x = torch.from_numpy(data[f'session_{session}_x']).transpose(1, 2).contiguous().to(device)
            y = torch.from_numpy(data[f'session_{session}_y']).long().to(device)
            if x.size(0) != 50 or torch.unique(y).numel() != 10:
                raise ValueError('each session must have 10 classes and 50 support records')
            weights, losses = register_session(encoder, weights, x, y)
            torch.save({'encoder': encoder.state_dict(), 'weights': weights, 'stage': session,
                        'support_losses': losses, 'config': config}, output / f'stage_{session}.pt')
    (output / 'training_complete.json').write_text(json.dumps({'status': 'TRAINING_COMPLETE',
        'stages': 8, 'prediction_complete': False, 'scoring_complete': False}), encoding='utf8')


if __name__ == '__main__':
    main()
