"""Train LoRa extractor on author source data only, with author-code settings."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from experiments.pamf_phase2_practical.vendor_extra.metric_models import LoRaExtractor


def sample_triplets(groups, batch_size, rng):
    if len(groups) < 2 or batch_size <= 0 or any(len(group) == 0 for group in groups):
        raise ValueError('triplets require at least two nonempty source classes')
    anchor_classes = rng.integers(len(groups), size=batch_size)
    negative_classes = rng.integers(len(groups) - 1, size=batch_size)
    negative_classes += negative_classes >= anchor_classes
    anchor = np.array([groups[c][rng.integers(len(groups[c]))] for c in anchor_classes])
    positive = np.array([groups[c][rng.integers(len(groups[c]))] for c in anchor_classes])
    negative = np.array([groups[c][rng.integers(len(groups[c]))] for c in negative_classes])
    return anchor, positive, negative


def triplet_loss(anchor, positive, negative, *, margin=.1):
    return F.relu((anchor - positive).square().sum(1) - (anchor - negative).square().sum(1) + margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Source-only prepared spectrogram directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026092701)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    source, root = Path(args.source), Path(args.output)
    preparation = json.loads((source / 'preparation.json').read_text(encoding='utf8'))
    if preparation.get('count') != 30000:
        raise ValueError('source preparation incomplete')
    x = np.load(source / 'spectrograms.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(source / 'labels.npy', allow_pickle=False)
    if x.shape != (30000, 102, 62, 1) or y.shape != (30000,) or not np.all(np.bincount(y) == 1000):
        raise ValueError('source data do not match author training configuration')
    train_indices, validation_indices = train_test_split(np.arange(len(y)), test_size=.1,
                                                          random_state=args.seed, shuffle=True)
    groups = [[indices[y[indices] == class_id] for class_id in range(30)]
              for indices in (train_indices, validation_indices)]
    if any(len(group) == 0 for split in groups for group in split):
        raise ValueError('training or validation is missing a source class')
    root.mkdir(parents=True, exist_ok=False)
    config = {'source': str(source.resolve()), 'seed': args.seed, 'batch_size': 32,
              'max_epochs': 1000, 'early_stop_patience': 20, 'reduce_lr_patience': 10,
              'reduce_lr_factor': .2, 'optimizer': 'PyTorch RMSprop rho=0.9 eps=1e-7',
              'learning_rate': .001, 'triplet_margin': .1, 'validation_fraction': .1,
              'train_steps_per_epoch': len(train_indices) // 32,
              'validation_steps_per_epoch': len(validation_indices) // 32,
              'compatibility_note': 'Author code uses Keras, 20-epoch stopping patience and no fixed seed; '
                                    'this port uses PyTorch and a fixed seed. Optimizer kernels may differ. '
                                    'Author train/validation generators share mutable data state; '
                                    'this port keeps the two source splits separate.'}
    (root / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = LoRaExtractor(height=102, width=62).to(device)
    torch.save({'model': model.state_dict(), 'config': config}, root / 'initial.pt')
    model.load_state_dict(torch.load(root / 'initial.pt', map_location=device, weights_only=False)['model'])
    model.eval()
    with torch.no_grad():
        smoke = torch.tensor(np.asarray(x[:2]), device=device).permute(0, 3, 1, 2).contiguous()
        out = model(smoke)
    if out.shape != (2, 512) or not bool(torch.isfinite(out).all()):
        raise ValueError('source-only initial checkpoint smoke failed')
    (root / 'smoke.json').write_text(json.dumps({'status': 'PASS', 'source_records': 2,
                                                'query_records': 0}), encoding='utf8')
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.001, alpha=.9, eps=1e-7)
    random_streams = [np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1)]
    best, early_stale, lr_stale = float('inf'), 0, 0
    with (root / 'epochs.jsonl').open('x', encoding='utf8') as log:
        for epoch in range(1, 1001):
            started, losses = time.perf_counter(), []
            for is_train in (True, False):
                model.train(is_train)
                position = 0 if is_train else 1
                step_count = config['train_steps_per_epoch' if is_train else 'validation_steps_per_epoch']
                total = 0.
                for _ in range(step_count):
                    picks = sample_triplets(groups[position], 32, random_streams[position])
                    indexes = np.concatenate(picks)
                    batch = torch.tensor(np.asarray(x[indexes]), dtype=torch.float32, device=device)
                    batch = batch.permute(0, 3, 1, 2).contiguous()
                    with torch.set_grad_enabled(is_train):
                        anchor, positive, negative = model(batch).split(32)
                        value = triplet_loss(anchor, positive, negative).mean()
                    if not bool(torch.isfinite(value)):
                        raise FloatingPointError('nonfinite LoRa triplet loss')
                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                        value.backward()
                        checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                        if not checks or not bool(torch.stack(checks).all()):
                            raise FloatingPointError('nonfinite LoRa training gradient')
                        optimizer.step()
                    total += float(value.detach())
                losses.append(total / step_count)
            val = losses[1]
            if val < best:
                best, early_stale, lr_stale = val, 0, 0
            else:
                early_stale += 1
                lr_stale += 1
                if lr_stale >= 10:
                    for group in optimizer.param_groups:
                        group['lr'] *= .2
                    lr_stale = 0
            item = {'epoch': epoch, 'training_loss': losses[0], 'validation_loss': val,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'elapsed_seconds': time.perf_counter() - started}
            log.write(json.dumps(item) + '\n')
            log.flush()
            if early_stale >= 20:
                break
    torch.save({'model': model.state_dict(), 'config': config, 'final_epoch': epoch}, root / 'final.pt')
    (root / 'source_complete.json').write_text(json.dumps({'status': 'SOURCE_TRAINED',
        'final_epoch': epoch, 'best_validation_loss': best, 'prediction_complete': False}), encoding='utf8')


if __name__ == '__main__':
    main()
