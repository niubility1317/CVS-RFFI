"""Freeze source and support-adapted POSTER models without query input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.poster_paper_repro.fit import fit_source, fit_support
from experiments.poster_paper_repro.model import PosterHomegrownAuthor
from experiments.pamf_phase2_practical.vendor_extra.li_backbones import PosterHomegrown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026092701)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = {'purpose': 'WiSig 256-point cross-day diagnostic, not POSTER paper data',
              'source_train_per_class': 200, 'source_validation_per_class': 40,
              'support_per_class': [5, 10], 'classes': 5, 'source_epochs_max': 50,
              'source_early_stop_patience': 10, 'support_epochs': 10,
              'support_validation_fraction': .1, 'batch_size': 256, 'optimizer': 'Adam eps=1e-7',
              'seed': args.seed, 'author_runtime': 'compile-before-freeze behavior inferred from Keras 2.2.4 source',
              'paper_last_layer': 'retained last layer; paper text interpretation'}
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    with np.load(args.training, allow_pickle=False) as data:
        def values(name):
            return torch.from_numpy(data[name + '_x']).transpose(1, 2).contiguous(), torch.from_numpy(data[name + '_y']).long()
        train_x, train_y = values('train')
        val_x, val_y = values('val')
        support_x, support_y = values('support')
    if len(train_y) != 1000 or len(val_y) != 200 or len(support_y) != 50:
        raise ValueError('unexpected fixed diagnostic data size')
    initial_models = {}
    for architecture, cls in [('author', PosterHomegrownAuthor), ('legacy', PosterHomegrown)]:
        torch.manual_seed(args.seed)
        model = cls(5, input_len=256).to(args.device)
        initial = root / f'{architecture}_initial.pt'
        torch.save(model.state_dict(), initial)
        model.load_state_dict(torch.load(initial, map_location=args.device, weights_only=False))
        model.eval()
        with torch.no_grad():
            output = model(train_x[:2].to(args.device))
        if output.shape != (2, 5) or not bool(torch.isfinite(output).all()):
            raise ValueError('real source checkpoint smoke failed')
        initial_models[architecture] = model
    (root / 'source_smoke.json').write_text(json.dumps({'status': 'PASS', 'source_records_per_model': 2,
                                                       'query_records': 0}), encoding='utf8')
    sources = {}
    for architecture, model in initial_models.items():
        source, history = fit_source(model, train_x, train_y, val_x, val_y, seed=args.seed)
        sources[architecture] = source
        torch.save({'architecture': architecture, 'model': source.state_dict()}, root / f'{architecture}_source.pt')
        (root / f'{architecture}_source_history.json').write_text(json.dumps(history), encoding='utf8')
        print(f'{architecture} source trained; epochs={len(history)}', flush=True)
    for k in (5, 10):
        indices = torch.cat([torch.where(support_y == label)[0][:k] for label in range(5)])
        if len(indices) != 5 * k:
            raise ValueError('support is missing a class')
        for architecture, mode in [('author', 'paper_last_layer'), ('author', 'repository'),
                                   ('legacy', 'legacy_project')]:
            adapted, history = fit_support(sources[architecture], support_x[indices], support_y[indices],
                                           mode=mode, seed=args.seed)
            name = f'{architecture}_{mode}_K{k}'
            torch.save({'architecture': architecture, 'model': adapted.state_dict()}, root / f'{name}.pt')
            (root / f'{name}_history.json').write_text(json.dumps(history), encoding='utf8')
            print(f'{name} fitted', flush=True)
    (root / 'models_fixed.json').write_text(json.dumps({'status': 'MODELS_FIXED', 'models': 8}), encoding='utf8')


if __name__ == '__main__':
    main()
