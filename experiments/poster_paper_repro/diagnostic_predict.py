"""Predict on unlabeled WiSig cross-day IQ with the frozen POSTER candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.poster_paper_repro.model import PosterHomegrownAuthor
from experiments.pamf_phase2_practical.vendor_extra.li_backbones import PosterHomegrown


MODEL_NAMES = ['author_source', 'legacy_source'] + [
    f'{architecture}_{mode}_K{k}' for k in (5, 10) for architecture, mode in
    [('author', 'paper_last_layer'), ('author', 'repository'), ('legacy', 'legacy_project')]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.models)
    marker = json.loads((root / 'models_fixed.json').read_text(encoding='utf8'))
    if marker.get('status') != 'MODELS_FIXED' or marker.get('models') != len(MODEL_NAMES):
        raise ValueError('not all models are fixed')
    x = np.load(args.query, mmap_mode='r', allow_pickle=False)
    if x.shape != (500, 256, 2):
        raise ValueError('unexpected unlabeled query shape')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name in MODEL_NAMES:
        saved = torch.load(root / f'{name}.pt', map_location=args.device, weights_only=False)
        expected_arch = 'author' if name.startswith('author_') else 'legacy'
        if saved['architecture'] != expected_arch:
            raise ValueError('model architecture binding mismatch')
        cls = PosterHomegrownAuthor if expected_arch == 'author' else PosterHomegrown
        model = cls(5, input_len=256).to(args.device)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                batch = torch.tensor(np.asarray(x[start:start + 128]), device=args.device).transpose(1, 2)
                logits = model(batch.contiguous())
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('nonfinite target logits')
                predictions.append(logits.argmax(1).cpu().numpy())
        np.save(output / f'{name}.npy', np.concatenate(predictions), allow_pickle=False)
    (output / 'predictions_fixed.json').write_text(json.dumps({'status': 'PREDICTIONS_FIXED',
                                                              'models': MODEL_NAMES}), encoding='utf8')


if __name__ == '__main__':
    main()
