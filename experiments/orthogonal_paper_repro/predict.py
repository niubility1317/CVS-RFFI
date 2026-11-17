"""Predict from fixed checkpoints and unlabeled inputs; this module never reads truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.orthogonal_incremental_sei.model import SixBlockConv1DEncoder


def predict_records(encoder, weights, inputs, *, device, batch_size=256):
    if batch_size <= 0 or weights.ndim != 2 or len(inputs) == 0:
        raise ValueError('invalid input, batch size or classifier weights')
    encoder.eval()
    weights = F.normalize(weights.to(device), dim=1)
    output = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.tensor(np.asarray(inputs[start:start + batch_size]), dtype=torch.float32, device=device)
            z = encoder(x.transpose(1, 2).contiguous())
            if z.ndim != 2 or z.size(1) != weights.size(1) or not bool(torch.isfinite(z).all()):
                raise ValueError('invalid feature shape or nonfinite features')
            output.append((F.normalize(z, dim=1) @ weights.t()).argmax(dim=1).cpu().numpy())
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--inputs', required=True, help='Unlabeled prediction-input NPY')
    parser.add_argument('--stage-counts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    checkpoints = Path(args.checkpoints)
    marker = json.loads((checkpoints / 'training_complete.json').read_text(encoding='utf8'))
    if marker.get('status') != 'TRAINING_COMPLETE' or marker.get('stages') != 8:
        raise ValueError('training is incomplete')
    counts = json.loads(Path(args.stage_counts).read_text(encoding='utf8'))
    inputs = np.load(args.inputs, mmap_mode='r', allow_pickle=False)
    if (len(counts) != 8 or any(not isinstance(x, int) or x <= 0 for x in counts)
            or counts != sorted(set(counts)) or counts[-1] != len(inputs)):
        raise ValueError('invalid prediction stage counts')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for stage, count in enumerate(counts):
        state = torch.load(checkpoints / f'stage_{stage}.pt', map_location=args.device, weights_only=False)
        if state['stage'] != stage or state['weights'].size(0) != 60 + 10 * stage:
            raise ValueError('checkpoint stage or class count mismatch')
        encoder = SixBlockConv1DEncoder(embedding_dim=256).to(args.device)
        encoder.load_state_dict(state['encoder'], strict=True)
        predictions = predict_records(encoder, state['weights'], inputs[:count], device=args.device)
        with (root / f'stage_{stage}.npy').open('xb') as file:
            np.save(file, predictions, allow_pickle=False)
    (root / 'predictions_complete.json').write_text(json.dumps({'status': 'PREDICTIONS_FIXED',
        'stages': 8, 'counts': counts}), encoding='utf8')


if __name__ == '__main__':
    main()
