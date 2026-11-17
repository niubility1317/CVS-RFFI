"""Independent scoring only after predictions are fixed."""
import argparse
import json
from pathlib import Path
import numpy as np


def score(predictions, truth):
    predictions, truth = np.asarray(predictions), np.asarray(truth)
    if predictions.ndim != 1 or truth.ndim != 1 or predictions.shape != truth.shape or not len(truth):
        raise ValueError('prediction and truth vectors are not aligned')
    if not np.array_equal(np.unique(truth), np.arange(30, 40)):
        raise ValueError('truth does not contain the ten predeclared devices')
    if not np.isin(predictions, np.arange(30, 40)).all():
        raise ValueError('prediction contains an unenrolled device')
    result = {'accuracy': float((predictions == truth).mean()), 'count': int(len(truth))}
    result['per_device_accuracy'] = {str(int(device + 1)): float((predictions[truth == device] == device).mean())
                                     for device in np.unique(truth)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.predictions)
    marker = json.loads((root / 'predictions_complete.json').read_text(encoding='utf8'))
    if marker.get('status') != 'PREDICTIONS_FIXED':
        raise ValueError('prediction must be fixed before accessing truth')
    predicted = np.load(root / 'predictions.npy', allow_pickle=False)
    if marker.get('count') != len(predicted):
        raise ValueError('prediction count mismatch')
    truth = np.load(args.truth, allow_pickle=False)
    with Path(args.output).open('x', encoding='utf8') as output:
        json.dump(score(predicted, truth), output, indent=2)


if __name__ == '__main__':
    main()
