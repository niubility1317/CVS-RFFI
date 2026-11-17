"""Score only fixed POSTER predictions with independently held truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.poster_paper_repro.diagnostic_predict import MODEL_NAMES


def score(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.ndim != 1 or truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError('prediction and truth must be aligned nonempty vectors')
    if not np.array_equal(np.unique(truth), np.arange(5)) or not np.isin(prediction, np.arange(5)).all():
        raise ValueError('expected exactly five known classes')
    return {'accuracy': float((prediction == truth).mean()), 'count': int(len(truth)),
            'per_class': {str(label): float((prediction[truth == label] == label).mean()) for label in range(5)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.predictions)
    marker = json.loads((root / 'predictions_fixed.json').read_text(encoding='utf8'))
    if marker.get('status') != 'PREDICTIONS_FIXED' or marker.get('models') != MODEL_NAMES:
        raise ValueError('all model predictions must be fixed before scoring')
    predictions = {name: np.load(root / f'{name}.npy', allow_pickle=False) for name in MODEL_NAMES}
    truth = np.load(args.truth, allow_pickle=False)
    results = {name: score(prediction, truth) for name, prediction in predictions.items()}
    with Path(args.output).open('x', encoding='utf8') as file:
        json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
