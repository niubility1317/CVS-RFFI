"""Independent truth-side scoring after every stage prediction is fixed."""
import argparse
import json
from pathlib import Path
import numpy as np


def score_stage(predicted, truth, *, old_classes, seen_classes):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.ndim != 1 or truth.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError('predictions and truth must be aligned nonempty vectors')
    if seen_classes < 1 or not 0 <= old_classes < seen_classes:
        raise ValueError('invalid number of old and seen classes')
    if not np.array_equal(np.unique(truth), np.arange(seen_classes)):
        raise ValueError('truth does not cover every seen class')
    if not np.issubdtype(predicted.dtype, np.integer) or np.any(predicted < 0) or np.any(predicted >= seen_classes):
        raise ValueError('invalid predicted class')
    correct = predicted == truth
    result = {'accuracy': float(correct.mean()), 'count': int(len(truth))}
    if old_classes:
        old = truth < old_classes
        old_accuracy, new_accuracy = float(correct[old].mean()), float(correct[~old].mean())
        total = old_accuracy + new_accuracy
        result.update(old_accuracy=old_accuracy, new_accuracy=new_accuracy,
                      harmonic=2 * old_accuracy * new_accuracy / total if total else 0.)
    return result


def score_all(predictions, truths, *, base_classes=60, increment_size=10):
    if len(predictions) != 8 or len(truths) != 8:
        raise ValueError('expected base and seven incremental stages')
    rows = []
    for stage, (prediction, truth) in enumerate(zip(predictions, truths)):
        seen = base_classes + stage * increment_size
        row = score_stage(prediction, truth, old_classes=0 if stage == 0 else seen - increment_size,
                          seen_classes=seen)
        rows.append({'stage': stage, 'seen_classes': seen, **row})
    return {'stages': rows, 'final_accuracy': rows[-1]['accuracy'],
            'mean_increment_accuracy': float(np.mean([row['accuracy'] for row in rows[1:]])),
            'mean_increment_harmonic': float(np.mean([row['harmonic'] for row in rows[1:]]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.predictions)
    marker = json.loads((root / 'predictions_complete.json').read_text(encoding='utf8'))
    if marker.get('status') != 'PREDICTIONS_FIXED' or marker.get('stages') != 8:
        raise ValueError('all stage predictions must be fixed before scoring')
    predictions = [np.load(root / f'stage_{stage}.npy', allow_pickle=False) for stage in range(8)]
    with np.load(args.truth, allow_pickle=False) as truth_file:
        truths = [truth_file[f'stage_{stage}'] for stage in range(8)]
    result = score_all(predictions, truths)
    with Path(args.output).open('x', encoding='utf8') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
