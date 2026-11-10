"""Truth-last class confusion and paired changes for frozen PAMF predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_route.paired_report import READOUTS, TASKS, VIEWS, prepare_row


def analyze(root, rows):
    results = []
    for row in rows:
        cfg, entries = prepare_row(root, row)
        classes = cfg['classes']
        seed = cfg['model_seed']
        populations = {}
        for task in TASKS:
            truth = read(cfg['tests'][task]['truth'])
            predictions = {}
            ids = None
            for name, (path, overall) in entries.items():
                with np.load(path / task / 'predictions.npz', allow_pickle=False) as data:
                    current = data['ids'].copy()
                    if ids is None:
                        ids = current
                        if len(np.unique(ids)) != len(ids):
                            raise ValueError('Physical IDs must be unique')
                    elif not np.array_equal(ids, current):
                        raise ValueError('Physical IDs must match across frozen predictions')
                    predictions[name] = {(route, view): data[f'{route}__{view}'].copy()
                                         for route in READOUTS for view in VIEWS}
                labels = np.array([truth[str(i)]['label'] for i in ids], dtype=np.int64)
                for (route, view), values in predictions[name].items():
                    score = overall[task, route, view]
                    if score['count'] != len(ids) or int((values == labels).sum()) != round(score['accuracy'] * len(ids)):
                        raise ValueError('Frozen predictions do not match independent scores')
            populations[task] = (ids, labels, predictions)
        if np.intersect1d(populations[TASKS[0]][0], populations[TASKS[1]][0]).size:
            raise ValueError('Combined target populations must be disjoint')
        labels = np.concatenate([populations[t][1] for t in TASKS])
        for route in READOUTS:
            for view in VIEWS:
                preds = {name: np.concatenate([populations[t][2][name][route, view] for t in TASKS])
                         for name in entries}
                matrices = {}
                for name, values in preds.items():
                    if np.any((values < 0) | (values >= len(classes))):
                        raise ValueError('Prediction outside registered classes')
                    matrices[name] = np.bincount(labels * len(classes) + values,
                                                 minlength=len(classes) ** 2).reshape(len(classes), -1).tolist()
                transitions = {}
                for left, right in (('candidate', 'PL'), ('candidate', 'Tri-SIG-TIME'), ('Tri-SIG-TIME', 'PL')):
                    masks = []
                    for label in range(len(classes)):
                        selected = labels == label
                        a, b = preds[left], preds[right]
                        masks.append(dict(count=int(selected.sum()),
                            fixed=int((selected & (a == label) & (b != label)).sum()),
                            broken=int((selected & (a != label) & (b == label)).sum())))
                    transitions[left + '_vs_' + right] = masks
                results.append(dict(seed=seed, row=row, route=route, view=view, classes=classes,
                                    count=len(labels), confusion=matrices, paired=transitions))
    return dict(status='CONFUSION_COMPLETE', scope='combined_disjoint_cross_rx_cross_both',
                truth_last=True, rows=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--row', action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root, args.row), allow_nan=False))
