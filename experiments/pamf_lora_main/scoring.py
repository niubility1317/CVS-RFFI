"""Read official test labels only after verifying all LoRa predictions."""
import argparse
from pathlib import Path
from statistics import mean, stdev
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import CLASSES, TARGET, VIEWS, label_ids, packet_indices, recording
from .recognition import METHODS, SEEDS


def readouts(extended=()):
    return {**{method: ('raw', 'dura_lora') if method == 'pamf_adaptive' else ('raw',) for method in METHODS},
            **{method: ('raw', 'dura_lora') for method in extended}}


def prediction_matrix(root, methods):
    result = {}
    for seed in SEEDS:
        for method, modes in methods.items():
            path = root/f'{method}-s{seed}'
            state = read(path/'complete.json')
            if (state['status'] != 'PREDICTIONS_FROZEN' or state['target_truth_read']
                    or state['receivers'] != list(TARGET) or state['views'] != list(VIEWS)
                    or state['modes'] != list(modes) or state['per_receiver'] != 2000):
                raise ValueError('Incomplete or unexpected prediction row')
            for mode in modes:
                for view in VIEWS:
                    for rx in TARGET:
                        pred = np.load(path/f'{rx}_{view}_{mode}.npy', allow_pickle=False)
                        if (pred.shape != (2000,) or pred.dtype.kind not in 'iu'
                                or pred.min() < 0 or pred.max() >= len(CLASSES)):
                            raise ValueError('Malformed LoRa test decisions')
                        result[seed, method, mode, view, rx] = pred
    return result


def macro_f1(y, pred):
    k = len(CLASSES)
    matrix = np.bincount(y*k + pred, minlength=k*k).reshape(k, k)
    count = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.mean(2 * matrix.diagonal() / np.maximum(1, count)))


def score(predictions, dataset, output, scratch, *, extended=()):
    methods = readouts(extended)
    p = prediction_matrix(predictions, methods)
    truth = {}
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', scratch) as h5:
                truth[rx] = label_ids(h5, 2000)[packet_indices(2000, i)]

    overall_y = np.concatenate(list(truth.values()))
    res = []
    for seed in SEEDS:
        for method, modes in methods.items():
            for mode in modes:
                for view in VIEWS:
                    by_rx = {rx: p[seed, method, mode, view, rx] for rx in TARGET}
                    selected = np.concatenate(list(by_rx.values()))
                    correct = selected == overall_y
                    pl = np.concatenate([p[seed, 'pamf_pl', 'raw', view, rx] for rx in TARGET])
                    pl_correct = pl == overall_y
                    row = dict(seed=seed, method=method, readout=mode, view=view, count=len(correct),
                        accuracy=float(correct.mean()), macro_f1=macro_f1(overall_y, selected),
                        vs_pl_gain=int((correct & ~pl_correct).sum()), vs_pl_damage=int((~correct & pl_correct).sum()),
                        by_receiver={rx: dict(accuracy=float((a == truth[rx]).mean()),
                                    macro_f1=macro_f1(truth[rx], a), count=len(a)) for rx, a in by_rx.items()})
                    if mode == 'dura_lora':
                        direct = np.concatenate([p[seed, method, 'raw', view, rx] for rx in TARGET]) == overall_y
                        row.update(vs_direct_gain=int((correct & ~direct).sum()),
                                   vs_direct_damage=int((~correct & direct).sum()))
                    res.append(row)
    groups = {}
    for row in res:
        key = f"{row['method']}|{row['readout']}|{row['view']}"
        groups.setdefault(key, []).append(row['accuracy'])
    summary = {k: dict(mean=mean(a), sample_sd=stdev(a)) for k,a in groups.items()}
    write(output, dict(status='SCORED_COMPLETE', truth_last=True, rows=res, three_seed_accuracy=summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    score(args.predictions, args.dataset, args.output, args.scratch)
