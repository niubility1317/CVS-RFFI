"""Truth-side sample packing, executable only after all checkpoints are frozen."""
import argparse
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np


def select_query_records(payload, train, truth):
    labels = [str(value) for value in payload['node_list']]
    if len(set(labels)) != len(labels) or len(labels) != len(payload['data']):
        raise ValueError('ambiguous transmitter metadata')
    data = dict(zip(labels, payload['data']))
    sessions = [train['base']] + train['increments']
    rows = [row for session in sessions for row in session]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate class in training metadata')
    inputs, label_chunks, counts, truth_rows = [], [], [], []
    total = 0
    class_id = 0
    for session in sessions:
        for row in session:
            physical = row['id']
            if physical not in data or physical not in truth['classes']:
                raise ValueError('missing test class')
            indices = np.asarray(truth['classes'][physical], dtype=np.int64)
            if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
                raise ValueError('invalid or duplicate test indices')
            if indices.min() < 0 or indices.max() >= len(data[physical]):
                raise ValueError('test index out of bounds')
            if set(indices.tolist()).intersection(row['indices']):
                raise ValueError('test and training indices overlap')
            inputs.append(np.asarray(data[physical][indices], dtype=np.float32))
            label_chunks.append(np.full(len(indices), class_id, dtype=np.int64))
            total += len(indices)
            class_id += 1
        counts.append(total)
        truth_rows.append(np.concatenate(label_chunks).copy())
    return np.concatenate(inputs), counts, truth_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-output', required=True)
    parser.add_argument('--training-indices', required=True)
    parser.add_argument('--truth-indices', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training_root = Path(args.training_output)
    marker = json.loads((training_root / 'training_complete.json').read_text(encoding='utf8'))
    if marker.get('status') != 'TRAINING_COMPLETE' or marker.get('stages') != 8:
        raise ValueError('all checkpoints must be frozen before test data is opened')
    if not all((training_root / f'stage_{i}.pt').is_file() for i in range(8)):
        raise ValueError('a frozen checkpoint is missing')
    train = json.loads(Path(args.training_indices).read_text(encoding='utf8'))
    truth = json.loads(Path(args.truth_indices).read_text(encoding='utf8'))
    if train['source'] != truth['source']:
        raise ValueError('training and truth source identities differ')
    source = train['source']
    with zipfile.ZipFile(source['archive']) as archive, archive.open(source['member']) as file:
        payload = pickle.load(file)
    inputs, counts, labels = select_query_records(payload, train, truth)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / 'prediction_input.npy', inputs, allow_pickle=False)
    (root / 'stage_counts.json').write_text(json.dumps(counts), encoding='utf8')
    with (root / 'truth_side.npz').open('xb') as file:
        np.savez(file, **{f'stage_{i}': row for i, row in enumerate(labels)})


if __name__ == '__main__':
    main()
