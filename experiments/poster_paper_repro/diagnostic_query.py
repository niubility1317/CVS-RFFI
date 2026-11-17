"""Pack fixed target query inputs and independent truth after models are frozen."""
import argparse
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    marker = json.loads((Path(args.models) / 'models_fixed.json').read_text(encoding='utf8'))
    if marker.get('status') != 'MODELS_FIXED' or marker.get('models') != 10:
        raise ValueError('models must be fixed before accessing query records')
    metadata = json.loads(Path(args.indices).read_text(encoding='utf8'))
    with zipfile.ZipFile(metadata['archive']) as z, z.open(metadata['member']) as file:
        payload = pickle.load(file)
    ids = list(map(str, payload['node_list']))
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate physical ID')
    data = dict(zip(ids, payload['data']))
    xs, labels = [], []
    if len(metadata['classes']) != 5 or len(metadata['support']) != 5:
        raise ValueError('five classes are required')
    for label, (query, support) in enumerate(zip(metadata['classes'], metadata['support'])):
        if query['id'] != support['id'] or set(query['query']) & set(support['support']):
            raise ValueError('query overlaps support or physical IDs differ')
        indices = np.asarray(query['query'], dtype=np.int64)
        if len(indices) != 100 or len(np.unique(indices)) != 100 or indices.min() < 0 or indices.max() >= len(data[query['id']]):
            raise ValueError('invalid query indices')
        xs.append(np.asarray(data[query['id']][indices], dtype=np.float32))
        labels.append(np.full(100, label, dtype=np.int64))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / 'query_input.npy', np.concatenate(xs), allow_pickle=False)
    np.save(root / 'truth_side_labels.npy', np.concatenate(labels), allow_pickle=False)


if __name__ == '__main__':
    main()
