"""Select only predeclared base and new-class support records for training."""
import numpy as np


def select_training_records(payload, metadata, *, validation_fraction=.1):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation fraction must be between zero and one')
    labels = [str(label) for label in payload['node_list']]
    if len(set(labels)) != len(labels) or len(labels) != len(payload['data']):
        raise ValueError('ambiguous physical transmitter identifiers')
    data = dict(zip(labels, payload['data']))
    all_rows = list(metadata['base']) + [row for session in metadata['increments'] for row in session]
    if len({row['id'] for row in all_rows}) != len(all_rows):
        raise ValueError('a physical class cannot be used in multiple sessions')

    def select(row):
        indices = np.asarray(row['indices'], dtype=np.int64)
        if row['id'] not in data:
            raise ValueError('unknown physical class')
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError('duplicate or invalid indices')
        if not len(indices) or indices.min() < 0 or indices.max() >= len(data[row['id']]):
            raise ValueError('record index out of bounds')
        return np.asarray(data[row['id']][indices], dtype=np.float32)

    train_x, train_y, val_x, val_y = [], [], [], []
    for class_id, row in enumerate(metadata['base']):
        records = select(row)
        n_val = max(1, int(len(records) * validation_fraction))
        if n_val >= len(records):
            raise ValueError('class has no training records after validation split')
        train_x.append(records[:-n_val])
        train_y.append(np.full(len(records) - n_val, class_id, dtype=np.int64))
        val_x.append(records[-n_val:])
        val_y.append(np.full(n_val, class_id, dtype=np.int64))
    output = {'base_train_x': np.concatenate(train_x), 'base_train_y': np.concatenate(train_y),
              'base_validation_x': np.concatenate(val_x), 'base_validation_y': np.concatenate(val_y)}
    for session_id, session in enumerate(metadata['increments'], 1):
        values = [select(row) for row in session]
        output[f'session_{session_id}_x'] = np.concatenate(values)
        output[f'session_{session_id}_y'] = np.concatenate([
            np.full(len(value), class_id, dtype=np.int64) for class_id, value in enumerate(values)])
    return output
